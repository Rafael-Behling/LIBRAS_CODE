"""Vetores de caracteristicas para reconhecimento de sinais.

Duas escalas, de proposito:
  * `HandFeatures.geometry`  -- foto da pose atual, invariante a posicao e
    escala no frame. E o que descreve a maioria dos sinais estaticos.
  * `MotionFeatures`         -- como a pose mudou, em unidades de articulacao
    por segundo. E o que separa sinais com a mesma pose inicial e final
    (movimento circular, oscillatorio).

Todas as medidas sao derivadas dos 21 landmarks, entao valem tanto para
datilologia quanto para sinais do vocabulario.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .hand_tracker import (
    FINGERS,
    FINGERS_4,
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    Hand,
)


# Sequencias dos dedos, da base a ponta, para as diferencas de segunda ordem
_CHAIN4 = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP, None),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, None),
    "ring": (RING_MCP, None, RING_TIP, None),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP, None),
}

_TIP_OF = {"index": INDEX_TIP, "middle": MIDDLE_TIP, "ring": RING_TIP, "pinky": PINKY_TIP}


@dataclass
class HandFeatures:
    """Pose de uma mao, independente de onde ela esta no frame."""

    # dobradilha de cada dedo, 0 estendido -> 1 dobrado
    curl: np.ndarray               # (5,) thumb, index, middle, ring, pinky
    # angulo de cada articulacao, em graus
    angles: np.ndarray             # (4, 2) PIP e DIP de index..pinky
    thumb_angles: np.ndarray       # (2,) IP e spread
    # posicao de cada ponta relativa ao pulso, em unidades de tamanho de mao
    tips: np.ndarray               # (5, 2)
    # distancia de cada ponta ao pulso, normalizada
    tip_radius: np.ndarray         # (5,)
    # posicao de cada ponta no referencial da mao (eixo x = indicador->mininho)
    tips_local: np.ndarray         # (5, 2)
    # orientacao da mao e inclinacao do plano
    orientation: float             # graus, pulso -> meio da mao
    plane_tilt: float              # graus, quanto a palma esta de frente para a camera
    # comprimento de cada dedo em unidades de tamanho de mao
    finger_len: np.ndarray         # (5,)

    def vector(self) -> np.ndarray:
        """Vetor plano para classificadores lineares / MLP."""
        return np.concatenate([
            self.curl,
            self.angles.ravel() / 180.0,
            self.thumb_angles / 180.0,
            self.tips.ravel(),
            self.tip_radius,
            self.tips_local.ravel(),
            [math.sin(math.radians(self.orientation)),
             math.cos(math.radians(self.orientation)),
             math.sin(math.radians(self.plane_tilt)),
             math.cos(math.radians(self.plane_tilt))],
            self.finger_len,
        ]).astype(np.float32)

    @staticmethod
    def names() -> list:
        n = []
        n += [f"curl_{f}" for f in FINGERS]
        n += [f"ang_{f}_{j}" for f in FINGERS_4 for j in ("pip", "dip")]
        n += ["thumb_ip", "thumb_spread"]
        n += [f"tip_{f}_{ax}" for f in FINGERS for ax in ("x", "y")]
        n += [f"tip_r_{f}" for f in FINGERS]
        n += [f"tipl_{f}_{ax}" for f in FINGERS for ax in ("x", "y")]
        n += ["ori_sin", "ori_cos", "tilt_sin", "tilt_cos"]
        n += [f"len_{f}" for f in FINGERS]
        return n


def _flat_features() -> HandFeatures:
    """Vetor neutro, para quando o esqueleto esta degenerado."""
    return HandFeatures(
        curl=np.zeros(5, np.float32), angles=np.zeros((4, 2), np.float32),
        thumb_angles=np.zeros(2, np.float32), tips=np.zeros((5, 2), np.float32),
        tip_radius=np.zeros(5, np.float32), tips_local=np.zeros((5, 2), np.float32),
        orientation=0.0, plane_tilt=0.0, finger_len=np.zeros(5, np.float32),
    )


def hand_features(hand: Hand) -> HandFeatures:
    lm = hand.landmarks
    size = max(hand.hand_size, 1e-6)
    palm = max(hand.palm_size, 1e-6)
    # landmarks vem normalizados pela imagem, entao um crop muito folgado
    # produz medidas absurdas (medido: tips_local de 21.0 numa deteccao com
    # hand_size de 0.010). O limite e frouxo de proposito: a razao
    # pulso/palma varia entre a foto real e o recorte de teste, entao nao serve
    # como invariante. Aqui so cortamos o caso extremo, em que a mao nao esta
    # no quadro de jeito nenhum.
    if size < 0.004 or size > 0.6 or palm > 0.9:
        return _flat_features()

    curl = np.array([hand.curl(f) for f in FINGERS], np.float32)

    angles = np.zeros((4, 2), np.float32)
    for i, f in enumerate(FINGERS_4):
        mcp, pip, tip = FINGER_CHAINS[f]
        angles[i, 0] = hand.angle(mcp, pip, tip)          # dobradilha na PIP
        angles[i, 1] = hand.angle(pip, tip, _TIP_OF[f])  # dobradilha na DIP

    thumb_angles = np.array([
        hand.angle(THUMB_MCP, THUMB_IP, THUMB_TIP),
        hand.angle(WRIST, THUMB_MCP, THUMB_TIP),
    ], np.float32)

    tips = np.array([lm[FINGER_CHAINS[f][2], :2] for f in FINGERS], np.float32)
    tips = (tips - lm[WRIST, :2]) / size

    tip_radius = np.linalg.norm(tips, axis=1).astype(np.float32)

    # referencial da propria mao: x ao longo da palma, y na perpendicular.
    # Isso torna as posicoes das pontas independentes de como a mao esta
    # girada no frame -- um "A" apontando para cima e para baixo tem o mesmo
    # perfil neste referencial.
    ex = (lm[INDEX_MCP, :2] - lm[PINKY_MCP, :2]) / palm
    ey = np.array([-ex[1], ex[0]], np.float32)
    rel = (lm[:, :2] - lm[WRIST, :2]) / size
    tips_local = np.empty((len(FINGERS), 2), np.float32)
    for i, f in enumerate(FINGERS):
        p = rel[FINGER_CHAINS[f][2]]
        tips_local[i, 0] = p[0] * ex[0] + p[1] * ex[1]
        tips_local[i, 1] = p[0] * ey[0] + p[1] * ey[1]

    orientation = hand.orientation()
    # normal do plano da palma via produto vetorial dos eixos da mao
    v1 = lm[MIDDLE_MCP, :2] - lm[WRIST, :2]
    v2 = lm[PINKY_MCP, :2] - lm[INDEX_MCP, :2]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    area = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    plane_tilt = math.degrees(math.asin(float(np.clip(cross / area, -1.0, 1.0)))) if area else 0.0

    finger_len = np.array([
        float(np.linalg.norm(lm[FINGER_CHAINS[f][1], :2] - lm[FINGER_CHAINS[f][0], :2])) / size
        for f in FINGERS
    ], np.float32)

    return HandFeatures(
        curl=curl, angles=angles, thumb_angles=thumb_angles, tips=tips,
        tip_radius=tip_radius, tips_local=tips_local, orientation=orientation,
        plane_tilt=plane_tilt, finger_len=finger_len,
    )


@dataclass
class MotionFeatures:
    """Como a pose mudou, em unidades de articulacao por segundo.

    `delta` e a variacao por frame normalizada; os valores sao pequenos em
    videos com pouca mudanca, e por isso sao multiplicados por 10 para ficar
    na mesma ordem de grandeza das features estaticas.
    """

    velocity: np.ndarray            # (5,) velocidade de dobradilha
    tip_velocity: np.ndarray        # (5, 2) velocidade de cada ponta
    center_velocity: np.ndarray     # (2,) velocidade do centro da mao
    acceleration: np.ndarray        # (5,) segunda derivada da dobradilha
    spread: float                   # 0..1, abertura dos dedos
    path_length: float              # deslocamento acumulado normalizado
    straightness: float             # 0 ida-e-volta, 1 movimento em linha reta

    def vector(self) -> np.ndarray:
        return np.concatenate([
            self.velocity * 10.0,
            self.tip_velocity.ravel() * 10.0,
            self.center_velocity.ravel() * 10.0,
            self.acceleration * 10.0,
            [self.spread, self.path_length, self.straightness],
        ]).astype(np.float32)

    @staticmethod
    def names() -> list:
        n = [f"vel_{f}" for f in FINGERS]
        n += [f"tipvel_{f}_{ax}" for f in FINGERS for ax in ("x", "y")]
        n += ["cvel_x", "cvel_y"]
        n += [f"acc_{f}" for f in FINGERS]
        n += ["spread", "path_len", "straight"]
        return n


def motion_features(prev: HandFeatures, cur: HandFeatures,
                    prev_center: np.ndarray, cur_center: np.ndarray,
                    prev_vel: np.ndarray | None = None,
                    dt: float = 1.0 / 30.0) -> MotionFeatures:
    """Diferencas entre dois quadros consecutivos.

    `prev_vel` e a velocidade do quadro anterior, para a aceleracao; sem ela a
    aceleracao e zero.
    """
    dt = max(dt, 1e-3)
    vel = (cur.curl - prev.curl) / dt
    acc = ((vel - prev_vel) / dt) if prev_vel is not None else np.zeros_like(vel)
    tip_vel = (cur.tips - prev.tips) / dt
    cen_vel = (cur_center - prev_center) / dt

    # abertura: quanto as pontas estao separadas da menor, em unidades de mao
    spread = float(np.clip(cur.tip_radius[1:].mean() - cur.tip_radius[1:].min(), 0.0, 1.0))
    path = float(np.linalg.norm(cur_center - prev_center))

    return MotionFeatures(
        velocity=vel.astype(np.float32),
        tip_velocity=tip_vel.astype(np.float32),
        center_velocity=cen_vel.astype(np.float32),
        acceleration=acc.astype(np.float32),
        spread=spread, path_length=path, straightness=0.0,
    )


@dataclass
class HandHistory:
    """Janela deslizante de pose de uma mao.

    Acumula os ultimos N quadros para que o classificador possa usar
    contexto temporal -- necessario para sinais dinamicos, que so se
    distinguem de outros pela trajetoria.
    """

    maxlen: int = 30
    poses: list = field(default_factory=list)      # HandFeatures
    centers: list = field(default_factory=list)    # xy normalizado
    times: list = field(default_factory=list)      # timestamps

    def push(self, hand: Hand, t: float) -> HandFeatures:
        f = hand_features(hand)
        self.poses.append(f)
        self.centers.append(hand.center()[:2].copy())
        self.times.append(float(t))
        if len(self.poses) > self.maxlen:
            self.poses.pop(0)
            self.centers.pop(0)
            self.times.pop(0)
        return f

    def __len__(self) -> int:
        return len(self.poses)

    def motion(self) -> MotionFeatures | None:
        if len(self.poses) < 2:
            return None
        dt = self.times[-1] - self.times[-2]
        prev_vel = getattr(self, "_last_vel", None)
        m = motion_features(self.poses[-2], self.poses[-1],
                            self.centers[-2], self.centers[-1], prev_vel,
                            dt or 1 / 30.0)
        self._last_vel = m.velocity
        return m

    def displacement(self) -> float:
        """Distancia total percorrida pelo centro da mao."""
        if len(self.centers) < 2:
            return 0.0
        c = np.array(self.centers)
        return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))

    def straightness(self) -> float:
        """1.0 = deslocamento em linha reta, 0.0 = ida e volta.

        Raiz entre a distancia percorrida e a distancia reta entre os
        extremos. Vale para sinais de movimento (ir/vir, subir/descer).
        """
        if len(self.centers) < 3:
            return 0.0
        c = np.array(self.centers)
        path = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        if path < 1e-9:
            return 0.0
        straight_dist = float(np.linalg.norm(c[-1] - c[0]))
        return float(np.clip(2.0 * straight_dist / path, 0.0, 1.0))

    def vector(self) -> np.ndarray:
        """Estatico + movimento + resumo da trajetoria."""
        if not self.poses:
            return np.zeros(0, np.float32)
        parts = [self.poses[-1].vector()]
        m = self.motion()
        if m is not None:
            parts.append(m.vector())
        parts.append(np.array([self.displacement(), self.straightness()], np.float32))
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def names() -> list:
        return HandFeatures.names() + MotionFeatures.names() + ["displacement", "straightness"]
