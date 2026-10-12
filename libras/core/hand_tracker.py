"""Rastreamento de mao em frames BGR: detector de palma -> ROI -> landmarks.

Por que nao usar `mediapipe` direto: a Tasks API mata o processo neste
ambiente durante `HandLandmarker.create_from_options` (SIGKILL ao criar o
delegate XNNPACK -- exit=137, tanto no delegate default quanto no CPU
explicitado). Os mesmos `.tflite` do bundle rodam bem pelo `ai-edge-litert`,
entao o pipeline do MediaPipe foi reimplementado aqui em duas etapas:

  1. `hand_detector.tflite` (192x192) acha a PALMA: caixa + keypoints.
  2. `hand_landmarks_detector.tflite` (224x224) roda sobre o recorte da palma,
     girado para o eixo punho -> metacarpal do medio ficar vertical.

A etapa 2 e o que decide tudo. O landmarker foi treinado com os recortes que o
detector de palma produz, e medir a diferenca e o que separa as duas hipoteses que
explicavam o app nunca achar a mao:

    recorte arbitrario (grade de busca, 96-256 px)   presenca 0.00-0.43
    ROI da palma, girada, 2.6x a caixa                presenca 0.89-0.99

A presenca e a cabeca `presence` do landmarker, com limiar 0.55. Ela e o que
distingue "achou a mao" de "achou um boneco com 21 pontos".

Decodificacao do detector de palma (medida neste modelo, nao documentada):
  * saida (2016, 18) = caixa e keypoints por ancora; (2016, 1) = logit de score
  * as 1152 primeiras ancoras formam uma grade de 48 colunas x 24 linhas,
    espacadas 4 px em x e 8 px em y no canvas de 192, com `index = linha*48 +
    coluna`. Medido varrendo uma mao real pelo frame: o indice do ancor mais
    forte acompanha a posicao com 32 px de grade, sem excecao.
  * canais 0..3 = (ymin, xmin, ymax, xmax), em px a partir do centro da ancora
  * canais 4..13 = pulso, MCP do indicador, do medio, do anelar e do mininho
    (erro medio de 1.5-4.5 px contra os landmarks do landmarker, em 12 poses
    giradas em 0/90/180/270 graus)

A rotacao vem do vetor punho -> MCP do medio, e importa: com a ROI quadrada e
sem girar, a presenca cai de 0.99 para 0.16-0.31.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import ensure_models

NUM_LANDMARKS = 21
LANDMARK_CANVAS = 224
PALM_CANVAS = 192

# Indices dos landmarks da mao (padrao MediaPipe)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_CHAINS = {
    "thumb": (THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGERS_4 = ("index", "middle", "ring", "pinky")

# grade de ancoras do detector de palma (medida: ver docstring do modulo)
PALM_GRID_COLS = 48
PALM_GRID_ROWS = 24
PALM_GRID_STEP_X = 4.0
PALM_GRID_STEP_Y = 8.0
PALM_GRID_SIZE = PALM_GRID_COLS * PALM_GRID_ROWS


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angulo em graus no ponto b, entre os segmentos b->a e b->c."""
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


@dataclass
class Hand:
    """Uma mao rastreada; landmarks em coordenadas normalizadas da imagem."""

    landmarks: np.ndarray  # (21, 3): x, y em [0,1]; z relativo ao pulso
    presence: float
    handedness: float  # <0.5 direita, >0.5 esquerda (convenção do modelo)
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    smooth: float = 0.0  # 0..1, quao estavel a track esta

    # --- escala ---

    @property
    def hand_size(self) -> float:
        """Distancia pulso -> metacarpal do medio, em unidades de imagem."""
        return float(np.linalg.norm(self.landmarks[WRIST, :2] - self.landmarks[MIDDLE_MCP, :2]))

    @property
    def palm_size(self) -> float:
        """Largura da palma: metacarpal do indicador ate do mininho."""
        return float(np.linalg.norm(self.landmarks[INDEX_MCP] - self.landmarks[PINKY_MCP]))

    # --- articulation ---

    def angle(self, a: int, b: int, c: int) -> float:
        return _angle(self.landmarks[a, :2], self.landmarks[b, :2], self.landmarks[c, :2])

    def curl(self, finger: str) -> float:
        """0.0 = estendido, 1.0 = totalmente dobrado (mede a dobradilha distal)."""
        mcp, pip, tip = FINGER_CHAINS[finger]
        return float(np.clip((180.0 - max(self.angle(mcp, pip, tip), 1.0)) / 90.0, 0.0, 1.0))

    def is_extended(self, finger: str) -> bool:
        """Dedo estendido: cadeia distal reta e ponta afastada do pulso."""
        mcp, pip, tip = FINGER_CHAINS[finger]
        return bool(self.angle(mcp, pip, tip) > 150.0 and self.angle(WRIST, mcp, tip) > 135.0)

    def is_curled(self, finger: str) -> bool:
        mcp, pip, tip = FINGER_CHAINS[finger]
        return bool(self.angle(mcp, pip, tip) < 105.0)

    def extended_fingers(self) -> list:
        return [f for f in FINGERS if self.is_extended(f)]

    def n_extended(self, fingers=FINGERS_4) -> int:
        return sum(1 for f in fingers if self.is_extended(f))

    def fist_score(self) -> float:
        return float(np.mean([self.curl(f) for f in FINGERS]))

    # --- forma ---

    def orientation(self) -> float:
        """Angulo pulso -> meio da mao, em graus no plano da imagem."""
        v = self.landmarks[MIDDLE_MCP, :2] - self.landmarks[WRIST, :2]
        return float(math.degrees(math.atan2(v[1], v[0])))

    def center(self) -> np.ndarray:
        return self.landmarks.mean(axis=0)

    @property
    def label(self) -> str:
        return "E" if self.handedness > 0.5 else "D"

    def tip(self, finger: str) -> np.ndarray:
        return self.landmarks[FINGER_CHAINS[finger][2]]

    def mcp(self, finger: str) -> np.ndarray:
        return self.landmarks[FINGER_CHAINS[finger][0]]

    def distance(self, a: str, b: str) -> float:
        return float(np.linalg.norm(self.tip(a) - self.tip(b)))

    def pinch(self) -> float:
        """Distancia polegar-indicador relativa ao tamanho da palma."""
        return self.distance("thumb", "index") / max(self.palm_size, 1e-9)

    def spread(self) -> float:
        """Abertura entre as pontas dos dedos, relativa ao tamanho da palma."""
        d = 0.0
        for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
            d += self.distance(a, b)
        return d / (3.0 * max(self.palm_size, 1e-9))


@dataclass
class Palm:
    """Palma detectada.

    `cx` e `cy` sao pixels da imagem, mas `side` e um lado em px do canvas de
    192, que e um resize esticado do frame. Num frame 640x960 a escala y e 1.5x a
    de x, entao a ROI precisa ser um retangulo (`side * scale_x` por
    `side * scale_y`) e nao um quadrado em px da imagem. `angle` deixa o eixo
    punho -> MCP do medio na vertical.
    """

    cx: float
    cy: float
    side: float          # lado da caixa da palma, em px do canvas de 192
    angle: float         # graus
    score: float
    scale_x: float = 1.0  # px_da_imagem por px_do_canvas, no eixo x
    scale_y: float = 1.0


@dataclass
class _Track:
    """Esqueleto do quadro anterior, para suavizar o tremor da deteccao."""

    landmarks: np.ndarray
    mid: np.ndarray   # ponto medio pulso -> metacarpal do medio
    presence: float
    handedness: float
    hits: int = 1


class HandTracker:
    """Detecta e rastreia maos em frames BGR."""

    def __init__(
        self,
        models_dir: Path | str = "models",
        min_presence: float = 0.55,
        min_landmark: float = 0.30,
        min_palm: float = 0.40,
        max_hands: int = 2,
        roi_expand: float = 2.6,
        smoothing: float = 0.5,
        max_coast: int = 4,
        num_threads: int = 4,
    ):
        from ai_edge_litert.interpreter import Interpreter

        path = ensure_models(Path(models_dir))
        self.min_presence = min_presence
        self.min_landmark = min_landmark
        self.min_palm = min_palm
        self.max_hands = max_hands
        self.roi_expand = roi_expand
        self.smoothing = smoothing
        # quantos quadros sem deteccao a ultima mao ainda e devolvida, para o
        # detector piscar sem picar a leitura (0 desliga a ponte)
        self.max_coast = max_coast

        self._palm_it = Interpreter(model_path=str(path.parent / "hand_detector.tflite"),
                                    num_threads=num_threads)
        self._palm_it.allocate_tensors()
        self._palm_in = self._palm_it.get_input_details()[0]
        palm_outs = self._palm_it.get_output_details()
        self._palm_boxes = next(d for d in palm_outs if d["shape"][-1] == 18)
        self._palm_scores = next(d for d in palm_outs if d["shape"][-1] == 1)

        self._it = Interpreter(model_path=str(path), num_threads=num_threads)
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]
        outs = self._it.get_output_details()
        self._o_pts = next(d for d in outs if len(d["shape"]) == 2 and d["shape"][-1] == 63)
        scalars = [d for d in outs if len(d["shape"]) == 2 and d["shape"][-1] == 1]
        self._o_pres, self._o_hand = scalars[0], scalars[1]

        self._tracks: list = []
        self._last_hands: list = []
        self._misses = 0
        self._frame_shape = (0, 0, 3)

    def close(self) -> None:
        self._it = None
        self._palm_it = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset(self) -> None:
        self._tracks = []
        self._last_hands = []
        self._misses = 0

    # ---------------- deteccao de palma ----------------

    def detect_palms(self, frame: np.ndarray) -> list:
        """Palmas no frame, da mais confiante para a menos.

        A deduplicacao acontece em px do canvas de 192, que e onde as caixas
        sao decodificadas: o detector acende varias ancoras da mesma mao
        (medido: 6 ancoras com prob 0.71-0.91 sobre a mesma mao) e cada uma
        traz uma caixa um pouco diferente.
        """
        import cv2

        h, w = frame.shape[:2]
        blob = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                          (PALM_CANVAS, PALM_CANVAS), interpolation=cv2.INTER_LINEAR)
        self._palm_it.set_tensor(self._palm_in["index"],
                                 np.ascontiguousarray(blob)[None, ...])
        self._palm_it.invoke()
        raw = self._palm_it.get_tensor(self._palm_boxes["index"])[0]
        logits = self._palm_it.get_tensor(self._palm_scores["index"])[0].ravel()

        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        found: list = []
        for i in np.argsort(-prob):
            p = float(prob[i])
            if p < self.min_palm:
                break
            i = int(i)
            if i >= PALM_GRID_SIZE:
                # so a primeira grade foi medida; uma ancora fora dela ficaria
                # sem posicao e viraria recorte no lugar errado
                continue
            row, col = divmod(i, PALM_GRID_COLS)
            acx, acy = col * PALM_GRID_STEP_X, row * PALM_GRID_STEP_Y
            ymin = min(PALM_CANVAS, max(0.0, acy + raw[i, 0]))
            xmin = min(PALM_CANVAS, max(0.0, acx + raw[i, 1]))
            ymax = min(PALM_CANVAS, max(0.0, acy + raw[i, 2]))
            xmax = min(PALM_CANVAS, max(0.0, acx + raw[i, 3]))
            bw, bh = xmax - xmin, ymax - ymin
            if bw < 4 or bh < 4:
                continue
            # angulo que deixa o vetor punho -> MCP do medio apontando para cima
            vx = raw[i, 8] - raw[i, 4]
            vy = raw[i, 9] - raw[i, 5]
            if abs(vx) + abs(vy) < 1e-6:
                continue
            side = max(bw, bh)
            pcx, pcy = (xmin + xmax) / 2, (ymin + ymax) / 2
            if any(math.hypot(pcx - q["cx"], pcy - q["cy"])
                   < 0.5 * max(side, q["side"]) for q in found):
                continue
            found.append({"cx": pcx, "cy": pcy, "side": side,
                          "angle": -math.degrees(math.atan2(vx, -vy)), "score": p})
            if len(found) >= self.max_hands * 3:
                break

        scale_x, scale_y = w / PALM_CANVAS, h / PALM_CANVAS
        return [Palm(cx=d["cx"] * scale_x, cy=d["cy"] * scale_y, side=d["side"],
                     angle=d["angle"], score=d["score"], scale_x=scale_x, scale_y=scale_y)
                for d in found]

    # ---------------- landmarks na ROI ----------------

    def _infer(self, frame, palm: Palm):
        """Roda o landmarker no recorte da palma, alinhado com o antebraço.

        Devolve (presence, handed, landmarks normalizados, caixa em px) ou None.
        A ROI e girada antes do resize, que e o que o MediaPipe faz com a
        regiao de interesse; sem a rotacao a presenca cai para 0.16-0.31.
        """
        import cv2

        h, w = frame.shape[:2]
        self._frame_shape = frame.shape
        # o canvas de 192 e um resize esticado do frame: em 640x960 a escala y
        # e 1.5x a de x, entao a ROI e um retangulo. Um quadrado em px da
        # imagem corta a mao em um dos eixos e a presenca desaba.
        grow = palm.side * self.roi_expand
        rw = int(round(grow * palm.scale_x))
        rh = int(round(grow * palm.scale_y))
        lim = int(0.95 * min(w, h))
        rw = max(16, min(rw, lim))
        rh = max(16, min(rh, lim))
        cx, cy = palm.cx, palm.cy

        m = cv2.getRotationMatrix2D((cx, cy), palm.angle, 1.0)
        m[0, 2] += rw / 2.0 - cx
        m[1, 2] += rh / 2.0 - cy
        crop = cv2.warpAffine(frame, m, (rw, rh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if min(crop.shape[:2]) < 16:
            return None
        blob = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                          (LANDMARK_CANVAS, LANDMARK_CANVAS), interpolation=cv2.INTER_LINEAR)

        self._it.set_tensor(self._in["index"], np.ascontiguousarray(blob)[None, ...])
        self._it.invoke()
        presence = float(self._it.get_tensor(self._o_pres["index"])[0].ravel()[0])
        handed = float(self._it.get_tensor(self._o_hand["index"])[0].ravel()[0])
        pts = self._it.get_tensor(self._o_pts["index"])[0].reshape(-1, 3).astype(np.float32)
        if not np.isfinite(pts).all():
            return None

        px = pts[:, :2] * np.array([rw / LANDMARK_CANVAS, rh / LANDMARK_CANVAS])
        inv = cv2.invertAffineTransform(m)
        xy = (inv @ np.c_[px, np.ones(len(px))].T).T
        lm = np.empty((NUM_LANDMARKS, 3), np.float32)
        lm[:, 0] = xy[:, 0] / w
        lm[:, 1] = xy[:, 1] / h
        lm[:, 2] = pts[:, 2] / LANDMARK_CANVAS

        xs = xy[:, 0]
        ys = xy[:, 1]
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return presence, handed, lm, box

    # ---------------- plausibilidade ----------------

    def _quality(self, lm: np.ndarray) -> float:
        """Nota 0..1 de plausibilidade do esqueleto.

        A `presence` chega a 0.9 com landmarks degenerados: ROI folgada demais
        (expansao 6x) devolve presenca 0.46 e a mao fantasma, e ROI apertada
        demais devolve presenca 0.09 com o esqueleto todo amontoado. Sem esta
        checagem a frase enche de letras inventadas.

        Todas as razoes sao medidas em pixels, nao em coordenadas normalizadas:
        normalizar x por W e y por H mistura escalas diferentes e torna o
        limiar dependente do aspecto do frame.
        """
        h, w = self._frame_shape[:2]
        px = lm[:, :2] * np.array([w, h], np.float32)
        span = float(np.ptp(px[:, 0]) + np.ptp(px[:, 1]))
        if span < 1e-6:
            return 0.0
        size = float(np.linalg.norm(px[WRIST] - px[MIDDLE_MCP]))
        palm = float(np.linalg.norm(px[INDEX_MCP] - px[PINKY_MCP]))
        if size < 1e-3 or palm < 1e-3:
            return 0.0

        # largura da palma / comprimento da palma: 1.09-1.14 na foto de teste
        # (duas maos, presenca 0.89-0.98). O limite superior e absoluto: palma
        # maior que 2.2x o comprimento nao sai de nenhuma vista legitima.
        # O limite inferior NAO pode ser absoluto: numa mao de perfil (C, O,
        # sinal visto do lado) a largura projetada encolhe por geometria, nao
        # por degeneracao -- chega a ~0.3 em perfil, e vetar isso e o que fazia
        # o app nao achar a mao ao sinalizar C de lado.
        pr = palm / size
        score = 1.0
        if pr > 2.2:
            score -= 1.0
        elif pr > 1.8:
            score -= 0.3

        # pulso -> metacarpal do medio sobre a extensao do esqueleto: 0.21-0.25
        # na mao real. So o limite inferior vale a pena: em punho fechado o
        # esqueleto encolhe e a razao sobe, entao um teto aqui rejeitaria o proprio
        # punho que queremos ler.
        ratio = size / span
        if ratio < 0.05:
            score -= 1.0
        elif ratio < 0.08:
            score -= 0.3

        # o pulso -> meio do indicador tem comprimento parecido com a palma
        idx = float(np.linalg.norm(px[WRIST] - px[INDEX_MCP])) / size
        if idx < 0.5 or idx > 1.35:
            score -= 0.5

        # ossos vizinhos nao podem diferir em mais de ~12x (o dobro do medido)
        bones = [(WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_TIP),
                 (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_TIP), (PINKY_MCP, PINKY_TIP),
                 (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_TIP)]
        lens = [float(np.linalg.norm(px[a] - px[b])) for a, b in bones]
        if min(lens) < 1e-3:
            return 0.0
        rb = max(lens) / min(lens)
        if rb > 12.0:
            score -= 0.8
        elif rb > 6.0:
            score -= 0.3

        # dedos nao sao uma regua: matriz rastreada, so penaliza muito
        for base, pip, tip in ((INDEX_MCP, INDEX_PIP, INDEX_TIP),
                               (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
                               (PINKY_MCP, PINKY_PIP, PINKY_TIP)):
            b = float(np.linalg.norm(px[base] - px[pip]))
            t = float(np.linalg.norm(px[pip] - px[tip]))
            if b < 1e-3:
                return 0.0
            r = t / b
            if r > 1.6:
                score -= 0.3

        # as 4 falanges proximais sao praticamente paralelas mesmo em punho
        # fechado (o que dobra e a partir do PIP): coseno minimo de +0.99 na
        # foto de teste contra <= +0.06 em 21 pontos jogados ao acaso. E o que
        # separa ruido do detector de um esqueleto de verdade.
        pairs = ((INDEX_MCP, INDEX_PIP), (MIDDLE_MCP, MIDDLE_PIP),
                 (RING_MCP, RING_PIP), (PINKY_MCP, PINKY_PIP))
        dirs = np.array([px[b] - px[a] for a, b in pairs], np.float32)
        norms = np.linalg.norm(dirs, axis=1)
        if norms.min() < 1e-3:
            return 0.0
        unit = dirs / norms[:, None]
        cos = np.clip((unit @ unit.T)[np.triu_indices(4, 1)], -1.0, 1.0)
        if cos.min() < 0.0:
            score -= 0.8
        elif cos.min() < 0.3:
            score -= 0.3

        # a mao nao pode estar metade fora do quadro
        out = ((lm[:, 0] < -0.05) | (lm[:, 0] > 1.05) |
               (lm[:, 1] < -0.05) | (lm[:, 1] > 1.05)).mean()
        if out > 0.3:
            score -= 0.6

        # largura projetada pequena: so derruba a nota quando o esqueleto inteiro
        # ja mostrou degeneracao (os bem conhecidos recortes folgados combinavam
        # largura pequena e geometria torta). Sozinha, largura pequena significa
        # que a mao esta de perfil -- e o que o app precisa ler em C e O.
        if pr < 0.45 and score < 0.75:
            score -= 0.6
        elif pr < 0.12:
            score -= 0.4
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _landmarks_overlap(a, b, tol: float = 0.02) -> float:
        """Fracao dos landmarks de `a` que caem perto de algum ponto de `b`."""
        d = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
        return float((d.min(axis=1) <= tol).mean())

    # ---------------- ciclo principal ----------------

    def process(self, frame: np.ndarray) -> list:
        self._frame_shape = frame.shape
        palms = self.detect_palms(frame)

        hands: list = []
        if palms:
            found: list = []
            for p in palms:
                out = self._infer(frame, p)
                if out is None:
                    continue
                presence, handed, lm, box = out
                if presence < self.min_landmark:
                    continue
                q = self._quality(lm)
                if q < 0.5:
                    continue
                found.append((p, presence, handed, lm, box, q))

            # a mesma mao pode ser achada por varias palmas (duas ancoras com
            # caixas parecidas): fica a de maior presenca
            found.sort(key=lambda r: -r[1])
            for p, presence, handed, lm, box, q in found:
                if any(self._landmarks_overlap(lm, h.landmarks) > 0.5 for h in hands):
                    continue
                hands.append(Hand(lm, presence, handed, box, 0.0))
                if len(hands) >= self.max_hands:
                    break

            self._smooth(hands)
            hands.sort(key=lambda hd: -hd.presence)

        if hands:
            self._misses = 0
            self._last_hands = hands
            return hands

        # Nada valido neste quadro. O detector de palma e o `_quality` piscam
        # (a mao some por um quadro, a pose de perfil cai abaixo do limiar), e
        # devolver [] picaria a leitura. Segura a ultima mao por alguns quadros,
        # com a presenca decaindo, para o vazio virar espaco so quando for de
        # verdade.
        self._misses += 1
        if self._last_hands and self._misses <= self.max_coast:
            decay = 1.0 - self._misses / (self.max_coast + 1.0)
            return [Hand(h.landmarks.copy(), float(h.presence * decay),
                         h.handedness, h.box, h.smooth) for h in self._last_hands]
        self._tracks = []
        self._last_hands = []
        return []

    def _smooth(self, hands: list) -> None:
        """Suaviza os landmarks contra a track do quadro anterior.

        A deteccao de palma roda em todo frame, entao nao ha track morta para
        manter: o que a track guarda e o esqueleto anterior, so para tirar o
        tremor que viraria jitter nas features. Uma mao que some da tela por
        alguns quadros volta sem historico, e `smooth` volta a 0.
        """
        tracks: list = []
        for h in hands:
            mid = (h.landmarks[WRIST] + h.landmarks[MIDDLE_MCP]) / 2
            best, best_d = None, 1e9
            for tr in self._tracks:
                d = float(np.linalg.norm(tr.mid - mid))
                if d < best_d:
                    best, best_d = tr, d
            if best is not None and best_d < 0.12:
                h.landmarks = (self.smoothing * best.landmarks
                               + (1.0 - self.smoothing) * h.landmarks).astype(np.float32)
                hits = best.hits + 1
            else:
                hits = 1
            h.smooth = min(1.0, hits / 5.0)
            tracks.append(_Track(h.landmarks, mid, h.presence, h.handedness,
                                 hits=hits))
        self._tracks = tracks
