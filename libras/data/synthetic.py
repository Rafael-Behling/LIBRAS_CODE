"""Gera poses sinteticas de mao para testar as regras de datilologia.

Nao substitui dados reais de Libras -- o alvo e apenas exercitar a
classificacao de forma reprodutivel, ja que nao ha dataset no projeto.
Uma mao de 21 landmarks e montada como uma arvore: pulso -> cinco cadeia de
dedos, cada uma com angulos de dobradilha controlados.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.hand_tracker import (
    FINGERS,
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    RING_MCP,
    THUMB_CMC,
    THUMB_MCP,
    WRIST,
    Hand,
)

# nos percorridos por _chain, do MCP ate a ponta (o primeiro segmento vai do
# pulso ate o MCP; nao ha no no meio desse trecho)
CHAIN_NODES = {
    "thumb": (THUMB_MCP, 3, 4),
    "index": (INDEX_MCP, 6, 7, 8),
    "middle": (MIDDLE_MCP, 10, 11, 12),
    "ring": (RING_MCP, 14, 15, 16),
    "pinky": (PINKY_MCP, 18, 19, 20),
}

# comprimento do segmento do pulso ate o MCP, e dos segmentos seguintes
PALM_LEN = {"thumb": 0.30, "index": 0.62, "middle": 0.62, "ring": 0.60, "pinky": 0.54}

# no inicial de cada cadeia: o que _chain consome
CHAIN_ROOTS = {
    "thumb": THUMB_CMC,
    "index": WRIST,
    "middle": WRIST,
    "ring": WRIST,
    "pinky": WRIST,
}

# comprimento das falanges, em unidades de "tamanho de mao"
SEG_LEN = {
    "thumb": (0.42, 0.26),
    "index": (0.26, 0.20, 0.18),
    "middle": (0.28, 0.21, 0.19),
    "ring": (0.26, 0.20, 0.18),
    "pinky": (0.22, 0.18, 0.16),
}

# direcao base de cada dedo quando totalmente estendido (graus, 0 = +x, y
# para baixo no sistema de coordenadas da imagem)
BASE_DIR = {"thumb": 150.0, "index": 92.0, "middle": 90.0, "ring": 88.0, "pinky": 86.0}

# abertura entre os dedos, em graus. O valor padrao reproduz dedos colados;
# `spread` afasta um par para desambiguar U/V e K/H.
FAN = {"index": 0.0, "middle": -7.0, "ring": -14.0, "pinky": -21.0}


def _chain(pts, nodes, root_idx, direction, curls, lengths, palm=0.6):
    """Percorre a cadeia do dedo a partir da base, ja posicionada.

    O primeiro segmento chega ate o MCP; as juntas seguintes somam o angulo
    de dobradilha. Com todos os curls em zero a cadeia fica reta e a PIP
    mede ~180 graus, que e o que o classificador espera para "estendido".
    """
    x, y = pts[root_idx][0], pts[root_idx][1]
    a = direction
    # pulso -> MCP: a palma nao dobra, so aponta
    mcp = nodes[0]
    x += palm * math.cos(math.radians(a))
    y += palm * math.sin(math.radians(a))
    pts[mcp] = (x, y)
    # MCP -> PIP -> DIP -> ponta: cada junta soma o seu angulo
    for i, (node, seg) in enumerate(zip(nodes[1:], lengths, strict=False)):
        a += curls[min(i, len(curls) - 1)]
        x += seg * math.cos(math.radians(a))
        y += seg * math.sin(math.radians(a))
        pts[node] = (x, y)
    return pts


def make_pose(
    curl=(0.0, 0.0, 0.0, 0.0, 0.0),
    thumb_dir=None,
    scale=0.18,
    rotation=0.0,
    spread=0.0,
    thumb_spread=0.0,
) -> Hand:
    """Monta uma `Hand` com os curls pedidos (0 = estendido, 1 = dobrado).

    `curl` segue a ordem de FINGERS: polegar, indicador, medio, anelar,
    mininho; o valor e a dobradilha por junta. `rotation` gira a mao toda,
    `spread` afasta indicador e medio (para distinguir U de V), e
    `thumb_spread` afasta o polegar da palma -- sem ele o sintetizador produz
    so dois estados de polegar e as letras C, D, R, S, X ficam indistintas.
    """
    pts = [None] * 21
    pts[WRIST] = (0.0, 0.0)
    pts[THUMB_CMC] = (-0.18, 0.10)

    # o polegar parte do CMC; os demais dedos sao construidos a partir do pulso
    pts[THUMB_MCP] = pts[THUMB_CMC]

    for f in FINGERS:
        nodes = CHAIN_NODES[f]
        root = CHAIN_ROOTS[f]
        if f == "thumb":
            direction = 60.0 if thumb_dir is None else thumb_dir
            direction -= thumb_spread
            curls = [curl[0] * 70.0, curl[0] * 70.0]
        else:
            direction = BASE_DIR[f] + FAN[f]
            if f == "index":
                direction -= spread
            elif f == "middle":
                direction += spread
            c = curl[FINGERS.index(f)]
            # dobradilha por junta: 0.5 por junta leva a ~100 graus na PIP,
            # que e o que o MediaPipe ve num dedo fechado
            curls = [c * 100.0, c * 100.0, c * 100.0]
        _chain(pts, nodes, root, direction, curls, SEG_LEN[f], PALM_LEN[f])

    arr = np.zeros((21, 3), np.float32)
    for i, p in enumerate(pts):
        if p is None:
            p = (0.0, 0.0)
        arr[i, 0], arr[i, 1] = p
    # recentraliza e leva para o meio do frame
    c = arr[:, :2].mean(axis=0)
    arr[:, 0] = arr[:, 0] - c[0] + 0.5
    arr[:, 1] = arr[:, 1] - c[1] + 0.5
    # escala para o tamanho desejado no quadro
    hand_size = float(np.linalg.norm(arr[WRIST, :2] - arr[MIDDLE_MCP, :2]))
    if hand_size > 1e-6:
        arr[:, :2] = 0.5 + (arr[:, :2] - 0.5) * (scale / hand_size)
    if rotation:
        th = math.radians(rotation)
        rel = arr[:, :2] - arr[WRIST, :2]
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
        arr[:, :2] = rel @ rot.T + arr[WRIST, :2]
    return Hand(arr, presence=0.9, handedness=0.2, box=(0.0, 0.0, 1.0, 1.0), smooth=1.0)
