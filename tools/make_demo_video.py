"""Gera um video sintetico de Libras para testar a app sem webcam.

As letras aparecem em sequencia, com a mao mantida por `hold` quadros,
pausada por `gap` (que vira espaco/pausa) e com rotulo na tela.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.data.synthetic import make_pose
from libras.signs.dactylology import LETTERS

W, H = 640, 480
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

CURL = {"lateral": 0.0, "dobrado": 0.9, "c": 0.45, "apoiado": 0.55, "cruz": 0.95}
TSPREAD = {"lateral": 0.0, "dobrado": 0.0, "c": 35.0, "apoiado": 15.0, "cruz": 30.0}
SPREAD = {"U": 0.0, "V": 25.0, "W": 18.0}


def pose_for(letter):
    spec = next(l for l in LETTERS if l.name == letter)
    curl = [1.0] * 5
    for i, up in enumerate(spec.fingers):
        curl[i + 1] = 0.0 if up else 1.0
    curl[0] = CURL[spec.thumb]
    return make_pose(curl=tuple(curl), scale=0.20,
                     rotation=0.0, spread=SPREAD.get(letter, 0.0),
                     thumb_spread=TSPREAD[spec.thumb])


SKIN = (150, 175, 205)     # BGR, dentro da faixa YCrCb usada pelo tracker
SKIN_DARK = (120, 145, 175)
BG = (30, 30, 30)


def draw(hand, letter):
    """Desenha a mao com pele, para o detector de pele e o landmarker
    encontrarem algo. Um boneco de linhas nao e detectado."""
    img = np.full((H, W, 3), BG, np.uint8)
    px = hand.landmarks[:, :2] * np.array([W, H])
    pi = px.astype(int)

    chains = {
        "thumb": (0, 1, 2, 3, 4),
        "index": (0, 5, 6, 7, 8),
        "middle": (0, 9, 10, 11, 12),
        "ring": (0, 13, 14, 15, 16),
        "pinky": (0, 17, 18, 19, 20),
    }

    # antebraco e palma: poligono preenchido
    palm = np.array([pi[0], pi[5], pi[9], pi[13], pi[17]], np.int32)
    cv2.fillPoly(img, [palm], SKIN_DARK)
    cv2.line(img, tuple(pi[0]), (W // 2, H + 40), SKIN_DARK, 46)

    # dedos: capsulas grossas com articulation
    for name, chain in chains.items():
        width = 30 if name == "thumb" else 24
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(img, tuple(pi[a]), tuple(pi[b]), SKIN, width)
        for i in chain[1:-1]:
            cv2.circle(img, tuple(pi[i]), width // 2, SKIN, -1)
        cv2.circle(img, tuple(pi[chain[-1]]), width // 2, SKIN, -1)
        cv2.circle(img, tuple(pi[chain[-1]]), width // 2 - 6, SKIN_DARK, -1)

    # marcadores dos landmarks, para conferir a pose a olho nu
    for i, p in enumerate(pi):
        cv2.circle(img, tuple(int(v) for v in p), 4,
                   (0, 0, 255) if i == 0 else (40, 220, 60), -1)

    cv2.rectangle(img, (0, 0), (W, 120), (0, 0, 0), -1)
    cv2.putText(img, letter, (W // 2 - 40, 90), cv2.FONT_HERSHEY_SIMPLEX,
                3.0, (255, 255, 255), 5)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/opencode/libras.mp4")
    ap.add_argument("--text", default="OLAR")
    ap.add_argument("--hold", type=int, default=18)
    ap.add_argument("--gap", type=int, default=14)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

    blank = np.full((H, W, 3), 30, np.uint8)
    for letter in args.text:
        frame = draw(pose_for(letter), letter)
        for _ in range(args.hold):
            vw.write(frame)
        for _ in range(args.gap):
            vw.write(blank.copy())
    vw.release()
    print(f"{args.out}  ({len(args.text)} letras, "
          f"{(args.hold + args.gap) * len(args.text)} quadros)")


if __name__ == "__main__":
    main()
