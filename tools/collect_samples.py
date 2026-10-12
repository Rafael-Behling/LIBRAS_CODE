"""Grava amostras rotuladas de datilologia para treinar um classificador.

Uso:
    python tools/collect_samples.py                     # webcam 0
    python tools/collect_samples.py --camera 1
    python tools/collect_samples.py --out data/samples.npz

Controles (janela):
    A..Z     escolhe a letra e comeca a gravar essa letra
    ESPACO   pausa / continua a gravacao
    0        reinicia as amostras DESTA sessao (nao mexe nas ja salvas)
    ESC      salva e sai

O que e salvo sao os 21 landmarks (x, y, z normalizados) de cada quadro
gravado, nao as features prontas: assim o dataset sobrevive a mudancas no
extrator de features. O treino recompoe as features a partir dos landmarks.

Cada letra quer umas 20-40 amostras, variando um pouco o angulo e a distancia
da mao. Grava em rajada: um quadro a cada `--stride` (padrao 2), para as
amostras nao ficarem quase identicas.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.core.hand_tracker import HandTracker
from libras.signs.dactylology import LETTERS

WINDOW = "Gravar amostras (A-Z)"
ALPHABET = [l.name for l in LETTERS]


def load_dataset(path: Path):
    """Le um .npz anterior, se existir; senao devolve arrays vazios."""
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return (d["landmarks"], d["labels"], d["presence"], d["handedness"])
    return (np.zeros((0, 21, 3), np.float32), np.zeros(0, "U1"),
            np.zeros(0, np.float32), np.zeros(0, np.float32))


def save_dataset(path: Path, lm, labels, presence, handedness) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, landmarks=lm.astype(np.float32),
                        labels=labels.astype("U1"),
                        presence=presence.astype(np.float32),
                        handedness=handedness.astype(np.float32))


def draw(frame, label, recording, counts, session, total, fps):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 84), (18, 18, 18), -1)
    letra = label or "-"
    cor = (0, 220, 0) if recording else (140, 140, 140)
    estado = "GRAVANDO" if recording else ("pausado" if label else "escolha uma letra")
    cv2.putText(out, f"{letra}   {estado}", (16, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3)
    cv2.putText(out, f"desta sessao: {session}   total: {total}   {fps:.0f} fps",
                (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    # contagem por letra ja gravada
    parts = [f"{k}:{v}" for k, v in sorted(counts.items())]
    linha = "  ".join(parts)
    cv2.putText(out, linha[-90:], (16, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(out, "[A-Z] letra  [espaco] pausa  [0] limpar sessao  [ESC] sair",
                (16, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="data/samples.npz")
    ap.add_argument("--models", default="models")
    ap.add_argument("--stride", type=int, default=2,
                    help="grava um quadro a cada N (evita amostras quase iguais)")
    ap.add_argument("--min-presence", type=float, default=0.6)
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    lm, labels, presence, handedness = load_dataset(out_path)
    n_base = len(labels)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"nao consegui abrir a camera {args.camera}", file=sys.stderr)
        return 1
    tracker = HandTracker(args.models, max_hands=1)

    label = None
    recording = False
    session = 0
    frame_i = 0
    prev = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        frame_i += 1

        hands = tracker.process(frame)
        hand = hands[0] if hands else None
        gravou = False
        if hand is not None and recording and label is not None:
            if hand.presence >= args.min_presence and frame_i % max(args.stride, 1) == 0:
                lm = np.concatenate([lm, hand.landmarks[None, :, :3].astype(np.float32)])
                labels = np.concatenate([labels, np.array([label], "U1")])
                presence = np.append(presence, hand.presence).astype(np.float32)
                handedness = np.append(handedness, hand.handedness).astype(np.float32)
                session += 1
                gravou = True

        if hand is not None:
            px = hand.landmarks[:, :2] * np.array([frame.shape[1], frame.shape[0]])
            for x, y in px.astype(int):
                cv2.circle(frame, (x, y), 3, (0, 180, 255), -1)

        counts = {L: int(np.sum(labels[n_base:] == L)) for L in ALPHABET}
        counts = {k: v for k, v in counts.items() if v}
        if gravou:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.imshow(WINDOW, draw(frame, label, recording, counts, session,
                                len(labels), fps))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                                # ESC: sai (Q e uma letra!)
            break
        elif key == ord(" "):
            recording = not recording
        elif key == ord("0"):
            lm, labels, presence, handedness = lm[:n_base], labels[:n_base], \
                presence[:n_base], handedness[:n_base]
            session = 0
        elif 97 <= key <= 122:                       # a..z
            letra = chr(key).upper()
            if letra in ALPHABET:
                label = letra
                recording = True

    cap.release()
    cv2.destroyAllWindows()
    save_dataset(out_path, lm, labels, presence, handedness)
    print(f"salvo {len(labels)} amostras em {out_path} (+{len(labels) - n_base} nesta sessao)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
