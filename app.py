"""Aplicacao webcam: reconhece Libras e mostra a frase em portugues.

Uso:
    python app.py                # webcam 0
    python app.py --camera 1
    python app.py --source video.mp4

Controles (Janela de video):
    ESPACO  segura a letra atual (forca a entrada na frase)
    B       apaga o ultimo caractere
    C       limpa a frase
    Q/ESC   sai
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from libras.core.features import HandHistory
from libras.core.hand_tracker import HandTracker
from libras.core.phrase import PhraseBuilder
from libras.signs.dactylology import recognise

WINDOW = "Libras -> portugues"
BAR_H = 110


def draw_overlay(frame, hands, phrase, fps):
    """Desenha landmarks, caixa e a frase na parte inferior."""
    out = frame.copy()
    for hand in hands:
        px = hand.landmarks[:, :2] * np.array([frame.shape[1], frame.shape[0]])
        for i, (x, y) in enumerate(px.astype(int)):
            color = (0, 220, 0) if i == 0 else (0, 160, 255)
            cv2.circle(out, (x, y), 4, color, -1)
        bx = [int(v) for v in hand.box]
        cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), (0, 220, 0), 2)

    h, w = out.shape[:2]
    # barra da frase
    cv2.rectangle(out, (0, h - BAR_H), (w, h), (24, 24, 24), -1)
    cv2.line(out, (0, h - BAR_H), (w, h - BAR_H), (90, 90, 90), 1)

    # letra em leitura
    if phrase.current:
        cv2.putText(out, phrase.current, (18, h - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3)
    if phrase.candidates and len(phrase.candidates) > 1:
        opts = " / ".join(phrase.candidates)
        cv2.putText(out, f"ambiguo: {opts}", (18, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        if phrase.needs:
            cv2.putText(out, phrase.needs, (18, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)

    # frase acumulada
    text = phrase.text
    if text:
        scale = 1.0
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        while tw > w - 40 and scale > 0.4:
            scale -= 0.05
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(out, text, (18, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 2)

    # status
    n = len(hands)
    cv2.putText(out, f"maos: {n}   fps: {fps:.0f}", (w - 220, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 1)
    if n == 0:
        cv2.putText(out, "mostre a mao para a camera", (w // 2 - 170, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="arquivo de video no lugar da webcam")
    ap.add_argument("--models", default="models")
    ap.add_argument("--hold", type=int, default=5,
                    help="quadros que a letra precisa se sustentar")
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--no-gui", action="store_true",
                    help="nao abre janela: util para testar com --source em maquina "
                         "sem display ou em servidor")
    args = ap.parse_args(argv)

    src = args.source
    if src is None:
        src = args.camera
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"nao consegui abrir a camera {src}", file=sys.stderr)
            return 1
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"nao consegui abrir {src}", file=sys.stderr)
            return 1

    tracker = HandTracker(args.models, max_hands=args.max_hands)
    builder = PhraseBuilder(hold_frames=args.hold)
    histories: dict = {}

    prev = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.source is None:
            frame = cv2.flip(frame, 1)     # espelho: movementos como no espelho
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        hands = tracker.process(frame)

        guess = None
        if hands:
            # usa a mao mais confiavel, pela ordem de presenca
            hand = hands[0]
            key = int(round(hand.center()[0] * 1000))
            hist = histories.setdefault(key, HandHistory())
            hist.push(hand, now)
            guess = recognise(hand, hist)
            if len(hist) > hist.maxlen:
                histories = {key: hist}
        phrase = builder.push(guess)

        if not args.no_gui:
            cv2.imshow(WINDOW, draw_overlay(frame, hands, phrase, fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                builder._cooldown = 0     # confirma a letra atual
                builder._hold = args.hold
            elif key in (ord("b"), ord("B")):
                builder.backspace()
            elif key in (ord("c"), ord("C")):
                builder.reset()
        else:
            if guess:
                print(f"letra={guess} maos={len(hands)} frase={builder.text!r}")
            elif hands:
                print(f"sem letra confiavel maos={len(hands)} frase={builder.text!r}")

    cap.release()
    if not args.no_gui:
        cv2.destroyAllWindows()
    print("frase final:", builder.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
