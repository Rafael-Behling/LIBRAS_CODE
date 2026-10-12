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
from libras.signs.classifier import LetterClassifier
from libras.signs.dactylology import recognise

WINDOW = "Libras -> Português"
BAR_H = 120


def draw_overlay(frame, hands, phrase, fps, hold_progress=0.0, hold_total=1.0):
    """Desenha landmarks, caixa e a frase na parte inferior."""
    out = frame.copy()
    for hand in hands:
        px = hand.landmarks[:, :2] * np.array([frame.shape[1], frame.shape[0]])
        for i, (x, y) in enumerate(px.astype(int)):
            if i == 0:
                cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
                cv2.circle(out, (x, y), 3, (255, 255, 255), -1)
            else:
                cv2.circle(out, (x, y), 4, (0, 180, 255), -1)
        bx = [int(v) for v in hand.box]
        cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)

    h, w = out.shape[:2]
    # barra da frase com gradiente visual
    cv2.rectangle(out, (0, h - BAR_H), (w, h), (18, 18, 18), -1)
    cv2.line(out, (0, h - BAR_H), (w, h - BAR_H), (60, 60, 60), 1)

    # barra de progresso para confirmar letra
    if hold_total > 0 and hold_progress > 0:
        prog_w = int((w - 36) * min(hold_progress / hold_total, 1.0))
        if prog_w > 0:
            cv2.rectangle(out, (18, h - BAR_H + 8), (18 + prog_w, h - BAR_H + 12), (0, 180, 255), -1)

    # letra em leitura
    if phrase.current:
        cv2.putText(out, phrase.current, (18, h - 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 200, 255), 3)
    if phrase.candidates and len(phrase.candidates) > 1:
        # votacao: mostra para onde a leitura esta convergindo
        votes = " / ".join(f"{name}:{weight:g}" for name, weight in phrase.votes[:4])
        cv2.putText(out, votes, (18, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        if phrase.needs:
            cv2.putText(out, phrase.needs, (18, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    # frase acumulada
    text = phrase.text
    if text:
        scale = 1.0
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        while tw > w - 40 and scale > 0.4:
            scale -= 0.05
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(out, text, (18, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 2)
    else:
        cv2.putText(out, "Frase aparecerá aqui...", (18, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)

    # status
    n = len(hands)
    status = f"Mãos: {n}  •  FPS: {fps:.0f}"
    cv2.putText(out, status, (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # controles
    controls = "[Espaço] confirmar  [B] apagar  [C] limpar  [Q/ESC] sair"
    cv2.putText(out, controls, (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if n == 0:
        cv2.putText(out, "Mostre a mão para a câmera", (w // 2 - 190, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(out, "Posicione sua mão de frente para a câmera", (w // 2 - 180, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="arquivo de video no lugar da webcam")
    ap.add_argument("--models", default="models")
    ap.add_argument("--hold", type=int, default=5,
                    help="quadros que a letra precisa se sustentar")
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--width", type=int, default=None,
                    help="largura da janela de exibicao")
    ap.add_argument("--height", type=int, default=None,
                    help="altura da janela de exibicao")
    ap.add_argument("--fullscreen", action="store_true",
                    help="abre em tela cheia")
    ap.add_argument("--no-gui", action="store_true",
                    help="nao abre janela: util para testar com --source em maquina "
                         "sem display ou em servidor")
    ap.add_argument("--classifier", default=None,
                    help="modelo treinado (padrao: models/dactylology_clf.npz "
                         "se existir)")
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

    clf_path = Path(args.classifier) if args.classifier \
        else Path(args.models) / "dactylology_clf.npz"
    classifier = None
    if clf_path.exists():
        classifier = LetterClassifier.load(clf_path)
        print(f"classificador: {clf_path} ({len(classifier.classes)} letras)")
    else:
        print("classificador: nenhum (usando so as regras)", file=sys.stderr)

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
            guess = recognise(hand, hist, classifier=classifier)
            if len(hist) > hist.maxlen:
                histories = {key: hist}
        phrase = builder.push(guess)

        hold_progress = getattr(builder, "_hold", 0)
        hold_total = getattr(builder, "hold_frames", args.hold)

        if not args.no_gui:
            display = draw_overlay(frame, hands, phrase, fps,
                                   hold_progress=hold_progress,
                                   hold_total=hold_total)
            if args.fullscreen:
                cv2.namedWindow(WINDOW, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                if args.width and args.height:
                    cv2.resizeWindow(WINDOW, args.width, args.height)
                else:
                    h, w = display.shape[:2]
                    cv2.resizeWindow(WINDOW, int(w * 1.3), int(h * 1.3))
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                builder.confirm()     # confirma a letra em leitura
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
