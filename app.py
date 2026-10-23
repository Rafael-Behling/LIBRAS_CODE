"""Aplicacao webcam: reconhece Libras e mostra a frase em portugues.

Uso:
    python app.py                # webcam 0
    python app.py --camera 1
    python app.py --source video.mp4

Controles (Janela de video):
    ESPACO  segura a letra atual (forca a entrada na frase)
    B       apaga o ultimo caractere
    C       limpa a frase
    ENTER   salva a frase no historico
    H       abre o historico
    T       alterna o tema claro/escuro
    ?       abre a lista de atalhos
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
from libras.ui import Hud, HudState, get_theme

WINDOW = "Libras -> Português"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--theme", choices=["dark", "light"], default="dark",
                    help="tema da interface (T alterna em tempo de execucao)")
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
    history: list = []

    hud = Hud(get_theme(args.theme))
    state = HudState(history=history)

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

        if not args.no_gui:
            state.hands = hands
            state.phrase = phrase
            state.fps = fps
            state.hold_progress = getattr(builder, "_hold", 0)
            state.hold_total = getattr(builder, "hold_frames", args.hold)
            display = hud.render(frame, state)

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
            elif key == 13:            # ENTER: guarda a frase no historico
                if builder.text.strip():
                    history.append(builder.text.strip())
                    builder.reset()
                    hud.notify("frase salva no histórico")
                else:
                    hud.notify("nada para salvar")
            elif key in (ord("h"), ord("H")):
                if state.overlay == "history":
                    state.overlay = ""
                else:
                    state.overlay = "history"
            elif key in (ord("t"), ord("T")):
                theme = hud.toggle_theme()
                hud.notify(f"tema {theme.name}")
            elif key == ord("?") or key == ord("/"):
                state.overlay = "" if state.overlay == "help" else "help"
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
