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
    + / -   sobe / desce o piso de confianca (min_score)
    [ / ]   encolhe / amplia a margem de votos (vote_margin)
    H       abre o historico
    T       alterna o tema claro/escuro
    ?       abre a lista de atalhos
    Q/ESC   sai

Os ajustes de + / - / [ / ] ficam salvos em models/tuning.json e valem
tambem nas proximas execucoes. Com --log-misses, apagar (B) uma letra que
acabou de entrar grava os landmarks dela no arquivo: depois
`tools/relabel_misses.py --as <letra certa>` rotula e anexa ao dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
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

# distancia maxima (em unidades de frame) entre o centro da mao e a track
# anterior para continuarem sendo a mesma mao; mesmo limiar do _smooth
TRACK_MATCH_DIST = 0.12

TUNING_PATH = Path("models/tuning.json")
MIN_SCORE_RANGE = (0.2, 0.9)
VOTE_MARGIN_RANGE = (0.3, 0.95)
MISS_FRAMES = 30  # quadros de contexto guardados por leitura errada


def load_tuning(path: Path = TUNING_PATH) -> dict:
    """Ajustes persistidos das teclas + / - / [ / ]."""
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {k: float(d[k]) for k in ("min_score", "vote_margin") if k in d}


def save_tuning(t: dict, path: Path = TUNING_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(t, indent=2))
    except OSError:
        pass


def append_misses(path: Path, frames: list, letter: str, times: list, group: int) -> None:
    """Anexa os quadros de uma leitura rejeitada ao arquivo de misses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lm_novos = np.stack([f.landmarks[:, :3] for f in frames]).astype(np.float32)
    pr_novos = np.array([f.presence for f in frames], np.float32)
    hd_novos = np.array([f.handedness for f in frames], np.float32)
    ts_novos = np.array(times[: len(frames)], np.float64)
    lt_novos = np.array([letter] * len(frames), "U1")
    gi_novos = np.array([group] * len(frames), np.int32)
    if path.exists():
        d = np.load(path, allow_pickle=False)
        np.savez_compressed(
            path,
            landmarks=np.concatenate([d["landmarks"], lm_novos]),
            presence=np.concatenate([d["presence"], pr_novos]),
            handedness=np.concatenate([d["handedness"], hd_novos]),
            times=np.concatenate([d["times"], ts_novos]),
            letters=np.concatenate([d["letters"], lt_novos]),
            groups=np.concatenate([d["groups"], gi_novos]),
        )
    else:
        np.savez_compressed(
            path,
            landmarks=lm_novos,
            presence=pr_novos,
            handedness=hd_novos,
            times=ts_novos,
            letters=lt_novos,
            groups=gi_novos,
        )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="arquivo de video no lugar da webcam")
    ap.add_argument("--models", default="models")
    ap.add_argument("--hold", type=int, default=5, help="quadros que a letra precisa se sustentar")
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument(
        "--palm-every",
        type=int,
        default=2,
        help="detector de palma a cada N quadros (1 = todo quadro); o landmarker roda sempre",
    )
    ap.add_argument("--width", type=int, default=None, help="largura da janela de exibicao")
    ap.add_argument("--height", type=int, default=None, help="altura da janela de exibicao")
    ap.add_argument("--fullscreen", action="store_true", help="abre em tela cheia")
    ap.add_argument(
        "--no-gui",
        action="store_true",
        help="nao abre janela: util para testar com --source em maquina sem display ou em servidor",
    )
    ap.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="tema da interface (T alterna em tempo de execucao)",
    )
    ap.add_argument(
        "--classifier",
        default=None,
        help="modelo treinado (padrao: models/dactylology_clf.npz se existir)",
    )
    ap.add_argument(
        "--motion-classifier",
        default=None,
        help="modelo temporal para as letras de movimento "
        "(padrao: models/dactylology_motion_clf.npz se existir)",
    )
    ap.add_argument(
        "--log-misses",
        metavar="ARQUIVO",
        help="grava os quadros das leituras que o usuario apagar "
        "(ex.: data/misses.npz); depois "
        "tools/relabel_misses.py --as <letra> anexa ao dataset",
    )
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

    tracker = HandTracker(args.models, max_hands=args.max_hands, palm_every=args.palm_every)
    tuning = load_tuning()
    builder = PhraseBuilder(hold_frames=args.hold, **tuning)
    histories: list = []  # HandHistory da(s) mao(s) vista(s) recentemente
    history: list = []  # frases salvas (ENTER), para o overlay

    hud = Hud(get_theme(args.theme))
    state = HudState(history=history)

    clf_path = (
        Path(args.classifier) if args.classifier else Path(args.models) / "dactylology_clf.npz"
    )
    classifier = None
    if clf_path.exists():
        classifier = LetterClassifier.load(clf_path)
        print(f"classificador: {clf_path} ({len(classifier.classes)} letras)")
    else:
        print("classificador: nenhum (usando so as regras)", file=sys.stderr)

    mclf_path = (
        Path(args.motion_classifier)
        if args.motion_classifier
        else Path(args.models) / "dactylology_motion_clf.npz"
    )
    motion_classifier = None
    if mclf_path.exists():
        motion_classifier = LetterClassifier.load(mclf_path)
        print(f"classificador de movimento: {mclf_path} ({len(motion_classifier.classes)} letras)")

    prev = time.time()
    fps = 0.0
    # contexto da leitura em curso, para o --log-misses: landmarks dos
    # ultimos quadros com leitura; quando o usuario apaga a letra que acabou
    # de entrar, e isso que vai para o arquivo
    miss_buf: deque = deque(maxlen=MISS_FRAMES)
    miss_ids: deque = deque(maxlen=MISS_FRAMES)
    miss_letter = ""
    miss_group = -1
    miss_pending: list = []
    miss_ids_pending: list = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.source is None:
                frame = cv2.flip(frame, 1)  # espelho: movementos como no espelho
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            hands = tracker.process(frame)

            guess = None
            if hands:
                # usa a mao mais confiavel, pela ordem de presenca
                hand = hands[0]
                c = hand.center()[:2]
                # casa a mao com o historico anterior pela distancia do centro.
                # Uma chave pela posicao (round(x*1000)) picaria a track a cada
                # quadro -- o centro anda ~0.001/frame e nasceria um historico
                # novo por quadro, matando as features de movimento (J/Z).
                best, best_d = None, TRACK_MATCH_DIST
                for h in histories:
                    if h.centers:
                        d = float(np.linalg.norm(h.centers[-1] - c))
                        if d < best_d:
                            best, best_d = h, d
                if best is None:
                    best = HandHistory()
                best.push(hand, now)
                histories = [h for h in histories if h is not best]
                histories.insert(0, best)
                del histories[2:]  # memoria curta: mao ativa + a anterior
                guess = recognise(
                    hand, best, classifier=classifier, motion_classifier=motion_classifier
                )
            antes = len(builder.text)
            phrase = builder.push(guess)
            if args.log_misses:
                if guess is not None:
                    miss_buf.append(hand)
                    miss_ids.append(now)
                elif miss_buf:
                    miss_buf.clear()  # mao sumiu no meio: contexto virou ruido
                    miss_ids.clear()
                if len(builder.text) > antes and builder.text:
                    miss_letter = builder.text[-1]
                    miss_group += 1
                    miss_pending = list(miss_buf)
                    miss_ids_pending = list(miss_ids)
            if not args.no_gui:
                state.hands = hands
                state.phrase = phrase
                state.fps = fps
                state.hold_progress = builder.hold_progress
                state.hold_total = builder.hold_frames
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
                    builder.confirm()  # confirma a letra em leitura
                elif key in (ord("b"), ord("B")):
                    if (
                        args.log_misses
                        and miss_letter
                        and miss_pending
                        and builder.text.endswith(miss_letter)
                    ):
                        # o usuario rejeitou a letra que acabou de entrar:
                        # guarda o contexto para rotular depois
                        append_misses(
                            Path(args.log_misses),
                            miss_pending,
                            miss_letter,
                            miss_ids_pending,
                            miss_group,
                        )
                        hud.notify(f"leitura de '{miss_letter}' gravada em {args.log_misses}")
                    builder.backspace()
                    miss_letter, miss_pending = "", []
                elif key in (ord("c"), ord("C")):
                    builder.reset()
                elif key == 13:  # ENTER: guarda a frase no historico
                    if builder.text.strip():
                        history.append(builder.text.strip())
                        builder.reset()
                        hud.notify("frase salva no histórico")
                    else:
                        hud.notify("nada para salvar")
                elif key in (ord("+"), ord("=")):
                    novo = min(
                        MIN_SCORE_RANGE[1], max(MIN_SCORE_RANGE[0], builder.min_score + 0.05)
                    )
                    builder.min_score = novo
                    save_tuning({"min_score": novo, "vote_margin": builder.vote_margin})
                    hud.notify(f"piso de confianca: {novo:.2f}")
                elif key in (ord("-"), ord("_")):
                    novo = min(
                        MIN_SCORE_RANGE[1], max(MIN_SCORE_RANGE[0], builder.min_score - 0.05)
                    )
                    builder.min_score = novo
                    save_tuning({"min_score": novo, "vote_margin": builder.vote_margin})
                    hud.notify(f"piso de confianca: {novo:.2f}")
                elif key in (ord("["), ord("{")):
                    novo = min(
                        VOTE_MARGIN_RANGE[1], max(VOTE_MARGIN_RANGE[0], builder.vote_margin - 0.05)
                    )
                    builder.vote_margin = novo
                    save_tuning({"min_score": builder.min_score, "vote_margin": novo})
                    hud.notify(f"margem de votos: {novo:.2f}")
                elif key in (ord("]"), ord("}")):
                    novo = min(
                        VOTE_MARGIN_RANGE[1], max(VOTE_MARGIN_RANGE[0], builder.vote_margin + 0.05)
                    )
                    builder.vote_margin = novo
                    save_tuning({"min_score": builder.min_score, "vote_margin": novo})
                    hud.notify(f"margem de votos: {novo:.2f}")
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
    finally:
        cap.release()
        tracker.close()
        if not args.no_gui:
            cv2.destroyAllWindows()
    print("frase final:", builder.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
