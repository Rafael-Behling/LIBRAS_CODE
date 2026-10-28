"""Treina o classificador de letras a partir de `data/samples.npz`.

Uso:
    python tools/train_classifier.py
    python tools/train_classifier.py --samples data/samples.npz \
        --out models/dactylology_clf.npz
    python tools/train_classifier.py --motion   # modelo temporal (H,J,K,X,Y,Z)

O dataset guarda landmarks; aqui eles viram `HandFeatures.vector()` e
alimentam a regressao softmax de `libras.signs.classifier`. No modo estatico
(padrao) as 6 letras dinamicas (H, J, K, X, Y, Z) ficam de fora -- o movimento
nao aparece num quadro so; elas seguem pelas regras com historico. No modo
`--motion` o treino usa `HandHistory.vector()` sobre as sequencias do dataset
(grupos do `tools/collect_samples.py`) e salva
`models/dactylology_motion_clf.npz`, que cobre as dinamicas.

A avaliacao segura 20% das amostras de CADA letra para teste, imprime a
acuracia e as confusoes mais comuns (onde o modelo troca A por B).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.core.features import HandHistory, hand_features
from libras.core.hand_tracker import Hand
from libras.signs.classifier import fit
from libras.signs.dactylology import DYNAMIC_LETTERS

# Em Libras 6 letras sao dinamicas (tem movimento): H, J, K, X, Y, Z. O
# movimento nao aparece num quadro estatico, entao elas saem do modelo e
# seguem pelas regras com historico. As outras 20 sao estaticas.
DEFAULT_DROP = set(DYNAMIC_LETTERS)


def load_features(path: Path, drop=DEFAULT_DROP):
    """Le o .npz e devolve (X, y), descartando features degeneradas."""
    d = np.load(path, allow_pickle=False)
    lm, labels = d["landmarks"], [str(x) for x in d["labels"]]
    presence, handedness = d["presence"], d["handedness"]

    X, y = [], []
    for i in range(len(labels)):
        if labels[i] in drop:
            continue
        hand = Hand(landmarks=lm[i], presence=float(presence[i]), handedness=float(handedness[i]))
        v = hand_features(hand).vector()
        if not np.any(v):  # features neutras: mao invalida
            continue
        X.append(v)
        y.append(labels[i])
    return np.array(X, np.float32), y


def report(model, X, y, classes, titulo="teste") -> float:
    pred = model.predict_proba(X).argmax(axis=1)
    idx = {c: i for i, c in enumerate(classes)}
    true = np.array([idx[c] for c in y])
    acc = float((pred == true).mean())

    print(f"acuracia de {titulo}: {acc:.3f}  ({len(y)} amostras, {len(classes)} letras)")
    # confusoes: onde o modelo erra, com que frequencia
    conf: dict = {}
    for t, p in zip(true, pred, strict=False):
        if t != p:
            key = (classes[t], classes[p])
            conf[key] = conf.get(key, 0) + 1
    if conf:
        piores = sorted(conf.items(), key=lambda kv: -kv[1])[:10]
        print("confusoes mais comuns (verdade -> previsto):")
        for (a, b), n in piores:
            print(f"  {a} -> {b}: {n}")
    return acc


def split(X, y, frac=0.2, seed=0):
    """Segura `frac` de cada letra para teste."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in sorted(set(y)):
        ids = np.array([i for i, v in enumerate(y) if v == c])
        rng.shuffle(ids)
        n = max(1, int(round(len(ids) * frac))) if len(ids) > 2 else 0
        te.extend(ids[:n])
        tr.extend(ids[n:])
    return (X[tr], [y[i] for i in tr]), (X[te], [y[i] for i in te])


def load_sequences(path: Path, min_len: int = 4):
    """Sequencias por grupo: (HandHistory parcial, letra) de cada rajada.

    O dataset antigo nao tem `groups`/`times` (so quadros soltos); sem
    sequencia nao ha movimento para aprender, e o treino temporal sai de
    cena com um aviso -- o modelo estatico continua valendo.
    """
    d = np.load(path, allow_pickle=False)
    if "groups" not in d or "times" not in d:
        return None
    lm = d["landmarks"]
    labels = [str(x) for x in d["labels"]]
    presence = d["presence"]
    handedness = d["handedness"]
    groups = d["groups"].astype(np.int64)
    times = d["times"].astype(np.float64)

    seqs: list = []
    for g in np.unique(groups):
        ids = np.flatnonzero(groups == g)
        if len(ids) < min_len:
            continue
        hist = HandHistory()
        for i in ids:
            hand = Hand(
                landmarks=lm[i], presence=float(presence[i]), handedness=float(handedness[i])
            )
            v = hand_features(hand).vector()
            if not np.any(v):  # mao degenerada no meio da rajada: pula
                continue
            hist.push(hand, float(times[i]))
        if len(hist) >= min_len:
            seqs.append((hist, labels[ids[0]]))
    return seqs


def train_motion(src: Path, out: Path, iters: int, lr: float) -> int:
    """Modelo temporal: HandHistory.vector() (pose + movimento + trajetoria).

    Cobre as 6 letras dinamicas (H, J, K, X, Y, Z), que o modelo estatico nao
    representa. O vetor temporal inclui a pose estatica tambem, mas quem
    decide sobre as 20 estaticas segue sendo o modelo delas -- este aqui so
    e consultado quando a mao esta se mexendo.
    """
    seqs = load_sequences(src)
    if not seqs:
        print(
            "sem sequencias no dataset: regrave com tools/collect_samples.py "
            "(cada rajada de uma letra forma um grupo)",
            file=sys.stderr,
        )
        return 1

    X = np.array([h.vector() for h, _ in seqs], np.float32)
    y = [letra for _, letra in seqs]
    print(f"{len(y)} sequencias, {len(set(y))} letras")
    cont = {c: y.count(c) for c in sorted(set(y))}
    print("por letra:", " ".join(f"{c}:{cont[c]}" for c in cont))

    (Xtr, ytr), (Xte, yte) = split(X, y)
    classes = sorted(set(y))
    model = fit(Xtr, ytr, classes=classes, iters=iters, lr=lr)
    print("--- treino ---")
    report(model, Xtr, ytr, classes, "treino")
    if yte:
        print("--- teste ---")
        report(model, Xte, yte, classes, "teste")
    full = fit(X, y, classes=classes, iters=iters, lr=lr)
    full.save(out)
    print(f"modelo temporal salvo em {out} ({len(classes)} classes)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default="data/samples.npz")
    ap.add_argument("--out", default="models/dactylology_clf.npz")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument(
        "--keep-motion",
        action="store_true",
        help="inclui as 6 dinamicas (H,J,K,X,Y,Z) no modelo estatico",
    )
    ap.add_argument(
        "--motion",
        action="store_true",
        help="treina o modelo TEMPORAL (HandHistory.vector()) em vez "
        "do estatico: cobre H, J, K, X, Y, Z; salva em "
        "models/dactylology_motion_clf.npz",
    )
    args = ap.parse_args(argv)

    src = Path(args.samples)
    if not src.exists():
        print(f"dataset nao encontrado: {src}. Rode tools/collect_samples.py", file=sys.stderr)
        return 1

    if args.motion:
        out = (
            Path(args.out)
            if args.out != "models/dactylology_clf.npz"
            else Path("models/dactylology_motion_clf.npz")
        )
        return train_motion(src, out, args.iters, args.lr)

    drop = set() if args.keep_motion else DEFAULT_DROP
    X, y = load_features(src, drop)
    if len(y) == 0:
        print("dataset vazio depois dos filtros", file=sys.stderr)
        return 1

    print(f"{len(y)} amostras, {len(set(y))} letras")
    cont = {c: y.count(c) for c in sorted(set(y))}
    print("por letra:", " ".join(f"{c}:{cont[c]}" for c in cont))

    (Xtr, ytr), (Xte, yte) = split(X, y)
    classes = sorted(set(y))
    model = fit(Xtr, ytr, classes=classes, iters=args.iters, lr=args.lr)

    print("--- treino ---")
    report(model, Xtr, ytr, classes, "treino")
    if yte:
        print("--- teste ---")
        report(model, Xte, yte, classes, "teste")

    # modelo final treinado com TODAS as amostras
    full = fit(X, y, classes=classes, iters=args.iters, lr=args.lr)
    full.save(args.out)
    print(f"modelo salvo em {args.out} ({len(classes)} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
