"""Confere como o modelo le letras especificas (por padrao L e C).

Uso:
    python tools/check_letters.py                # L e C
    python tools/check_letters.py --letters LCM
    python tools/check_letters.py --samples data/samples.backup.npz

Serve para comparar antes/depois de regravar: mostra acuracia e para quais
letras cada amostra esta sendo trocada. Se as amostras de L continuarem
virando M, o rotulo regravado ainda tem o mesmo vicio.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.core.features import hand_features
from libras.core.hand_tracker import Hand
from libras.signs.classifier import LetterClassifier


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default="data/samples.npz")
    ap.add_argument("--model", default="models/dactylology_clf.npz")
    ap.add_argument("--letters", default="LC",
                    help="letras a inspecionar (ex.: LCM)")
    args = ap.parse_args(argv)

    src = Path(args.samples)
    if not src.exists():
        print(f"dataset nao encontrado: {src}", file=sys.stderr)
        return 1

    d = np.load(src, allow_pickle=False)
    lm, labels = d["landmarks"], [str(x) for x in d["labels"]]
    presence, handedness = d["presence"], d["handedness"]
    clf = LetterClassifier.load(args.model)
    idx = {c: i for i, c in enumerate(clf.classes)}

    for L in args.letters.upper():
        if L not in idx:
            print(f"{L}: nao esta no modelo (letra dinamica?)")
            continue
        preds = Counter()
        curl_p, curl = [], []
        for i in range(len(labels)):
            if labels[i] != L:
                continue
            h = Hand(landmarks=lm[i], presence=float(presence[i]),
                     handedness=float(handedness[i]))
            f = hand_features(h)
            p = clf.predict_proba(f.vector())[0]
            preds[clf.classes[int(p.argmax())]] += 1
            curl.append(f.curl)
        total = sum(preds.values())
        certos = preds.get(L, 0)
        print(f"\n{L}: {certos}/{total} corretos "
              f"({100.0 * certos / max(total, 1):.0f}%)")
        for letra, n in preds.most_common():
            marca = "ok" if letra == L else "  <- troca"
            print(f"   {L} -> {letra}: {n}{marca}")
        if curl:
            print(f"   curl medio: {np.array(curl).mean(0).round(2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
