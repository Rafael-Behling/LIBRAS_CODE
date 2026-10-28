"""Rotula e anexa ao dataset as leituras que o usuario rejeitou no app.

Uso:
    python tools/relabel_misses.py data/misses.npz --list
    python tools/relabel_misses.py data/misses.npz --as B
    python tools/relabel_misses.py data/misses.npz --as B --ids 0-2
    python tools/relabel_misses.py data/misses.npz --as B --samples data/samples.npz

O app com `--log-misses data/misses.npz` grava os quadros de cada leitura que
o usuario apagou com B: os landmarks estao la, mas o rotulo e a letra ERRADA
(a que foi rejeitada). Este comando lista os registros, rotula um intervalo
com `--as` (a letra correta que voce estava fazendo) e anexa ao dataset de
treino. Cada registro vira um grupo temporal, entao as letras dinamicas
tambem podem ser salvas por aqui.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(path: Path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in ("landmarks", "presence", "handedness", "times", "letters", "groups")}


def records(d: dict) -> list:
    """(id_do_grupo, letra_rejeitada, quadros) por grupo."""
    groups = d["groups"]
    out = []
    for g in np.unique(groups):
        ids = np.flatnonzero(groups == g)
        out.append((int(g), str(d["letters"][ids[0]]), ids))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("misses", help="arquivo gravado pelo app com --log-misses")
    ap.add_argument("--list", action="store_true", help="so lista os registros")
    ap.add_argument(
        "--ids", default="", help="registros a rotular, ex.: 0-2 ou 1 (o padrao e todos)"
    )
    ap.add_argument("--as", dest="label", default="", help="letra correta desses registros (A-Z)")
    ap.add_argument(
        "--samples", default="data/samples.npz", help="dataset para onde os registros anexados vao"
    )
    args = ap.parse_args(argv)

    src = Path(args.misses)
    if not src.exists():
        print(f"arquivo nao encontrado: {src}", file=sys.stderr)
        return 1
    d = load(src)
    recs = records(d)
    if not recs:
        print("nenhum registro no arquivo")
        return 0

    print(f"{len(recs)} registros (grupos) em {src}:")
    for _g, letra, ids in recs:
        print(f"  id {int(_g):3d}: rejeitada '{letra}'  {len(ids)} quadros")

    if args.list or not args.label:
        if not args.list:
            print("informe --as <letra> para rotular (ou --list para so listar)", file=sys.stderr)
        return 0

    label = args.label.strip().upper()
    if len(label) != 1 or not ("A" <= label <= "Z"):
        print(f"--as espera uma letra A-Z, veio {args.label!r}", file=sys.stderr)
        return 1

    sel = set()
    for part in args.ids.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            sel.update(range(int(a), int(b) + 1))
        else:
            sel.add(int(part))
    escolhidos = [r for r in recs if not sel or r[0] in sel]
    if sel and not escolhidos:
        print(f"nenhum id de {sorted(sel)} no arquivo", file=sys.stderr)
        return 1

    # concatena os quadros escolhidos, com o rotulo CORRETO
    lms, prs, hds, tss, lts, gis = [], [], [], [], [], []
    next_group = 0
    out = Path(args.samples)
    if out.exists():
        s = np.load(out, allow_pickle=False)
        n = len(s["labels"])
        base = {
            "landmarks": s["landmarks"],
            "labels": s["labels"],
            "presence": s["presence"],
            "handedness": s["handedness"],
            "groups": s["groups"] if "groups" in s else np.zeros(n, np.int32),
            "times": s["times"] if "times" in s else np.zeros(n, np.float64),
        }
        if len(base["groups"]):
            next_group = int(base["groups"].max()) + 1
    else:
        base = {
            "landmarks": np.zeros((0, 21, 3), np.float32),
            "labels": np.zeros(0, "U1"),
            "presence": np.zeros(0, np.float32),
            "handedness": np.zeros(0, np.float32),
            "groups": np.zeros(0, np.int32),
            "times": np.zeros(0, np.float64),
        }

    for _g, _, ids in escolhidos:
        lms.append(d["landmarks"][ids])
        prs.append(d["presence"][ids])
        hds.append(d["handedness"][ids])
        tss.append(d["times"][ids])
        lts.append(np.array([label] * len(ids), "U1"))
        gis.append(np.array([next_group] * len(ids), np.int32))
        next_group += 1

    np.savez_compressed(
        out,
        landmarks=np.concatenate([base["landmarks"], *lms]).astype(np.float32),
        labels=np.concatenate([base["labels"], *lts]).astype("U1"),
        presence=np.concatenate([base["presence"], *prs]).astype(np.float32),
        handedness=np.concatenate([base["handedness"], *hds]).astype(np.float32),
        groups=np.concatenate([base["groups"], *gis]).astype(np.int32),
        times=np.concatenate([base["times"], *tss]).astype(np.float64),
    )
    n_quadros = sum(len(ids) for _, _, ids in escolhidos)
    print(f"anexados {len(escolhidos)} registros ({n_quadros} quadros) como '{label}' em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
