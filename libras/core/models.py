"""Download e extracao dos modelos TFLite do hand_landmarker."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DOWNLOAD_TIMEOUT = 30.0

# (nome dentro do .task, tamanho minimo em bytes para considerar valido)
NEEDED = (
    ("hand_detector.tflite", 1_000_000),
    ("hand_landmarks_detector.tflite", 1_000_000),
)


def _task_valido(task_file: Path) -> bool:
    """O .task e um zip com os dois .tflite dentro.

    Um download interrompido deixa um arquivo truncado que `ZipFile` nao abre
    (ou abre sem os membros); validar aqui evita que a primeira execucao
    quebrada contamine todas as seguintes com um .task ruim em disco.
    """
    try:
        with zipfile.ZipFile(task_file) as z:
            nomes = z.namelist()
        return all(any(n.endswith(nome) for n in nomes) for nome, _ in NEEDED)
    except (zipfile.BadZipFile, OSError):
        return False


def _baixar(task_file: Path) -> None:
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "libras-code"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, open(task_file, "wb") as f:  # noqa: S310 (URL fixa e https)
        while True:
            bloco = r.read(1 << 16)
            if not bloco:
                break
            f.write(bloco)
    if not _task_valido(task_file):
        task_file.unlink(missing_ok=True)
        raise RuntimeError("download do hand_landmarker.task veio corrompido; rode de novo")


def ensure_models(models_dir: Path | str = "models") -> Path:
    """Garante os dois .tflite e devolve o caminho do de landmarks.

    O .task do MediaPipe e um zip com os dois detectores: o de palma
    (`hand_detector.tflite`,acha a ROI) e o de landmarks
    (`hand_landmarks_detector.tflite`,acha as 21 juntas). O tracker precisa dos
    dois, entao extrair so o de landmarks deixava a primeira execucao quebrada.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    targets = [(models_dir / name, size) for name, size in NEEDED]
    if all(t.exists() and t.stat().st_size > size for t, size in targets):
        return targets[-1][0]

    task_file = models_dir / "hand_landmarker.task"
    if not task_file.exists() or not _task_valido(task_file):
        _baixar(task_file)

    with zipfile.ZipFile(task_file) as z:
        names = z.namelist()
        for target, _ in targets:
            if target.exists() and target.stat().st_size > 1_000_000:
                continue
            for name in names:
                if name.endswith(target.name):
                    target.write_bytes(z.read(name))
                    break
            else:
                raise RuntimeError(f"{target.name} nao encontrado dentro do .task")

    return targets[-1][0]
