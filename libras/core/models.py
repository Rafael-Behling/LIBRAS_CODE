"""Download e extracao dos modelos TFLite do hand_landmarker."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# (nome dentro do .task, tamanho minimo em bytes para considerar valido)
NEEDED = (
    ("hand_detector.tflite", 1_000_000),
    ("hand_landmarks_detector.tflite", 1_000_000),
)


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
    if not task_file.exists():
        urllib.request.urlretrieve(MODEL_URL, task_file)

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
