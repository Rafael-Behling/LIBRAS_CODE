"""Teste de integracao do loop do app (ponta a ponta, sem camera).

O `app.main` com `--source` e `--no-gui` e o mesmo loop da webcam, so sem
janela: para exercita-lo sem dependender da camera nem dos .tflite, o
`HandTracker` e substituido por um esqueleto sintetico (make_pose) estavel, e
o video e um .avi minimo feito na hora. O que se verifica: o loop consome o
video inteiro, a letra sustentada entra na frase e o programa sai com 0.
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from libras.data.synthetic import make_pose


class TestAppIntegration(unittest.TestCase):
    @staticmethod
    def _video(path: Path, n=20, size=(240, 320)):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
        self_ = np.full((size[1], size[0], 3), 60, np.uint8)
        for _ in range(n):
            w.write(self_)
        w.release()

    def _run(self, extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "v.avi"
            self._video(video)
            saida = io.StringIO()
            fake = mock.MagicMock(side_effect=lambda *a, **k: _FakeTracker())
            with mock.patch.object(app, "HandTracker", fake), contextlib.redirect_stdout(saida):
                code = app.main(
                    [
                        "--source",
                        str(video),
                        "--no-gui",
                        "--classifier",
                        str(Path(tmp) / "sem_modelo.npz"),
                        *extra,
                    ]
                )
            return code, saida.getvalue()

    def test_loop_roda_e_saida_e_zero(self):
        code, saida = self._run()
        self.assertEqual(code, 0)
        self.assertIn("frase final:", saida)

    def test_letra_sustentada_entra_na_frase(self):
        code, saida = self._run()
        self.assertEqual(code, 0)
        for linha in saida.splitlines():
            if linha.startswith("frase final:"):
                frase = linha.split(":", 1)[1].strip()
                self.assertEqual(frase, "B", f"frase inesperada: {frase!r}")


class _FakeTracker:
    """Tracker sintetico: sempre a mesma mao aberta (le B pelas regras)."""

    def __init__(self):
        self._pose = make_pose(curl=(0.0, 0.0, 0.0, 0.0, 0.0), thumb_spread=0.0)

    def process(self, frame):
        # Hand e imutavel na pratica; aqui devolvemos sempre o mesmo
        # esqueleto, e o app so le features dele
        return [self._pose]

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
