"""Testes do rastreador e das features.

A fixture e uma foto real com duas maos (imagem de exemplo do proprio
MediaPipe, em `tests/data/hands.jpg`): boneco desenhado nao e detectado pelo
detector de palma, entao nao da para montar o frame de teste. O que se vary
sao enquadramentos e escalas da foto inteira, que e o que o app ve de fato.
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.core.features import HandFeatures, HandHistory, hand_features
from libras.core.hand_tracker import (INDEX_MCP, MIDDLE_MCP, PALM_GRID_SIZE, PINKY_MCP,
                                     WRIST, HandTracker)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PHOTO = Path(__file__).resolve().parent / "data" / "hands.jpg"


def _photo():
    return cv2.imread(str(PHOTO)) if PHOTO.exists() else None


@unittest.skipUnless(_photo() is not None, "fixture tests/data/hands.jpg ausente")
class TestTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = _photo()
        cls.tracker = HandTracker(str(MODELS), max_hands=3)

    def test_two_hands_in_a_real_photo(self):
        """O app nunca achava mao nenhuma: a presenca ficava em 0.00-0.43 porque
        o recorte nao vinha do detector de palma."""
        hands = self.tracker.process(self.img)
        self.assertEqual(len(hands), 2, f"so {len(hands)} maos")
        for h in hands:
            self.assertGreaterEqual(h.presence, 0.55, f"presenca {h.presence:.2f}")
            self.assertGreaterEqual(self.tracker._quality(h.landmarks), 0.5)

    def test_no_hands_on_empty_frames(self):
        for frame in (np.full((720, 960, 3), 90, np.uint8),
                      np.random.default_rng(0).integers(0, 255, (720, 960, 3), dtype=np.uint8),
                      self.img[0:180, 0:640]):
            self.assertEqual(self.tracker.process(frame), [], "mao fantasma em frame vazio")

    def test_detects_across_scale_flip_and_letterbox(self):
        """Escala, espelhamento e mudanca de aspecto nao podem derrubar a
        deteccao: a ROI e um retangulo justamente porque o canvas de 192 estica
        o frame, e um quadrado em px da imagem corta a mao num dos eixos."""
        cases = {
            "espelhada": cv2.flip(self.img, 1),
            "0.6x": cv2.resize(self.img, None, fx=0.6, fy=0.6),
            "1.4x": cv2.resize(self.img, None, fx=1.4, fy=1.4),
            "recorte 4:3": self.img[60:870, 40:620],
            "letterbox 1280x960": np.pad(self.img, ((80, 240), (0, 640), (0, 0)),
                                         constant_values=30),
        }
        for name, frame in cases.items():
            with self.subTest(case=name):
                self.assertEqual(len(self.tracker.process(frame)), 2, f"falhou em {name}")

    def test_boxes_stay_hand_sized(self):
        """A track nao pode crescer ate cobrir o frame: era o modo de falha
        recorrente, com caixas de 514 px numa mao de 230 px."""
        for name, frame in (("inteira", self.img),
                            ("1.4x", cv2.resize(self.img, None, fx=1.4, fy=1.4))):
            h, w = frame.shape[:2]
            for hd in self.tracker.process(frame):
                side = max(hd.box[2] - hd.box[0], hd.box[3] - hd.box[1])
                self.assertLessEqual(side, 0.45 * min(w, h) + 1,
                                     f"caixa grande demais em {name}: {side}")
                self.assertGreater(side, 0.05 * min(w, h),
                                  f"caixa pequena demais em {name}: {side}")

    def test_landmarks_are_not_degenerate(self):
        """Landmarks normalizados sao 0..1; trata-los como pixels fazia a track
        encolher para 36 px."""
        for hd in self.tracker.process(self.img):
            lm = hd.landmarks
            self.assertEqual(lm.shape, (21, 3))
            self.assertTrue(np.isfinite(lm).all())
            size = np.linalg.norm(lm[WRIST, :2] - lm[MIDDLE_MCP, :2])
            self.assertGreater(size, 1e-3, "esqueleto colapsado")
            self.assertLessEqual(size, 0.6, "esqueleto absurdo")
            self.assertTrue(((lm[:, :2] > -0.2) & (lm[:, :2] < 1.2)).all(),
                            "landmark fora do quadro")

    def test_single_hand_is_returned_once(self):
        """Duas palmas podem cair na mesma mao (ancoras com caixas parecidas);
        a deduplicacao por sobreposicao de landmarks tem que manter uma so."""
        hands = self.tracker.process(self.img)
        for i, a in enumerate(hands):
            for b in hands[i + 1:]:
                self.assertLess(self.tracker._landmarks_overlap(a.landmarks, b.landmarks), 0.5)

    def test_palm_grid_constant(self):
        """A grade de ancoras foi medida no modelo, nao documentada: se mudar de
        tamanho, a deteccao de palma volta a apontar para o lugar errado."""
        self.assertEqual(PALM_GRID_SIZE, 48 * 24)
        for p in self.tracker.detect_palms(self.img):
            self.assertGreater(p.score, 0.5)
            self.assertTrue(0 < p.cx < self.img.shape[1])
            self.assertTrue(0 < p.cy < self.img.shape[0])
            self.assertGreater(p.side, 0)


@unittest.skipUnless(_photo() is not None, "fixture tests/data/hands.jpg ausente")
class TestQuality(unittest.TestCase):
    """A `presence` chega a 0.9 com landmarks degenerados, entao a plausibilidade
    do esqueleto e o que segura a frase contra letras inventadas."""

    @classmethod
    def setUpClass(cls):
        cls.img = _photo()
        cls.tracker = HandTracker(str(MODELS), max_hands=3)

    def test_real_hand_scores_well(self):
        for hd in self.tracker.process(self.img):
            self.assertGreaterEqual(self.tracker._quality(hd.landmarks), 0.5)

    def test_degenerate_skeleton_scores_zero(self):
        lm = np.zeros((21, 3), np.float32)
        self.assertEqual(self.tracker._quality(lm), 0.0)

    def test_quality_is_independent_of_frame_aspect(self):
        """As razoes sao medidas em pixels. Normalizar x por W e y por H mistura
        escalas e torna o limiar dependente do aspecto: a mesma mao dava nota 0.0
        num frame 640x960 e 0.7 num 640x640. Aqui a mao e a mesma em pixels e so
        o canvas muda, entao a nota nao pode mudar."""
        hands = self.tracker.process(self.img)
        self.assertEqual(len(hands), 2)
        px = hands[0].landmarks[:, :2] * np.array([self.img.shape[1], self.img.shape[0]],
                                                  np.float32)
        scores = []
        for h, w in ((960, 640), (640, 640), (720, 1280), (960, 1600)):
            self.tracker._frame_shape = (h, w, 3)
            lm = np.empty((21, 3), np.float32)
            lm[:, :2] = px / np.array([w, h], np.float32)
            lm[:, 2] = hands[0].landmarks[:, 2]
            scores.append(self.tracker._quality(lm))
        self.assertGreaterEqual(min(scores), 0.5, f"nota muda com o aspecto: {scores}")
        self.assertAlmostEqual(max(scores), min(scores), places=5,
                               msg=f"nota varia com o aspecto: {scores}")

    def test_random_landmarks_score_low(self):
        """21 pontos jogados ao acaso tem falanges proximais em direcoes
        qualquer; numa mao de verdade elas sao quase paralelas."""
        rng = np.random.default_rng(7)
        for _ in range(5):
            lm = rng.random((21, 3)).astype(np.float32)
            self.assertLess(self.tracker._quality(lm), 0.5,
                            "landmarks aleatorios aceitos como mao")


@unittest.skipUnless(_photo() is not None, "fixture tests/data/hands.jpg ausente")
class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = _photo()
        cls.tracker = HandTracker(str(MODELS), max_hands=3)

    def test_vector_matches_named_features(self):
        for hd in self.tracker.process(self.img):
            v = hand_features(hd).vector()
            self.assertEqual(len(v), len(HandFeatures.names()))
            self.assertTrue(np.isfinite(v).all(), "feature nao finita")

    def test_history_vector_dimension(self):
        hist = HandHistory()
        for k in range(6):
            frame = cv2.resize(self.img, None, fx=1.0 + 0.02 * k, fy=1.0 + 0.02 * k)
            hands = self.tracker.process(frame)
            if hands:
                hist.push(hands[0], k / 30.0)
        if len(hist):
            self.assertEqual(len(hist.vector()), len(HandHistory.names()))

    def test_features_are_scale_invariant(self):
        """A mesma pose em dois tamanhos de frame deve gerar o mesmo vetor --
        e o que permite ler a mao a qualquer distancia da camera."""
        vecs = []
        for scale in (1.0, 0.6):
            frame = cv2.resize(self.img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            hands = self.tracker.process(frame)
            self.assertEqual(len(hands), 2, f"nada detectado em {scale}x")
            vecs.append(hand_features(hands[0]).curl)
        self.assertEqual(len(vecs[0]), len(vecs[1]))
        self.assertLess(abs(float(np.subtract(vecs[0], vecs[1]).sum())), 1.0,
                        "curl mudou demais com a escala")


if __name__ == "__main__":
    unittest.main(verbosity=2)
