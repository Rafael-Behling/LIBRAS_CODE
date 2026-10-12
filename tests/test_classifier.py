"""Testes do classificador de letras (regressao softmax em numpy).

O dataset real so existe depois de gravar com webcam, entao aqui o treino usa
poses sinteticas: o objetivo nao e medir acuracia real, e fixar o
comportamento -- que o modelo separa poses distintas, que salva/carrega sem
perder a predicao, e que a integracao com `recognise` prefere o classificador.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.core.features import HandHistory, hand_features
from libras.signs.classifier import LetterClassifier, fit
from libras.signs.dactylology import LETTERS, recognise

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dactylology import pose  # noqa: E402


# letras de poses bem distintas, para o modelo ter o que aprender
DISTINTAS = ("B", "C", "D", "L", "F", "W", "Y", "O")


def _samples(letters, rots=(-6.0, 0.0, 6.0), scales=(0.16, 0.20)):
    X, y = [], []
    for name in letters:
        for r in rots:
            for s in scales:
                hand = pose(name, rotation=r, scale=s)
                X.append(hand_features(hand).vector())
                y.append(name)
    return X, y


class TestLetterClassifier(unittest.TestCase):
    def test_learns_distinct_poses(self):
        X, y = _samples(DISTINTAS)
        model = fit(X, y, iters=800, lr=0.5)
        pred = model.predict_proba(X).argmax(axis=1)
        idx = {c: i for i, c in enumerate(model.classes)}
        acc = sum(p == idx[t] for p, t in zip(pred, y)) / len(y)
        self.assertGreater(acc, 0.9, f"acuracia de treino baixa: {acc:.2f}")

    def test_predicts_letter_and_confidence(self):
        X, y = _samples(DISTINTAS)
        model = fit(X, y, iters=800)
        letra, prob = model.predict(hand_features(pose("C")).vector())
        self.assertIn(letra, DISTINTAS)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_to_guess_lists_candidates(self):
        X, y = _samples(DISTINTAS)
        model = fit(X, y, iters=800)
        g = model.to_guess(hand_features(pose("B")).vector())
        self.assertIn("B", g.candidates)
        self.assertEqual("B", g.letter)
        self.assertTrue(g.confident, f"B deveria ser certa: {g.candidates}")

    def test_save_load_roundtrip(self):
        X, y = _samples(DISTINTAS)
        model = fit(X, y, iters=400)
        vec = hand_features(pose("D")).vector()
        antes = model.predict_proba(vec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clf.npz"
            model.save(path)
            loaded = LetterClassifier.load(path)
        self.assertEqual(model.classes, loaded.classes)
        self.assertTrue((abs(antes - loaded.predict_proba(vec)) < 1e-6).all())

    def test_recognise_prefers_classifier(self):
        X, y = _samples(DISTINTAS)
        model = fit(X, y, iters=800)
        vec = hand_features(pose("D")).vector()
        esperado = model.predict(vec)[0]
        g = recognise(pose("D"), classifier=model)
        self.assertEqual(esperado, g.letter)

    def test_moving_dynamic_letter_defers_to_rules(self):
        """H e uma letra dinamica (com movimento) e nao esta no modelo. Com a
        mao se movendo, a leitura tem que vir das regras (com candidatos), nao
        de um chute estatico confiante do classificador."""
        X, y = _samples(DISTINTAS)
        model = fit(X, y, iters=800)
        self.assertNotIn("H", model.classes)

        hist = HandHistory()
        for k in range(10):
            h = pose("H")
            h.landmarks = h.landmarks + np.array([k * 0.01, 0, 0], "float32")
            hist.push(h, k / 30.0)

        g = recognise(pose("H"), hist, classifier=model)
        self.assertIsNotNone(g)
        self.assertIn("H", g.candidates)


if __name__ == "__main__":
    unittest.main()
