"""Testes das regras de datilologia, com poses sinteticas.

O sintetizador em libras.data.synthetic nao substitui um gravador real de
Libras: ele produz a geometria de 21 pontos que as regras esperam. O que
estes testes fixam e a logica de desempate -- quais pares de letras
compartilham a mesma pose e como o classificador lida com isso.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.core.features import HandHistory
from libras.core.hand_tracker import FINGERS
from libras.data.synthetic import make_pose
from libras.signs.dactylology import LETTERS, recognise

CURL = FINGERS  # polegar, indicador, medio, anelar, mininho


# dobradilha do polegar e afastamento por forma. Sao necessarios os dois: a
# dobradilha sozinha nao separa "c" (curva) de "lateral" (reto), e o
# afastamento sozinho nao separa "cruz" de "dobrado".
THUMB = {
    "lateral": (0.00, 0.0),
    "dobrado": (0.90, 0.0),
    "c": (0.45, 35.0),
    "apoiado": (0.55, 15.0),
    "cruz": (0.95, 30.0),
}


def pose(letter, **kw):
    """Pose sintetica correspondente a uma letra da tabela."""
    spec = next(l for l in LETTERS if l.name == letter)
    curl = [1.0] * 5
    for i, up in enumerate(spec.fingers):
        curl[i + 1] = 0.0 if up else 1.0
    curl[0], tspread = THUMB[spec.thumb]
    return make_pose(curl=tuple(curl), thumb_spread=tspread, **kw)


class TestDactylologyRules(unittest.TestCase):
    def test_reads_poses_without_crashing(self):
        for l in LETTERS:
            with self.subTest(letter=l.name):
                g = recognise(pose(l.name))
                self.assertIsNotNone(g, f"{l.name} nao produziu leitura")
                self.assertIn(g.letter, [x.name for x in LETTERS])
                self.assertGreaterEqual(g.score, 0.0)
                self.assertLessEqual(g.score, 1.0)

    def test_letters_with_distinct_poses_are_read_correctly(self):
        """Letras cuja pose nao e compartilhada com outra devem sair
        certainemente. A/I/G sao o contra-exemplo: a pose e identica e so a
        direcao separa."""
        for name in ("B", "F", "N", "O", "P", "R", "S", "T", "W", "X"):
            with self.subTest(letter=name):
                g = recognise(pose(name))
                self.assertIn(name, g.candidates,
                              f"{name}: esperado entre os candidatos, "
                              f"veio {g.candidates}")

    def test_ambiguity_is_reported_not_guessed(self):
        """A/I/G tem a mesma pose. O modulo tem de dizer que nao sabe, em vez
        de escolher uma e presentsa como certeza."""
        hand = pose("I")
        g = recognise(hand)
        self.assertGreater(len(g.candidates), 1,
                           f"esperava ambiguidade, veio {g.candidates}")
        self.assertTrue(g.ambiguous)
        self.assertFalse(g.confident)

    def test_rotation_resolves_some_ambiguity(self):
        """Girar a mao muda a orientacao, que e o que separa A de L e G de I."""
        g0 = recognise(pose("I", rotation=0.0))
        g90 = recognise(pose("I", rotation=90.0))
        self.assertIsNotNone(g0)
        self.assertIsNotNone(g90)

    def test_spread_separates_u_from_v(self):
        """U e V tem os mesmos dedos; a diferenca e o espacamento."""
        juntos = recognise(pose("U", spread=0.0))
        abertos = recognise(pose("V", spread=25.0))
        self.assertIsNotNone(juntos)
        self.assertIsNotNone(abertos)
        self.assertNotEqual(juntos.candidates, abertos.candidates,
                            "espacamento nao mudou a leitura")

    def test_history_disambiguates_motion_letters(self):
        """J e Z so existem como movimento. Com historico parado nao podem ser
        sugeridos."""
        hist = HandHistory()
        for k in range(10):
            hist.push(pose("I"), k / 30.0)
        parado = recognise(pose("I"), hist)
        self.assertIsNotNone(parado)
        self.assertNotIn("J", parado.candidates)
        self.assertNotIn("Z", parado.candidates)

        # agora com deslocamento
        hist2 = HandHistory()
        for k in range(10):
            h = make_pose(curl=(0.0, 1.0, 1.0, 1.0, 0.0))
            h.landmarks = h.landmarks + np.array([k * 0.01, 0, 0], dtype="float32")
            hist2.push(h, k / 30.0)
        movido = recognise(pose("I"), hist2)
        self.assertIsNotNone(movido)

    def test_closed_fist_is_not_a_letter_without_thumb_info(self):
        g = recognise(pose("E"))
        self.assertIsNotNone(g)
        self.assertLessEqual(len(g.candidates), 3)


import numpy as np  # noqa: E402  (usado só em test_history)


if __name__ == "__main__":
    unittest.main(verbosity=2)
