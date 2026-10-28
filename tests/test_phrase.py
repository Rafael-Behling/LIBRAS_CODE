"""Testes do acumulador de frase.

O que importa aqui e a histerese: uma letra so entra depois de se sustentar
por varios quadros, e uma pose instavel nao pode gerar texto ilegivel.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.core.phrase import PhraseBuilder
from libras.signs.dactylology import LetterGuess


def g(letter, score=0.9, candidates=None, needs=""):
    return LetterGuess(letter=letter, score=score, candidates=candidates or [letter], needs=needs)


def feed(builder, guess, n):
    for _ in range(n):
        builder.push(guess)


class TestPhraseBuilder(unittest.TestCase):
    def test_single_letter_after_hold(self):
        b = PhraseBuilder(hold_frames=5, cooldown=2)
        feed(b, g("A"), 4)
        self.assertEqual(b.text, "", "aceitou antes do tempo")
        feed(b, g("A"), 1)
        self.assertEqual(b.text, "A")

    def test_letter_changes_resets_the_hold(self):
        """Pose instavel entre duas letras nao pode produzir nada."""
        b = PhraseBuilder(hold_frames=5, cooldown=0, space_after=999)
        for i in range(20):
            b.push(g("A" if i % 2 == 0 else "L"))
        self.assertEqual(b.text, "", f"texto indevido: {b.text!r}")

    def test_single_wrong_frame_does_not_reset_hold(self):
        """Um quadro lido como outra letra (flicker) nao reinicia a leitura."""
        b = PhraseBuilder(hold_frames=5, cooldown=0, space_after=999)
        feed(b, g("C"), 4)
        feed(b, g("D"), 1)  # classificador piscou para D
        feed(b, g("C"), 2)  # volta para C
        self.assertEqual(b.text, "C", f"um quadro isolado derrubou: {b.text!r}")

    def test_single_quiet_frame_does_not_reset_hold(self):
        """Um quadro sem leitura (mao perdida) nao reinicia a leitura."""
        b = PhraseBuilder(hold_frames=5, cooldown=0, space_after=999)
        feed(b, g("C"), 4)
        feed(b, None, 1)  # rastreador perdeu a mao
        feed(b, g("C"), 2)
        self.assertEqual(b.text, "C", f"um quadro sem leitura derrubou: {b.text!r}")

    def test_word_sequence(self):
        b = PhraseBuilder(hold_frames=3, cooldown=2, space_after=4)
        for letter in "OLAR":
            feed(b, g(letter), 6)
        self.assertEqual(b.text, "OLAR")

    def test_silence_becomes_space(self):
        b = PhraseBuilder(hold_frames=3, cooldown=2, space_after=4)
        feed(b, g("O"), 6)
        feed(b, None, 8)
        feed(b, g("L"), 6)
        self.assertIn(" ", b.text)
        self.assertEqual(b.text.replace(" ", ""), "OL")

    def test_cooldown_prevents_immediate_repeat(self):
        b = PhraseBuilder(hold_frames=3, cooldown=6)
        feed(b, g("A"), 20)
        self.assertEqual(b.text, "A", f"repetiu durante o cooldown: {b.text!r}")

    def test_ambiguous_letters_are_not_accepted(self):
        """A/I/G tem a mesma pose. Aceitar uma delas produziria palavra
        errada; o certo e nao entrar nada e reportar a ambiguidade."""
        b = PhraseBuilder(hold_frames=3, cooldown=0)
        amb = g("I", score=0.9, candidates=["G", "I", "Y"], needs="direcao do dedo")
        feed(b, amb, 30)
        self.assertEqual(b.text, "", f"aceitou letra ambigua: {b.text!r}")
        self.assertIn("I", builder_candidates(b))

    def test_low_score_is_ignored(self):
        b = PhraseBuilder(hold_frames=3, min_score=0.6)
        feed(b, g("A", score=0.3), 30)
        self.assertEqual(b.text, "")

    def test_backspace_and_reset(self):
        b = PhraseBuilder(hold_frames=2, cooldown=0)
        feed(b, g("O"), 5)
        feed(b, g("L"), 5)
        self.assertEqual(b.text, "OL")
        b.backspace()
        self.assertEqual(b.text, "O")
        b.reset()
        self.assertEqual(b.text, "")

    def test_max_length_is_bounded(self):
        b = PhraseBuilder(hold_frames=1, cooldown=0, space_after=9999, max_len=10)
        for i in range(40):
            b.push(g(chr(ord("A") + i % 26)))
        self.assertLessEqual(len(b.text), 10)


def builder_candidates(b):
    return b.push(g("I", candidates=["G", "I", "Y"])).candidates


if __name__ == "__main__":
    unittest.main(verbosity=2)
