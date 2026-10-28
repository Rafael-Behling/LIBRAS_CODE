"""Testes da camada de interface (HUD).

Nao ha o que comparar pixel a pixel: o objetivo aqui e garantir que o HUD
desenha em qualquer resolucao, preserva o frame recebido e reage a tema,
overlay e estado vazio sem quebrar.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libras.core.phrase import Phrase
from libras.data.synthetic import make_pose
from libras.ui import Hud, HudState, get_theme

RESOLUCOES = ((240, 320), (480, 640), (720, 1280))

FRASE = Phrase(
    text="OLA MUNDO",
    current="R",
    candidates=["R", "P"],
    needs="polegar",
    votes=[("R", 2.4), ("P", 0.8)],
    confidence=0.75,
)


def fundo(w, h):
    return np.full((h, w, 3), 90, np.uint8)


class TestHud(unittest.TestCase):
    def test_render_preserva_o_frame(self):
        for h, w in RESOLUCOES:
            frame = fundo(w, h)
            out = Hud(get_theme("dark")).render(frame, HudState())
            self.assertIs(out, frame, "devia desenhar no proprio frame")
            self.assertEqual(out.shape, (h, w, 3))
            self.assertEqual(out.dtype, np.uint8)

    def test_frase_longa_cabe_na_tela(self):
        p = Phrase(text="A" * 120, current="A", candidates=["A"])
        st = HudState(hands=[], phrase=p)
        for h, w in RESOLUCOES:
            out = Hud(get_theme("dark")).render(fundo(w, h), st)
            self.assertEqual(out.shape, (h, w, 3))

    def test_tema_claro_e_escuro_diferentes(self):
        st = HudState(hands=[], phrase=FRASE, fps=12.0)
        escuro = Hud(get_theme("dark")).render(fundo(640, 480), st).copy()
        claro = Hud(get_theme("light")).render(fundo(640, 480), st).copy()
        self.assertGreater(
            int(claro.mean()), int(escuro.mean()), "o tema claro devia clarear os paineis"
        )

    def test_toggle_troca_o_tema(self):
        hud = Hud(get_theme("dark"))
        self.assertTrue(hud.theme.dark)
        self.assertFalse(hud.toggle_theme().dark)
        self.assertTrue(hud.toggle_theme().dark)

    def test_texto_aparece_no_cartao(self):
        out = Hud(get_theme("dark")).render(fundo(480, 640), HudState(phrase=FRASE))
        rodape = out[480 - 200 :, :]
        brilho = int(rodape.max())
        self.assertGreater(brilho, 200, "nenhum texto claro no rodape")

    def test_estado_vazio_desenha_algo(self):
        base = fundo(480, 640)
        out = Hud(get_theme("dark")).render(base.copy(), HudState())
        self.assertFalse(np.array_equal(out, base))

    def test_overlays_nao_quebram(self):
        for overlay in ("help", "history"):
            st = HudState(phrase=FRASE, history=["OLA", "MUNDO"], overlay=overlay)
            for h, w in ((240, 320), (480, 640)):
                out = Hud(get_theme("dark")).render(fundo(w, h), st)
                self.assertEqual(out.shape, (h, w, 3))

    def test_cor_do_tema_fica_em_bgr(self):
        """Ciano do tema nao pode sair laranja por troca de canal."""
        hud = Hud(get_theme("dark"))
        frame = fundo(480, 640)
        hud._glow(frame, 320, 240, 30, (56, 189, 248), 1.0)
        b, _g, r = frame[240, 320]
        self.assertGreater(int(b), int(r), "o azul devia dominar o ciano")

    def test_escala_fica_no_intervalo(self):
        from libras.ui.hud import _scale

        self.assertAlmostEqual(_scale(1280), 1.0, places=3)
        self.assertGreaterEqual(_scale(320), 0.60)
        self.assertLessEqual(_scale(3840), 1.20)

    def test_cache_de_texto_estabiliza(self):
        """O caret pisca, entao a chave muda a cada quadro: o que nao pode e
        o cache crescer sem limite com o mesmo conteudo."""
        hud = Hud(get_theme("dark"))
        st = HudState(phrase=FRASE, fps=30.0)
        for _ in range(4):
            hud.render(fundo(480, 640), st)
        cheio = len(hud._text_cache)
        for _ in range(20):
            hud.render(fundo(480, 640), st)
        self.assertLessEqual(
            len(hud._text_cache), cheio + 8, "o cache cresceu demais com o mesmo conteudo"
        )

    def test_skeleton_com_mao(self):
        mao = make_pose(curl=(0, 0, 0, 0, 0), scale=0.25)
        st = HudState(hands=[mao], phrase=FRASE, fps=30.0, hold_progress=2, hold_total=5)
        out = Hud(get_theme("dark")).render(fundo(640, 480), st)
        self.assertEqual(out.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
