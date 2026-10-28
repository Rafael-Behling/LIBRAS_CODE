"""HUD de linha sobre o video, no estilo sci-fi discreto.

O desenho e quase todo contorno: brackets nos cantos, um rotulo tecnico em
caixa alta, a frase grande em monoespacada e um badge com a letra em leitura
mais um anel de progresso. Nada de paineis preenchidos; onde o texto precisa
de contraste entra so um veu escuro em degradê, que some nas bordas.

Tudo e desenhado direto no frame (BGR, uint8). Este projeto escreve os
comentarios sem acentos, como no resto do codigo.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .theme import DARK, Theme, font_size, toggle

MARGIN = 34
FONT_ROLES = ("mono", "mono_medium", "label", "label_light", "sans_bold")

# ligacoes do esqueleto: do pulso aos MCPs e cada cadeia de dedo
BONES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)
TIPS = (4, 8, 12, 16, 20)


# --- utilidades -------------------------------------------------------------


def _scale(width: int) -> float:
    """Fator de escala da interface pela largura do frame."""
    return max(0.60, min(1.20, width / 1280.0))


def _ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _mix(a, b, t: float):
    return tuple(int(round(x * (1.0 - t) + y * t)) for x, y in zip(a, b, strict=False))


def _blend(frame, x0, y0, x1, y1, color, alpha: float) -> None:
    """Mistura uma cor solida sobre um retangulo do frame (em BGR)."""
    if alpha <= 0:
        return
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1].astype(np.float32)
    a = min(1.0, alpha)
    region *= 1.0 - a
    region += np.asarray(color, np.float32)[None, None, :] * a
    frame[y0:y1, x0:x1] = region.astype(np.uint8)


# --- estado -----------------------------------------------------------------


@dataclass
class HudState:
    """O que o HUD precisa saber a cada quadro."""

    hands: list = field(default_factory=list)
    phrase: object = None
    fps: float = 0.0
    hold_progress: float = 0.0
    hold_total: float = 1.0
    history: list = field(default_factory=list)
    overlay: str = ""  # "", "help" ou "history"


class Hud:
    """Camada de interface sobre o video.

    Guarda o tema, o cache de tiles de texto e o estado das animacoes entre
    quadros. `render` altera o frame recebido e o devolve.
    """

    def __init__(self, theme: Theme = DARK):
        self.theme = theme
        self._t = 0.0
        self._last_t: float | None = None
        self._letter: str | None = None
        self._letter_at = -10.0
        self._toast = ""
        self._toast_at = -10.0
        self._overlay_at = -10.0
        # tiles de texto ja rasterizados: os rotulos e a marca se repetem muito
        # entre quadros, so a frase muda
        self._text_cache: dict = {}

    # --- tema e avisos ---

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme

    def toggle_theme(self) -> Theme:
        self.theme = toggle(self.theme)
        return self.theme

    def notify(self, text: str) -> None:
        """Mostra um aviso temporario no rodape."""
        self._toast = text
        self._toast_at = self._t

    # --- loop ---

    def render(self, frame: np.ndarray, state: HudState) -> np.ndarray:
        """Desenha o HUD por cima do frame e o devolve."""
        now = time.time()
        dt = 0.033 if self._last_t is None else min(0.1, max(0.0, now - self._last_t))
        self._last_t = now
        self._t += dt

        phrase = state.phrase
        current = getattr(phrase, "current", None)
        if current != self._letter:
            self._letter = current
            self._letter_at = self._t
        if state.overlay and self._overlay_at < 0:
            self._overlay_at = self._t
        elif not state.overlay:
            self._overlay_at = -10.0

        h, w = frame.shape[:2]
        self._skeleton(frame, state.hands)
        self._frame_marks(frame, w, h, state)
        self._phrase(frame, w, h, state)
        self._reading_badge(frame, w, h, state)
        if not state.hands and not state.overlay:
            self._idle(frame, w, h)
        self._overlay(frame, w, h, state)
        self._toast_pill(frame, w, h)
        return frame

    # --- texto ---------------------------------------------------------------

    def _font(self, role: str, size: int):
        return font_size(role, max(9, int(round(size))))

    def _fonts(self, width: int) -> dict:
        """Conjunto de fontes do quadro, escalado pela largura da janela."""
        s = _scale(width)
        return {
            "label": self._font("label", 12 * s),
            "tiny": self._font("label_light", 11 * s),
            "mark": self._font("mono", 15 * s),
            "letter": self._font("mono", 46 * s),
            "phrase": self._font("mono", 40 * s),
            "body": self._font("label", 14 * s),
            "key": self._font("mono_medium", 12 * s),
            "h2": self._font("label", 22 * s),
        }

    def _text(
        self,
        frame,
        x: float,
        y: float,
        text: str,
        font,
        color,
        alpha: int = 255,
        align: str = "left",
        tracking: float = 0.0,
    ) -> float:
        """Escreve texto com fonte real e devolve a largura usada.

        O tile rasterizado fica em cache: rotulos e a marca se repetem muito
        entre quadros, e redesenhar a fonte a cada quadro seria desperdicio.
        """
        if not text or font is None:
            return 0.0
        fill = (*self.theme.rgb(color), int(max(0, min(255, alpha))))
        key = (text, id(font), fill, round(float(tracking), 2))
        hit = self._text_cache.get(key)
        if hit is not None:
            tile, width, ox, oy = hit
            # toca o item: o dict guarda a ordem de uso, e os tiles quentes
            # (rotulos fixos, que se repetem todo quadro) nao podem cair
            self._text_cache[key] = self._text_cache.pop(key)
        else:
            if tracking:
                width = sum(font.getlength(c) for c in text) + tracking * (len(text) - 1)
            else:
                width = font.getlength(text)
            asc, desc = font.getmetrics()
            pad = 4
            tw = max(1, int(math.ceil(width)) + 2 * pad)
            th = max(1, asc + desc + 2 * pad)
            from PIL import Image, ImageDraw

            tile = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            d = ImageDraw.Draw(tile)
            if tracking:
                cx = float(pad)
                for ch in text:
                    d.text((cx, pad), ch, font=font, fill=fill)
                    cx += font.getlength(ch) + tracking
            else:
                d.text((pad, pad), text, font=font, fill=fill)
            ox, oy = pad, pad
            if len(self._text_cache) > 512:
                # LRU simples: em vez de limpar tudo aos 512 itens (o que
                # re-rasterizaria ate os rotulos fixos do proximo quadro),
                # evicta os mais antigos e preserva os quentes
                excedente = len(self._text_cache) - 384
                for antigo in list(self._text_cache)[:excedente]:
                    self._text_cache.pop(antigo, None)
            self._text_cache[key] = (tile, float(width), ox, oy)

        if align == "center":
            px = x - width / 2.0
        elif align == "right":
            px = x - width
        else:
            px = x
        self._paste(frame, tile, px - ox, y - oy)
        return float(width)

    def _paste(self, frame, tile, x: float, y: float) -> None:
        """Compoe um tile RGBA no frame, com alpha blend."""
        tw, th = tile.size
        fh, fw = frame.shape[:2]
        ix, iy = int(round(x)), int(round(y))
        x0, y0 = max(0, ix), max(0, iy)
        x1, y1 = min(fw, ix + tw), min(fh, iy + th)
        if x1 <= x0 or y1 <= y0:
            return
        crop = tile.crop((x0 - ix, y0 - iy, x0 - ix + (x1 - x0), y0 - iy + (y1 - y0)))
        t = np.asarray(crop, np.float32)
        region = frame[y0:y1, x0:x1].astype(np.float32)
        a = t[:, :, 3:4] / 255.0
        out = region * (1.0 - a) + t[:, :, :3] * a
        frame[y0:y1, x0:x1] = out.astype(np.uint8)

    def _text_w(self, font, text: str, tracking: float = 0.0) -> float:
        if not text or font is None:
            return 0.0
        if tracking:
            return sum(font.getlength(c) for c in text) + tracking * (len(text) - 1)
        return font.getlength(text)

    def _fit_size(self, text: str, max_w: float, hi: int, lo: int, role: str = "mono") -> int:
        """Maior tamanho de fonte em que o texto cabe na largura."""
        size = hi
        while size > lo:
            f = self._font(role, size)
            if f is not None and f.getlength(text) <= max_w:
                return size
            size -= 2
        return lo

    # --- primitivas de desenho ----------------------------------------------

    def _vgrad(self, frame, x0, y0, x1, y1, top: float, bottom: float, color) -> None:
        """Veu vertical: opacidade cai de `top` para `bottom` (ligado, suave)."""
        h, w = frame.shape[:2]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0 or max(top, bottom) <= 0:
            return
        rows = y1 - y0
        a = np.linspace(top, bottom, rows, dtype=np.float32)[:, None, None]
        region = frame[y0:y1, x0:x1].astype(np.float32)
        region *= 1.0 - a
        region += np.asarray(color, np.float32)[None, None, :] * a
        frame[y0:y1, x0:x1] = region.astype(np.uint8)

    def _glow(self, frame, cx: float, cy: float, radius: float, color, strength: float) -> None:
        """Halo radial aditivo, em baixa resolucao (e so uma mancha suave)."""
        if strength <= 0 or radius <= 1:
            return
        h, w = frame.shape[:2]
        x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
        x1, y1 = min(w, int(cx + radius) + 1), min(h, int(cy + radius) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        q = 3
        sw, sh = max(1, (x1 - x0) // q), max(1, (y1 - y0) // q)
        yy, xx = np.mgrid[0:sh, 0:sw]
        gx = (cx - x0) / q
        gy = (cy - y0) / q
        d = np.hypot(xx - gx, yy - gy) / max(radius / q, 1e-6)
        g = (np.clip(1.0 - d, 0.0, 1.0) ** 2)[:, :, None] * strength
        region = cv2.resize(frame[y0:y1, x0:x1], (sw, sh), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
        region += np.asarray(self.theme.bgr(color), np.float32)[None, None, :] * g
        out = cv2.resize(region, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)

    def _ring(
        self,
        frame,
        cx: float,
        cy: float,
        radius: float,
        thickness: float,
        frac: float,
        color,
        alpha: float = 1.0,
    ) -> None:
        """Arco de progresso, do topo no sentido horario."""
        if frac <= 0:
            return
        frac = min(1.0, frac)
        fh, fw = frame.shape[:2]
        pad = thickness + 3
        x0, y0 = max(0, int(cx - radius - pad)), max(0, int(cy - radius - pad))
        x1 = min(fw, int(cx + radius + pad) + 1)
        y1 = min(fh, int(cy + radius + pad) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - cx, yy - cy)
        band = np.abs(dist - radius) <= thickness
        ang = (np.degrees(np.arctan2(xx - cx, -(yy - cy))) + 360.0) % 360.0
        within = ang <= frac * 360.0
        m = (band * within * alpha)[:, :, None]
        region = frame[y0:y1, x0:x1].astype(np.float32)
        region *= 1.0 - m
        region += np.asarray(self.theme.bgr(color), np.float32)[None, None, :] * m
        frame[y0:y1, x0:x1] = region.astype(np.uint8)

    def _brackets(
        self, frame, x0, y0, x1, y1, size: float, color, alpha: float, thickness: int = 1
    ) -> None:
        """Cantos em 'L': marca a area sem fechar o retangulo."""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        c = self.theme.bgr(color)
        over = cv2.LINE_AA
        for px, py, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(frame, (px, py), (px + int(sx * size), py), c, thickness, over)
            cv2.line(frame, (px, py), (px, py + int(sy * size)), c, thickness, over)

    # --- esqueleto da mao ----------------------------------------------------

    def _skeleton(self, frame, hands: list) -> None:
        """Ossos, juntas e pontas da mao, com um halo nas pontas."""
        if not hands:
            return
        th = self.theme
        fh, fw = frame.shape[:2]
        for hand in hands:
            px = hand.landmarks[:, :2] * np.array([fw, fh])
            pts = px.astype(int)
            pad = 46
            x0 = max(0, int(px[:, 0].min()) - pad)
            y0 = max(0, int(px[:, 1].min()) - pad)
            x1 = min(fw, int(px[:, 0].max()) + pad)
            y1 = min(fh, int(px[:, 1].max()) + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = frame[y0:y1, x0:x1]
            local = pts - np.array([x0, y0])

            # halo das pontas, em baixa resolucao
            gh, gw = sub.shape[:2]
            gq = 4
            gs = cv2.resize(
                sub, (max(1, gw // gq), max(1, gh // gq)), interpolation=cv2.INTER_LINEAR
            )
            tip_col = self.theme.bgr(th.tip)
            for i in TIPS:
                cv2.circle(
                    gs,
                    (int(local[i][0] // gq), int(local[i][1] // gq)),
                    max(1, 12 // gq),
                    tip_col,
                    -1,
                    lineType=cv2.LINE_AA,
                )
            gs = cv2.GaussianBlur(gs, (0, 0), 8.0 / gq)
            gs = cv2.resize(gs, (gw, gh), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(gs, 0.35 * (0.6 + self.theme.glow), sub, 1.0, 0, dst=sub)

            layer = sub.copy()
            for a, b in BONES:
                cv2.line(
                    layer,
                    tuple(local[a]),
                    tuple(local[b]),
                    self.theme.bgr(th.bone),
                    2,
                    lineType=cv2.LINE_AA,
                )
            cv2.addWeighted(layer, th.bone_alpha, sub, 1.0 - th.bone_alpha, 0, dst=sub)

            for i in range(1, 21):
                r = 3 if i in TIPS else 2
                cv2.circle(
                    sub, tuple(local[i]), r, self.theme.bgr(th.joint), -1, lineType=cv2.LINE_AA
                )
            for i in TIPS:
                cv2.circle(
                    sub, tuple(local[i]), 5, self.theme.bgr(th.tip), -1, lineType=cv2.LINE_AA
                )
                cv2.circle(sub, tuple(local[i]), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            # pulso em anel: marca onde a mao comeca
            cv2.circle(sub, tuple(local[0]), 9, self.theme.bgr(th.primary), 1, lineType=cv2.LINE_AA)
            self._glow(frame, pts[0][0], pts[0][1], 26, th.primary, self.theme.glow * 0.6)

            bx = [int(v) for v in hand.box]
            self._brackets(
                frame, bx[0], bx[1], bx[2], bx[3], 18, th.primary, self.theme.line_alpha * 0.9
            )

    # --- marcas de moldura ---------------------------------------------------

    def _frame_marks(self, frame, w: int, h: int, state: HudState) -> None:
        """Brackets da area util + rotulos tecnicos nos cantos."""
        sc = _scale(w)
        m = int(MARGIN * sc)
        fonts = self._fonts(w)
        th = self.theme

        self._brackets(frame, m, m, w - m, h - m, 26 * sc, th.border, th.line_alpha * 0.55, 1)

        self._text(frame, m + 14 * sc, m + 8 * sc, "LIBRAS", fonts["mark"], th.text, tracking=4.0)
        self._text(
            frame, m + 14 * sc, m + 42 * sc, "→ PORTUGUÊS", fonts["tiny"], th.primary, tracking=3.0
        )

        n = len(state.hands)
        status = "RECONHECENDO" if n else "AGUARDANDO"
        col = th.success if n else th.accent
        self._text(
            frame,
            w - m - 14 * sc,
            m + 8 * sc,
            status,
            fonts["tiny"],
            col,
            align="right",
            tracking=2.4,
        )
        if w >= 700:
            self._text(
                frame,
                w - m - 14 * sc,
                m + 30 * sc,
                f"{state.fps:4.1f} FPS",
                fonts["tiny"],
                th.text_faint,
                align="right",
                tracking=1.4,
            )

    # --- frase ---------------------------------------------------------------

    def _phrase(self, frame, w: int, h: int, state: HudState) -> None:
        """Frase grande, com um veu em degradê por tras para dar contraste."""
        phrase = state.phrase
        text = (getattr(phrase, "text", "") or "") if phrase else ""
        current = getattr(phrase, "current", None) if phrase else None
        sc = _scale(w)
        fonts = self._fonts(w)
        th = self.theme

        shown = text + (current or "")
        m = int(MARGIN * sc)
        x = m + 96 * sc
        base = h - m - 24 * sc
        max_w = max(60.0, (w - m - 14 * sc) - x)

        size = self._fit_size(shown, max_w, int(40 * sc), int(24 * sc))
        f = fonts["phrase"] if abs(size - fonts["phrase"].size) < 1 else self._font("mono", size)

        self._text(frame, x, base - 46 * sc, "FRASE", fonts["tiny"], th.text_faint, tracking=3.0)
        self._vgrad(
            frame,
            x - 16 * sc,
            base - 20 * sc,
            w - m,
            base + 40 * sc,
            0.0,
            th.glass_alpha,
            th.glass_tint,
        )
        if shown.strip():
            self._text(frame, x, base, shown, f, th.text)
            if current:
                cw = self._text_w(f, text)
                self._text(frame, x + cw, base, current, f, th.primary)
                caret = x + cw + self._text_w(f, current) + 3 * sc
                blink = 0.5 + 0.5 * math.sin(self._t * 3.2)
                level = round(blink * 12) / 12.0
                self._text(frame, caret, base, "▌", f, th.primary, alpha=int(90 + 165 * level))
        else:
            self._text(frame, x, base, "mostre uma letra à câmera", fonts["body"], th.text_faint)

        self._vline(frame, x - 16 * sc, base - 6 * sc, base + 40 * sc, th.primary, th.line_alpha)

    # --- badge da letra em leitura ------------------------------------------

    def _reading_badge(self, frame, w: int, h: int, state: HudState) -> None:
        phrase = state.phrase
        current = getattr(phrase, "current", None) if phrase else None
        sc = _scale(w)
        fonts = self._fonts(w)
        th = self.theme

        m = int(MARGIN * sc)
        cx = m + 40 * sc
        cy = h - m - 56 * sc
        base_r = 30 * sc

        anim = _ease_out((self._t - self._letter_at) / 0.22)
        r = base_r * (0.90 + 0.10 * anim)
        self._glow(frame, cx, cy, r + 22 * sc, th.primary, self.theme.glow * (0.6 + 0.4 * anim))

        cv2.circle(
            frame, (int(cx), int(cy)), int(r), self.theme.bgr(th.border), 1, lineType=cv2.LINE_AA
        )
        frac = 0.0
        if state.hold_total > 0:
            frac = max(0.0, min(1.0, state.hold_progress / state.hold_total))
        if frac > 0:
            self._ring(frame, cx, cy, r + 7 * sc, max(2.0, 2.6 * sc), frac, th.primary, 0.95)

        if current:
            self._text(frame, cx, cy, current, fonts["letter"], th.text, align="center")
        else:
            self._text(frame, cx, cy, "-", fonts["letter"], th.text_faint, align="center")

        label = "LEITURA" if current else "SEM LETRA"
        self._text(
            frame,
            cx,
            cy + r + 14 * sc,
            label,
            fonts["tiny"],
            th.text_faint,
            align="center",
            tracking=2.0,
        )

    # --- estado ocioso -------------------------------------------------------

    def _idle(self, frame, w: int, h: int) -> None:
        """Dica discreta quando nao ha mao nem overlay."""
        fonts = self._fonts(w)
        th = self.theme
        pulse = 0.5 + 0.5 * math.sin(self._t * 1.6)
        # quantiza a pulsacao: sem isso a cor muda a cada quadro e o cache de
        # texto nunca acerta (rasterizar fonte por quadro e caro)
        level = round(pulse * 12) / 12.0
        self._text(
            frame,
            w / 2,
            h * 0.16,
            "POSICIONE A MÃO NA CÂMERA",
            fonts["body"],
            _mix(th.text_faint, th.primary, 0.5 * level),
            align="center",
            tracking=4.0,
        )

    # --- overlays ------------------------------------------------------------

    def _overlay(self, frame, w: int, h: int, state: HudState) -> None:
        if not state.overlay:
            return
        sc = _scale(w)
        m = int(MARGIN * sc)
        panel = (m + 40 * sc, m + 70 * sc, w - 2 * m - 80 * sc, h - 2 * m - 140 * sc)
        self._panel_bg(frame, panel, 0.90)
        if state.overlay == "help":
            self._help(frame, panel, w)
        elif state.overlay == "history":
            self._history(frame, panel, w, state.history)

    def _panel_bg(self, frame, rect, darkness: float) -> None:
        """Veu solido + brackets: um painel sem superficie pintada."""
        x, y, cw, ch = [int(v) for v in rect]
        th = self.theme
        _blend(frame, x, y, x + cw, y + ch, th.bgr(th.bg), darkness)
        self._brackets(frame, x, y, x + cw, y + ch, 22 * _scale(cw), th.border, th.line_alpha, 1)

    def _help(self, frame, rect, w: int) -> None:
        x, y, cw, ch = [int(v) for v in rect]
        sc = _scale(w)
        fonts = self._fonts(w)
        th = self.theme
        rows = [
            ("ESPAÇO", "confirmar a letra em leitura"),
            ("B", "apagar o último caractere"),
            ("C", "limpar a frase"),
            ("ENTER", "salvar a frase no histórico"),
            ("H", "abrir o histórico"),
            ("T", "alternar tema claro/escuro"),
            ("?  /", "abrir esta ajuda"),
            ("Q / ESC", "sair"),
        ]
        self._text(frame, x + 26 * sc, y + 22 * sc, "ATALHOS", fonts["h2"], th.text, tracking=3.0)
        self._text(
            frame,
            x + 26 * sc,
            y + 58 * sc,
            "letras entram sozinhas após a pose se manter",
            fonts["tiny"],
            th.text_faint,
            tracking=1.0,
        )
        self._hline(
            frame, x + 26 * sc, y + 82 * sc, x + cw - 26 * sc, th.border, th.line_alpha * 0.6
        )

        row_y = y + 100 * sc
        step = min(34 * sc, max(18.0, (ch - 130 * sc) / len(rows)))
        key_w = 96 * sc
        for i, (keys, desc) in enumerate(rows):
            ry = row_y + i * step
            self._keycap(frame, x + 26 * sc, ry - 6 * sc, key_w, 22 * sc, keys)
            self._text(frame, x + 26 * sc + key_w + 18 * sc, ry, desc, fonts["body"], th.text_dim)

    def _history(self, frame, rect, w: int, items: list) -> None:
        x, y, cw, ch = [int(v) for v in rect]
        sc = _scale(w)
        fonts = self._fonts(w)
        th = self.theme
        self._text(frame, x + 26 * sc, y + 22 * sc, "HISTÓRICO", fonts["h2"], th.text, tracking=3.0)
        self._hline(
            frame, x + 26 * sc, y + 58 * sc, x + cw - 26 * sc, th.border, th.line_alpha * 0.6
        )
        if not items:
            self._text(
                frame, x + 26 * sc, y + 80 * sc, "nenhuma frase salva", fonts["body"], th.text_faint
            )
            return
        row_y = y + 80 * sc
        step = min(34 * sc, max(18.0, (ch - 100 * sc) / max(1, len(items))))
        for i, item in enumerate(items[-10:]):
            ry = row_y + i * step
            self._text(
                frame, x + 26 * sc, ry, f"{i + 1:02d}", fonts["key"], th.primary, tracking=1.0
            )
            self._text(frame, x + 70 * sc, ry, item, fonts["phrase"], th.text)

    def _keycap(self, frame, x, y, w, h, text: str) -> None:
        x, y, w, h = int(x), int(y), int(w), int(h)
        th = self.theme
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.theme.bgr(th.surface), -1)
        cv2.rectangle(
            frame, (x, y), (x + w, y + h), self.theme.bgr(th.border), 1, lineType=cv2.LINE_AA
        )
        f = self._font("mono_medium", h * 0.62)
        self._text(frame, x + w / 2, y + h * 0.20, text, f, th.text_dim, align="center")

    # --- rodape --------------------------------------------------------------

    def _toast_pill(self, frame, w: int, h: int) -> None:
        """Aviso rapido no centro do rodape."""
        if not self._toast:
            return
        age = self._t - self._toast_at
        if age > 2.4:
            return
        alpha = 1.0 if age < 1.8 else max(0.0, 1.0 - (age - 1.8) / 0.6)
        sc = _scale(w)
        fonts = self._fonts(w)
        th = self.theme
        text = self._toast
        fw = self._text_w(fonts["body"], text)
        pw = fw + 44 * sc
        ph = 34 * sc
        x = (w - pw) / 2
        y = h - int(MARGIN * sc) - ph - 8 * sc
        _blend(frame, x, y, x + pw, y + ph, th.bgr(th.surface), 0.88 * alpha)
        self._brackets(frame, x, y, x + pw, y + ph, 10 * sc, th.primary, th.line_alpha * alpha)
        self._text(
            frame,
            w / 2,
            y + (ph - 16 * sc) / 2,
            text,
            fonts["body"],
            th.text,
            alpha=int(255 * alpha),
            align="center",
        )

    # --- linha ---------------------------------------------------------------

    def _hline(self, frame, x0, y, x1, color, alpha: float) -> None:
        """Linha horizontal de 1 px, misturada com o fundo."""
        y = int(max(0, min(frame.shape[0] - 1, y)))
        x0, x1 = int(max(0, x0)), int(min(frame.shape[1], x1))
        if x1 <= x0:
            return
        col = np.asarray(self.theme.bgr(color), np.float32)
        region = frame[y : y + 1, x0:x1].astype(np.float32)
        region *= 1.0 - alpha
        region += col[None, None, :] * alpha
        frame[y : y + 1, x0:x1] = region.astype(np.uint8)

    def _vline(self, frame, x, y0, y1, color, alpha: float) -> None:
        """Linha vertical de 1 px, misturada com o fundo."""
        x = int(max(0, min(frame.shape[1] - 1, x)))
        y0, y1 = int(max(0, y0)), int(min(frame.shape[0], y1))
        if y1 <= y0:
            return
        col = np.asarray(self.theme.bgr(color), np.float32)
        region = frame[y0:y1, x : x + 1].astype(np.float32)
        region *= 1.0 - alpha
        region += col[None, None, :] * alpha
        frame[y0:y1, x : x + 1] = region.astype(np.uint8)
