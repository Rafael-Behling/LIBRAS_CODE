"""Tema visual do HUD: paleta, tipografia e tokens de estilo.

O HUD e um overlay discreto de linha, no estilo sci-fi: quase sem superficie
preenchida, so contornos finos, rotulos em caixa alta espaçados e um brilho
ciano. A paleta sai do `slides.html` (ciano #38bdf8, ambar #f59e0b) sobre um
fundo quase preto.

O tema claro inverte os valores para ambientes iluminados; a estrutura de
tokens e a mesma, nao um desenho novo -- o HUD so le a paleta.

Este projeto escreve os comentarios sem acentos, como no resto do codigo.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

RGB = tuple  # (r, g, b)


@dataclass(frozen=True)
class Theme:
    """Tokens de cor e de estilo de um tema.

    `glass_tint` e `glass_alpha` descrevem um veu escuro por tras do texto: o
    suficiente para o texto aparecer sobre qualquer video, e nada mais -- e um
    degradê de escurecimento, nao um painel de vidro.
    """

    name: str
    dark: bool

    bg: RGB  # fundo da moldura (tambem o veu do degrade)
    surface: RGB  # superficie solida minima (chips, keycaps)
    glass_tint: RGB  # cor do veu atras do texto
    glass_alpha: float
    border: RGB  # linha/base dos contornos
    border_alpha: float

    text: RGB  # texto principal
    text_dim: RGB  # texto secundario
    text_faint: RGB  # texto terciario / placeholders

    primary: RGB  # cor de destaque (ciano)
    accent: RGB  # cor de acento (ambar)
    success: RGB
    danger: RGB

    line_alpha: float  # opacidade das linhas finas (brackets, trilhas)
    glow: float  # intensidade do brilho dos elementos primarios

    bone: RGB  # ossos do esqueleto da mao
    bone_alpha: float
    joint: RGB
    tip: RGB  # pontas dos dedos

    def rgb(self, color: RGB) -> RGB:
        return tuple(int(round(c)) for c in color)

    def bgr(self, color: RGB) -> RGB:
        """Cor em ordem BGR, como o OpenCV e o fundo do frame esperam."""
        r, g, b = self.rgb(color)
        return (b, g, r)

    def with_alpha(self, color: RGB, alpha: float) -> tuple:
        """RGBA a partir de uma cor RGB e uma opacidade 0..1."""
        r, g, b = self.rgb(color)
        return (r, g, b, int(round(max(0.0, min(1.0, alpha)) * 255)))


DARK = Theme(
    name="dark",
    dark=True,
    bg=(6, 10, 18),
    surface=(14, 21, 34),
    glass_tint=(7, 12, 22),
    glass_alpha=0.46,
    border=(120, 150, 180),
    border_alpha=0.30,
    text=(233, 240, 250),
    text_dim=(150, 168, 190),
    text_faint=(92, 108, 130),
    primary=(56, 189, 248),
    accent=(245, 158, 11),
    success=(52, 211, 153),
    danger=(248, 113, 113),
    line_alpha=0.55,
    glow=0.22,
    bone=(190, 214, 238),
    bone_alpha=0.58,
    joint=(120, 150, 180),
    tip=(56, 189, 248),
)

LIGHT = Theme(
    name="light",
    dark=False,
    bg=(244, 247, 251),
    surface=(255, 255, 255),
    glass_tint=(246, 249, 253),
    glass_alpha=0.52,
    border=(30, 41, 59),
    border_alpha=0.26,
    text=(15, 23, 42),
    text_dim=(71, 85, 105),
    text_faint=(130, 144, 164),
    primary=(2, 132, 199),
    accent=(217, 119, 6),
    success=(5, 150, 105),
    danger=(220, 38, 38),
    line_alpha=0.42,
    glow=0.12,
    bone=(51, 65, 85),
    bone_alpha=0.52,
    joint=(100, 116, 139),
    tip=(2, 132, 199),
)

THEMES = {"dark": DARK, "light": LIGHT}


def get_theme(name: str | None) -> Theme:
    """Tema pelo nome ('dark'/'light'); unknown cai no escuro."""
    if not name:
        return DARK
    return THEMES.get(str(name).strip().lower(), DARK)


def toggle(theme: Theme) -> Theme:
    """O outro tema."""
    return LIGHT if theme.dark else DARK


def dim(theme: Theme, amount: float) -> Theme:
    """Copia do tema com menos brilho de linha e de glow (para o degrade)."""
    a = max(0.0, min(1.0, amount))
    return replace(theme, line_alpha=theme.line_alpha * a, glow=theme.glow * a)


# --- tipografia -------------------------------------------------------------

# Onde procurar as fontes. A ordem dentro de cada lista e a preferencia:
# a primeira que existir no sistema vence.
_FONT_DIRS = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".local/share/fonts",
    Path.home() / ".fonts",
)

# (padrao de glob, altura em px, weight)
_FONT_SPECS = {
    "mono": (
        (
            "JetBrainsMono-Bold.ttf",
            "JetBrainsMono-ExtraBold.ttf",
            "JetBrainsMono-SemiBold.ttf",
            "DejaVuSansMono-Bold.ttf",
            "LiberationMono-Bold.ttf",
        ),
        40,
        "bold",
    ),
    "mono_medium": (
        (
            "JetBrainsMono-Medium.ttf",
            "JetBrainsMono-Regular.ttf",
            "DejaVuSansMono.ttf",
            "LiberationMono-Regular.ttf",
        ),
        40,
        "regular",
    ),
    # rotulos: sans leve e espaçado, a voz "tecnica" do HUD
    "label": (
        ("Roboto-Medium.ttf", "Roboto-Regular.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
        40,
        "medium",
    ),
    "label_light": (
        ("Roboto-Regular.ttf", "Roboto-Light.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
        40,
        "regular",
    ),
    "sans_bold": (
        ("Roboto-Bold.ttf", "Roboto-Black.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"),
        40,
        "bold",
    ),
}

# cache: papel -> (caminho, tamanho, weight)
_RESOLVED: dict = {}


def _search(names) -> Path | None:
    """Primeiro arquivo que casar com um dos nomes, em qualquer diretorio de fonte."""
    for name in names:
        for base in _FONT_DIRS:
            if not base.is_dir():
                continue
            # o nome pode aparecer com prefixo de familia (ex.: NotoSansDisplay-Bold)
            for path in base.rglob(name):
                if path.is_file():
                    return path
    return None


def font_path(role: str) -> Path | None:
    """Caminho da fonte de um papel (mono, label, sans_bold, ...)."""
    spec = _FONT_SPECS.get(role)
    if spec is None:
        return None
    return _search(spec[0])


def font_size(role: str, size: int):
    """Carrega a fonte de um papel num tamanho, com cache.

    Sem Pillow instalado a funcao devolve None e o HUD cai para um desenho
    sem texto (nao deve quebrar o app inteiro por causa de fonte).
    """
    key = (role, size)
    if key in _RESOLVED:
        return _RESOLVED[key]
    from PIL import ImageFont

    spec = _FONT_SPECS.get(role)
    font = None
    if spec is not None:
        path = _search(spec[0])
        if path is not None:
            try:
                font = ImageFont.truetype(str(path), size)
            except OSError:
                font = None
    _RESOLVED[key] = font
    return font


def scale_font(role: str, size: int, factor: float):
    """Fonte de um papel escalada por um fator (para o alto contraste do video)."""
    return font_size(role, max(6, int(round(size * factor))))
