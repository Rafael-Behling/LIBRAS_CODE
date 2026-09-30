"""Reconhecedor de datilologia (A-Z) por regras geometricas.

Escolha: regras em vez de classificador treinado. O projeto nao tem dataset,
e a datilologia e descrita por posicoes bem definidas -- quantos dedos
estendidos e onde esta o polegar. Um classificador sem treino daria
acertarias arbitrarias; regras sao legiveis e ajustaveis.

Ambiguidade assumida e explicita: varias letras tem a MESMA pose e so se
distinguem por posicao, movimento ou orientacao. 'A', 'L' e 'D' sao o mesmo
configuracao de dedos com o polegar em tres lugares; 'E', 'M', 'O' e 'X' sao
mao fechada. Este modulo devolve `candidates` com todas as leituras plausiveis
e um `needs` indicando o que falta para decidir. O app usa isso para pedir ao
usuario um ajuste, em vez de chutar uma letra.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.features import HandFeatures, hand_features
from ..core.hand_tracker import FINGERS, FINGERS_4, Hand

# --- descricao das letras -------------------------------------------------

# polegar
TL, TD, TC, TA, TX = "lateral", "dobrado", "c", "apoiado", "cruz"

# (polegar, index, middle, ring, pinky)
@dataclass(frozen=True)
class Letter:
    name: str
    thumb: str
    fingers: tuple
    # o que diferencia esta letra das outras de mesma pose
    needs: str = ""


LETTERS: tuple = (
    # A e um punho fechado com o polegar ao lado -- nao indicador estendido.
    # Com o indicador estendido, A ficaria indistinguivel de L e D.
    Letter("A", TL, (0, 0, 0, 0), "posicao do polegar"),
    Letter("B", TL, (1, 1, 1, 1)),
    # C tem os quatro dedos curvados em arco (indicador a mininho), nao so
    # os dois primeiros -- senao C colide com P e o proprio teste que
    # separa U de V nao tem como distingui-lo.
    Letter("C", TC, (1, 1, 1, 1)),
    Letter("D", TA, (1, 0, 0, 0), "polegar apoiado no indicador"),
    Letter("E", TD, (0, 0, 0, 0), "posicao e curvatura do polegar"),
    Letter("F", TA, (0, 0, 0, 1)),
    Letter("G", TL, (0, 0, 0, 1), "direcao do dedo"),
    Letter("H", TL, (1, 1, 0, 0), "direcao dos dedos"),
    Letter("I", TL, (0, 0, 0, 1), "direcao do dedo"),
    Letter("J", TL, (0, 0, 0, 1), "movimento em J"),
    Letter("K", TA, (1, 1, 0, 0)),
    Letter("L", TL, (1, 0, 0, 0), "direcao e angulacao do indicador"),
    Letter("M", TD, (0, 0, 0, 0), "posicao do polegar"),
    Letter("N", TD, (1, 1, 0, 0)),
    Letter("O", TC, (0, 0, 0, 0), "abertura dos dedos"),
    Letter("P", TC, (1, 1, 0, 0)),
    Letter("Q", TC, (0, 0, 0, 1), "direcao do indicador"),
    Letter("R", TX, (1, 1, 0, 0)),
    Letter("S", TX, (0, 0, 0, 0)),
    Letter("T", TD, (0, 0, 0, 1)),
    Letter("U", TL, (1, 1, 0, 0), "posicao: lado a lado ou juntos"),
    Letter("V", TL, (1, 1, 0, 0), "posicao: separados ou juntos"),
    Letter("W", TL, (1, 1, 1, 0)),
    Letter("X", TX, (0, 0, 0, 0)),
    Letter("Y", TL, (0, 0, 0, 1), "direcao e curvatura do mininho"),
    Letter("Z", TL, (0, 0, 0, 1), "movimento em Z"),
)

# letras que so se distinguem por trajetoria, nao por pose
MOTION_LETTERS = frozenset("JZW")


@dataclass
class LetterGuess:
    letter: str
    score: float
    candidates: list = field(default_factory=list)
    needs: str = ""          # o que falta para decidir entre os candidatos

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def confident(self) -> bool:
        return not self.ambiguous and self.score >= 0.6


# --- leitura da pose ------------------------------------------------------

def _extended(f: HandFeatures) -> list:
    """Dedos de index..pinky estendidos, com faixa de histerese.

    Um dedo a 150 graus conta como estendido, a 170 tambem; a 130 nao. A
    faixa evita que uma pose no limite produza letras diferentes a cada
    quadro.
    """
    return [bool(f.angles[i, 0] > 145.0
                 or (f.angles[i, 0] > 125.0 and f.curl[FINGERS.index(FINGERS_4[i])] < 0.45))
            for i in range(4)]


def _thumb_shape(f: HandFeatures) -> str:
    """Posicao do polegar, por dobradilha e abertura.

    Limiares medidos nas poses de teste:

        forma      curl   tip_radius   ip
        lateral    0.00   1.61         180
        apoiado    0.43   1.53         142
        c          0.35   1.45         148
        dobrado    0.70   1.38         117
        cruz       0.74   1.25         114

    O par (curl, tip_radius) separa as cinco: o polegar dobrado fica mais
    perto do pulso, e o cruz tambem, mas com dobradilha maior.
    """
    curl = float(f.curl[0])
    radius = float(f.tip_radius[0])
    if curl < 0.18:
        return TL                      # estendido ao lado
    if curl > 0.66:
        return TX if radius < 1.31 else TD
    if radius > 1.49:
        return TA                      # apoiado, quase reto
    return TC


def _upright(f: HandFeatures) -> bool:
    """A mao esta com os dedos para cima, como nas letras A-E?"""
    a = f.orientation % 360.0
    return not (60.0 <= a < 120.0 or 240.0 <= a < 300.0)


def _spread_index_middle(f: HandFeatures) -> float:
    """Distancia entre as pontas do indicador e do medio, em unidades de mao.

    Separa 'U' (juntos) de 'V' (abertos) e 'K' de 'H'.
    """
    return float(np.linalg.norm(f.tips_local[1] - f.tips_local[2]))


def _closed(f: HandFeatures) -> float:
    """0 = mao aberta, 1 = punho fechado. Separa 'O' de 'E' e 'S' de 'X'."""
    return float(np.clip(f.curl[1:].mean(), 0.0, 1.0))


# --- reconhecimento -------------------------------------------------------

def recognise(hand: Hand, history=None) -> LetterGuess | None:
    """Leitura de datilologia para uma pose.

    `history` (HandHistory) e opcional; quando ha trajetoria, as letras de
    movimento (J, Z, W) deixam de ser ambiguas.
    """
    f = hand_features(hand)
    ext = _extended(f)
    thumb = _thumb_shape(f)
    pattern = tuple(1 if x else 0 for x in ext)
    upright = _upright(f)
    spread = _spread_index_middle(f)
    closed = _closed(f)

    # desempate dentro do mesmo padrao de dedos
    def refine(l: Letter) -> float:
        """0.5 base + sinais que separam as letras de mesma pose."""
        s = 0.5
        if l.name in ("U", "V", "H", "K"):
            # U e H tem os dedos juntos; V e K tem o par afastado. A divisao
            # certa e (U,H) juntos x (V,K) afastados: H nao e "aberto", e o
            # que separa H de U e a orientacao da mao, nao o espacamento.
            s += 0.3 if ((spread > 0.55) == (l.name in ("V", "K"))) else -0.3
        if l.name in ("E", "M", "S", "X", "A", "B", "C", "D"):
            s += 0.2 if upright else -0.2
        if l.name == "L":
            # indicador quase vertical e reto, demais dedos recolhidos
            s += 0.3 if (f.angles[0, 0] > 165.0 and closed > 0.5) else -0.3
        return s

    same = [l for l in LETTERS if l.fingers == pattern and l.thumb == thumb]
    if not same:
        # Nenhuma letra tem exatamente esta combinacao. Em vez de devolver
        # nada, devolve as letras com o mesmo conjunto de dedos para que o
        # app possa mostrar as opcoes -- um "nada reconhecido" nao ajuda o
        # usuario a saber o que ajustar.
        near = [l for l in LETTERS if l.fingers == pattern]
        if not near:
            return None
        return LetterGuess(
            letter=near[0].name, score=0.0,
            candidates=[l.name for l in near],
            needs="posicao do polegar",
        )

    # J, Z e W so existem como movimento. Sem historico -- ou com a mao
    # parada -- nao ha como separa-las da letra estatica de mesma pose, entao
    # sao retiradas em vez de sorteadas.
    moving = False
    if history is not None and len(history) >= 4:
        moving = history.displacement() > 0.04
    if not moving:
        static = [l for l in same if l.name not in MOTION_LETTERS]
        if static:
            same = static

    ranked = sorted(((refine(l), l) for l in same), key=lambda t: -t[0])
    top_score, top = ranked[0]
    cands = [l.name for _, l in ranked if _ >= top_score - 0.15]

    # `refine` existe para desempatar, nao para julgar o quao boa e a pose. Se
    # sobra uma letra so, a combinacao exata de polegar e dedos ja e evidencia
    # forte: a mao pode estar um pouco torta sem que a leitura deixe de valer.
    score = top_score if len(cands) > 1 else max(top_score, 0.7)

    return LetterGuess(
        letter=top.name, score=float(np.clip(score, 0.0, 1.0)),
        candidates=cands, needs=top.needs if len(cands) > 1 else "",
    )
