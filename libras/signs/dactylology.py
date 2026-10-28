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

# Corte de curvatura (curl medio dos 4 dedos) que separa B (retos) de C (arco
# suave). Calibrado no data/samples.npz: acerta ~94% das amostras de B e C.
ARC_BC = 0.065

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

# Em Libras, 6 letras do alfabeto manual sao dinamicas (tem movimento): H, J,
# K, X, Y, Z. As outras 20 sao estaticas. Um classificador de quadro unico nao
# enxerga movimento, entao essas 6 ficam de fora dele e seguem pelas regras com
# historico.
DYNAMIC_LETTERS = frozenset("HJKXYZ")


@dataclass
class LetterGuess:
    letter: str
    score: float
    candidates: list = field(default_factory=list)
    needs: str = ""  # o que falta para decidir entre os candidatos

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
    return [
        bool(
            f.angles[i, 0] > 145.0
            or (f.angles[i, 0] > 125.0 and f.curl[FINGERS.index(FINGERS_4[i])] < 0.45)
        )
        for i in range(4)
    ]


def _out(f: HandFeatures) -> list:
    """Dedos fora da palma: estendidos OU em arco, mas nao dobrados no punho.

    O corte do `_extended` fica em ~139 graus, bem no meio do arco do C
    (PIP 120-150). Por isso a leitura oscilava: ora os quatro dedos contavam,
    ora um caia e o padrao deixava de ser (1,1,1,1). Aqui o arco vale como
    "fora", que e o que a mao faz em C -- o que decide B x C e a curvatura.
    """
    return [bool(f.angles[i, 0] > 118.0) for i in range(4)]


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
        return TL  # estendido ao lado
    if curl > 0.66:
        return TX if radius < 1.31 else TD
    if radius > 1.49:
        return TA  # apoiado, quase reto
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


def recognise(
    hand: Hand, history=None, classifier=None, motion_classifier=None
) -> LetterGuess | None:
    """Leitura de datilologia para uma pose.

    `history` (HandHistory) e opcional; quando ha trajetoria, as letras que
    dependem de movimento deixam de ser ambiguas.

    `classifier` (LetterClassifier) e opcional. Quando existe, ele e a leitura
    principal -- foi treinado com as amostras do usuario e separa os pares que
    as regras so desempatam por limiares frageis. As regras continuam sendo o
    fallback e as unicas que enxergam movimento, porque o classificador e
    estatico e nao cobre as 6 letras dinamicas (H, J, K, X, Y, Z).

    `motion_classifier` (LetterClassifier treinado sobre HandHistory.vector())
    e opcional e cobre justamente as 6 dinamicas: e consultado quando a mao
    esta se mexendo e o historico ja tem contexto, e a leitura dele vale
    apenas para letras de movimento -- as estaticas seguem pelo caminho de
    sempre.
    """
    f = hand_features(hand)
    if motion_classifier is not None and history is not None:
        guess = _recognise_motion(history, motion_classifier)
        if guess is not None:
            return guess
    # Transicao rapida entre poses: nao ha letra ali. Votar no que as regras
    # ou o modelo leem no meio do voo enche a frase de lixo (era o que fazia o
    # alfabeto em sequencia sair errado). A janela de 30 quadros acumulava o
    # deslocamento de TODAS as transicoes do ultimo segundo e parecia "em
    # movimento" sempre; o teste certo e o caminho dos ultimos quadros.
    if history is not None and len(history) >= 4 \
            and history.recent_path() > TRANSITION_PATH:
        return None
    if classifier is not None:
        guess = _recognise_model(f, history, classifier)
        if guess is not None:
            return guess
    return _recognise_rules(f, history)


# Caminho do centro da mao nos ultimos 4 quadros (unidades de frame):
#   pose segurada      < 0.02 (so tremor; o tracker suaviza)
#   sinal deliberado   ~0.05-0.10 (J, Z)
#   transicao rapida   > 0.14 (mao voando entre duas poses)
TRANSITION_PATH = 0.14
MOTION_PATH = 0.04          # movimento deliberado comeca por aqui
MOTION_CLEAR_PATH = 0.10    # acima disso, so as letras de movimento valem


def _recognise_motion(history, motion_clf) -> LetterGuess | None:
    """Leitura temporal das letras de movimento.

    So fala quando a mao esta em movimento deliberado (caminho recente
    acima do piso) e o historico acumulou contexto: com a mao parada, pose
    estatica e quem manda; com poucos quadros, o vetor temporal nao vale
    nada. A leitura so entra se a letra for uma das 6 dinamicas -- para as
    estaticas, o modelo estatico tem mais treino e segue mandando.
    """
    if len(history) < 4 or history.recent_path() < MOTION_PATH:
        return None
    guess = motion_clf.to_guess(history.vector())
    if not guess.candidates:
        return None
    dyn = [c for c in guess.candidates if c in _MOTION_ONLY]
    if not dyn or guess.letter not in _MOTION_ONLY:
        return None
    return LetterGuess(
        letter=guess.letter,
        score=guess.score,
        candidates=dyn,
        needs="trajetoria" if len(dyn) > 1 else "",
    )


# letras que o classificador estatico nao representa: so existem com trajetoria
_MOTION_ONLY = DYNAMIC_LETTERS


def _recognise_model(f: HandFeatures, history, classifier) -> LetterGuess | None:
    """Usa o classificador, deixando as poses de letra dinamica para as regras.

    O modelo estatico nunca viu H, J, K, X, Y, Z (nao cabem num quadro so).
    Quando a pose aponta para uma delas -- as regras sao quem reconhece a
    geometria delas -- o chute do modelo seria a vizinha estatica errada
    (H lido como U/V, X como S). Nesses casos as regras mandam, mesmo com a
    mao parada: um H segurado tem de virar H, nao U.
    """
    rules = _recognise_rules(f, history)
    if rules is not None and rules.letter in _MOTION_ONLY \
            and rules.score >= 0.5:
        return rules
    guess = classifier.to_guess(f.vector())
    if not guess.candidates:
        return None
    return _disambiguate(f, guess)


# C e L sao as duas letras que o classificador mais troca em maos reais, porque
# dependem de um dedo que muda com a pose: C vira F quando o indicador fecha, L
# vira M quando os outros dedos relaxam. As duas trocas tem uma feature
# geometrica forte por tras, entao aqui a geometria manda quando o modelo
# contradiz o que a mao esta fazendo de fato.
_INDEX_CURL_F = 0.45  # acima disso o indicador esta dobrado: e F, nao C
_INDEX_CURL_L = 0.35  # L tem o indicador estendido; M nao
_PINKY_CURL_L = 0.45  # L tem o mindinho dobrado; M tambem, mas junto ao resto


def _disambiguate(f: HandFeatures, guess: LetterGuess) -> LetterGuess:
    """Corrige trocas do classificador que a geometria desmente.

    So age nas duas confusoes medidas (C<->F e L<->M) e so quando o modelo
    votou numa delas; fora disso devolve a leitura como veio.
    """
    letra = guess.letter
    idx = float(f.curl[1])
    outros = f.curl[2:]

    if letra == "F" and idx < _INDEX_CURL_F and float(outros.mean()) < 0.4:
        # indicador reto com os outros abertos: e a pose de C, nao F
        return LetterGuess(
            letter="C", score=guess.score, candidates=_swap(guess.candidates, "F", "C")
        )
    if letra == "C" and idx >= _INDEX_CURL_F:
        # indicador fechado: a pose virou F
        return LetterGuess(
            letter="F", score=guess.score, candidates=_swap(guess.candidates, "C", "F")
        )
    if letra == "M" and idx < _INDEX_CURL_L and float(f.curl[4]) < _PINKY_CURL_L:
        # indicador estendido: e L (o M dobra o indicador junto ao punho)
        return LetterGuess(
            letter="L", score=guess.score, candidates=_swap(guess.candidates, "M", "L")
        )
    if letra == "L" and idx >= _INDEX_CURL_L and float(f.curl[4]) >= _PINKY_CURL_L:
        # indicador dobrado e mindinho fechado: e M
        return LetterGuess(
            letter="M", score=guess.score, candidates=_swap(guess.candidates, "L", "M")
        )
    return guess


def _swap(candidates: list, velho: str, novo: str) -> list:
    """Troca `velho` por `novo` na lista de candidatos, sem duplicar."""
    out = []
    for c in candidates:
        c2 = novo if c == velho else c
        if c2 not in out:
            out.append(c2)
    if novo not in out:
        out.insert(0, novo)
    return out


def _recognise_rules(f: HandFeatures, history=None) -> LetterGuess | None:
    """Leitura por regras geometricas (usada como fallback e para movimento)."""
    ext = _extended(f)
    out = _out(f)
    thumb = _thumb_shape(f)
    pattern = tuple(1 if x else 0 for x in ext)
    upright = _upright(f)
    spread = _spread_index_middle(f)
    # curvatura media dos quatro dedos (indicador a mininho): e o que separa B
    # (retos) de C (em arco). Medido no data/samples.npz: B fica em ~0.025-0.056
    # (mediana 0.043) e C em ~0.042-0.193 (mediana 0.111). O corte que mais
    # acerta os dois e ~0.065 (acerta 94% das amostras); 0.17 jogava quase todo
    # C para B, porque o C real e um arco bem mais suave do que parecia.
    arc = float(f.curl[1:].mean())

    # desempate dentro do mesmo padrao de dedos
    def refine(let: Letter) -> float:
        """0.5 base + sinais que separam as letras de mesma pose."""
        s = 0.5
        if let.name in ("U", "V", "H", "K"):
            # U e H tem os dedos juntos; V e K tem o par afastado. A divisao
            # certa e (U,H) juntos x (V,K) afastados: H nao e "aberto", e o
            # que separa H de U e a orientacao da mao, nao o espacamento.
            apart = spread > 0.55
            s += 0.3 if (apart == (let.name in ("V", "K"))) else -0.3
            # H e K geralmente com mao mais lateral/orientada; U/V mais "para cima"
            if let.name in ("H", "K") and upright:
                s -= 0.1
            if let.name in ("U", "V") and not upright:
                s -= 0.1
        if let.name in ("E", "M", "S", "X", "A", "B", "C", "D"):
            s += 0.2 if upright else -0.2
        if let.name in ("B", "C"):
            # B e C compartilham a pose-base (quatro dedos para fora da palma) e o
            # polegar costuma sair lateral nos dois; o que os separa e a
            # curvatura: B retos, C em arco. Corte em ARC_BC (calibrado no
            # dataset), com o polegar so como empurrao.
            arched = arc > ARC_BC
            if let.name == "C":
                s += 0.4 if arched else -0.4
                s += 0.1 if thumb == TC else 0.0
            else:  # B
                s += 0.4 if not arched else -0.4
                s -= 0.1 if thumb == TC else 0.0
        if let.name in ("D", "L"):
            # D e L sao a MESMA pose de dedos (so o indicador estendido); separa
            # pela abertura polegar-indicador: em L os dois ficam estendidos e
            # afastados, formando o "L", com as pontas longe (~1.2 medidas); em
            # D o polegar dobra junto ao indicador, pontas perto (~0.7). A forma
            # do polegar sozinha nao serve: o D real sai lido como lateral/c.
            gap = float(np.linalg.norm(f.tips_local[0] - f.tips_local[1]))
            wide = gap > 1.0
            if let.name == "L":
                s += 0.35 if wide else -0.35
            else:  # D
                s += 0.35 if not wide else -0.35
        # G/I/J/Y/Z: todos com padrao (0,0,0,1) - mininho estendido
        if let.name in ("G", "I", "J", "Y", "Z"):
            if let.name == "G" and not upright:
                s += 0.1
            if let.name in ("I", "Y") and upright:
                s += 0.05
            if let.name == "Y":
                s += 0.1 if f.curl[4] > 0.3 else -0.02
            # historico: usar trajetoria para ajudar a distinguir J/Z
            if history is not None and len(history) >= 6:
                strt = history.straightness()
                if let.name == "Z" and strt > 0.6:
                    s += 0.1
                if let.name == "J" and strt < 0.4:
                    s += 0.1
        return s

    same = [let for let in LETTERS if let.fingers == pattern and let.thumb == thumb]

    # B e C tem a mesma pose-base e se separam so pela curvatura. Junta os dois
    # quando os quatro dedos estao no padrao estendido (nao existe outra letra
    # com esse padrao) e deixa o `refine` decidir. Quando um dedo em arco faz o
    # padrao binario perder uma posicao, C ainda entra se o arco for claro --
    # sem isso, arco apertado virava A e arco medio virava B.
    if pattern == (1, 1, 1, 1):
        same = [let for let in LETTERS if let.name in ("B", "C")]
    elif tuple(out) == (1, 1, 1, 1) and arc > ARC_BC:
        same = [let for let in LETTERS if let.name == "C"]
    elif pattern == (1, 0, 0, 0):
        # D e L so diferem pelo polegar; junta os dois e deixa o `refine`
        # decidir pela forma do polegar.
        same = [let for let in LETTERS if let.name in ("D", "L")]

    if not same:
        # Nenhuma letra tem exatamente esta combinacao. Em vez de devolver
        # nada, devolve as letras com o mesmo conjunto de dedos para que o
        # app possa mostrar as opcoes -- um "nada reconhecido" nao ajuda o
        # usuario a saber o que ajustar.
        near = [let for let in LETTERS if let.fingers == pattern]
        if not near:
            return None
        return LetterGuess(
            letter=near[0].name,
            score=0.0,
            candidates=[let.name for let in near],
            needs="posicao do polegar",
        )

    # J, Z e W so existem como movimento. Sem historico -- ou com a mao
    # parada -- nao ha como separa-las da letra estatica de mesma pose, entao
    # sao retiradas em vez de sorteadas. O teste e pelo caminho RECENTE (a
    # janela inteira acumula as transicoes do ultimo segundo e todo alfabeto
    # em sequencia parecia "em movimento").
    moving = False
    recent = 0.0
    if history is not None and len(history) >= 4:
        recent = history.recent_path()
        moving = recent > MOTION_PATH
    if not moving:
        static = [let for let in same if let.name not in MOTION_LETTERS]
        if static:
            same = static
    elif recent > MOTION_CLEAR_PATH:
        dyn = [let for let in same if let.name in MOTION_LETTERS]
        if dyn:
            same = dyn

    ranked = sorted(((refine(let), let) for let in same), key=lambda t: -t[0])
    top_score, top = ranked[0]
    cands = [let.name for score_rank, let in ranked if score_rank >= top_score - 0.15]

    # `refine` existe para desempatar, nao para julgar o quao boa e a pose. Se
    # sobra uma letra so, a combinacao exata de polegar e dedos ja e evidencia
    # forte: a mao pode estar um pouco torta sem que a leitura deixe de valer.
    score = top_score if len(cands) > 1 else max(top_score, 0.7)

    return LetterGuess(
        letter=top.name,
        score=float(np.clip(score, 0.0, 1.0)),
        candidates=cands,
        needs=top.needs if len(cands) > 1 else "",
    )
