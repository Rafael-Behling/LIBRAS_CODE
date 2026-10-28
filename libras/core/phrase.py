"""Acumulo de letras em uma frase, com histerese temporal e votacao.

Uma letra so entra na frase quando a leitura se sustenta por varios quadros.
Sem isso, uma pose no limite entre duas letras produz texto ilegivel
('OLRLO' em vez de 'OLA'). As opcoes:

  * `hold_frames`  quadros que a leitura precisa se sustentar
  * `cooldown`     quadros ignorados depois de aceitar uma letra, para a
                   mao poder sair da pose e voltar sem repetir a letra
  * `space_after`  quantos quadros sem letra viram um espaco na frase

O desenho de estabilidade tem tres partes, e as tres trabalham juntas:

  1. **Votacao por quadro, sempre ligada.** Cada quadro, das regras ou do
     classificador, vota nas suas leituras -- o candidato melhor com peso
     maior, e todos os votos escalados pela confianca do quadro. A letra que
     entra na frase e o lider da votacao, nao a leitura do ultimo quadro:
     um piscar do classificador no exato quadro da decisao nao muda a letra.

  2. **Permanencia vazada.** Um quadro ruim (letra vizinha ou sem leitura)
     custa 1 de progresso em vez de zerar. O classificador piscar por um
     quadro nao reinicia a corrida; uma troca sustentada de verdade chega a
     zerar e comeca outra corrida.

  3. **Sair para repetir.** Depois de aceitar uma letra, a corrida fica
     "armada": ela so libera a proxima quando a leitura sai da letra aceita
     (a mao solta a pose e volta). E assim que se escreve "AA" -- segurar e
     soltar -- e nao "AAA", que sairia sozinho enquanto a pose se mantem.

Algumas letras tem a MESMA pose e so a orientacao as separa (U/V, D/L,
B/C). O reconhecedor devolve essas leituras como `candidates`; a votacao
temporal decide, entao um quadro isolado nao decide nada e uma pose mantida
decide mesmo que cada quadro, sozinho, fosse ambiguo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..signs.dactylology import LetterGuess


def _rank_weight(rank: int, score: float) -> float:
    """Peso de um candidato: pela posicao na leitura e pela confianca do quadro.

    O melhor candidato do quadro vale `score`; o seguinte, metade disso, e
    assim por diante. Assim um quadro confiante pesa mais que um duvidoso, e
    dentro do quadro a ordem entre os candidatos e respeitada.
    """
    return float(max(0.0, score)) / (rank + 1)


@dataclass
class Phrase:
    text: str = ""
    current: str | None = None  # letra em leitura, ainda nao aceita
    current_score: float = 0.0
    candidates: list = field(default_factory=list)
    needs: str = ""
    votes: list = field(default_factory=list)  # [(letra, peso), ...]
    confidence: float = 0.0  # lider / total, 0..1


class PhraseBuilder:
    """Converte uma sequencia de leituras em texto.

    Ver o docstring do modulo para o desenho de estabilidade. O essencial:
    a corrida vota a cada quadro e a letra vem do lider dos votos, nao do
    ultimo quadro; o `cooldown` conta em quadros de video (nao so quando a
    mao some) e a repeticao exige soltar a pose.
    """

    def __init__(
        self,
        hold_frames: int = 5,
        cooldown: int = 8,
        space_after: int = 12,
        min_score: float = 0.5,
        max_len: int = 200,
        ambiguous_floor: float = 0.25,
        vote_margin: float = 0.6,
        min_votes: int | None = None,
        release_frames: int = 3,
    ):
        self.hold_frames = hold_frames
        self.cooldown = cooldown
        self.space_after = space_after
        # limiar para leitura de pose unica; leitura ambigua nao precisa de
        # score alto, precisa de consistencia entre quadros (votacao)
        self.min_score = min_score
        self.ambiguous_floor = ambiguous_floor
        # o lider precisa de vantagem minima sobre o segundo colocado
        self.vote_margin = vote_margin
        self.min_votes = hold_frames if min_votes is None else min_votes
        # quadros de "leitura fora da letra" que liberam a proxima letra
        self.release_frames = release_frames
        self.max_len = max_len
        self._text: list = []
        self._hold = 0
        self._quiet = 0
        self._cooldown = 0
        self._locked = False
        self._current: str | None = None
        self._tally: dict = {}  # letra -> peso acumulado na corrida
        self._frames = 0  # quadros que votaram na corrida
        self._away = 0  # quadros consecutivos fora da letra

    # --- API ---

    def reset(self) -> None:
        self._text.clear()
        self._hold = self._quiet = self._cooldown = 0
        self._locked = False
        self._current = None
        self._tally = {}
        self._frames = 0
        self._away = 0

    @property
    def text(self) -> str:
        return "".join(self._text)

    @property
    def hold_progress(self) -> int:
        """Quadros ja sustentados na corrida atual (o anel de progresso do HUD)."""
        return self._hold

    def backspace(self) -> None:
        if self._text:
            self._text.pop()
        self._hold = self._quiet = self._cooldown = 0
        self._locked = False
        self._current = None
        self._tally = {}
        self._frames = 0
        self._away = 0

    def confirm(self) -> None:
        """Forca a entrada da letra em leitura (tecla ESPACO).

        E a saida manual para uma pose que o reconhecedor le como ambigua: o
        usuario confirma a leitura em vez de esperar a votacao.
        """
        if self._current:
            self._commit(self._current)
            self._quiet = 0

    def push(self, guess: LetterGuess | None) -> Phrase:
        """Acrescenta um quadro de leitura. Retorna o estado da frase.

        Modelo de interacao: uma letra entra quando a pose se sustenta por
        `hold_frames`. Depois de aceita, a corrida fica "travada" e so libera
        a proxima quando a mao sai da pose por `release_frames` quadros -- e
        assim que se escreve "AA" (soltar e refazer) sem virar "AAA".
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        if guess is None or guess.score < self._floor(guess):
            return self._miss_frame(guess)

        # corrida de uma letra ja aceita: fica travada ate a mao soltar a
        # pose (ou trocar para outra letra por `release_frames` seguidos)
        if self._locked:
            if self._current is not None and guess.letter != self._current:
                self._away += 1
                if self._away >= self.release_frames:
                    self._locked = False
                    self._start(guess)
            else:
                self._away = 0
            return self._state(guess)

        if guess.letter == self._current and self._hold > 0:
            # a leitura sustenta a corrida
            self._hold += 1
            self._away = 0
            self._vote(guess)
        elif self._hold > 0:
            # quadro lido como outra letra no meio da corrida: pode ser so um
            # piscar do classificador. Vota nela mas cobra 1 de progresso em
            # vez de zerar; uma troca sustentada chega a zerar e comeca outra.
            self._hold -= 1
            self._vote(guess)
        else:
            # a corrida morreu: comeca uma nova
            self._start(guess)

        if self._hold >= self.hold_frames and self._cooldown <= 0:
            letter = self._decide(guess)
            if letter is not None:
                self._commit(letter)

        return self._state(guess)

    # --- interno ---

    def _vote(self, guess: LetterGuess) -> None:
        """Cada quadro vota nos seus candidatos, ponderado pela confianca."""
        cands = guess.candidates or [guess.letter]
        self._frames += 1
        for rank, name in enumerate(cands):
            self._tally[name] = self._tally.get(name, 0.0) + _rank_weight(rank, guess.score)

    def _start(self, guess: LetterGuess) -> None:
        """Comeca uma corrida nova para a letra lida."""
        self._current = guess.letter
        self._hold = 1
        self._locked = False
        self._away = 0
        self._tally = {}
        self._frames = 0
        self._vote(guess)

    def _floor(self, guess: LetterGuess) -> float:
        """Score minimo por quadro.

        A leitura ambigua tem score baixo de proposito (o reconhecedor nao
        sabe qual das letras e), entao quem a valida e a votacao, nao o
        score. O piso evita que um quadro de ruido entre na corrida.
        """
        return self.ambiguous_floor if guess.ambiguous else self.min_score

    def _decide(self, guess: LetterGuess) -> str | None:
        """Qual letra entra depois da permanencia; None se ainda nao der.

        A letra vem do lider da votacao acumulada, nao do ultimo quadro: um
        piscar do classificador no quadro da decisao nao muda o resultado.

        Leitura inequivoca (um candidato) nao tem com o que votar: aceita
        assim que a permanencia chega, como validado pelo chamador.

        Leitura ambigua exige que o lider domine a votacao -- uma fracao
        minima do total -- senao a pose esta genuinamente dividida entre as
        letras (D/L, U/V, B/C) e nao se escreve nada. Isso e o que evita
        "OLRLO" no lugar de "OLA".
        """
        if len(guess.candidates) <= 1 or not self._tally:
            return guess.letter
        leader, lead = max(self._tally.items(), key=lambda kv: kv[1])
        total = sum(self._tally.values())
        if self._frames < self.min_votes:
            return None
        if total > 0.0 and (lead / total) < self.vote_margin:
            return None
        return leader

    def _commit(self, letter: str) -> None:
        # Nao se remove o espaco final antes de escrever: o usuario precisa
        # ver onde a palavra terminou, senao "OLA " e "OLA" ficam idênticos.
        self._text.append(letter)
        del self._text[: -self.max_len]
        self._hold = 0
        self._locked = True  # so libera quando a mao sair da pose
        self._cooldown = self.cooldown
        self._quiet = 0
        self._current = letter
        self._away = 0
        self._tally = {}
        self._frames = 0

    def _miss_frame(self, guess: LetterGuess | None) -> Phrase:
        """Quadro sem leitura (mao sumiu, pose ambigua abaixo do piso).

        Um quadro assim tambem custa 1 de progresso em vez de zerar: o
        rastreador perde a mao por um frame e a letra nao pode reiniciar.
        """
        self._quiet += 1
        if self._hold > 0:
            self._hold -= 1
            self._away += 1
        else:
            self._locked = False
            self._current = None
            self._tally = {}
            self._frames = 0
            self._away = 0
        # soltar a mao tambem libera a repeticao da mesma letra
        if self._locked and self._current is not None and self._away >= self.release_frames:
            self._locked = False
        if self._quiet >= self.space_after and self._text and self._text[-1] != " ":
            self._text.append(" ")
            self._quiet = 0
        return self._state(guess)

    def _state(self, guess: LetterGuess | None) -> Phrase:
        total = sum(self._tally.values())
        votes = sorted(self._tally.items(), key=lambda kv: -kv[1])
        confidence = (votes[0][1] / total) if total else 0.0
        return Phrase(
            text=self.text,
            current=self._current,
            current_score=float(guess.score) if guess else 0.0,
            candidates=list(guess.candidates) if guess else [],
            needs=guess.needs if guess else "",
            votes=[(name, round(weight, 2)) for name, weight in votes],
            confidence=float(confidence),
        )
