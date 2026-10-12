"""Acumulo de letras em uma frase, com histerese temporal e votacao.

Uma letra so entra na frase quando a leitura se sustenta por varios quadros.
Sem isso, uma pose no limite entre duas letras produz texto ilegivel
('OLRLO' em vez de 'OLA'). As opcoes:

  * `hold_frames`  quantos quadros a leitura precisa se sustentar
  * `cooldown`     quadros ignorados depois de aceitar uma letra, para a
                   mao poder sair da pose e voltar sem repetir a letra
  * `space_after`  quantos quadros sem letra viram um espaco na frase

Algumas letras tem a MESMA pose e so a orientacao/movimento as separa
(A/I/G, J/Z, U/V, E/M/O/S/X). O reconhecedor devolve essas leituras como
`candidates` e nao como certeza. Em vez de descartar uma leitura ambigua --
o que fazia a app "nunca aceitar nada" -- a votacao temporal decide: cada
quadro vota nos seus candidatos, com peso pela posicao (o melhor vale mais),
e a letra entra quando o lider se sustentou e abriu vantagem sobre o
segundo. Um quadro isolado nao decide nada; uma pose mantida decide mesmo
que cada quadro, sozinho, fosse ambiguo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..signs.dactylology import LetterGuess


def _rank_weight(rank: int) -> float:
    """Peso do candidato pela posicao na leitura: 1, 1/2, 1/3, ..."""
    return 1.0 / (rank + 1)


@dataclass
class Phrase:
    text: str = ""
    current: str | None = None          # letra em leitura, ainda nao aceita
    current_score: float = 0.0
    candidates: list = field(default_factory=list)
    needs: str = ""
    votes: list = field(default_factory=list)      # [(letra, peso), ...]
    confidence: float = 0.0                        # lider / total, 0..1


class PhraseBuilder:
    """Converte uma sequencia de leituras em texto.

    A letra em leitura guarda a corrida atual. A contagem de permanencia e
    'vazada': um quadro ruim (letra vizinha ou sem leitura) tira 1 de
    progresso em vez de zerar, entao o classificador piscar por um frame nao
    reinicia a letra. Uma troca sustentada zera e comeca a corrida nova.
    """

    def __init__(self, hold_frames: int = 5, cooldown: int = 8,
                 space_after: int = 12, min_score: float = 0.5,
                 max_len: int = 200, ambiguous_floor: float = 0.25,
                 vote_margin: float = 1.5, min_votes: int | None = None):
        self.hold_frames = hold_frames
        self.cooldown = cooldown
        self.space_after = space_after
        # limiar para leitura de pose unica; leitura ambigua nao precisa de
        # score alto, precisa de consistencia entre quadros (votacao)
        self.min_score = min_score
        self.ambiguous_floor = ambiguous_floor
        # o lider precisa de tantos votos quanto `hold_frames` e de vantagem
        # minima sobre o segundo colocado
        self.vote_margin = vote_margin
        self.min_votes = hold_frames if min_votes is None else min_votes
        self.max_len = max_len
        self._text: list = []
        self._hold = 0
        self._quiet = 0
        self._cooldown = 0
        self._accepted = False
        self._current: str | None = None
        self._tally: dict = {}          # letra -> peso acumulado na corrida

    # --- API ---

    def reset(self) -> None:
        self._text.clear()
        self._hold = self._quiet = self._cooldown = 0
        self._accepted = False
        self._current = None
        self._tally = {}

    @property
    def text(self) -> str:
        return "".join(self._text)

    def backspace(self) -> None:
        if self._text:
            self._text.pop()
        self._hold = self._quiet = self._cooldown = 0
        self._accepted = False
        self._current = None
        self._tally = {}

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

        O cooldown conta a partir do momento em que a mao sai da pose, nao
        enquanto ela a mantem: e assim que se escreve "AA" (segurar a letra,
        soltar, voltar) e nao "AAA" (a letra entrando sozinha).
        """
        if guess is None or guess.score < self._floor(guess):
            return self._miss_frame(guess)

        if guess.letter == self._current:
            self._hold += 1
            self._vote(guess)
        elif self._hold > 0:
            # um quadro isolado fora da corrida nao derruba a leitura
            # (o classificador pisca entre letras vizinhas); custa 1 de
            # progresso em vez de zerar. Uma troca de verdade, sustentada,
            # chega a zerar e comeca nova corrida.
            self._hold -= 1
        else:
            self._start(guess)

        if self._hold >= self.hold_frames and not self._accepted:
            letter = self._decide(guess)
            if letter is not None:
                self._commit(letter)
                self._quiet = 0

        return self._state(guess)

    # --- interno ---

    def _vote(self, guess: LetterGuess) -> None:
        """Cada quadro vota nos seus candidatos, o melhor com peso maior."""
        for rank, name in enumerate(guess.candidates or [guess.letter]):
            self._tally[name] = self._tally.get(name, 0.0) + _rank_weight(rank)

    def _start(self, guess: LetterGuess) -> None:
        """Comeca uma corrida nova para a letra lida."""
        self._current = guess.letter
        self._hold = 1
        self._accepted = False
        self._tally = {}
        self._vote(guess)

    def _floor(self, guess: LetterGuess) -> float:
        """Score minimo por quadro.

        A leitura ambigua tem score baixo de proposito (o reconhecedor nao
        sabe qual das letras e), entao quem a valida e a votacao, nao o
        score. O piso evita que um quadro de ruido entre na corrida.
        """
        return self.ambiguous_floor if guess.ambiguous else self.min_score

    def _decide(self, guess: LetterGuess) -> str | None:
        """Qual letra entra depois da permanencia; None se ainda nao der."""
        if not guess.ambiguous:
            return guess.letter
        leader = guess.letter
        lead = self._tally.get(leader, 0.0)
        runner = max((v for k, v in self._tally.items() if k != leader), default=0.0)
        if lead < self.min_votes:
            return None
        if runner > 0.0 and lead < self.vote_margin * runner:
            return None
        return leader

    def _commit(self, letter: str) -> None:
        # Nao se remove o espaco final antes de escrever: o usuario precisa
        # ver onde a palavra terminou, senao "OLA " e "OLA" ficam idênticos.
        self._text.append(letter)
        del self._text[:-self.max_len]
        self._hold = 0
        self._accepted = True
        self._cooldown = self.cooldown
        self._quiet = 0
        self._current = letter
        self._tally = {}

    def _miss_frame(self, guess: LetterGuess | None) -> Phrase:
        """Quadro sem leitura (mao sumiu, pose ambigua abaixo do piso).

        Um quadro assim tambem custa 1 de progresso em vez de zerar: o
        rastreador perde a mao por um frame e a letra nao pode reiniciar.
        """
        self._quiet += 1
        if self._hold > 0:
            self._hold -= 1
        else:
            self._accepted = False
            self._current = None
            self._tally = {}
        if self._quiet >= self.space_after and self._text and self._text[-1] != " ":
            self._text.append(" ")
            self._quiet = 0
        if self._cooldown > 0:
            self._cooldown -= 1
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
