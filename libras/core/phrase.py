"""Acumulo de letras em uma frase, com histerese temporal.

Uma letra so entra na frase quando permanece a mesma por varios quadros.
Sem isso, uma pose no limite entre duas letras produz texto ilegivel
('OLRLO' em vez de 'OLA'). As opcoes:

  * `hold_frames`  quantos quadros a letra precisa se sustentar
  * `cooldown`     quadros ignorados depois de aceitar uma letra, para a
                   mao poder sair da pose e voltar sem repetir a letra
  * `space_after`  quantos quadros sem letra viram um espaco na frase
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..signs.dactylology import LetterGuess


@dataclass
class Phrase:
    text: str = ""
    current: str | None = None          # letra em leitura, ainda nao aceita
    current_score: float = 0.0
    candidates: list = field(default_factory=list)
    needs: str = ""


class PhraseBuilder:
    """Converte uma sequencia de leituras em texto."""

    def __init__(self, hold_frames: int = 5, cooldown: int = 8,
                 space_after: int = 12, min_score: float = 0.5,
                 max_len: int = 200):
        self.hold_frames = hold_frames
        self.cooldown = cooldown
        self.space_after = space_after
        self.min_score = min_score
        self.max_len = max_len
        self._text: list = []
        self._hold = 0
        self._quiet = 0
        self._cooldown = 0
        self._accepted = False
        self._current: str | None = None

    # --- API ---

    def reset(self) -> None:
        self._text.clear()
        self._hold = self._quiet = self._cooldown = 0
        self._accepted = False
        self._current = None

    @property
    def text(self) -> str:
        return "".join(self._text)

    def backspace(self) -> None:
        if self._text:
            self._text.pop()
        self._hold = self._quiet = self._cooldown = 0
        self._accepted = False
        self._current = None

    def push(self, guess: LetterGuess | None) -> Phrase:
        """Acrescenta um quadro de leitura. Retorna o estado da frase.

        O cooldown conta a partir do momento em que a mao sai da pose, nao
        enquanto ela a mantem: e assim que se escreve "AA" (segurar a letra,
        soltar, voltar) e nao "AAA" (a letra entrando sozinha).
        """
        if guess is None or guess.score < self.min_score:
            self._quiet += 1
            self._hold = 0
            self._accepted = False
            self._current = None
            if self._quiet >= self.space_after and self._text and self._text[-1] != " ":
                self._text.append(" ")
                self._quiet = 0
            if self._cooldown > 0:
                self._cooldown -= 1
            return self._state(guess)

        if guess.letter != self._current:
            # pose mudou: nova contagem de permanencia
            self._current = guess.letter
            self._hold = 1
            self._accepted = False
        elif not self._accepted:
            self._hold += 1

        # O cooldown segura a MESMA letra depois que ela ja entrou, evitando
        # que um ruido de jitter reescreva o ultimo caractere. Nao bloqueia a
        # letra seguinte, que e o caso normal de escrita.
        repeating = self._text and self._text[-1] == guess.letter
        if self._hold >= self.hold_frames and not self._accepted \
                and (self._cooldown == 0 or not repeating):
            self._accept(guess)

        return self._state(guess)

    # --- interno ---

    def _accept(self, guess: LetterGuess) -> None:
        if guess.ambiguous:
            # letra ambigua nao entra: o usuario precisa ajustar a pose ou o
            # sistema precisa de mais informacao (direcao, movimento).
            # Mantem o contador um abaixo do limite, para tentar de novo no
            # proximo quadro em vez de ficar travado.
            self._hold = self.hold_frames - 1
            return
        # Nao se remove o espaco final antes de escrever: o usuario precisa
        # ver onde a palavra terminou, senao "OLA " e "OLA" ficam idênticos.
        self._text.append(guess.letter)
        del self._text[:-self.max_len]
        self._hold = 0
        self._accepted = True
        self._cooldown = self.cooldown
        self._quiet = 0

    def _state(self, guess: LetterGuess | None) -> Phrase:
        return Phrase(
            text=self.text,
            current=self._current,
            current_score=float(guess.score) if guess else 0.0,
            candidates=list(guess.candidates) if guess else [],
            needs=guess.needs if guess else "",
        )
