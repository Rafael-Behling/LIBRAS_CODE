"""Classificador de letras por regressao softmax, em numpy puro.

Por que existe: as regras geometricas de `dactylology` acertam a maioria das
letras, mas os pares de mesma pose (D/L, B/C, ...) dependem de limiares
frageis, medidos para uma mao e um angulo. Com amostras rotuladas do proprio
usuario, um classificador aprende esses limiares sozinho e generaliza melhor.

Sem sklearn de proposito: o projeto so depende de numpy/opencv/litert, e uma
regressao softmax cabe em poucas linhas. O modelo e pequeno (49 features x
poucas dezenas de classes), entao treina em segundos na CPU.

O vetor de entrada e o `HandFeatures.vector()` de `libras.core.features`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax estavel por linha (z: (N, K))."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class LetterClassifier:
    """Modelo linear padronizado (z-score) + softmax.

    Guarda a media/desvio do treino para normalizar a entrada -- features como
    `tip_radius` e `curl` tem escalas bem diferentes, e sem padronizar o
    gradiente fica dominado pela de maior escala.
    """

    classes: list  # nomes das letras, na ordem das colunas de W
    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)
    W: np.ndarray  # (D, K)
    b: np.ndarray  # (K,)

    @property
    def n_features(self) -> int:
        return self.W.shape[0]

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probabilidade por classe para uma amostra (D,) ou lote (N, D)."""
        x = np.atleast_2d(np.asarray(features, np.float32))
        return softmax(self._standardize(x) @ self.W + self.b)

    def predict(self, features: np.ndarray):
        """(letra, prob) mais provavel para uma amostra (D,)."""
        p = self.predict_proba(features)[0]
        k = int(np.argmax(p))
        return self.classes[k], float(p[k])

    def to_guess(self, features: np.ndarray, margin: float = 0.5):
        """Converte as probabilidades em `LetterGuess` com candidatos.

        Entram como candidatos todas as classes com probabilidade a menos de
        `margin` relativo da lider (prob >= top * margin). Se sobrar uma so,
        a leitura e certa; se varias, o acumulador de frases vota entre elas.
        """
        from .dactylology import LetterGuess

        p = self.predict_proba(features)[0]
        order = np.argsort(-p)
        top = int(order[0])
        cands = [self.classes[k] for k in order if p[k] >= p[top] * margin]
        return LetterGuess(letter=self.classes[top], score=float(p[top]), candidates=cands)

    # --- persistencia ---

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, classes=np.array(self.classes), mean=self.mean, std=self.std, W=self.W, b=self.b
        )

    @staticmethod
    def load(path) -> LetterClassifier:
        d = np.load(path, allow_pickle=False)
        return LetterClassifier(
            classes=[str(c) for c in d["classes"]], mean=d["mean"], std=d["std"], W=d["W"], b=d["b"]
        )


def fit(
    X: np.ndarray,
    y: list,
    classes: list | None = None,
    iters: int = 1500,
    lr: float = 0.5,
    l2: float = 1e-4,
    batch: int = 64,
    seed: int = 0,
    verbose: bool = False,
) -> LetterClassifier:
    """Treina a regressao softmax por descida de gradiente (mini-batch).

    `X` (N, D) sao as features; `y` os rotulos (letras). Devolve o modelo ja
    padronizado. `classes` fixa a ordem das colunas; sem ele, usa a ordem
    alfabetica das letras presentes.
    """
    X = np.asarray(X, np.float32)
    y = list(y)
    if classes is None:
        classes = sorted(set(y))
    index = {c: i for i, c in enumerate(classes)}
    Y = np.array([index[c] for c in y], np.int64)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xs = (X - mean) / std

    n, d = Xs.shape
    k = len(classes)
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.01, (d, k)).astype(np.float32)
    b = np.zeros(k, np.float32)

    # one-hot dos rotulos do lote, reaproveitada por iteracao
    for it in range(iters):
        idx = rng.integers(0, n, size=min(batch, n))
        xb, yb = Xs[idx], Y[idx]
        p = softmax(xb @ W + b)
        oh = np.zeros_like(p)
        oh[np.arange(len(idx)), yb] = 1.0
        g = (p - oh) / len(idx)  # (b, K)
        grad_W = xb.T @ g + l2 * W
        grad_b = g.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
        if verbose and (it + 1) % 250 == 0:
            acc = (p.argmax(1) == yb).mean()
            print(f"  it {it + 1:4d}  lote_acc={acc:.2f}")

    return LetterClassifier(classes=classes, mean=mean, std=std, W=W, b=b)
