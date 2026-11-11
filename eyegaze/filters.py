"""
One-Euro Filter (Casiez, Roussel & Vogel, CHI 2012).

Filtro passa-baixa cuja frequência de corte aumenta com a velocidade do
sinal: suaviza bastante quando o olhar está parado (reduz tremor) e pouco
quando o olhar se move rápido (reduz atraso). Substitui a média móvel de
tamanho fixo do script de referência, que atrasa igualmente em qualquer
situação.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filtra um vetor de dimensão fixa (cada componente independente)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, value, t: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self._x is None:
            self._x = value.copy()
            self._dx = np.zeros_like(value)
            self._t = t
            return self._x.copy()

        dt = max(t - self._t, 1e-3)
        self._t = t

        dx = (value - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = np.array([_alpha(c, dt) for c in np.atleast_1d(cutoff)]).reshape(value.shape)
        self._x = a * value + (1 - a) * self._x
        return self._x.copy()


class FixationStabilizer:
    """Estabilizador do cursor baseado em fixações e sacadas.

    O olhar humano alterna entre FIXAÇÕES (olho parado num ponto por
    200-600 ms, com micro-movimentos involuntários) e SACADAS (saltos
    rápidos). Com webcam, o ruído dos landmarks durante uma fixação é da
    ordem de dezenas de pixels na tela. Um filtro passa-baixa genérico ou
    deixa o cursor tremendo, ou atrasa demais as sacadas.

    Estratégia (inspirada em detectores de fixação por dispersão, I-DT):

    * Durante uma fixação, o cursor segue a MÉDIA das últimas amostras
      (janela deslizante). Ruído com desvio s vira s/sqrt(N). Uma zona
      morta impede movimentos de poucos pixels.
    * Uma amostra a mais de ``limiar`` px da fixação atual é candidata a
      sacada. Só depois de ``confirm_frames`` amostras seguidas, coerentes
      entre si, o cursor salta para o novo ponto. Picos isolados (piscada
      parcial, erro de detecção) são ignorados.
    * O limiar se adapta ao ruído medido: max(limiar_base, k * desvio).
      Com mais ruído (luz ruim, rosto longe), fica mais difícil confundir
      tremor com sacada.
    """

    def __init__(self, saccade_px: float = 130.0, confirm_frames: int = 3,
                 window: int = 20, follow: float = 0.3, deadzone_px: float = 6.0,
                 noise_k: float = 3.0):
        self.saccade_px = saccade_px
        self.confirm_frames = confirm_frames
        self.window = window
        self.follow = follow
        self.deadzone_px = deadzone_px
        self.noise_k = noise_k
        self.reset()

    def reset(self):
        self.cursor: np.ndarray | None = None
        self._buf: list[np.ndarray] = []
        self._pending: list[np.ndarray] = []
        self.threshold = self.saccade_px
        self.noise_px = 0.0

    def _fix_mean(self) -> np.ndarray:
        return np.mean(self._buf, axis=0)

    def __call__(self, point) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        if self.cursor is None:
            self.cursor = p.copy()
            self._buf = [p]
            return self.cursor.copy()

        if len(self._buf) >= 8:
            arr = np.asarray(self._buf)
            self.noise_px = float(np.sqrt(np.sum(arr.var(axis=0))))
        self.threshold = max(self.saccade_px, self.noise_k * self.noise_px)

        if np.linalg.norm(p - self._fix_mean()) > self.threshold:
            self._pending.append(p)
            if len(self._pending) >= self.confirm_frames:
                pend = np.asarray(self._pending)
                pm = pend.mean(axis=0)
                if np.max(np.linalg.norm(pend - pm, axis=1)) < self.threshold:
                    # Sacada confirmada: nova fixação.
                    self._buf = list(pend)
                    self._pending = []
                    self.cursor = pm.copy()
                else:
                    self._pending.pop(0)
            return self.cursor.copy()

        self._pending = []
        self._buf.append(p)
        if len(self._buf) > self.window:
            self._buf.pop(0)
        target = self._fix_mean()
        if np.linalg.norm(target - self.cursor) > self.deadzone_px:
            self.cursor = self.cursor + self.follow * (target - self.cursor)
        return self.cursor.copy()
