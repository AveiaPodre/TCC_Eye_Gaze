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
    """Estabilizador do cursor baseado em fixações e sacadas, com TRAVA.

    O olhar humano alterna entre FIXAÇÕES (olho parado num ponto por
    200-600 ms) e SACADAS (saltos rápidos). Com webcam, a estimativa
    durante uma fixação não treme só de quadro a quadro: ela PASSEIA. Nas
    sessões gravadas, a autocorrelação entre quadros seguidos era 0,97, e o
    ponto se afastava 50-130 px do início da fixação em 1,2 s, sem o olho
    se mover (≈1-2,5°, bem acima do movimento fisiológico na fixação). A
    versão anterior seguia a média de uma janela deslizante. Contra ruído
    branco isso funciona, mas a média acompanha esse passeio e o cursor
    "andava" com o olho parado (70% dos quadros com movimento).

    Estratégia:

    * **Sacada.** Uma amostra a mais de ``limiar`` px da fixação atual é
      candidata. Depois de ``confirm_frames`` amostras seguidas, coerentes
      entre si, o cursor salta para a amostra mais recente. Picos isolados
      (piscada parcial, erro de detecção) são ignorados. O limiar se adapta
      ao ruído medido: max(limiar_base, k * dispersão). Nas sessões
      gravadas, o atraso extra do cursor depois que o ponto bruto chega ao
      alvo é de 42 ms (mediana) e 110 ms (p90); antes, 92 e 292 ms.
    * **Acomodação.** Nos primeiros ``settle_frames`` quadros de uma
      fixação, o cursor desliza até a média das amostras (melhora a
      estimativa logo após o salto).
    * **Trava.** Depois disso, o cursor PARA. Só volta a se mover se as
      últimas ``shift_frames`` amostras estiverem TODAS a mais de
      ``raio de trava`` da posição travada (deslocamento pequeno, mas
      sustentado; o passeio do ruído vai e volta). Aí começa uma nova
      fixação no ponto médio dessas amostras.
    * O raio de trava é max(``lock_px``, ``lock_k`` x passeio típico), e o
      passeio típico é a dispersão das fixações anteriores (média móvel).
      Com mais ruído (luz ruim, rosto longe), a trava fica mais larga. O
      raio nunca passa de ``lock_cap`` (90%) do limiar de sacada.

    Com os dados gravados, os quadros com o cursor em movimento durante a
    fixação caíram de 71% para ~6% (a maior parte no deslizamento logo
    após a sacada), sem atraso nas sacadas (mediana de 2 quadros).
    """

    def __init__(self, saccade_px: float = 130.0, confirm_frames: int = 2,
                 window: int = 30, follow: float = 0.6, lock_px: float = 65.0,
                 lock_k: float = 3.5, shift_frames: int = 8, settle_frames: int = 8,
                 noise_k: float = 3.0, lock_cap: float = 0.9):
        self.saccade_px = saccade_px
        self.confirm_frames = confirm_frames
        self.window = window
        self.follow = follow
        self.lock_px = lock_px
        self.lock_k = lock_k
        self.shift_frames = shift_frames
        self.settle_frames = settle_frames
        self.noise_k = noise_k
        self.lock_cap = lock_cap
        self.reset()

    def reset(self):
        self.cursor: np.ndarray | None = None
        self.goal: np.ndarray | None = None   # para onde o cursor desliza / onde trava
        self._buf: list[np.ndarray] = []      # amostras da fixação atual
        self._pending: list[np.ndarray] = []  # candidatas a sacada
        self._age = 0                          # amostras desde o início da fixação
        self.threshold = self.saccade_px
        self.noise_px = 0.0     # dispersão da fixação atual (HUD)
        self.wander_px = 0.0    # passeio típico (média das fixações anteriores)
        self.fixation_id = 0    # incrementa a cada nova fixação (sacada ou deslocamento)

    @property
    def lock_radius(self) -> float:
        return min(max(self.lock_px, self.lock_k * self.wander_px), self.lock_cap * self.threshold)

    @property
    def settling(self) -> bool:
        """True nos primeiros ``settle_frames`` quadros de uma fixação."""
        return self.goal is not None and self._age <= self.settle_frames

    @property
    def locked(self) -> bool:
        """True quando o cursor está parado numa fixação."""
        return (self.goal is not None and self._age > self.settle_frames
                and self.cursor is not None and np.array_equal(self.cursor, self.goal))

    def _start_fixation(self, pts):
        if len(self._buf) >= 10:
            arr = np.asarray(self._buf)
            disp = float(np.sqrt(np.mean(np.sum((arr - arr.mean(0)) ** 2, axis=1))))
            self.wander_px = disp if self.wander_px == 0 else 0.8 * self.wander_px + 0.2 * disp
        self._buf = [np.asarray(q, dtype=np.float64) for q in pts]
        self._age = len(self._buf)
        self.goal = np.mean(self._buf, axis=0)
        self.fixation_id += 1

    def __call__(self, point) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        if self.cursor is None:
            self.cursor = p.copy()
            self._start_fixation([p])
            return self.cursor.copy()

        if len(self._buf) >= 8:
            arr = np.asarray(self._buf)
            self.noise_px = float(np.sqrt(np.sum(arr.var(axis=0))))
        self.threshold = max(self.saccade_px, self.noise_k * self.noise_px)

        center = np.mean(self._buf, axis=0)
        if np.linalg.norm(p - center) > self.threshold:
            self._pending.append(p)
            if len(self._pending) >= self.confirm_frames:
                pend = np.asarray(self._pending)
                pm = pend.mean(axis=0)
                # As amostras pendentes precisam concordar entre si (um pico
                # isolado não é sacada). A tolerância cresce com o tamanho do
                # salto: o ponto bruto leva alguns quadros para atravessar a
                # tela (o MediaPipe suaviza os landmarks), e nessa travessia
                # as amostras ficam espalhadas.
                tol = max(self.threshold, 0.5 * np.linalg.norm(pend[-1] - center))
                if np.max(np.linalg.norm(pend - pm, axis=1)) < tol:
                    # Sacada confirmada: o cursor salta para a amostra MAIS
                    # RECENTE e começa ali uma nova fixação (a acomodação
                    # desliza até a média das amostras seguintes). A média das
                    # pendentes ficava no meio do caminho da travessia, e o
                    # cursor ainda tinha de deslizar até o destino: nas
                    # sessões gravadas, o atraso p90 caiu de 235 para 110 ms.
                    self._start_fixation(pend[-1:])
                    self._pending = []
                    self.cursor = pend[-1].copy()
                else:
                    self._pending.pop(0)
            return self.cursor.copy()

        self._pending = []
        self._buf.append(p)
        self._age += 1
        if len(self._buf) > self.window:
            self._buf.pop(0)

        if self._age <= self.settle_frames:
            self.goal = np.mean(self._buf, axis=0)
        else:
            recent = np.asarray(self._buf[-self.shift_frames:])
            if (len(recent) >= self.shift_frames and
                    np.all(np.linalg.norm(recent - self.goal, axis=1) > self.lock_radius)):
                # Deslocamento pequeno e sustentado: nova fixação ali perto.
                self._start_fixation(recent)

        d = self.goal - self.cursor
        if np.linalg.norm(d) < 0.5:
            self.cursor = self.goal.copy()
        else:
            self.cursor = self.cursor + self.follow * d
        return self.cursor.copy()


class HeadFineTuner:
    """Ajuste fino do cursor pela cabeça (modo híbrido olhar + cabeça).

    O olhar leva o cursor à região (sacada) e o estabilizador o trava na
    fixação. Durante a fixação, pequenos movimentos da cabeça deslocam o
    cursor a partir dali:

        cursor = ponto da fixação + ganho x (pose da cabeça - pose no início
                 da fixação), descontada uma zona morta.

    Por quê: com o olhar parado, a estimativa do olhar treme ~40 px (limite
    das medidas do olho do MediaPipe), mas a pose da cabeça varia só ~0,1°
    (~2-4 px no cursor com o ganho padrão). A cabeça dá precisão de poucos
    pixels para acertar alvos pequenos.

    Uma nova fixação (sacada ou deslocamento do olhar) zera o ajuste, e o
    olhar volta a mandar; um giro grande da cabeça (``reset_deg``) também,
    porque aí a pessoa está mudando de postura, não ajustando o cursor.
    """

    def __init__(self, gain_px_per_deg: float = 25.0, deadzone_deg: float = 0.3,
                 max_px: float = 250.0, reset_deg: float = 6.0, smooth_hz: float = 2.0):
        self.gain = gain_px_per_deg
        self.deadzone = deadzone_deg
        self.max_px = max_px
        self.reset_deg = reset_deg
        self._filter = OneEuroFilter(smooth_hz, 0.0, 1.0)
        self.reset()

    def reset(self):
        self._fix_id: int | None = None
        self._ref: np.ndarray | None = None
        self.offset = np.zeros(2)
        self._filter.reset()

    def __call__(self, cursor: np.ndarray, fixation_id: int, head_yaw_pitch, t: float,
                 settling: bool = False) -> np.ndarray:
        """Cursor ajustado. head_yaw_pitch: (yaw, pitch) em graus (yaw > 0 =
        direita do usuário, pitch > 0 = para cima). settling: a fixação ainda
        está se acomodando; a referência da cabeça acompanha (a pessoa pode
        estar terminando de mover a cabeça junto com o olhar)."""
        head = self._filter(np.asarray(head_yaw_pitch, dtype=np.float64), t)
        if fixation_id != self._fix_id or self._ref is None or settling:
            self._fix_id = fixation_id
            self._ref = head.copy()
            self.offset = np.zeros(2)
            return np.asarray(cursor, dtype=np.float64).copy()
        delta = head - self._ref
        if np.max(np.abs(delta)) > self.reset_deg:
            # Mudança de postura: nova referência, o olhar volta a mandar.
            self._ref = head.copy()
            self.offset = np.zeros(2)
            return np.asarray(cursor, dtype=np.float64).copy()
        mag = np.abs(delta)
        eff = np.sign(delta) * np.maximum(mag - self.deadzone, 0.0)
        off = self.gain * np.array([eff[0], -eff[1]])   # cima na cabeça = cima na tela (-y)
        n = float(np.linalg.norm(off))
        if n > self.max_px:
            off *= self.max_px / n
        self.offset = off
        return np.asarray(cursor, dtype=np.float64) + off
