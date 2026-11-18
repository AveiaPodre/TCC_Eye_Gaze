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


class GazeFollower:
    """Estabilizador do cursor: salta nas sacadas; na fixação, fica PARADO
    enquanto o olhar estimado está perto e desliza até ele quando o olhar se
    move de verdade (histerese).

    O olhar humano alterna entre FIXAÇÕES (olho parado) e SACADAS (saltos
    rápidos); dentro de uma fixação há também movimentos SUTIS (microssacadas
    e pequenos reajustes de dezenas de pixels). Com webcam, a estimativa
    numa fixação treme de quadro a quadro e também passeia devagar.

    * **Sacada.** Uma amostra a mais de ``limiar`` px da mediana recente é
      candidata. Depois de ``confirm_frames`` amostras seguidas, coerentes
      entre si, o cursor salta para a mais recente (``fixation_id`` aumenta)
      e, nos ``settle_frames`` quadros seguintes, desliza até a mediana das
      amostras (o salto foi para UMA amostra, ruidosa). Picos isolados são
      ignorados. limiar = max(limiar_base, k × ruído).
    * **Fixação.** O destino é a mediana das últimas ``median_frames``
      amostras. Parado, o cursor só começa a andar quando o destino se
      afasta mais que ``start_k`` × ruído; aí desliza (``alpha`` do caminho
      por quadro) até ficar a menos de ``stop_k`` × ruído e para de novo. O
      ruído é o de quadro a quadro (robusto), não o passeio.

    Histórico (bancada com as validações gravadas, 20 sessões, 180 alvos,
    degraus sintéticos sobre o ruído real; seção 4.0 da metodologia):

    * trava (até 16/11): o raio crescia com o passeio (188 px de mediana numa
      sessão de uso); seguia 0%/5%/11% dos degraus de 40/70/100 px (o
      cursor andava em blocos), cursor parado em 97% dos quadros;
    * perseguição suave (16/11): deslizava sempre um pouco até o olhar;
      seguia 38%/62%/66%, mas parado em só 72% (o cursor acompanhava o
      passeio com o olho fixo);
    * histerese (atual): segue 28%/58%/63%, parado em 86%, tremor 3,1 contra
      3,9 px/quadro, mesmo erro no alvo.
    """

    def __init__(self, saccade_px: float = 134.0, confirm_frames: int = 2,
                 median_frames: int = 7, start_k: float = 4.0, stop_k: float = 1.0,
                 alpha: float = 0.4, noise_k: float = 3.0, noise_floor_px: float = 15.0,
                 settle_frames: int = 8, window: int = 30):
        self.saccade_px = saccade_px
        self.confirm_frames = confirm_frames
        self.median_frames = median_frames
        self.start_k, self.stop_k, self.alpha = start_k, stop_k, alpha
        self.noise_k = noise_k
        self.noise_floor_px = noise_floor_px
        self.settle_frames = settle_frames
        self.window = window
        self.reset()

    def reset(self):
        self.cursor: np.ndarray | None = None
        self._buf: list[np.ndarray] = []      # amostras da fixação atual
        self._pending: list[np.ndarray] = []  # candidatas a sacada
        self._age = 0                          # amostras desde o início da fixação
        self.moving = False                    # deslizando até o olhar
        self.noise_px = self.noise_floor_px    # ruído de quadro a quadro (HUD)
        self.threshold = self.saccade_px
        self.fixation_id = 0    # incrementa a cada sacada

    @property
    def settling(self) -> bool:
        """True nos primeiros ``settle_frames`` quadros de uma fixação."""
        return self.cursor is not None and self._age <= self.settle_frames

    def _start_fixation(self, p):
        self._buf = [np.asarray(p, dtype=np.float64)]
        self._age = 1
        self.cursor = self._buf[0].copy()
        self.moving = True
        self.fixation_id += 1

    def __call__(self, point) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        if self.cursor is None:
            self._start_fixation(p)
            return self.cursor.copy()

        center = np.median(self._buf[-10:], axis=0)
        self.threshold = max(self.saccade_px, self.noise_k * self.noise_px)
        if np.linalg.norm(p - center) > self.threshold:
            self._pending.append(p)
            if len(self._pending) >= self.confirm_frames:
                pend = np.asarray(self._pending)
                # As pendentes precisam concordar entre si (um pico isolado não
                # é sacada); a tolerância cresce com o tamanho do salto (o
                # ponto bruto leva alguns quadros para atravessar a tela).
                tol = max(self.threshold, 0.5 * np.linalg.norm(pend[-1] - center))
                if np.max(np.linalg.norm(pend - pend.mean(axis=0), axis=1)) < tol:
                    self._pending = []
                    self._start_fixation(pend[-1])
                else:
                    self._pending.pop(0)
            return self.cursor.copy()

        self._pending = []
        self._buf.append(p)
        self._age += 1
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if len(self._buf) >= 8:
            # Ruído de quadro a quadro, robusto (MAD das diferenças), por amostra.
            dif = np.diff(np.asarray(self._buf), axis=0)
            mad = np.median(np.linalg.norm(dif - np.median(dif, axis=0), axis=1))
            self.noise_px = max(self.noise_floor_px, 1.4826 * mad / np.sqrt(2))

        goal = np.median(self._buf[-self.median_frames:], axis=0)
        e = goal - self.cursor
        dist = float(np.linalg.norm(e))
        if self._age <= self.settle_frames:
            self.moving = True                 # acomodação depois da sacada
        elif not self.moving and dist > self.start_k * self.noise_px:
            self.moving = True                 # o olhar se moveu de verdade
        if self.moving:
            # Na acomodação, desliza um pouco mais depressa (0,5 por quadro).
            a = max(self.alpha, 0.5) if self._age <= self.settle_frames else self.alpha
            self.cursor = self.cursor + a * e
            # Para assim que o passo deixa o cursor perto do destino. Com
            # alpha 0,7 isso costuma acontecer no 1º passo: amortece o
            # movimento (na bancada, exigir que o deslizamento durasse até a
            # mediana se renovar seguia mais, 64% contra 53% dos degraus de
            # 40 px, mas deixava o cursor parado em só 65% dos quadros de
            # olho fixo, contra 87%, e o erro subia de 25 para 29 px).
            if (self._age > self.settle_frames
                    and dist * (1.0 - self.alpha) < self.stop_k * self.noise_px):
                self.moving = False
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

    @property
    def adjusting(self) -> bool:
        """A cabeça está deslocando o cursor (passou da zona morta)."""
        return bool(np.any(self.offset != 0.0))

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


def make_stabilizer(st, screen_width: int) -> GazeFollower:
    """Estabilizador do cursor a partir da configuração (config.StabilizerConfig)."""
    return GazeFollower(
        saccade_px=st.saccade_ratio * screen_width, confirm_frames=st.confirm_frames,
        median_frames=st.median_frames, start_k=st.start_k, stop_k=st.stop_k,
        alpha=st.alpha, noise_k=st.noise_k, noise_floor_px=st.noise_floor_px,
        settle_frames=st.settle_frames)
