"""
Etapas 2 e 3 do pipeline: globo ocular, feixe de cada olho e intersecção.

GLOBO (ANATÔMICO). Cada globo é uma esfera de raio r que gira em torno de um
centro fixo no crânio. O raio vem do próprio olho: r = 0,4 × largura do olho
(distância entre os cantos; ~12 mm / ~30 mm), medida a cada quadro antes de
calibrar e fixada na calibração (mediana). O centro fica atrás da pupila, num
deslocamento constante no referencial da cabeça a partir da âncora do olho:

    centro = âncora + R · (c_x, c_y, r)

com (c_x, c_y) a posição da pupila quando o olho aponta para a frente da
cabeça. Como no projeto de referência (Webcam3DTracker), o globo tem o
tamanho de um olho real e fica "travado" à cabeça.

A âncora padrão (``"misto"``, ver EyeballModel) usa o ponto médio dos
cantos do olho na horizontal e na profundidade e a ponte do nariz na
vertical: os cantos sobem e descem com as pálpebras quando o olhar muda na
vertical, e isso "comia" o sinal vertical da pupila.

FEIXE. Sai do centro do globo na direção da pupila. No referencial da
cabeça, com p = Rᵀ(pupila − âncora):

    direção_cabeça ∝ ( g_x·(p_x − c_x)/r , g_y·(p_y − c_y)/r , −1 )
    direção_câmera = R · direção_cabeça

g = (g_x, g_y) é o GANHO da pupila, ajustado na calibração. O MediaPipe
"encolhe" o movimento da íris (nas sessões gravadas, para ~1/3 do real, mais
na vertical): a íris medida anda menos que a pupila de verdade. O ganho
corrige essa compressão na própria medida da pupila, ANTES de somar a
rotação da cabeça, então girar a cabeça continua movendo o olhar 1:1. Num
olho medido sem compressão, g = 1 e o feixe passa exatamente pela pupila. Com
g > 1, ele passa pela "pupila corrigida", o ponto da superfície do globo
onde a pupila de verdade estaria (``GazeRay.corrected``). O projeto de
referência faz o mesmo no mapeamento ângulo → tela (±15° na horizontal e só
±5° na vertical, ou seja, 3× mais ganho na vertical).

INTERSECÇÃO. Os dois feixes (olho direito e esquerdo) apontam para o mesmo
ponto fixado. Duas retas no espaço quase nunca se cruzam exatamente; a
intersecção é o ponto médio do segmento perpendicular comum
(geometry.closest_point_between_rays). O VETOR DO OLHAR vai do ponto médio
entre os centros dos globos ("olho ciclópico") até a intersecção. Quando
os feixes não convergem à frente do rosto (ruído, retas quase paralelas
ou divergentes), usa-se a bissetriz das duas direções; com um olho
fechado, o feixe do outro.

CALIBRAÇÃO (``EyeballModel.fit``). Olhando para um alvo conhecido T, a
direção do centro até T é conhecida, e a pupila medida deve satisfazer, em
cada eixo,

    p_j = c_j + (r / g_j) · tan_j(T)          (tan no referencial da cabeça)

Para cada olho e cada eixo, isso é uma regressão linear com 2 incógnitas
(c_j e r/g_j), resolvida por mínimos quadrados ponderados ROBUSTOS (Huber,
por repesagem iterativa): quadros em que a pessoa piscou, olhou para fora
ou o landmark falhou pesam menos, em vez de puxar o globo. Uma regularização
fraca puxa o ganho para 1. A etapa da cabeça (olhar fixo, cabeça girando)
entra no mesmo ajuste: é ela que separa o movimento do olho do movimento da
cabeça. Como o centro depende de c e a direção até o alvo depende do
centro, o ajuste é repetido duas vezes (a mudança é de milímetros a ~60 cm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import closest_point_between_rays
from .gaze import RAW_EYE_WIDTH, RAW_NOSE, raw_parts

RADIUS_CM = 1.2                # raio anatômico típico do globo ocular
RADIUS_RATIO = 0.40            # raio / largura do olho (cantos)
GAIN_MIN, GAIN_MAX = 0.2, 25.0
PRIOR_WEIGHT = 2e-3            # regularização do ganho em direção a 1 (unidades de tan²)
HUBER_K = 1.345                # constante de Huber (em desvios robustos)
HUBER_ITERS = 4
# Intersecção aceita entre estas distâncias ao longo de cada feixe (cm).
VERGENCE_MIN_CM, VERGENCE_MAX_CM = 5.0, 400.0
EYE_NAMES = ("direito", "esquerdo")


@dataclass
class GazeRay:
    """Feixes e vetor do olhar de UMA amostra (cm, referencial da câmera)."""

    centers: np.ndarray       # (2, 3) centro do globo de cada olho
    pupils: np.ndarray        # (2, 3) pupila medida de cada olho
    directions: np.ndarray    # (2, 3) direção unitária de cada feixe
    open: np.ndarray          # (2,) olho aberto
    origin: np.ndarray        # (3,) ponto médio entre os centros (ou o olho aberto)
    direction: np.ndarray     # (3,) vetor do olhar (unitário)
    vergence: np.ndarray | None   # (3,) intersecção dos feixes, se convergem
    vergence_gap_cm: float | None  # distância entre os feixes na intersecção
    radii: np.ndarray | None = None  # (2,) raio anatômico de cada globo (cm)

    @property
    def corrected(self) -> np.ndarray:
        """(2, 3) onde cada feixe sai do globo: a pupila corrigida pelo ganho."""
        r = np.full(2, RADIUS_CM) if self.radii is None else self.radii
        return self.centers + r[:, None] * self.directions


def _rt(R, V):
    """Rᵀ·v para cada amostra: R (n, 3, 3), V (n, k, 3)."""
    return np.einsum("nji,nkj->nki", R, V)


def _r(R, V):
    return np.einsum("nij,nkj->nki", R, V)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _huber_line(x, y, w, prior_slope, prior_weight):
    """y ≈ a + b·x por mínimos quadrados ponderados com pesos de Huber
    (repesagem iterativa) e regularização λ·W·(b − prior_slope)²."""
    wk = w.copy()
    a = b = 0.0
    for it in range(HUBER_ITERS + 1):
        W = wk.sum()
        lam = prior_weight * W
        Sx, Sxx = wk @ x, wk @ (x * x)
        M = np.array([[W, Sx], [Sx, Sxx + lam]])
        rhs = np.array([wk @ y, wk @ (x * y) + lam * prior_slope])
        a, b = np.linalg.solve(M, rhs)
        if it == HUBER_ITERS:
            break
        res = y - a - b * x
        scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        u = np.abs(res) / (HUBER_K * scale)
        wk = w * np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    return float(a), float(b)


class EyeballModel:
    """Globo de cada olho: raio anatômico r (2,), repouso da pupila c (2 olhos,
    (x, y)) e ganho da pupila g (2 olhos, (x, y)).

    anchor: de onde parte o deslocamento até o centro do globo.

    * ``"cantos"``: meio dos cantos do olho, a cada quadro;
    * ``"nariz"``: ponte do nariz + deslocamento fixo de cada olho no
      referencial da cabeça, medido na calibração (como o projeto de
      referência faz com a região do nariz);
    * ``"misto"`` (padrão da calibração): horizontal e profundidade pelos
      cantos, VERTICAL pela ponte do nariz. Os cantos fazem parte do
      contorno das pálpebras e sobem/descem com o olhar, o que reduzia o
      sinal vertical da pupila; a ponte do nariz não se move com as
      pálpebras (docs/metodologia.md, seção 5.6).

    Antes de calibrar (r = None), o raio vem da largura do olho de cada
    quadro e o ganho é 1 (olho anatômico, sem correção)."""

    def __init__(self, c=None, gain=None, source: str = "mediapipe", anchor: str = "cantos",
                 anchor_offsets=None, r=None):
        self.c = np.zeros((2, 2)) if c is None else np.asarray(c, dtype=np.float64)
        self.gain = np.ones((2, 2)) if gain is None else np.asarray(gain, dtype=np.float64)
        self.r = None if r is None else np.asarray(r, dtype=np.float64)
        self.source = source
        self.anchor = anchor
        self.anchor_offsets = (None if anchor_offsets is None
                               else np.asarray(anchor_offsets, dtype=np.float64))
        self.info: dict = {}

    @property
    def s(self) -> np.ndarray:
        """(2, 2) raio efetivo r/g: quanto a pupila MEDIDA anda por unidade de
        tangente do ângulo do olho."""
        r = np.full(2, RADIUS_CM) if self.r is None else self.r
        return r[:, None] / self.gain

    def _radii(self, raw):
        """(n, 2) raio do globo de cada olho em cada amostra."""
        if self.r is not None:
            return np.broadcast_to(self.r, (len(raw), 2))
        width = raw[:, RAW_EYE_WIDTH]
        return np.where(width > 0.5, RADIUS_RATIO * width, RADIUS_CM)

    def _parts(self, raw):
        """R, âncoras, pupilas e olhos abertos, com a âncora do modelo."""
        raw = np.atleast_2d(raw)
        R, A, P, op = raw_parts(raw, self.source)
        if self.anchor in ("nariz", "misto") and self.anchor_offsets is not None:
            N = raw[:, RAW_NOSE]
            An = N[:, None, :] + _r(R, np.broadcast_to(self.anchor_offsets, A.shape))
            if self.anchor == "nariz":
                A = An
            else:   # horizontal e profundidade pelos cantos, vertical pelo nariz
                a_h = _rt(R, A)
                a_h[..., 1] = _rt(R, An)[..., 1]
                A = _r(R, a_h)
        return R, A, P, op

    def _centers(self, R, A, radii):
        off = np.concatenate([np.broadcast_to(self.c, A.shape[:-1] + (2,)),
                              radii[..., None]], axis=-1)
        return A + _r(R, off)

    # ------------------------------------------------------------ feixes
    def beams(self, raw):
        """Feixe de cada olho (vetorizado): centros (n, 2, 3), pupilas (n, 2, 3),
        direções unitárias (n, 2, 3) e olhos abertos (n, 2)."""
        raw = np.atleast_2d(raw)
        R, A, P, op = self._parts(raw)
        radii = self._radii(raw)
        p = _rt(R, P - A)
        t = self.gain * (p[..., :2] - self.c) / radii[..., None]
        d_head = _unit(np.concatenate([t, -np.ones(t.shape[:-1] + (1,))], axis=-1))
        return self._centers(R, A, radii), P, _r(R, d_head), op

    def gaze(self, raw):
        """Etapa 3, vetorizada. Devolve (origem (n, 3), vetor do olhar (n, 3),
        intersecção (n, 3; NaN se não convergem), distância entre os feixes
        (n,), feixes = saída de ``beams``)."""
        centers, P, d, op = self.beams(raw)
        X, s1, t1, gap = closest_point_between_rays(centers[:, 0], d[:, 0],
                                                    centers[:, 1], d[:, 1])
        both = op[:, 0] & op[:, 1]
        with np.errstate(invalid="ignore"):
            conv = (both & (s1 > VERGENCE_MIN_CM) & (t1 > VERGENCE_MIN_CM)
                    & (s1 < VERGENCE_MAX_CM) & (t1 < VERGENCE_MAX_CM))
        only = np.where(op[:, 0], 0, 1)
        n = len(op)
        origin = np.where(both[:, None], centers.mean(axis=1), centers[np.arange(n), only])
        direction = np.where(both[:, None], _unit(d[:, 0] + d[:, 1]), d[np.arange(n), only])
        direction = np.where(conv[:, None], _unit(X - origin), direction)
        X = np.where(conv[:, None], X, np.nan)
        gap = np.where(conv, gap, np.nan)
        none = ~op.any(axis=1)
        direction[none] = np.nan
        return origin, direction, X, gap, (centers, P, d, op)

    def ray(self, raw) -> GazeRay | None:
        """Etapa 3 para UMA amostra (None se os dois olhos estão fechados)."""
        raw = np.atleast_2d(raw)
        origin, direction, X, gap, (centers, P, d, op) = self.gaze(raw)
        if not op[0].any():
            return None
        conv = bool(np.all(np.isfinite(X[0])))
        return GazeRay(centers[0], P[0], d[0], op[0], origin[0], direction[0],
                       X[0] if conv else None, float(gap[0]) if conv else None,
                       np.array(self._radii(raw)[0]))

    def pupil_residual(self, raw, targets_cm) -> np.ndarray:
        """(n, 2 olhos, 2 eixos) pupila medida menos a prevista pelo modelo
        para quem olha para ``targets_cm`` (cm, no referencial da cabeça; NaN
        com o olho fechado). Usado para medir a deriva da pupila."""
        raw = np.atleast_2d(raw)
        T = np.atleast_2d(np.asarray(targets_cm, dtype=np.float64))
        R, A, P, op = self._parts(raw)
        radii = self._radii(raw)
        p = _rt(R, P - A)
        tau = _rt(R, T[:, None, :] - self._centers(R, A, radii))
        tan = tau[..., :2] / np.maximum(-tau[..., 2:3], 1e-6)
        res = p[..., :2] - (self.c + (radii[..., None] / self.gain) * tan)
        res[~op] = np.nan
        return res

    # ------------------------------------------------------------ ajuste
    @classmethod
    def fit(cls, raw, targets_cm, weights=None, source: str = "mediapipe",
            prior_weight: float = PRIOR_WEIGHT, iterations: int = 2,
            anchor: str = "cantos") -> "EyeballModel":
        """Ajusta o globo de cada olho a partir de amostras com alvo conhecido.
        raw: (n, RAW_SIZE); targets_cm: (n, 3) alvo no referencial da câmera."""
        raw = np.atleast_2d(raw)
        T = np.atleast_2d(np.asarray(targets_cm, dtype=np.float64))
        w = np.ones(len(raw)) if weights is None else np.asarray(weights, dtype=np.float64)
        m = cls(source=source, anchor=anchor)
        R0, A0, _, op0 = raw_parts(raw, source)
        # Raio anatômico de cada olho: 0,4 × mediana da largura do olho.
        widths = raw[:, RAW_EYE_WIDTH]
        m.r = np.array([RADIUS_RATIO * np.median(widths[op0[:, k], k])
                        if op0[:, k].any() else RADIUS_CM for k in range(2)])
        if anchor in ("nariz", "misto"):
            # Deslocamento fixo do meio dos cantos em relação à ponte do nariz,
            # no referencial da cabeça (mediana das amostras da calibração).
            rel = _rt(R0, A0 - raw[:, RAW_NOSE][:, None, :])
            m.anchor_offsets = np.array([np.median(rel[op0[:, k], k], axis=0)
                                         if op0[:, k].any() else np.zeros(3) for k in range(2)])
        R, A, P, op = m._parts(raw)
        p = _rt(R, P - A)
        radii = m._radii(raw)
        for _ in range(iterations):
            tau = _rt(R, T[:, None, :] - m._centers(R, A, radii))
            tan_t = tau[..., :2] / np.maximum(-tau[..., 2:3], 1e-6)
            for k in range(2):
                ok = op[:, k] & (w > 0)
                if ok.sum() < 3:
                    continue
                for j in range(2):
                    # p = c + (r/g)·tan  ->  reta em tan com inclinação r/g
                    c, slope = _huber_line(tan_t[ok, k, j], p[ok, k, j], w[ok],
                                           m.r[k], prior_weight)
                    g = float(np.clip(m.r[k] / max(slope, 1e-9), GAIN_MIN, GAIN_MAX))
                    if slope <= 0 or g in (GAIN_MIN, GAIN_MAX):
                        c = float(np.average(p[ok, k, j] - (m.r[k] / g) * tan_t[ok, k, j],
                                             weights=w[ok]))
                    m.c[k, j], m.gain[k, j] = c, g
        # Erro angular médio (graus) de cada olho nas amostras de ajuste.
        centers, _, d, _ = m.beams(raw)
        want = _unit(T[:, None, :] - centers)
        ang = np.degrees(np.arccos(np.clip(np.sum(d * want, -1), -1, 1)))
        m.info = {"erro_angular_ajuste_graus": [
            float(np.average(ang[op[:, k] & (w > 0), k], weights=w[op[:, k] & (w > 0)]))
            if (op[:, k] & (w > 0)).any() else None for k in range(2)]}
        return m

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {"c": self.c.tolist(), "gain": self.gain.tolist(),
                "r": None if self.r is None else self.r.tolist(), "source": self.source,
                "anchor": self.anchor,
                "anchor_offsets": (None if self.anchor_offsets is None
                                   else self.anchor_offsets.tolist()),
                "info": self.info}

    @classmethod
    def from_dict(cls, d: dict) -> "EyeballModel":
        m = cls(np.array(d["c"]), np.array(d["gain"]), d.get("source", "mediapipe"),
                d.get("anchor", "cantos"), d.get("anchor_offsets"), d.get("r"))
        m.info = d.get("info", {})
        return m
