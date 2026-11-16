"""
Etapas 2 a 4 do pipeline: globo ocular, feixe de cada olho e projeção de
cada feixe no monitor virtual.

GLOBO (ANATÔMICO). Cada globo é uma esfera de raio r que gira em torno de um
centro fixo no crânio. O raio vem do próprio olho: r = 0,4 × largura do olho
(distância entre os cantos; ~12 mm / ~30 mm), medida a cada quadro antes de
calibrar e fixada na calibração (mediana). O centro fica atrás da pupila, num
deslocamento constante no referencial da cabeça a partir da âncora do olho:

    centro = âncora + R · (c_x, c_y, r)

com (c_x, c_y) a posição da pupila quando o olho aponta para a frente da
cabeça. Como no projeto de referência (Webcam3DTracker), o globo tem o
tamanho de um olho real e fica "travado" à cabeça.

ÂNCORA. Antes de calibrar, é o ponto médio dos cantos do olho. A calibração
mede o deslocamento fixo de cada olho em relação à ponte do nariz e passa a
usar a âncora "mista": horizontal e profundidade pelos cantos, VERTICAL pela
ponte do nariz. Os cantos fazem parte do contorno das pálpebras e sobem e
descem com o olhar, o que "comia" o sinal vertical da pupila; a ponte do
nariz não se move com as pálpebras (docs/metodologia.md, seção 5.6).

FEIXE. Sai do centro do globo na direção da pupila. No referencial da
cabeça, com p = Rᵀ(pupila − âncora):

    direção_cabeça ∝ ( g_x·(p_x − c_x)/r , g_y·(p_y − c_y)/r , −1 )
    direção_câmera = R · direção_cabeça

g = (g_x, g_y) é o GANHO da pupila, ajustado na calibração. O MediaPipe
"encolhe" o movimento da íris (nas sessões gravadas, para ~1/3 do real, mais
na vertical): a íris medida anda menos que a pupila de verdade. O ganho
corrige essa compressão na própria medida da pupila, ANTES de somar a
rotação da cabeça, então girar a cabeça continua movendo o olhar 1:1. Com
g > 1, o feixe passa pela "pupila corrigida", o ponto da superfície do globo
onde a pupila de verdade estaria (``GazeRay.corrected``).

PROJEÇÃO (etapa 4). Cada feixe é prolongado até o plano do monitor virtual
(``screen_points``), o que dá um ponto por olho. O ponto do olhar é a média
desses dois pontos, ponderada pela qualidade de cada olho na calibração
(``eye_weights``, inverso do erro angular ao quadrado); com um olho fechado,
vale o outro. Os rastreadores comerciais fazem o mesmo. A versão anterior
cruzava os dois feixes em 3D: com ~6 cm entre os olhos e a tela a ~60 cm, a
profundidade da intersecção oscilava 4-5 cm com o olhar parado, e usar
sempre a bissetriz dava o mesmo erro (metodologia, seção 5.6).

CALIBRAÇÃO (``EyeballModel.fit``). Olhando para um alvo conhecido T, a
direção do centro até T é conhecida, e a pupila medida deve satisfazer, em
cada eixo,

    p_j = c_j + (r / g_j) · tan_j(T)          (tan no referencial da cabeça)

Para cada olho e cada eixo, isso é uma regressão linear com 2 incógnitas
(c_j e r/g_j), resolvida por mínimos quadrados ponderados ROBUSTOS (Huber,
por repesagem iterativa): quadros em que a pessoa piscou, olhou para fora
ou o landmark falhou pesam menos, em vez de puxar o globo. Uma regularização
fraca puxa o ganho para 1. Como o centro depende de c e a direção até o
alvo depende do centro, o ajuste é repetido duas vezes (a mudança é de
milímetros a ~60 cm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaze import RAW_EYE_WIDTH, RAW_NOSE, raw_parts

RADIUS_CM = 1.2                # raio anatômico típico do globo ocular
RADIUS_RATIO = 0.40            # raio / largura do olho (cantos)
GAIN_MIN, GAIN_MAX = 0.2, 25.0
PRIOR_WEIGHT = 2e-3            # regularização do ganho em direção a 1 (unidades de tan²)
HUBER_K = 1.345                # constante de Huber (em desvios robustos)
HUBER_ITERS = 4
MIN_EYE_ERR_DEG = 0.5          # piso do erro angular no peso de cada olho
EYE_NAMES = ("direito", "esquerdo")


@dataclass
class GazeRay:
    """Feixes e vetor do olhar de UMA amostra (cm, referencial da câmera)."""

    centers: np.ndarray       # (2, 3) centro do globo de cada olho
    pupils: np.ndarray        # (2, 3) pupila medida de cada olho
    directions: np.ndarray    # (2, 3) direção unitária de cada feixe
    open: np.ndarray          # (2,) olho aberto
    origin: np.ndarray        # (3,) ponto médio entre os centros (ou o olho aberto)
    direction: np.ndarray     # (3,) vetor do olhar (bissetriz; unitário)
    radii: np.ndarray | None = None    # (2,) raio anatômico de cada globo (cm)
    weights: np.ndarray | None = None  # (2,) peso de cada olho no ponto da tela

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
    (x, y)), ganho da pupila g (2 olhos, (x, y)) e peso de cada olho no ponto
    da tela.

    ``anchor_offsets`` (2, 3): deslocamento do meio dos cantos de cada olho em
    relação à ponte do nariz, no referencial da cabeça (medido na
    calibração). Com ele, a vertical da âncora vem da ponte do nariz; sem ele
    (antes de calibrar), a âncora é o meio dos cantos.

    Antes de calibrar (r = None), o raio vem da largura do olho de cada
    quadro, o ganho é 1 e os dois olhos pesam igual."""

    def __init__(self, c=None, gain=None, source: str = "mediapipe", anchor_offsets=None,
                 r=None, eye_weights=None):
        self.c = np.zeros((2, 2)) if c is None else np.asarray(c, dtype=np.float64)
        self.gain = np.ones((2, 2)) if gain is None else np.asarray(gain, dtype=np.float64)
        self.r = None if r is None else np.asarray(r, dtype=np.float64)
        self.source = source
        self.anchor_offsets = (None if anchor_offsets is None
                               else np.asarray(anchor_offsets, dtype=np.float64))
        self.eye_weights = (np.full(2, 0.5) if eye_weights is None
                            else np.asarray(eye_weights, dtype=np.float64))
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
        if self.anchor_offsets is not None:
            # Horizontal e profundidade pelos cantos, vertical pela ponte do nariz.
            N = raw[:, RAW_NOSE]
            An = N[:, None, :] + _r(R, np.broadcast_to(self.anchor_offsets, A.shape))
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
        """Vetor do olhar (vetorizado): origem (n, 3) entre os centros dos
        globos e direção (n, 3) = bissetriz dos feixes (o feixe do olho aberto,
        se só um está aberto; NaN com os dois fechados). Usado na exibição e
        na detecção de fixação; o ponto na tela vem de ``screen_points``.
        Devolve também os feixes (saída de ``beams``)."""
        centers, P, d, op = self.beams(raw)
        both = op[:, 0] & op[:, 1]
        only = np.where(op[:, 0], 0, 1)
        n = len(op)
        origin = np.where(both[:, None], centers.mean(axis=1), centers[np.arange(n), only])
        direction = np.where(both[:, None], _unit(d[:, 0] + d[:, 1]), d[np.arange(n), only])
        direction[~op.any(axis=1)] = np.nan
        return origin, direction, (centers, P, d, op)

    def screen_points(self, raw, screen):
        """Etapa 4: (pontos de cada olho (n, 2, 2), ponto combinado (n, 2)) em
        pixels do monitor virtual ``screen`` (geometry.VirtualScreen). NaN
        onde o feixe não atinge a tela ou o olho está fechado."""
        centers, _, d, op = self.beams(raw)
        per_eye = np.stack([screen.project(centers[:, k], d[:, k]) for k in range(2)], axis=1)
        ok = op & np.all(np.isfinite(per_eye), axis=-1)
        w = np.where(ok, self.eye_weights[None, :], 0.0)
        tot = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            combined = np.einsum("nk,nkj->nj", w, np.nan_to_num(per_eye)) / tot[:, None]
        combined[tot <= 0] = np.nan
        per_eye[~ok] = np.nan
        return per_eye, combined

    def ray(self, raw) -> GazeRay | None:
        """Feixes de UMA amostra (None se os dois olhos estão fechados)."""
        raw = np.atleast_2d(raw)
        origin, direction, (centers, P, d, op) = self.gaze(raw)
        if not op[0].any():
            return None
        return GazeRay(centers[0], P[0], d[0], op[0], origin[0], direction[0],
                       np.array(self._radii(raw)[0]), self.eye_weights.copy())

    # ------------------------------------------------------------ ajuste
    @classmethod
    def fit(cls, raw, targets_cm, weights=None, source: str = "mediapipe",
            prior_weight: float = PRIOR_WEIGHT, iterations: int = 2,
            weight_eyes: bool = True) -> "EyeballModel":
        """Ajusta o globo de cada olho a partir de amostras com alvo conhecido.
        raw: (n, RAW_SIZE); targets_cm: (n, 3) alvo no referencial da câmera.
        weight_eyes: peso de cada olho no ponto da tela pelo erro angular de
        ajuste (False: os dois pesam igual)."""
        raw = np.atleast_2d(raw)
        T = np.atleast_2d(np.asarray(targets_cm, dtype=np.float64))
        w = np.ones(len(raw)) if weights is None else np.asarray(weights, dtype=np.float64)
        m = cls(source=source)
        R0, A0, _, op0 = raw_parts(raw, source)
        # Raio anatômico de cada olho: 0,4 × mediana da largura do olho.
        widths = raw[:, RAW_EYE_WIDTH]
        m.r = np.array([RADIUS_RATIO * np.median(widths[op0[:, k], k])
                        if op0[:, k].any() else RADIUS_CM for k in range(2)])
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
        errs = [float(np.average(ang[op[:, k] & (w > 0), k], weights=w[op[:, k] & (w > 0)]))
                if (op[:, k] & (w > 0)).any() else None for k in range(2)]
        m.info = {"erro_angular_ajuste_graus": errs}
        if weight_eyes and all(e is not None for e in errs):
            # Peso de cada olho = inverso da variância (erro angular²): em 7 de
            # 33 sessões gravadas, um olho errava 1,5x mais que o outro.
            inv = np.array([1.0 / max(e, MIN_EYE_ERR_DEG) ** 2 for e in errs])
            m.eye_weights = inv / inv.sum()
        return m

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {"c": self.c.tolist(), "gain": self.gain.tolist(),
                "r": None if self.r is None else self.r.tolist(), "source": self.source,
                "anchor_offsets": (None if self.anchor_offsets is None
                                   else self.anchor_offsets.tolist()),
                "eye_weights": self.eye_weights.tolist(),
                "info": self.info}

    @classmethod
    def from_dict(cls, d: dict) -> "EyeballModel":
        m = cls(np.array(d["c"]), np.array(d["gain"]), d.get("source", "mediapipe"),
                d.get("anchor_offsets"), d.get("r"), d.get("eye_weights"))
        m.info = d.get("info", {})
        return m
