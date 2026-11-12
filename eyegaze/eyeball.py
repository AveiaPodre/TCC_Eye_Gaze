"""
Etapas 2 e 3 do pipeline: globo ocular, feixe de cada olho e intersecção.

CENTRO DO GLOBO. Cada globo gira em torno de um centro fixo no crânio. Ele
não aparece na imagem; fica num deslocamento constante, no referencial da
cabeça, a partir da âncora do olho (ponto médio dos cantos, gaze.py):

    centro = âncora + R · (c_x, c_y, s)

com (c_x, c_y) a posição da pupila no plano do olho quando o olho aponta
para a frente da cabeça e s o raio (distância centro -> pupila, ~1,2 cm
num olho real).

FEIXE. A reta que sai do centro do globo e passa pela pupila. No
referencial da cabeça, com p = Rᵀ(pupila − âncora):

    direção_cabeça ∝ ( (p_x − c_x)/s_x , (p_y − c_y)/s_y , −1 )
    direção_câmera = R · direção_cabeça

O raio tem um valor por eixo (s_x, s_y): o MediaPipe "encolhe" o movimento
da íris (nas sessões gravadas, 18-47% do real), mais na vertical que na
horizontal. Com raio único, o feixe não acertaria os alvos nos dois eixos
ao mesmo tempo. O raio efetivo de cada eixo corrige essa compressão na
própria medida da pupila, ANTES de somar a rotação da cabeça, então girar a
cabeça continua movendo o olhar 1:1.

INTERSECÇÃO. Os dois feixes (olho direito e esquerdo) apontam para o mesmo
ponto fixado. Duas retas no espaço quase nunca se cruzam exatamente; a
intersecção é o ponto médio do segmento perpendicular comum
(geometry.closest_point_between_rays). O VETOR DO OLHAR vai do ponto médio
entre os centros dos globos ("olho ciclópico") até a intersecção. Quando
os feixes não convergem à frente do rosto (ruído, retas quase paralelas
ou divergentes), usa-se a bissetriz das duas direções; com um olho
fechado, o feixe do outro.

FORMA DO MODELO (``EYE_MODEL``). Num globo esférico ideal a pupila anda
s·sen(ângulo) (``"esfera"``: direção = ((p−c)/s, −√(1−|·|²))). Nos dados
reais o MediaPipe regulariza a íris em direção ao centro do olho e o raio
efetivo cai para ~0,3 cm; com esse raio, o ruído da pupila satura o seno e
a forma esférica errou mais nas 25 sessões gravadas (LOO 151 contra 145 px;
cabeça em poses novas 363 contra 342 px). O padrão é a forma tangente
(``"tangente"``), com a direção ∝ ((p−c)/s, −1) como acima.

CALIBRAÇÃO (``EyeballModel.fit``). Olhando para um alvo conhecido T, a
direção do centro até T é conhecida, e a pupila medida deve satisfazer, em
cada eixo,

    p_j = c_j + s_j · tan_j(T)          (tan no referencial da cabeça)

Para cada olho e cada eixo, isso é uma regressão linear com 2 incógnitas
(c_j, s_j): mínimos quadrados ponderados, com uma regularização fraca de
s_j em direção ao raio anatômico. A etapa da cabeça da calibração (olhar
fixo, cabeça girando) entra no mesmo ajuste: é ela que separa o movimento
do olho do movimento da cabeça. Como o centro depende de (c, s) e a direção
até o alvo depende do centro, o ajuste é repetido duas vezes (a mudança é
de milímetros a ~60 cm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import closest_point_between_rays
from .gaze import raw_parts

RADIUS_CM = 1.2                # raio anatômico típico do globo ocular
S_MIN, S_MAX = 0.05, 5.0       # limites do raio efetivo ajustado (cm)
PRIOR_WEIGHT = 2e-3            # regularização de s (unidades de tan²)
# Intersecção aceita entre estas distâncias ao longo de cada feixe (cm).
VERGENCE_MIN_CM, VERGENCE_MAX_CM = 5.0, 400.0
EYE_NAMES = ("direito", "esquerdo")
EYE_MODEL = "tangente"         # "tangente" (pupila = c + s·tan) ou "esfera" (c + s·sen); ver docstring
MAX_SIN = 0.95                 # |sen| máximo do olho na cabeça (ruído com raio pequeno)


@dataclass
class GazeRay:
    """Feixes e vetor do olhar de UMA amostra (cm, referencial da câmera)."""

    centers: np.ndarray       # (2, 3) centro do globo de cada olho
    pupils: np.ndarray        # (2, 3) pupila de cada olho
    directions: np.ndarray    # (2, 3) direção unitária de cada feixe
    open: np.ndarray          # (2,) olho aberto
    origin: np.ndarray        # (3,) ponto médio entre os centros (ou o olho aberto)
    direction: np.ndarray     # (3,) vetor do olhar (unitário)
    vergence: np.ndarray | None   # (3,) intersecção dos feixes, se convergem
    vergence_gap_cm: float | None  # distância entre os feixes na intersecção


def _rt(R, V):
    """Rᵀ·v para cada amostra: R (n, 3, 3), V (n, k, 3)."""
    return np.einsum("nji,nkj->nki", R, V)


def _r(R, V):
    return np.einsum("nij,nkj->nki", R, V)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class EyeballModel:
    """Parâmetros do globo de cada olho: c (2 olhos, (x, y)) e s (2 olhos, (x, y))."""

    def __init__(self, c=None, s=None, source: str = "mediapipe"):
        self.c = np.zeros((2, 2)) if c is None else np.asarray(c, dtype=np.float64)
        self.s = np.full((2, 2), RADIUS_CM) if s is None else np.asarray(s, dtype=np.float64)
        self.source = source
        self.info: dict = {}

    # ------------------------------------------------------------ feixes
    def center_offsets(self) -> np.ndarray:
        """(2, 3) deslocamento do centro de cada globo a partir da âncora,
        no referencial da cabeça (+z = para trás)."""
        return np.column_stack([self.c, self.s.mean(axis=1)])

    def beams(self, raw):
        """Feixe de cada olho (vetorizado): centros (n, 2, 3), pupilas (n, 2, 3),
        direções unitárias (n, 2, 3) e olhos abertos (n, 2)."""
        R, A, P, op = raw_parts(raw, self.source)
        p = _rt(R, P - A)
        q = (p[..., :2] - self.c) / self.s
        if EYE_MODEL == "esfera":
            n = np.linalg.norm(q, axis=-1, keepdims=True)
            q = q * np.minimum(1.0, MAX_SIN / np.maximum(n, 1e-12))
            dz = -np.sqrt(1.0 - np.sum(q * q, axis=-1, keepdims=True))
        else:
            dz = -np.ones(q.shape[:-1] + (1,))
        d_head = _unit(np.concatenate([q, dz], axis=-1))
        centers = A + _r(R, np.broadcast_to(self.center_offsets(), A.shape))
        return centers, P, _r(R, d_head), op

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
        origin, direction, X, gap, (centers, P, d, op) = self.gaze(np.atleast_2d(raw))
        if not op[0].any():
            return None
        conv = bool(np.all(np.isfinite(X[0])))
        return GazeRay(centers[0], P[0], d[0], op[0], origin[0], direction[0],
                       X[0] if conv else None, float(gap[0]) if conv else None)

    # ------------------------------------------------------------ ajuste
    @classmethod
    def fit(cls, raw, targets_cm, weights=None, source: str = "mediapipe",
            prior_weight: float = PRIOR_WEIGHT, iterations: int = 2) -> "EyeballModel":
        """Ajusta (c, s) de cada olho e eixo a partir de amostras com alvo
        conhecido. raw: (n, RAW_SIZE); targets_cm: (n, 3) alvo no referencial
        da câmera; weights: (n,)."""
        raw = np.atleast_2d(raw)
        T = np.atleast_2d(np.asarray(targets_cm, dtype=np.float64))
        w = np.ones(len(raw)) if weights is None else np.asarray(weights, dtype=np.float64)
        m = cls(source=source)
        R, A, P, op = raw_parts(raw, source)
        p = _rt(R, P - A)
        resid_deg = []
        for _ in range(iterations):
            centers = A + _r(R, np.broadcast_to(m.center_offsets(), A.shape))
            tau = _rt(R, T[:, None, :] - centers)
            if EYE_MODEL == "esfera":
                tan_t = _unit(tau)[..., :2]
            else:
                tan_t = tau[..., :2] / np.maximum(-tau[..., 2:3], 1e-6)
            for k in range(2):
                ok = op[:, k] & (w > 0)
                if ok.sum() < 3:
                    continue
                wk = w[ok]
                W = wk.sum()
                for j in range(2):
                    x, y = tan_t[ok, k, j], p[ok, k, j]
                    # [c, s] = argmin Σ w (y − c − s·x)² + λ·W·(s − s0)²
                    Sx, Sxx = wk @ x, wk @ (x * x)
                    lam = prior_weight * W
                    M = np.array([[W, Sx], [Sx, Sxx + lam]])
                    b = np.array([wk @ y, wk @ (x * y) + lam * RADIUS_CM])
                    c, s = np.linalg.solve(M, b)
                    if not S_MIN <= s <= S_MAX:
                        s = float(np.clip(s, S_MIN, S_MAX))
                        c = float(wk @ (y - s * x) / W)
                    m.c[k, j], m.s[k, j] = c, s
        # Erro angular médio (graus) de cada olho nas amostras de ajuste.
        centers, _, d, _ = m.beams(raw)
        want = _unit(T[:, None, :] - centers)
        ang = np.degrees(np.arccos(np.clip(np.sum(d * want, -1), -1, 1)))
        for k in range(2):
            ok = op[:, k] & (w > 0)
            resid_deg.append(float(np.average(ang[ok, k], weights=w[ok])) if ok.any() else None)
        m.info = {"erro_angular_ajuste_graus": resid_deg}
        return m

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {"c": self.c.tolist(), "s": self.s.tolist(), "source": self.source,
                "info": self.info}

    @classmethod
    def from_dict(cls, d: dict) -> "EyeballModel":
        m = cls(np.array(d["c"]), np.array(d["s"]), d.get("source", "mediapipe"))
        m.info = d.get("info", {})
        return m
