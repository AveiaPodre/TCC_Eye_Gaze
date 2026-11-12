"""
Geometria 3D compartilhada: câmera, monitor virtual e operações com retas.

Referencial da CÂMERA (usado em todo o pipeline), em centímetros:
origem no centro óptico da webcam, x para a direita da imagem, y para
baixo, z para longe da câmera (z > 0 = à frente dela).

Câmera pinhole com campo de visão horizontal suposto (``ASSUMED_HFOV_DEG``):
webcams comuns ficam entre 55° e 70°, e o erro dessa suposição é absorvido
pela calibração.

Monitor (real e virtual): plano z = 0, com a webcam centralizada logo acima
da borda superior (notebook ou webcam presa no monitor). Como a imagem NÃO
é espelhada, a esquerda da tela (do ponto de vista do usuário) fica em +x.

* MONITOR VIRTUAL: o retângulo nesse plano com a largura física suposta ou
  conhecida da tela. O vetor do olhar é projetado nele (etapa 4) e o ponto
  atingido vira um pixel "virtual" pela geometria pura.
* MONITOR REAL: os pixels onde estão de fato os alvos. A calibração
  (etapa 5) aprende a correção entre os dois.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ASSUMED_HFOV_DEG = 60.0
# Distância da webcam à borda superior da tela (cm).
SCREEN_GAP_CM = 1.0


# ------------------------------------------------------------------ câmera
def focal_px(image_width: int) -> float:
    return 0.5 * image_width / np.tan(np.radians(ASSUMED_HFOV_DEG / 2))


def px3d_to_cm(p, image_size: tuple[int, int], depth_cm: float) -> np.ndarray:
    """Ponto em "pixels 3D" do MediaPipe (x, y em px; z relativo ao rosto, na
    escala de x) -> cm no referencial da câmera, com o rosto a ``depth_cm``."""
    w, h = image_size
    f = focal_px(w)
    p = np.asarray(p, dtype=np.float64)
    z = depth_cm + p[..., 2] * depth_cm / f
    return np.stack([(p[..., 0] - w / 2) * z / f, (p[..., 1] - h / 2) * z / f, z], axis=-1)


def cm_to_px(P, image_size: tuple[int, int]) -> np.ndarray:
    """cm (referencial da câmera) -> pixels da imagem (x, y)."""
    w, h = image_size
    f = focal_px(w)
    P = np.asarray(P, dtype=np.float64)
    return np.stack([w / 2 + f * P[..., 0] / P[..., 2], h / 2 + f * P[..., 1] / P[..., 2]],
                    axis=-1)


def pixel_ray(xy, image_size: tuple[int, int]) -> np.ndarray:
    """Direção (não normalizada, z = 1) da linha de visada da câmera por um pixel."""
    w, h = image_size
    f = focal_px(w)
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([(xy[..., 0] - w / 2) / f, (xy[..., 1] - h / 2) / f,
                     np.ones(xy.shape[:-1])], axis=-1)


# ------------------------------------------------------------------- retas
def ray_plane(origin, direction, point, normal):
    """Interseção de retas com planos (vetorizado no primeiro eixo).

    Devolve (pontos, t) com ponto = origem + t·direção; NaN onde a reta é
    paralela ao plano."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    den = np.sum(d * normal, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(den) > 1e-12,
                     np.sum((np.asarray(point) - o) * normal, axis=-1) / den, np.nan)
    return o + t[..., None] * d, t


def closest_point_between_rays(o1, d1, o2, d2):
    """Ponto mais próximo entre duas retas (o + s·d), vetorizado.

    Duas retas no espaço quase nunca se cruzam exatamente; a "interseção" é o
    ponto médio do segmento perpendicular comum. Devolve (ponto, s, t,
    distância entre as retas); s e t são NaN se as retas são paralelas.
    """
    o1, d1, o2, d2 = (np.asarray(v, dtype=np.float64) for v in (o1, d1, o2, d2))
    w0 = o1 - o2
    a = np.sum(d1 * d1, -1)
    b = np.sum(d1 * d2, -1)
    c = np.sum(d2 * d2, -1)
    d = np.sum(d1 * w0, -1)
    e = np.sum(d2 * w0, -1)
    den = a * c - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = den > 1e-12 * a * c
        s = np.where(ok, (b * e - c * d) / den, np.nan)
        t = np.where(ok, (a * e - b * d) / den, np.nan)
    p1 = o1 + s[..., None] * d1
    p2 = o2 + t[..., None] * d2
    return 0.5 * (p1 + p2), s, t, np.linalg.norm(p1 - p2, axis=-1)


# ---------------------------------------------------------- monitor virtual
@dataclass
class VirtualScreen:
    """Monitor no plano z = 0 da câmera, com a webcam centralizada acima dele."""

    size_px: tuple[int, int]
    width_cm: float
    gap_cm: float = SCREEN_GAP_CM

    @property
    def px_per_cm(self) -> float:
        return self.size_px[0] / self.width_cm

    @property
    def height_cm(self) -> float:
        return self.size_px[1] / self.px_per_cm

    def px_to_cm(self, Y) -> np.ndarray:
        """Pixels da tela (n, 2) -> pontos 3D (n, 3) no plano z = 0."""
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        return np.column_stack([(0.5 * self.size_px[0] - Y[:, 0]) / self.px_per_cm,
                                self.gap_cm + Y[:, 1] / self.px_per_cm,
                                np.zeros(len(Y))])

    def cm_to_px(self, X) -> np.ndarray:
        """Pontos no plano (n, 2 ou 3) -> pixels (n, 2)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.column_stack([0.5 * self.size_px[0] - X[:, 0] * self.px_per_cm,
                                (X[:, 1] - self.gap_cm) * self.px_per_cm])

    def hit(self, origin, direction) -> np.ndarray:
        """Ponto (n, 3) onde as retas do olhar atingem o plano do monitor.

        NaN quando o olhar aponta para longe da tela (z não diminui)."""
        o = np.atleast_2d(np.asarray(origin, dtype=np.float64))
        d = np.atleast_2d(np.asarray(direction, dtype=np.float64))
        P, t = ray_plane(o, d, np.zeros(3), np.array([0.0, 0.0, 1.0]))
        P[~(t > 0)] = np.nan
        return P

    def project(self, origin, direction) -> np.ndarray:
        """Etapa 4: pixel (n, 2) do monitor virtual atingido pelo olhar."""
        return self.cm_to_px(self.hit(origin, direction))
