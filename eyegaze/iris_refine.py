"""
Refinamento sub-pixel do centro da íris direto na imagem.

Por que: nos dados reais, a íris do MediaPipe anda só ~4 pixels da imagem
quando o olhar vai de um lado ao outro da tela (olho com ~48 px de
largura), cerca de 3-4x menos do que a geometria do olho prevê. O modelo
regulariza a íris em direção ao centro do olho. Com isso, 1 px da imagem
vira ~400 px na tela e qualquer ruído do landmark domina o erro.

Como: dentro da abertura do olho (polígono formado pelo contorno das
pálpebras do MediaPipe, que é bem localizado), a íris é a região escura
e a esclera a clara. O centro é o centróide dos pixels escuros, ponderado
pelo quanto cada um é mais escuro que um limiar adaptativo (percentil da
própria região). O centróide é naturalmente sub-pixel e acompanha o olho
sem regularização.

O resultado é usado como uma fonte ALTERNATIVA da posição da íris: a
calibração ajusta o modelo com as duas fontes e fica com a de menor erro
em alvos não vistos (ver calibration.fit_from_samples).
"""

from __future__ import annotations

import cv2
import numpy as np

from .landmarks import LEFT_EYE, RIGHT_EYE, EyeIndices

UPSAMPLE = 4            # ampliação da região do olho (melhora o polígono e o centróide)
DARK_PERCENTILE = 35.0  # pixels abaixo deste percentil da região contam como íris
MIN_DARK_FRACTION = 0.05


def refine_eye(gray: np.ndarray, pts: np.ndarray, idx: EyeIndices) -> np.ndarray | None:
    """Centro (x, y) da íris em pixels da imagem, ou None se não der."""
    contour = pts[list(idx.contour), :2]
    x0, y0 = np.floor(contour.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(contour.max(axis=0)).astype(int) + 3
    h, w = gray.shape
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 - x0 < 6 or y1 - y0 < 4:
        return None

    roi = gray[y0:y1, x0:x1].astype(np.float32)
    roi = cv2.resize(roi, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    roi = cv2.GaussianBlur(roi, (0, 0), UPSAMPLE * 0.6)

    # Pixel (i, j) ampliado cobre o centro em ((j + 0.5)/U - 0.5 + x0, ...).
    poly = (contour - [x0, y0] + 0.5) * UPSAMPLE - 0.5
    mask = np.zeros(roi.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(poly * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    inside = mask.astype(bool)
    if inside.sum() < 20:
        return None

    vals = roi[inside]
    thr = np.percentile(vals, DARK_PERCENTILE)
    wgt = np.where(inside, np.clip(thr - roi, 0, None), 0.0)
    if (wgt > 0).sum() < MIN_DARK_FRACTION * inside.sum() or wgt.sum() <= 0:
        return None

    ys, xs = np.indices(roi.shape)
    cx = (xs * wgt).sum() / wgt.sum()
    cy = (ys * wgt).sum() / wgt.sum()
    return np.array([(cx + 0.5) / UPSAMPLE - 0.5 + x0, (cy + 0.5) / UPSAMPLE - 0.5 + y0])


def refine_irises(frame_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(2, 2): centro refinado do olho direito e do esquerdo (NaN se falhar)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    out = np.full((2, 2), np.nan)
    for k, idx in enumerate((RIGHT_EYE, LEFT_EYE)):
        c = refine_eye(gray, pts, idx)
        if c is not None:
            out[k] = c
    return out
