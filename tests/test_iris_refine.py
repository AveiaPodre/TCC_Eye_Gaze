import cv2
import numpy as np

from eyegaze.iris_refine import refine_eye, refine_irises
from eyegaze.landmarks import LEFT_EYE, RIGHT_EYE

SS = 8  # super-amostragem para desenhar a íris em posição sub-pixel


def eye_image(iris_xy, center=(100.0, 60.0), half=(24.0, 10.0), iris_r=11.0):
    """Olho sintético: pele, esclera (elipse clara) e íris escura."""
    h, w = 120, 200
    big = np.full((h * SS, w * SS), 120, np.uint8)
    c = (int(center[0] * SS), int(center[1] * SS))
    cv2.ellipse(big, c, (int(half[0] * SS), int(half[1] * SS)), 0, 0, 360, 225, -1)
    eye_mask = big == 225
    iris = np.zeros_like(big)
    cv2.circle(iris, (int(round(iris_xy[0] * SS)), int(round(iris_xy[1] * SS))),
               int(iris_r * SS), 1, -1)
    big[(iris == 1) & eye_mask] = 55
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def eye_points(center=(100.0, 60.0), half=(24.0, 10.0)):
    """Landmarks: contorno do olho direito sobre a elipse da esclera."""
    pts = np.zeros((478, 3))
    contour = RIGHT_EYE.contour
    # Ordem do contorno: canto externo -> pálpebra inferior -> canto interno -> superior.
    ang = np.linspace(np.pi, -np.pi, len(contour), endpoint=False)
    for k, a in zip(contour, ang):
        pts[k, :2] = [center[0] + half[0] * np.cos(a), center[1] - half[1] * np.sin(a)]
    return pts


def test_refined_center_is_subpixel_accurate_horizontally():
    pts = eye_points()
    for x in (92.3, 100.0, 107.6):
        c = refine_eye(eye_image((x, 60.0)), pts, RIGHT_EYE)
        assert c is not None
        # Perto do canto a pálpebra corta a íris e o centróide da parte
        # visível puxa um pouco para o centro (monotônico; a calibração absorve).
        assert abs(c[0] - x) < 1.0, (x, c)


def test_refined_center_follows_small_iris_motion():
    """O problema real: a íris do MediaPipe anda ~4 px para a tela toda.
    O refinamento tem de responder a deslocamentos de fração de pixel."""
    pts = eye_points()
    xs = np.linspace(97.0, 103.0, 7)
    cs = np.array([refine_eye(eye_image((x, 60.0)), pts, RIGHT_EYE)[0] for x in xs])
    slope = np.polyfit(xs, cs, 1)[0]
    assert 0.8 < slope < 1.2, slope
    assert np.all(np.diff(cs) > 0)


def test_refine_irises_returns_nan_when_eye_missing():
    pts = np.zeros((478, 3))   # contorno degenerado
    out = refine_irises(np.zeros((50, 50, 3), np.uint8), pts)
    assert out.shape == (2, 2) and np.isnan(out).all()
    assert LEFT_EYE.contour and RIGHT_EYE.contour
