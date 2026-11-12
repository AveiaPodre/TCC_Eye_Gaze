import cv2
import numpy as np

from eyegaze import scene3d


def test_screen_orientation():
    """Imagem não espelhada: a borda esquerda da tela (para o usuário) fica em +x."""
    left = scene3d.screen_px_to_cm((0, 0), (1920, 1080), 34.5)
    right = scene3d.screen_px_to_cm((1920, 1080), (1920, 1080), 34.5)
    assert left[0] > 0 > right[0]
    assert np.isclose(left[0] - right[0], 34.5)
    assert right[1] > left[1] > 0          # tela abaixo da webcam, y para baixo
    assert left[2] == right[2] == 0.0


def test_hit_screen():
    o = np.array([0.0, 10.0, 60.0])
    hit = scene3d.hit_screen(o, np.array([0.1, 0.0, -1.0]))
    assert np.allclose(hit, [6.0, 10.0, 0.0])
    assert scene3d.hit_screen(o, np.array([0.0, 0.0, 1.0])) is None   # olhando para trás


def test_orbit_controls_and_render_without_face():
    v = scene3d.OrbitView()
    v.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0)
    v.on_mouse(cv2.EVENT_MOUSEMOVE, 150, 120, 0)
    v.on_mouse(cv2.EVENT_LBUTTONUP, 150, 120, 0)
    assert v.yaw_deg != scene3d.OrbitView.DEFAULTS[0]
    assert v.pitch_deg != scene3d.OrbitView.DEFAULTS[1]
    v.on_mouse(cv2.EVENT_RBUTTONDOWN, 0, 0, 0)
    assert (v.yaw_deg, v.pitch_deg, v.radius_cm) == scene3d.OrbitView.DEFAULTS
    img = scene3d.render_scene(v, (1920, 1080), 34.5, None, None, size=(400, 300))
    assert img.shape == (300, 400, 3)
