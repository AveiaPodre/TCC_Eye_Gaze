import cv2
import numpy as np

from eyegaze import scene3d
from eyegaze.config import GazeConfig
from eyegaze.gaze import GazeEstimator
from eyegaze.geometry import VirtualScreen

from synthetic import ScreenGeometry, make_observation

SCREEN = VirtualScreen((1920, 1080), 34.5)


def test_orbit_controls_and_render_without_face():
    v = scene3d.OrbitView()
    v.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0)
    v.on_mouse(cv2.EVENT_MOUSEMOVE, 150, 120, 0)
    v.on_mouse(cv2.EVENT_LBUTTONUP, 150, 120, 0)
    assert v.yaw_deg != scene3d.OrbitView.DEFAULTS[0]
    assert v.pitch_deg != scene3d.OrbitView.DEFAULTS[1]
    v.on_mouse(cv2.EVENT_RBUTTONDOWN, 0, 0, 0)
    assert (v.yaw_deg, v.pitch_deg, v.radius_cm) == scene3d.OrbitView.DEFAULTS
    img = scene3d.render_scene(v, SCREEN, None, None, size=(400, 300))
    assert img.shape == (300, 400, 3)


def test_render_with_beams_and_intersection():
    """Com um rosto: globos, feixes, intersecção, vetor até o monitor virtual
    e cursor são desenhados sem erro (inclusive com um olho fechado)."""
    geom = ScreenGeometry()
    pos = (0.0, 10.0, 60.0)
    obs = make_observation(head_pos_cm=pos, with_transform=True,
                           gaze_vec=geom.gaze_vector(700, 400, pos))
    s = GazeEstimator(GazeConfig()).estimate(obs)
    assert s.gaze is not None
    img = scene3d.render_scene(scene3d.OrbitView(), SCREEN, s, obs, virtual_point=(710, 390),
                               cursor=(700, 400), raw_point=(705, 395),
                               trail=[(690, 400), (700, 400)], targets=[(100, 100)])
    blank = scene3d.render_scene(scene3d.OrbitView(), SCREEN, None, None)
    assert img.shape == blank.shape and np.any(img != blank)
