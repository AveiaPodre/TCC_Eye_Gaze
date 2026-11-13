import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _synthetic_camera():
    """O pipeline supõe a mesma câmera do rosto sintético (60°)."""
    from eyegaze import geometry
    from synthetic import SYNTH_HFOV_DEG
    old = geometry.ASSUMED_HFOV_DEG
    geometry.set_camera_hfov(SYNTH_HFOV_DEG)
    yield
    geometry.set_camera_hfov(old)
