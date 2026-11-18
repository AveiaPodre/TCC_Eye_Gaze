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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: roda o app inteiro (calibração simulada); pule com -m 'not slow'")


def pytest_collection_modifyitems(config, items):
    """Marca como ``slow`` os testes que rodam o app inteiro (cada um simula
    uma calibração completa, 5-25 s). A suíte rápida (``-m 'not slow'``)
    cobre a geometria, o modelo e os procedimentos em poucos segundos."""
    import inspect
    for item in items:
        fn = getattr(item, "function", None)
        try:
            src = inspect.getsource(fn) if fn else ""
        except (OSError, TypeError):
            src = ""
        if ("make_app(" in src or "_calibrated_app(" in src
                or item.module.__name__ == "test_recording"):
            item.add_marker(pytest.mark.slow)
