"""Gravação de sessão (vídeo + teclas) e reprodução determinística."""

import numpy as np

from eyegaze.app import EyeGazeApp
from eyegaze.config import AppConfig
from eyegaze.io_devices import NullMouse, ReplaySource, SessionRecorder

from test_app import Clock, FakeSource, FakeTracker, run_procedure, settle

SCREEN = (1920, 1080)


class SmallSource(FakeSource):
    """Quadros pequenos (o FakeTracker não usa a imagem): gravação rápida."""

    def __init__(self, clock):
        super().__init__(clock)
        self.frame = np.full((120, 160, 3), 70, np.uint8)


def _cfg(tmp_path, sub):
    cfg = AppConfig()
    cfg.calibration.save_path = tmp_path / sub / "cal.json"
    cfg.calibration.results_dir = tmp_path / sub
    cfg.validation.results_dir = tmp_path / sub
    from synthetic import SYNTH_HFOV_DEG
    cfg.camera.hfov_deg = SYNTH_HFOV_DEG
    return cfg


def test_record_then_replay_reproduces_session(tmp_path):
    # --- sessão "ao vivo" gravada -------------------------------------
    cfg = _cfg(tmp_path, "ao_vivo")
    cfg.calibration.random_seed = 42
    clock = Clock()
    rec = SessionRecorder(tmp_path / "sessao_teste", 30, (160, 120), SCREEN, seed=42)
    tracker = FakeTracker()
    app = EyeGazeApp(cfg, tracker, SmallSource(clock), SCREEN, mouse=NullMouse(),
                     headless=True, clock=clock, recorder=rec)
    tracker.app = app
    settle(app, 10)
    app.press(ord(" "))
    proc = run_procedure(app)
    order_live = list(proc.order)
    settle(app, 5)
    app.press(13)                      # fecha a tela de resultado
    settle(app, 10)
    app.press(ord("q"))
    rec.close()
    n_frames = len(rec.meta["timestamps_ms"])
    assert rec.video_path.exists() and rec.meta_path.exists()
    assert [k for _, k in rec.meta["teclas"]] == [ord(" "), 13, ord("q")]

    # --- reprodução ---------------------------------------------------
    src = ReplaySource(rec.meta_path)
    assert src.screen_size == SCREEN and src.seed == 42
    cfg2 = _cfg(tmp_path, "reproducao")
    cfg2.calibration.random_seed = src.seed
    tracker2 = FakeTracker()
    app2 = EyeGazeApp(cfg2, tracker2, src, src.screen_size, mouse=NullMouse(),
                      headless=True, clock=src.now)
    tracker2.app = app2
    seen_order = None
    while app2.step():
        if app2.procedure is not None and seen_order is None:
            seen_order = list(app2.procedure.order)
    assert seen_order == order_live              # mesma ordem de alvos
    assert src.frame_index <= n_frames - 1
    assert app2.calibrated                       # a calibração foi refeita
    assert cfg2.calibration.save_path.exists()
    assert list((tmp_path / "reproducao").glob("calibracao_*.npz"))
