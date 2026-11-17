"""
Conferência depois da calibração (desvio encolhido pela incerteza),
estabilizador mais responsivo e gravação com tempo por quadro.
"""

import json

import numpy as np

from eyegaze.config import StabilizerConfig
from eyegaze.filters import make_stabilizer
from eyegaze.io_devices import SessionRecorder
from eyegaze.procedures import RecenterProcedure, ValidationProcedure, shrink_offset

from test_app import make_app, run_procedure, settle


# ------------------------------------------------------------- rápidos
def test_shrink_offset_keeps_coherent_and_shrinks_noisy_offsets():
    coherent = np.array([[150, -100], [160, -95], [145, -110], [155, -98]], float)
    off, f = shrink_offset(coherent)
    assert f > 0.95 and np.allclose(off, coherent.mean(0) * f)
    noisy = np.array([[60, 10], [-50, -40], [70, 30], [-40, 20]], float)   # média ~ ruído
    off, f = shrink_offset(noisy)
    assert f < 0.2 and np.hypot(*off) < 5
    one = np.array([[30.0, 0.0]])
    assert np.allclose(shrink_offset(one)[0], one[0])


def test_stabilizer_follows_subtle_movement_and_stays_still():
    """Movimento sutil de 50 px com ruído real (~8 px por eixo, por quadro): o cursor
    chega perto em até 0,5 s e, com o olho parado, quase não se move."""
    rng = np.random.default_rng(1)
    stab = make_stabilizer(StabilizerConfig(), 1920)
    p0, p1 = np.array([900.0, 500.0]), np.array([950.0, 500.0])
    for _ in range(40):
        stab(p0 + rng.normal(0, 8, 2))
    path = [stab(p1 + rng.normal(0, 8, 2)) for _ in range(45)]
    # Cobre ao menos metade do movimento em 0,5 s (antes, start_k 4: ficava parado).
    assert min(np.linalg.norm(np.array(path[:15]) - p1, axis=1)) < 30
    tail = np.array(path[20:])
    moves = np.linalg.norm(np.diff(tail, axis=0), axis=1)
    assert np.mean(moves < 1.0) > 0.6


def test_recorder_writes_frames_in_background_and_logs_processing(tmp_path):
    rec = SessionRecorder(tmp_path / "s", 30, (64, 48), (1920, 1080), seed=1)
    for i in range(10):
        rec.write(np.full((48, 64, 3), i, np.uint8), i * 33)
        rec.log_frame(12.345, 0b101)
    rec.close()
    meta = json.loads((tmp_path / "s.json").read_text())
    assert len(meta["timestamps_ms"]) == 10 and meta["proc_ms"][0] == 12.3
    assert meta["estado"][0] == 5
    import cv2
    cap = cv2.VideoCapture(str(tmp_path / "s.avi"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10


# --------------------------------------------------------------- lentos
def test_calibration_is_followed_by_check_that_corrects_offset(tmp_path):
    app, tracker, _ = make_app(tmp_path)
    app.cfg.calibration.check_after_calibration = True
    app.handle_key(ord(" "))
    run_procedure(app)
    app.handle_key(13)
    proc = app.procedure
    assert isinstance(proc, RecenterProcedure) and proc.kind == "pos_calibracao"
    # Desvio que aparece depois de calibrar (postura mudou um pouco).
    app.model.offset = app.model.offset + np.array([0.0, 120.0])
    proc = run_procedure(app)
    assert proc.succeeded and proc.applied[1] < -90 and proc.factor > 0.8
    app.handle_key(13)
    assert app.procedure is None
    assert "recentralizacao_pos_calibracao_px" in app.model.info


def test_check_then_immediate_validation_in_research_mode(tmp_path):
    app, _, _ = make_app(tmp_path)
    app.cfg.calibration.check_after_calibration = True
    app.cfg.validation.after_calibration = True
    app.handle_key(ord(" "))
    run_procedure(app)
    app.handle_key(13)
    assert isinstance(app.procedure, RecenterProcedure)
    run_procedure(app)
    app.handle_key(13)
    assert isinstance(app.procedure, ValidationProcedure) and app.procedure.kind == "imediata"
    val = run_procedure(app)
    assert val.summary["acuracia_media_px"] < 80


def test_camera_preview_is_lighter_while_mouse_is_in_use(tmp_path):
    app, _, mouse = make_app(tmp_path)
    app.handle_key(ord(" "))
    run_procedure(app)
    app.handle_key(13)
    settle(app, 3)
    calls = []
    orig = app.render_camera_view
    app.render_camera_view = lambda f, n: calls.append(1) or orig(f, n)
    for _ in range(9):
        app.step()
    assert len(calls) == 9                      # mouse desligado: todo quadro
    app.handle_key(ord("m"))
    assert mouse.enabled
    calls.clear()
    for _ in range(9):
        app.step()
    assert len(calls) == 3                      # 1 a cada 3 quadros
    assert app.proc_ms > 0
