"""
Calibração por perseguição, métricas por olho e revalidação periódica.
"""

import numpy as np

from eyegaze.calibration import CalibrationData
from eyegaze.config import CalibrationConfig
from eyegaze.procedures import (CalibrationProcedure, ValidationProcedure, _along,
                                pursuit_path)

from test_app import make_app, run_procedure, settle
from test_procedures import StubSample

SCREEN = (1920, 1080)


# --------------------------------------------------------- caminho (rápidos)
def test_pursuit_path_covers_borders_and_middle():
    soma = pursuit_path(SCREEN, "soma", 0.05)
    alt = pursuit_path(SCREEN, "alternativa", 0.05)
    assert len(soma) == 5 and np.allclose(soma[0], soma[-1])       # perímetro fechado
    assert soma[:, 0].min() == 96 and soma[:, 0].max() == 1824
    assert soma[:, 1].min() == 54 and soma[:, 1].max() == 1026
    assert len(alt) == 9
    assert any(np.allclose(p, (960, 54)) for p in alt)              # linha vertical do meio
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(alt, axis=0).T))])
    p, seg = _along(alt, cum, cum[1] / 2)
    assert seg == 0 and np.allclose(p, (960, 54))
    p, seg = _along(alt, cum, 1e9)
    assert np.allclose(p, alt[-1])


def _cfg(mode, **kw):
    cfg = CalibrationConfig(pursuit=mode, random_seed=0, **kw)
    cfg.settle_s = cfg.settle_max_s = 0.0
    return cfg


def test_pursuit_runs_after_grid_in_soma_mode():
    proc = CalibrationProcedure(SCREEN, _cfg("soma", refine_points=()))
    assert proc.pursuit_at == proc.n_grid == 12 and not proc.pursuing
    t, s = 0.0, StubSample(0.0, 0.0)
    while proc.pos < proc.n_grid:
        t += 1 / 30
        proc.update(s, t)
    assert proc.pursuing and not proc.computing          # a grade acabou, falta a perseguição
    assert proc.status_line().startswith("Calibracao: siga o ponto")
    start = t
    while proc.pursuing:
        t += 1 / 30
        proc.update(s, t)
    assert proc.computing                                 # sem etapa 2: ajusta
    assert abs((t - start) - proc.pursuit_duration_s) < 0.1
    # Amostras do começo e logo depois dos cantos foram descartadas.
    n_moving = (proc.pursuit_duration_s - proc.cfg.pursuit_hold_s) * 30
    assert 0.6 * n_moving < len(proc.p_raw) < 0.95 * n_moving


def test_pursuit_first_in_alternativa_mode_with_5_points():
    proc = CalibrationProcedure(SCREEN, _cfg("alternativa"))
    assert len(proc.targets) == 5 and proc.pursuing
    assert np.allclose(proc.current_target(), proc.pursuit_path[0])
    t = 0.0
    proc.update(StubSample(0, 0), t)
    t = proc.cfg.pursuit_hold_s + 1.0
    proc.update(StubSample(0, 0), t)
    x, y = proc.current_target()
    assert 96 < x < 1824 and y == 54                     # andando pela borda de cima


def test_pursuit_rejects_saccade_jumps():
    proc = CalibrationProcedure(SCREEN, _cfg("alternativa"))
    t = proc.cfg.pursuit_hold_s + 1.0
    proc.update(StubSample(0, 0), 0.0)
    proc.update(StubSample(0, 0), t)
    n = len(proc.p_raw)
    proc.update(StubSample(5, 0), t + 1 / 30)            # salto de 5 graus
    assert len(proc.p_raw) == n
    proc.update(StubSample(5, 0), t + 2 / 30)
    assert len(proc.p_raw) == n + 1


# ----------------------------------------------------- app inteiro (lentos)
def _calibrate(app):
    app.handle_key(ord(" "))
    proc = run_procedure(app, max_steps=6000)
    app.handle_key(13)
    assert proc.succeeded and app.calibrated
    return proc


def test_pursuit_soma_end_to_end(tmp_path):
    app, _, _ = make_app(tmp_path)
    app.cfg.calibration.pursuit = "soma"
    # O usuário simulado segue o ponto com 1 quadro de atraso.
    app.cfg.calibration.pursuit_delay_s = 1 / 30
    proc = _calibrate(app)
    st = proc.stats
    assert len(proc.results) == 16                       # 12 da grade + 4 da etapa 2
    assert st["n_alvos_fixos"] == 16 and st["n_perseguicao"] > 250
    assert st["erro_ajuste_perseguicao_px"] < 60
    assert st["erro_loo_medio_px"] < 60
    data = CalibrationData.load(proc.raw_path)
    assert len(data.pursuit_raw) == st["n_perseguicao"] == len(data.pursuit_obs)
    assert data.pursuit_targets.shape == (st["n_perseguicao"], 2)


def test_pursuit_alternativa_end_to_end(tmp_path):
    app, _, _ = make_app(tmp_path)
    app.cfg.calibration.pursuit = "alternativa"
    app.cfg.calibration.pursuit_delay_s = 1 / 30
    proc = _calibrate(app)
    assert len(proc.results) == 5 and proc.stats["n_perseguicao"] > 400
    app.handle_key(ord("t"))
    val = run_procedure(app)
    assert val.summary["acuracia_media_px"] < 80         # alvos nunca vistos na calibração


def test_validation_reports_per_eye_metrics(tmp_path):
    app, tracker, _ = make_app(tmp_path)
    _calibrate(app)
    app.handle_key(ord("t"))
    val = run_procedure(app)
    eyes = val.summary["por_olho"]
    assert set(eyes) == {"direito", "esquerdo"}
    for e in eyes.values():
        assert e["disponibilidade"] > 0.9 and e["n_alvos"] == 9
        assert e["acuracia_media_px"] < 150 and e["precisao_rms_media_px"] > 0
    assert any(line.startswith("Por olho:") for line in val.message_lines)


def test_periodic_revalidation_starts_and_corrects_drift(tmp_path):
    app, _, _ = make_app(tmp_path)
    app.cfg.validation.revalidate_every_min = 0.05        # 3 s
    _calibrate(app)
    settle(app, 30)
    assert app.procedure is None                          # ainda não deu o tempo
    app.model.offset = app.model.offset + np.array([120.0, 0.0])  # deriva artificial
    for _ in range(120):
        app.step()
        if app.procedure is not None:
            break
    assert isinstance(app.procedure, ValidationProcedure) and app.procedure.kind == "periodica"
    val = run_procedure(app)
    assert val.applied_offset is not None and val.applied_offset[0] < -80
    assert "por_olho" in val.summary
    app.handle_key(13)
    assert app.procedure is None
    settle(app, 30)
    assert app.procedure is None                          # o relógio recomeçou


def test_periodic_revalidation_off_by_default(tmp_path):
    app, _, _ = make_app(tmp_path)
    _calibrate(app)
    app._last_check = app.clock() - 3600.0
    settle(app, 10)
    assert app.procedure is None
