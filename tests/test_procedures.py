"""
Testes do protocolo de calibração (procedures.py) e das partes novas do
modelo: ordem aleatória, coleta por fixação, etapa da cabeça guiada,
repetição de alvos ruins, filtro de faixa da cabeça e aviso de postura.
"""

import numpy as np

from eyegaze.calibration import (ALWAYS_TRAIN, HEAD_GROUP_BASE,
                                 fit_from_samples, split_head_stage)
from eyegaze.config import CalibrationConfig
from eyegaze.procedures import (CalibrationProcedure, TargetProcedure, head_progress,
                                quality_label)

from test_app import FakeTracker, make_app, run_procedure, settle
from test_pipeline import SimUser


class StubSample:
    """GazeSample mínimo para testar a máquina de estados."""

    def __init__(self, yaw, pitch):
        self._f = np.array([yaw, pitch], float)
        self.valid = True
        self.raw = np.zeros(20)
        self.obs = None

    def features(self):
        return self._f


# ------------------------------------------------------------ coleta
def test_targets_visited_in_random_order_and_all_collected(tmp_path):
    app, _, _ = make_app(tmp_path)
    app.cfg.calibration.random_seed = 3
    app.handle_key(ord(" "))
    proc = app.procedure
    assert sorted(proc.order) == list(range(16))
    assert proc.order != list(range(16))          # não é linha a linha
    visited = []
    for _ in range(4000):
        app.step()
        if proc.collecting_done:
            break
        if not proc.in_head_stage and (not visited or visited[-1] != proc.index):
            visited.append(proc.index)
    assert visited == proc.order
    assert all(len(s) >= proc.min_samples for s in proc.samples)


def test_collection_waits_for_fixation():
    proc = TargetProcedure([(100, 100)], (1920, 1080), settle_s=0.5, collect_s=0.5,
                           min_samples=5, settle_max_s=3.0, fixation_window=6,
                           fixation_max_deg=0.8)
    rng = np.random.default_rng(0)
    t = 0.0
    # 1,5 s com o olhar ainda "procurando" (variação de vários graus) ...
    while t < 1.5:
        proc.update(StubSample(*rng.normal(0, 3.0, 2)), t)
        assert not proc.collecting, t
        t += 1 / 30
    # ... depois fixa: a coleta começa logo (janela de 6 quadros).
    started = None
    while t < 2.5:
        proc.update(StubSample(*rng.normal(5, 0.1, 2)), t)
        if proc.collecting and started is None:
            started = t
        t += 1 / 30
    assert started is not None and started < 1.5 + 0.3
    # Só amostras da fixação foram guardadas.
    feats, _ = proc.arrays(0)
    assert np.all(np.abs(feats - 5) < 1.0)


def test_collection_starts_at_timeout_when_gaze_never_settles():
    proc = TargetProcedure([(100, 100)], (1920, 1080), settle_s=0.5, collect_s=0.5,
                           min_samples=5, settle_max_s=1.2)
    rng = np.random.default_rng(1)
    t = 0.0
    while not proc.collecting and t < 3:
        proc.update(StubSample(*rng.normal(0, 3.0, 2)), t)
        t += 1 / 30
    assert 1.2 <= t <= 1.3


# --------------------------------------------------------- etapa da cabeça
def test_head_progress_directions():
    target = (10.0, 8.0)
    assert head_progress(np.array([10.0, 0.0]), (1, 0), target) == 1.0     # direita
    assert head_progress(np.array([-5.0, 0.0]), (-1, 0), target) == 0.5    # esquerda
    assert head_progress(np.array([0.0, 8.0]), (0, -1), target) == 1.0     # cima = pitch +
    assert head_progress(np.array([0.0, -4.0]), (0, 1), target) == 0.5     # baixo
    assert head_progress(np.array([10.0, 0.0]), (-1, 0), target) < 0       # lado errado


class StillHeadTracker(FakeTracker):
    """Pessoa que não consegue mover a cabeça."""

    def _head_for_prompt(self, proc):
        return (0.0, 0.0, 0.0)


def test_head_stage_finishes_for_user_who_cannot_move_head(tmp_path):
    app, _, _ = make_app(tmp_path)
    tracker = StillHeadTracker()
    tracker.app = app
    app.tracker = tracker
    app.handle_key(ord(" "))
    proc = run_procedure(app, max_steps=6000)
    assert proc.collecting_done and proc.succeeded
    cfg = app.cfg.calibration
    expected = cfg.head_motion_settle_s + 4 * cfg.head_step_timeout_s + cfg.head_center_s
    n = len(proc.head_samples)
    assert abs(n / 30 - (expected - cfg.head_motion_settle_s)) < 1.0, n
    # Sem movimento, nenhuma direção vira dobra da validação da cabeça.
    assert "erro_cv_cabeca_px" not in proc.stats


class OverTurnTracker(FakeTracker):
    """Pessoa que gira a cabeça muito além do pedido (como nos dados reais)."""

    def _head_for_prompt(self, proc):
        prompt = getattr(proc, "head_prompt", None)
        if prompt is None:
            self._head_amount = 0.0
            return (0.0, 0.0, 0.0)
        _, (dx, dy) = prompt
        self._head_amount = min(1.0, self._head_amount + 1 / 40)
        return (30.0 * dx * self._head_amount, -25.0 * dy * self._head_amount, 0.0)


def test_head_samples_beyond_range_are_not_used(tmp_path):
    app, _, _ = make_app(tmp_path)
    tracker = OverTurnTracker()
    tracker.app = app
    app.tracker = tracker
    app.handle_key(ord(" "))
    proc = run_procedure(app)
    st = proc.stats
    assert st["n_cabeca_usadas"] < st["n_cabeca_total"]
    # A seta fica verde a 80% do alvo e a etapa avança: a pessoa passa pela
    # faixa útil antes de exagerar, então ainda sobram amostras para usar.
    assert st["n_cabeca_usadas"] >= 20


def test_head_arrow_renders_all_states(tmp_path):
    app, tracker, _ = make_app(tmp_path)
    app.handle_key(ord(" "))
    proc = app.procedure
    seen = set()
    for _ in range(4000):
        app.step()
        if proc.collecting_done:
            break
        if proc.head_prompt is not None:
            seen.add(proc.head_prompt[0])
            assert app.last_target_view.shape == (1080, 1920, 3)
    assert seen == {"DIREITA", "ESQUERDA", "CIMA", "BAIXO", "CENTRO"}


# ---------------------------------------------------- repetição de alvos
class DistractedTracker(FakeTracker):
    """Olha para o lugar errado num alvo na primeira passada."""

    def __init__(self, bad_index):
        super().__init__()
        self.bad_index = bad_index

    def process(self, frame, ts):
        proc = self.app.procedure
        if (isinstance(proc, CalibrationProcedure) and not proc.in_head_stage
                and not proc.collecting_done and not proc.retried_targets
                and proc.index == self.bad_index):
            tx, ty = proc.current_target()
            self.look_at_override = (tx + 450.0, ty + 250.0)
        else:
            self.look_at_override = None
        return super().process(frame, ts)


def test_bad_target_is_collected_again(tmp_path):
    app, _, _ = make_app(tmp_path)
    tracker = DistractedTracker(bad_index=5)
    tracker.app = app
    app.tracker = tracker
    app.handle_key(ord(" "))
    proc = run_procedure(app, max_steps=6000)
    assert proc.succeeded
    assert 5 in proc.retried_targets
    assert max(r.error_px for r in proc.results) < 60


def test_bad_targets_rule():
    proc = CalibrationProcedure((1920, 1080), CalibrationConfig())
    errs = [20.0] * 14 + [300.0, 25.0]
    stats = {"erro_loo_por_alvo_px": errs, "erro_loo_alvos": list(range(16))}
    assert proc.bad_targets(stats) == [14]
    # Abaixo do mínimo absoluto (8% da tela = 154 px) não repete.
    stats["erro_loo_por_alvo_px"] = [20.0] * 15 + [120.0]
    assert proc.bad_targets(stats) == []


# ------------------------------------------------------------ modelo
def test_split_head_stage_groups_and_range():
    user = SimUser(11)
    data = user.calibration_data()
    grid = np.vstack([r for _, _, r in data.grid])
    keep, groups, delta = split_head_stage(data.head_raw, grid, (12.0, 8.0))
    assert np.all(np.abs(delta[keep, 0]) <= 12) and np.all(np.abs(delta[keep, 1]) <= 8)
    assert (~keep).any()                     # a etapa sintética vai a ±15°/±10°
    assert set(np.unique(groups)) <= {ALWAYS_TRAIN, *range(HEAD_GROUP_BASE - 3, HEAD_GROUP_BASE + 1)}
    still = np.max(np.abs(delta), axis=1) < 2.0
    assert np.all(groups[still] == ALWAYS_TRAIN)


def test_model_reports_head_cv_and_candidates():
    user = SimUser(12)
    model, _, stats = fit_from_samples(user.calibration_data(), (1920, 1080), 2, None, 3.5)
    assert stats["erro_cv_cabeca_px"] > 0
    assert {c["grau"] for c in stats["candidatos"]} == {1, 2}
    assert stats["pontuacao_cv_px"] == model.cv_score


def test_posture_deviation(tmp_path):
    user = SimUser(13)
    model, _, _ = fit_from_samples(user.calibration_data(), (1920, 1080), 2, None, 3.5)
    _, r_ok = user.sample(960, 540, head=(0, 0, 0))
    _, r_far = user.sample(960, 540, head=(30, 0, 0), pos=(10.0, 10.0, 60.0))
    ok, far = model.posture_deviation(r_ok), model.posture_deviation(r_far)
    assert np.all(ok["ang"] == 0) and ok["pos_cm"] < 2
    assert far["ang"][0] > 10 and far["pos_cm"] > 8
    # Sobrevive a salvar e carregar.
    model.save(tmp_path / "c.json")
    from eyegaze.calibration import CalibrationModel
    loaded = CalibrationModel.load(tmp_path / "c.json", (1920, 1080))
    assert np.allclose(loaded.posture_deviation(r_far)["ang"], far["ang"])


def test_quality_label():
    assert quality_label(60, 1920)[0] == "boa"
    assert quality_label(120, 1920)[0] == "razoavel"
    assert quality_label(250, 1920)[0] == "ruim"


def test_validation_reports_constant_offset(tmp_path):
    app, tracker, _ = make_app(tmp_path)
    app.handle_key(ord(" "))
    run_procedure(app)
    app.handle_key(13)
    app.model.offset = app.model.offset + np.array([150.0, 0.0])   # desvio artificial
    app.handle_key(ord("t"))
    proc = run_procedure(app)
    s = proc.summary
    assert s["deslocamento_medio_px"][0] > 120
    assert s["acuracia_sem_deslocamento_px"] < s["acuracia_media_px"] - 100
    assert "fonte_iris" in s["calibracao"]


# ---------------------------------------------------- deriva durante o uso
def _calibrated_app(tmp_path):
    app, tracker, mouse = make_app(tmp_path)
    app.handle_key(ord(" "))
    run_procedure(app)
    app.handle_key(13)
    assert app.calibrated
    return app, tracker, mouse


def test_posture_warning_and_recenter_clears_it(tmp_path):
    app, tracker, _ = _calibrated_app(tmp_path)
    tracker.look_at = (960, 540)
    settle(app, 60)
    assert not app.posture_off
    # Rosto 8 cm para o lado: depois de ~1,5 s o aviso aparece.
    tracker.pos = np.array([8.0, 10.0, 60.0])
    settle(app, 20)
    assert not app.posture_off            # ainda não sustentado
    settle(app, 40)
    assert app.posture_off
    # Recentralizar nesta postura: ela vira a referência e o aviso some.
    tracker.proc_pos = tracker.pos
    app.handle_key(ord("r"))
    run_procedure(app)
    settle(app, 60)
    assert app.procedure is None
    settle(app, 60)
    assert not app.posture_off


def test_quick_head_turn_does_not_warn(tmp_path):
    app, tracker, _ = _calibrated_app(tmp_path)
    tracker.head = (25.0, 0.0, 0.0)
    settle(app, 20)                       # ~0,7 s
    tracker.head = (0.0, 0.0, 0.0)
    settle(app, 30)
    assert not app.posture_off


def test_known_target_correction_reduces_drift(tmp_path):
    app, tracker, _ = _calibrated_app(tmp_path)
    key = (700.0, 450.0)
    tracker.look_at = key

    def error():   # erro médio dos quadros recentes (um quadro só é ruidoso)
        return float(np.hypot(*(np.median(app.model.predict(np.array(app.recent_raw)), 0)
                                - key)))

    settle(app, 20)
    base = error()
    app.model.offset = app.model.offset + np.array([100.0, -60.0])   # deriva artificial
    errs = []
    for _ in range(8):                    # 8 "teclas" selecionadas no mesmo lugar
        settle(app, 20)
        errs.append(error())
        step = app.confirm_target(key)
        assert step is not None and np.linalg.norm(step) <= app.cfg.adapt.max_step_px + 1e-6
    assert errs[0] > base + 50                       # a deriva apareceu
    assert np.mean(errs[-3:]) < base + 0.3 * (errs[0] - base)   # e foi quase toda corrigida


def test_wrong_selection_moves_offset_only_a_little(tmp_path):
    app, tracker, _ = _calibrated_app(tmp_path)
    tracker.look_at = (700.0, 450.0)
    settle(app, 20)
    before = app.model.offset.copy()
    app.confirm_target((1700.0, 1000.0))  # "tecla" longe de onde a pessoa olhava
    assert np.linalg.norm(app.model.offset - before) <= app.cfg.adapt.max_step_px + 1e-6
