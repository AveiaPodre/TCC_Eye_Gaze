import math
from pathlib import Path

import numpy as np
import pytest

from eyegaze.calibration import (CalibrationData, CalibrationModel, build_grid,
                                 fit_from_samples, polynomial_terms, reject_outliers)
from eyegaze.config import GazeConfig
from eyegaze.filters import FixationStabilizer, OneEuroFilter
from eyegaze.gaze import GazeEstimator, direction_to_angles
from eyegaze.head_pose import estimate_head_pose
from eyegaze.landmarks import LEFT_EYE, RIGHT_EYE

from synthetic import (ScreenGeometry, gaze_dir, make_observation, make_points,
                       make_transform, rot)


# ---------------------------------------------------------------- geometria
def test_gaze_dir_roundtrip():
    for y, p in [(0, 0), (10, -5), (-20, 12)]:
        yy, pp = direction_to_angles(gaze_dir(y, p))
        assert yy == pytest.approx(y, abs=1e-9)
        assert pp == pytest.approx(p, abs=1e-9)


@pytest.mark.parametrize("head", [(0, 0, 0), (20, 0, 0), (-20, 0, 0),
                                  (0, 12, 0), (0, -12, 0), (0, 0, 10)])
def test_head_pose_signs(head):
    pose, _ = estimate_head_pose(make_points(head=head))
    yaw, pitch, roll = pose.euler_deg
    # O referencial da cabeça tem uma inclinação fixa (testa vs base do
    # nariz em profundidades diferentes): comparamos com a pose frontal.
    y0, p0, r0 = estimate_head_pose(make_points())[0].euler_deg
    for got, want in [(yaw - y0, head[0]), (pitch - p0, head[1]), (roll - r0, head[2])]:
        if want == 0:
            assert abs(got) < 3.0
        else:
            assert np.sign(got) == np.sign(want) and abs(got) > 0.5 * abs(want)


def test_gaze_signs_frontal():
    est = GazeEstimator(GazeConfig())
    base = est.estimate(make_observation(0, 0))
    right = est.estimate(make_observation(15, 0))
    up = est.estimate(make_observation(0, 10))
    assert right.yaw_deg > base.yaw_deg + 10      # olhar à direita do usuário => yaw +
    assert up.pitch_deg > base.pitch_deg + 6      # olhar para cima => pitch +
    assert abs(right.pitch_deg - base.pitch_deg) < 1.5


def test_gaze_invariant_to_head_rotation():
    """Mesma direção de olhar com cabeça em poses diferentes -> mesmos ângulos.

    É a propriedade central: a pessoa pode compensar com o olho o
    movimento da cabeça (ou vice-versa) e o ponto na tela não muda.
    """
    est = GazeEstimator(GazeConfig())
    ref = est.estimate(make_observation(5, -3)).features()
    for head in [(15, 0, 0), (-15, 0, 0), (0, 10, 0), (0, -10, 0), (8, 6, 7)]:
        f = est.estimate(make_observation(5, -3, head=head)).features()
        assert np.allclose(f, ref, atol=2.0), (head, f, ref)


def test_matrix_pose_matches_true_rotation():
    for head in [(0, 0, 0), (20, -10, 5), (-15, 12, -8)]:
        pose, used = estimate_head_pose(make_points(head=head), make_transform(head))
        assert used == "matrix"
        assert np.allclose(pose.R, rot(*head), atol=1e-9)


def test_incoherent_matrix_falls_back_to_landmarks():
    bad = make_transform((0, 0, 0))
    bad[:3, :3] = np.diag([-1.0, 1.0, -1.0])  # x espelhado (convenção errada)
    _, used = estimate_head_pose(make_points(), bad)
    assert used.startswith("landmarks")


def test_gaze_with_matrix_is_exact_and_head_invariant():
    est = GazeEstimator(GazeConfig())
    for g in [(0, 0), (12, -6), (-18, 9)]:
        for head in [(0, 0, 0), (15, 0, 0), (0, -10, 0), (8, 6, 7)]:
            f = est.estimate(make_observation(*g, head=head, with_transform=True)).features()
            assert np.allclose(f, g, atol=0.05), (g, head, f)


def test_gaze_invariant_to_distance_and_position():
    est = GazeEstimator(GazeConfig())
    ref = est.estimate(make_observation(-8, 4)).features()
    for scale, center in [(8, (300, 200, 0)), (18, (900, 500, 0))]:
        f = est.estimate(make_observation(-8, 4, scale=scale, center=center)).features()
        assert np.allclose(f, ref, atol=1e-6)


def test_blink_detected_and_invalid():
    est = GazeEstimator(GazeConfig())
    for _ in range(5):
        est.estimate(make_observation(0, 0))
    s = est.estimate(make_observation(0, 0, blink=True))
    assert s.blink and not s.valid
    assert all(not e.open for e in s.eyes)


def test_eye_indices_are_from_same_eye():
    """Regressão do bug da versão anterior (íris de um olho com cantos do outro)."""
    from eyegaze.landmarks import LEFT_EYE, RIGHT_EYE
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    def idx(conns):
        return {c.start for c in conns} | {c.end for c in conns}

    for eye, contour, iris in [(LEFT_EYE, C.FACE_LANDMARKS_LEFT_EYE, C.FACE_LANDMARKS_LEFT_IRIS),
                               (RIGHT_EYE, C.FACE_LANDMARKS_RIGHT_EYE, C.FACE_LANDMARKS_RIGHT_IRIS)]:
        ci = idx(contour)
        for i in (eye.outer_corner, eye.inner_corner, eye.upper_lid, eye.lower_lid):
            assert i in ci
        assert set(eye.iris_ring) == idx(iris)


# ---------------------------------------------------------------- calibração
# "Usuário" sintético com as fontes de erro que aparecem com webcam real:
# ganho do olho diferente do assumido (rho real 0.5 vs 0.4), movimento
# vertical da íris achatado pelas pálpebras e acoplamento com o pitch da cabeça.
# Íris quase parada na vertical (como nos dados reais do MediaPipe) e
# pálpebras acompanhando o olhar: o sinal vertical vem das pálpebras.
USER = dict(rho_true=0.5, v_scale=0.2, pitch_coupling=0.4, lid_follow=0.7)
EYE_POS = (0.0, 10.0, 60.0)


class SimUser:
    def __init__(self, seed=0, noise_px=0.0, **user):
        self.geom = ScreenGeometry()
        self.rng = np.random.default_rng(seed)
        self.est = GazeEstimator(GazeConfig())
        self.noise = noise_px
        self.user = user or USER

    def sample(self, tx, ty, head=(0, 0, 0), pos=EYE_POS):
        g = self.geom.gaze_vector(tx, ty, pos)
        obs = make_observation(head=head, head_pos_cm=pos, with_transform=True, gaze_vec=g,
                               noise_px=self.noise, rng=self.rng, **self.user)
        s = self.est.estimate(obs)
        return s.features(), s.raw

    def calibration_data(self, head_stage=True, grid=(3, 3), n=25):
        data = CalibrationData()
        for t in build_grid(1920, 1080, *grid, 0.08):
            F, Rw = zip(*[self.sample(*t, head=tuple(self.rng.normal(0, 1.5, 3)))
                          for _ in range(n)])
            data.grid.append((t, np.array(F), np.array(Rw)))
        if head_stage:
            hs = []
            for hy, hp in [(-15, 0), (15, 0), (0, 10), (0, -10), (0, 0)]:
                for k in range(30):
                    a = (k + 1) / 30
                    pos = np.array(EYE_POS) + self.rng.normal(0, 0.5, 3)
                    hs.append(self.sample(960, 540, head=(hy * a, hp * a, 0), pos=pos))
            F, Rw = zip(*hs)
            data.head_target = (960, 540)
            data.head_features, data.head_raw = np.array(F), np.array(Rw)
        return data

    def mean_error(self, model, head=(0, 0, 0), pos=EYE_POS):
        errs = []
        for tx, ty in [(480, 270), (1440, 270), (960, 540), (480, 810), (1440, 810),
                       (300, 540), (1600, 600)]:
            _, r = self.sample(tx, ty, head, pos)
            errs.append(math.hypot(*(model.predict(r) - [tx, ty])))
        return float(np.mean(errs))


def _fit(data, **kw):
    return fit_from_samples(data, (1920, 1080), 2, None, 3.5, head_weight_points=1.0, **kw)


def test_calibration_generalizes_to_unseen_points():
    user = SimUser(1)
    model, results, _ = _fit(user.calibration_data())
    assert max(r.error_px for r in results) < 30
    assert user.mean_error(model) < 25


def test_head_motion_stage_makes_gaze_head_invariant():
    """Problema relatado: a direção do olhar dependia da angulação da cabeça.

    A geometria da cabeça (P_cabeça) entra no modelo de forma explícita,
    então o erro fica pequeno em qualquer pose.
    """
    user = SimUser(2)
    model, _, stats = _fit(user.calibration_data())
    assert stats["usa_etapa_cabeca"]
    for head, pos in [((15, 0, 0), EYE_POS), ((-15, 0, 0), EYE_POS),
                      ((0, 10, 0), EYE_POS), ((0, -10, 0), EYE_POS),
                      ((10, -8, 5), (-4.0, 13.0, 55.0))]:
        e = user.mean_error(model, head, pos)
        assert e < 60, (head, e)   # pose mista (rotação + translação) ~55 px


def test_head_translation_is_compensated():
    """Chegar para o lado ou para perto da tela não desloca o cursor.

    Translação lateral é compensada quase exatamente. Mudar a distância em
    10 cm deixa um resíduo (~40 px): a aproximação linear de tan(olho) só
    é exata perto da distância da calibração.
    """
    user = SimUser(3)
    model, _, _ = _fit(user.calibration_data())
    for pos in [(6.0, 10.0, 60.0), (-6.0, 10.0, 60.0)]:
        assert user.mean_error(model, pos=pos) < 25, pos
    # Vertical: o sinal do olho vem das pálpebras, menos linear (~30 px).
    assert user.mean_error(model, pos=(0.0, 14.0, 60.0)) < 35
    for pos in [(0.0, 10.0, 50.0), (0.0, 10.0, 70.0)]:
        assert user.mean_error(model, pos=pos) < 50, pos


def test_head_invariance_with_landmark_noise():
    user = SimUser(4, noise_px=0.3)
    model, _, _ = _fit(user.calibration_data())
    for head in [(15, 0, 0), (-15, 0, 0), (0, 10, 0), (0, -10, 0)]:
        assert user.mean_error(model, head) < 70, head


def test_without_head_stage_still_works_with_head_still():
    user = SimUser(5)
    model, _, stats = _fit(user.calibration_data(head_stage=False))
    assert not stats["usa_etapa_cabeca"]
    assert user.mean_error(model) < 30  # cabeça parada continua funcionando


def test_vertical_comes_from_eyelids():
    """Problema real: a íris quase não se move na vertical. Sem as medidas
    das pálpebras, o y da tela "achatava" no centro."""
    user = SimUser(8, noise_px=0.2)
    model, _, stats = _fit(user.calibration_data())
    assert stats["erro_loo_medio_px"] < 40
    for ty in (150, 930):
        _, r = user.sample(960, ty)
        assert abs(model.predict(r)[1] - ty) < 40


def test_loo_error_reported_per_target():
    user = SimUser(9)
    _, _, stats = _fit(user.calibration_data(grid=(4, 4)))
    assert len(stats["erro_loo_por_alvo_px"]) == 16
    assert stats["ridge_alpha"] > 0


def test_raw_data_roundtrip(tmp_path):
    data = SimUser(10).calibration_data(n=5)
    data.save(tmp_path / "c.npz")
    loaded = CalibrationData.load(tmp_path / "c.npz")
    assert len(loaded.grid) == len(data.grid)
    assert np.allclose(loaded.grid[3][2], data.grid[3][2])
    assert np.allclose(loaded.head_raw, data.head_raw)
    assert not loaded.has_landmarks


def test_landmarks_saved_and_recomputed(tmp_path):
    """Os landmarks gravados permitem recalcular as medidas offline."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from reavaliar import recompute

    est = GazeEstimator(GazeConfig())
    obs = [make_observation(gaze_yaw=5, gaze_pitch=-3, with_transform=True, timestamp_ms=k)
           for k in range(6)]
    samples = [est.estimate(o) for o in obs]
    data = CalibrationData(grid=[((100.0, 200.0), np.array([s.features() for s in samples]),
                                  np.array([s.raw for s in samples]))],
                           grid_obs=[[s.obs for s in samples]])
    data.save(tmp_path / "c.npz")
    loaded = CalibrationData.load(tmp_path / "c.npz")
    assert loaded.has_landmarks
    again = recompute(loaded)
    assert np.allclose(again.grid[0][2], data.grid[0][2], atol=1e-4)


def test_iris_depth_noise_does_not_leak_into_eye_measures():
    """Bug real: o z da íris do MediaPipe é ruidoso e, com a cabeça girada,
    vazava para u e v (que variavam muito mais com a cabeça do que com o
    olho). A profundidade agora sai da esfera do olho; o z da íris é ignorado."""
    est = GazeEstimator(GazeConfig())
    for head in [(20, 0, 0), (0, -15, 0), (12, 10, 4)]:
        R = rot(*head)
        obs = make_observation(head=head, gaze_vec=R @ np.array([0.1, 0.05, -1.0]),
                               with_transform=True)
        base = est.estimate(obs)
        for eye in (RIGHT_EYE, LEFT_EYE):
            for k in (eye.iris_center,) + eye.iris_ring:
                obs.points[k, 2] += 8.0   # erro grande no z da íris
        noisy = est.estimate(obs)
        assert np.allclose(base.raw[12:], noisy.raw[12:], atol=1e-9), head


def test_calibration_falls_back_to_affine_with_few_points():
    user = SimUser(6)
    model, _, _ = _fit(user.calibration_data(head_stage=False, grid=(2, 2)))
    assert model.degree == 1


def test_calibration_save_load(tmp_path):
    user = SimUser(7)
    model, _, _ = _fit(user.calibration_data())
    model.offset = np.array([5.0, -3.0])
    path = tmp_path / "cal.json"
    model.save(path)
    loaded = CalibrationModel.load(path, (1920, 1080))
    _, r = user.sample(700, 400, head=(5, -3, 0))
    assert np.allclose(loaded.predict(r), model.predict(r))
    assert CalibrationModel.load(path, (1280, 720)) is None  # outra resolução


def test_polynomial_terms_shape():
    assert polynomial_terms(np.zeros((4, 2)), 1).shape == (4, 3)
    assert polynomial_terms(np.zeros((4, 2)), 2).shape == (4, 6)


def test_reject_outliers_removes_spike():
    rng = np.random.default_rng(4)
    s = rng.normal(0, 0.2, (30, 2))
    s[5] = [8, -8]
    assert len(reject_outliers(s, 3.5)) < 30
    assert not np.any(np.all(reject_outliers(s, 3.5) == [8, -8], axis=1))


# ------------------------------------------------------------------- filtro
def test_one_euro_reduces_jitter_and_follows_steps():
    rng = np.random.default_rng(5)
    f = OneEuroFilter(0.8, 0.03, 1.0)
    out = [f(np.array([rng.normal(0, 1.0)]), i / 30)[0] for i in range(300)]
    assert np.std(out[30:]) < 0.5
    f.reset()
    for i in range(30):
        f(np.array([0.0]), i / 30)
    vals = [f(np.array([20.0]), (30 + i) / 30)[0] for i in range(15)]
    assert vals[-1] > 18  # alcança o degrau em ~0.5 s


def _jitter_scenario(seed=0, noise=45.0):
    rng = np.random.default_rng(seed)
    truth = np.array([[500, 500]] * 60 + [[1400, 700]] * 60, float)
    return truth, truth + rng.normal(0, noise, truth.shape)


def test_stabilizer_removes_tremor_during_fixation():
    """Problema relatado: com o mouse ligado o cursor tremia sem parar."""
    truth, raw = _jitter_scenario()
    st = FixationStabilizer()
    out = np.array([st(p) for p in raw])
    step = np.linalg.norm(np.diff(out[25:60], axis=0), axis=1)
    raw_step = np.linalg.norm(np.diff(raw[25:60], axis=0), axis=1)
    assert step.mean() < 3.0 < raw_step.mean()          # ~2 px vs ~80 px por frame
    assert np.linalg.norm(out[40:60] - truth[40:60], axis=1).mean() < 25


def test_stabilizer_follows_saccades_quickly():
    truth, raw = _jitter_scenario(1)
    st = FixationStabilizer()
    out = np.array([st(p) for p in raw])
    arrived = np.argmax(np.linalg.norm(out[60:] - truth[60:], axis=1) < 60)
    assert arrived <= 5   # <= ~170 ms a 30 fps


def test_stabilizer_ignores_isolated_spikes():
    st = FixationStabilizer()
    for _ in range(30):
        st(np.array([800.0, 400.0]))
    for spike in ([1500.0, 900.0], [800.0, 400.0], [200.0, 100.0], [800.0, 400.0]):
        out = st(np.array(spike))
    assert np.linalg.norm(out - [800, 400]) < 1.0


def test_stabilizer_follows_slow_drift():
    st = FixationStabilizer()
    xs = np.linspace(500, 700, 120)   # 200 px em 4 s (perseguição lenta)
    for x in xs:
        out = st(np.array([x, 500.0]))
    assert abs(out[0] - 700) < 40
