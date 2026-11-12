import math
import sys
from pathlib import Path

import numpy as np
import pytest

from eyegaze.calibration import (CalibrationData, CalibrationModel, build_grid,
                                 correction_basis, fit_from_samples, reject_outliers)
from eyegaze.config import GazeConfig
from eyegaze.eyeball import RADIUS_CM, EyeballModel
from eyegaze.filters import FixationStabilizer, OneEuroFilter
from eyegaze.gaze import RAW_SIZE, GazeEstimator, direction_to_angles
from eyegaze.geometry import VirtualScreen, closest_point_between_rays
from eyegaze.head_pose import estimate_head_pose
from eyegaze.landmarks import LEFT_EYE, RIGHT_EYE

from synthetic import (ScreenGeometry, gaze_dir, make_observation, make_points,
                       make_transform, rot)

SCREEN = (1920, 1080)
EYE_POS = (0.0, 10.0, 60.0)


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
    # O referencial da cabeça por landmarks tem uma inclinação fixa (testa e
    # base do nariz em profundidades diferentes): compara com a pose frontal.
    y0, p0, r0 = estimate_head_pose(make_points())[0].euler_deg
    for got, want in [(yaw - y0, head[0]), (pitch - p0, head[1]), (roll - r0, head[2])]:
        if want == 0:
            assert abs(got) < 3.0
        else:
            assert np.sign(got) == np.sign(want) and abs(got) > 0.5 * abs(want)


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


def test_closest_point_between_rays():
    """Etapa 3: retas que se cruzam -> o ponto de cruzamento; retas reversas
    -> o meio do segmento perpendicular comum; paralelas -> NaN."""
    T = np.array([5.0, 12.0, 0.0])
    o1, o2 = np.array([-3.0, 10.0, 60.0]), np.array([3.0, 10.0, 60.0])
    X, s, t, gap = closest_point_between_rays(o1, T - o1, o2, T - o2)
    assert np.allclose(X, T) and gap < 1e-9 and s == pytest.approx(1.0)
    X, _, _, gap = closest_point_between_rays(np.zeros(3), np.array([1.0, 0, 0]),
                                              np.array([0, 0, 2.0]), np.array([0, 1.0, 0]))
    assert np.allclose(X, [0, 0, 1.0]) and gap == pytest.approx(2.0)
    X, s, _, _ = closest_point_between_rays(o1, np.array([0, 0, -1.0]), o2, np.array([0, 0, -1.0]))
    assert np.isnan(s)


def test_virtual_screen_orientation_and_roundtrip():
    """Imagem não espelhada: a borda esquerda da tela (para o usuário) fica em
    +x; a tela fica abaixo da webcam (y para baixo), no plano z = 0."""
    scr = VirtualScreen(SCREEN, 34.5)
    left, right = scr.px_to_cm([(0, 0), (1920, 1080)])
    assert left[0] > 0 > right[0] and np.isclose(left[0] - right[0], 34.5)
    assert right[1] > left[1] > 0 and left[2] == right[2] == 0.0
    Y = np.array([[100.0, 200.0], [1800.0, 900.0]])
    assert np.allclose(scr.cm_to_px(scr.px_to_cm(Y)), Y)
    o = np.array([[0.0, 10.0, 60.0]])
    T = scr.px_to_cm([(700.0, 300.0)])
    assert np.allclose(scr.project(o, T - o), [[700.0, 300.0]])
    assert np.all(np.isnan(scr.hit(o, np.array([[0, 0, 1.0]]))))   # olhando para trás


# ------------------------------------------------- etapas 1-2: medidas do olho
def test_anchor_and_pupil_measured_in_cm():
    """Âncoras (meio dos cantos) a ~6 cm uma da outra, à distância do rosto;
    pupilas no plano do olho, olhando em frente, sobre as âncoras."""
    est = GazeEstimator(GazeConfig())
    s = est.estimate(make_observation(head_pos_cm=EYE_POS, with_transform=True,
                                      gaze_vec=np.array([0, 0, -1.0])))
    a = np.array([e.anchor_cm for e in s.eyes])
    assert np.linalg.norm(a[0] - a[1]) == pytest.approx(6.0, abs=0.2)
    assert a[:, 2] == pytest.approx([60.0, 60.0], abs=1.5)
    for e in s.eyes:
        assert np.linalg.norm(e.pupil_cm[:2] - e.anchor_cm[:2]) < 0.1
    assert s.raw.shape == (RAW_SIZE,)


def test_pupil_ignores_iris_depth_noise():
    """O z da íris do MediaPipe é ruidoso e, com a cabeça girada, contaminava
    as outras coordenadas. A pupila 3D sai da linha de visada da câmera pelo
    pixel até o plano do olho; o z da íris não é usado."""
    est = GazeEstimator(GazeConfig())
    for head in [(20, 0, 0), (0, -15, 0), (12, 10, 4)]:
        obs = make_observation(head=head, head_pos_cm=EYE_POS, with_transform=True,
                               gaze_vec=rot(*head) @ np.array([0.1, 0.05, -1.0]))
        base = est.estimate(obs)
        for eye in (RIGHT_EYE, LEFT_EYE):
            for k in (eye.iris_center,) + eye.iris_ring:
                obs.points[k, 2] += 8.0
        assert np.allclose(base.raw, est.estimate(obs).raw, atol=1e-9), head


def test_blink_detected_and_invalid():
    est = GazeEstimator(GazeConfig())
    for _ in range(5):
        est.estimate(make_observation(0, 0))
    s = est.estimate(make_observation(0, 0, blink=True))
    assert s.blink and not s.valid and s.gaze is None
    assert all(not e.open for e in s.eyes)


def test_eye_indices_are_from_same_eye():
    """Regressão de um bug antigo (íris de um olho com os cantos do outro)."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    def idx(conns):
        return {c.start for c in conns} | {c.end for c in conns}

    for eye, contour, iris in [(LEFT_EYE, C.FACE_LANDMARKS_LEFT_EYE, C.FACE_LANDMARKS_LEFT_IRIS),
                               (RIGHT_EYE, C.FACE_LANDMARKS_RIGHT_EYE, C.FACE_LANDMARKS_RIGHT_IRIS)]:
        ci = idx(contour)
        for i in (eye.outer_corner, eye.inner_corner, eye.upper_lid, eye.lower_lid):
            assert i in ci
        assert set(eye.iris_ring) == idx(iris)


# ----------------------------------------------------- usuário sintético
# PHYS: olho "físico" (a íris anda o que a geometria do globo prevê).
# COMPRESSED: como o MediaPipe nos dados reais, o movimento da íris é
# encolhido (raio real 0,5 x largura do olho, vertical a 30%); a íris ainda
# contra-rotaciona quando a cabeça gira com o olhar fixo.
PHYS = dict(rho_true=0.4)
COMPRESSED = dict(rho_true=0.5, v_scale=0.3)


class SimUser:
    def __init__(self, seed=0, noise_px=0.0, **user):
        self.geom = ScreenGeometry()
        self.rng = np.random.default_rng(seed)
        self.est = GazeEstimator(GazeConfig())
        self.noise = noise_px
        self.user = user or COMPRESSED

    def observation(self, tx, ty, head=(0, 0, 0), pos=EYE_POS, mouth=0.0, ts=0):
        g = self.geom.gaze_vector(tx, ty, pos)
        return make_observation(head=head, head_pos_cm=pos, with_transform=True, gaze_vec=g,
                                noise_px=self.noise, rng=self.rng, mouth_open=mouth,
                                timestamp_ms=ts, **self.user)

    def sample(self, tx, ty, head=(0, 0, 0), pos=EYE_POS, mouth=0.0):
        s = self.est.estimate(self.observation(tx, ty, head, pos, mouth))
        return s.features(), s.raw

    def calibration_data(self, head_stage=True, grid=(4, 4), n=12, grid_head_sd=1.0,
                         mouth_stage=False):
        data = CalibrationData()
        for t in build_grid(1920, 1080, *grid, 0.05):
            F, Rw = zip(*[self.sample(*t, head=tuple(self.rng.normal(0, grid_head_sd, 3)))
                          for _ in range(n)])
            data.grid.append((t, np.array(F), np.array(Rw)))
        if head_stage:
            hs = []
            for hy, hp in [(10, 0), (-10, 0), (0, 8), (0, -8), (0, 0)]:
                for k in range(20):
                    a = (k + 1) / 20
                    pos = np.array(EYE_POS) + self.rng.normal(0, 0.5, 3)
                    hs.append(self.sample(960, 540, head=(hy * a, hp * a, 0), pos=pos))
            F, Rw = zip(*hs)
            data.head_target = (960, 540)
            data.head_features, data.head_raw = np.array(F), np.array(Rw)
        if mouth_stage:
            ms = [self.sample(960, 540, mouth=0.15 * abs(np.sin(k / 8.0))) for k in range(90)]
            F, Rw = zip(*ms)
            data.mouth_target = (960, 540)
            data.mouth_features, data.mouth_raw = np.array(F), np.array(Rw)
        return data

    def mean_error(self, model, head=(0, 0, 0), pos=EYE_POS, mouth=0.0):
        errs = []
        for tx, ty in [(480, 270), (1440, 270), (960, 540), (480, 810), (1440, 810),
                       (300, 540), (1600, 600)]:
            _, r = self.sample(tx, ty, head, pos, mouth)
            errs.append(math.hypot(*(model.predict(r) - [tx, ty])))
        return float(np.mean(errs))


def _fit(data, width_cm=53.0, **kw):
    return fit_from_samples(data, SCREEN, width_cm, 3.5, **kw)


# ------------------------------------------- etapas 2-3: globo, feixes, intersecção
def test_eyeball_fit_recovers_physical_eye():
    """Num olho físico, o ajuste encontra o raio real do globo (1,2 cm =
    0,4 x 3 cm de largura do olho) nos dois eixos, e o feixe sai do centro
    do globo e passa pela pupila."""
    user = SimUser(1, **PHYS)
    model, _, _ = _fit(user.calibration_data())
    assert np.allclose(model.eyeball.s, RADIUS_CM, atol=0.1), model.eyeball.s
    s = GazeEstimator(GazeConfig(), model.eyeball).estimate(user.observation(500, 300))
    for k in range(2):
        v = s.gaze.pupils[k] - s.gaze.centers[k]
        d = s.gaze.directions[k]
        assert np.linalg.norm(v - np.dot(v, d) * d) < 0.02    # pupila sobre o feixe (cm)
        assert np.dot(v, d) > 0                                # centro ATRÁS da pupila


def test_beams_intersect_at_the_fixated_target():
    """Os dois feixes se cruzam (quase) no alvo da tela; o vetor do olhar até a
    intersecção, projetado no monitor virtual, cai no alvo."""
    user = SimUser(2, **PHYS)
    model, _, _ = _fit(user.calibration_data())
    est = GazeEstimator(GazeConfig(), model.eyeball)
    for tx, ty in [(300, 200), (1500, 900), (960, 540)]:
        g = est.estimate(user.observation(tx, ty)).gaze
        T = model.screen.px_to_cm([(tx, ty)])[0]
        assert g.vergence is not None and np.linalg.norm(g.vergence - T) < 4.0, g.vergence
        hit = model.screen.project(g.origin, g.direction)[0]
        assert np.hypot(*(hit - [tx, ty])) < 40


def test_one_closed_eye_uses_the_other_beam():
    user = SimUser(3, **PHYS)
    model, _, _ = _fit(user.calibration_data(head_stage=False))
    _, r = user.sample(700, 400)
    r_one = r.copy()
    r_one[31] = 0.0                                  # olho esquerdo fechado
    o, d, X, _, _ = model.eyeball.gaze(r_one)
    assert np.all(np.isnan(X)) and np.all(np.isfinite(d))
    assert np.hypot(*(model.predict(r_one) - [700, 400])) < 60


# ------------------------------------------------- etapa 5: calibração
def test_calibration_generalizes_to_unseen_points():
    user = SimUser(4)
    model, results, stats = _fit(user.calibration_data())
    assert max(r.error_px for r in results) < 50
    assert stats["erro_loo_medio_px"] < 40
    assert user.mean_error(model) < 30


def test_compression_is_absorbed_by_effective_radius():
    """O MediaPipe encolhe o movimento da íris (mais na vertical): o raio
    efetivo de cada eixo absorve isso, e a cabeça continua valendo 1:1."""
    user = SimUser(5)
    model, _, _ = _fit(user.calibration_data())
    s = model.eyeball.s
    assert np.all(s[:, 1] < 0.6 * s[:, 0]), s     # vertical bem mais comprimido


def test_head_rotation_and_translation_are_compensated():
    """A pessoa pode girar ou deslocar a cabeça: o feixe acompanha a rotação
    (R) e a posição do rosto, e o ponto na tela quase não muda.

    O resíduo (até ~125 px em pose mista) vem de o usuário sintético
    comprimir o SENO do ângulo vertical do olho, enquanto o modelo do globo
    usa a tangente (eyeball.EYE_MODEL): com a cabeça girada, a componente
    vertical do alvo no referencial da cabeça muda de um jeito que só a
    tangente sente. A forma esférica reduz isso no sintético, mas errou mais
    nas sessões reais. Com a cabeça parada o erro é bem menor."""
    user = SimUser(6)
    model, _, stats = _fit(user.calibration_data())
    assert stats["usa_etapa_cabeca"]
    for head, pos in [((12, 0, 0), EYE_POS), ((-12, 0, 0), EYE_POS),
                      ((0, 9, 0), EYE_POS), ((0, -9, 0), EYE_POS),
                      ((8, -6, 4), (-4.0, 13.0, 55.0)), ((0, 0, 0), (6.0, 10.0, 68.0))]:
        e = user.mean_error(model, head, pos)
        assert e < 135, (head, pos, e)


def test_head_invariance_with_landmark_noise():
    user = SimUser(7, noise_px=0.3)
    model, _, _ = _fit(user.calibration_data())
    for head in [(12, 0, 0), (-12, 0, 0), (0, 9, 0), (0, -9, 0)]:
        assert user.mean_error(model, head) < 120, head


def test_without_head_stage_still_works():
    user = SimUser(8)
    model, _, stats = _fit(user.calibration_data(head_stage=False))
    assert not stats["usa_etapa_cabeca"]
    assert user.mean_error(model) < 35
    assert user.mean_error(model, (10, 0, 0)) < 80   # a geometria compensa a cabeça


def test_correction_absorbs_unknown_screen_width():
    """Sem a largura física, o monitor virtual tem o tamanho suposto (34,5 cm
    contra 53 cm reais); a correção monitor virtual -> real resolve a escala."""
    user = SimUser(9)
    model, _, stats = _fit(user.calibration_data(), width_cm=None)
    assert not stats["largura_conhecida"] and stats["correcao"] != "nenhuma"
    assert stats["erro_loo_monitor_virtual_px"] > 1.5 * stats["erro_loo_medio_px"]
    assert user.mean_error(model) < 45


def test_cv_report():
    user = SimUser(10)
    _, _, stats = _fit(user.calibration_data())
    assert len(stats["erro_loo_por_alvo_px"]) == 16
    assert "erro_cv_cabeca_px" in stats and stats["correcao"] in (
        "nenhuma", "deslocamento", "afim", "quadratica")
    assert {c["correcao"] for c in stats["candidatos"]} >= {"nenhuma", "afim", "quadratica"}
    assert stats["feixes_convergem_frac"] > 0.9


def test_few_targets_use_simpler_corrections():
    _, _, stats = _fit(SimUser(11).calibration_data(head_stage=False, grid=(2, 2)))
    assert stats["correcao"] in ("nenhuma", "deslocamento", "afim")


def test_correction_basis_shapes():
    V = np.zeros((4, 2))
    assert correction_basis(V, SCREEN, "deslocamento").shape == (4, 1)
    assert correction_basis(V, SCREEN, "afim").shape == (4, 3)
    assert correction_basis(V, SCREEN, "quadratica").shape == (4, 6)


def test_stale_raw_requires_recompute():
    data = SimUser(12).calibration_data(head_stage=False, grid=(2, 2), n=4)
    data.feature_version = 6
    with pytest.raises(ValueError):
        _fit(data)


def test_raw_data_roundtrip(tmp_path):
    data = SimUser(13).calibration_data(n=5)
    data.save(tmp_path / "c.npz")
    loaded = CalibrationData.load(tmp_path / "c.npz")
    assert not loaded.stale and len(loaded.grid) == len(data.grid)
    assert np.allclose(loaded.grid[3][2], data.grid[3][2])
    assert np.allclose(loaded.head_raw, data.head_raw)
    assert not loaded.has_landmarks


def test_landmarks_saved_and_recomputed(tmp_path):
    """Os landmarks gravados permitem recalcular as medidas offline."""
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


def test_mouth_stage_cancels_mouth_artifact():
    """Abrir a boca fazia o olhar descer: o MediaPipe deforma a malha inteira.
    A etapa da boca (olhar fixo, boca abrindo e fechando) ensina quanto
    descontar."""
    user = SimUser(14, **dict(COMPRESSED, mouth_coupling=0.6))
    with_m, _, stats = _fit(user.calibration_data(mouth_stage=True))
    without, _, _ = _fit(user.calibration_data())
    assert "boca_px_por_unidade" in stats
    e_without = user.mean_error(without, mouth=0.12)
    e_with = user.mean_error(with_m, mouth=0.12)
    assert e_without > 80
    assert e_with < 0.5 * e_without, (e_with, e_without)
    assert abs(user.mean_error(with_m) - user.mean_error(without)) < 15


def test_mouth_correction_needs_mouth_variation():
    user = SimUser(15)
    data = user.calibration_data()
    data.mouth_target = (960, 540)
    data.mouth_raw = np.array([user.sample(960, 540)[1] for _ in range(30)])
    data.mouth_features = np.zeros((30, 2))
    model, _, _ = _fit(data)
    assert not np.any(model.mouth_coef)


def test_calibration_save_load(tmp_path):
    user = SimUser(16)
    model, _, _ = _fit(user.calibration_data())
    model.offset = np.array([5.0, -3.0])
    path = tmp_path / "cal.json"
    model.save(path)
    loaded = CalibrationModel.load(path, SCREEN)
    _, r = user.sample(700, 400, head=(5, -3, 0))
    assert np.allclose(loaded.predict(r), model.predict(r))
    assert CalibrationModel.load(path, (1280, 720)) is None  # outra resolução


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
    # <= 200 ms a 30 fps. Com este ruído BRANCO de 45 px por eixo, uma amostra
    # isolada erra ~64 px e é preciso 1 quadro a mais de média. O cursor salta
    # para a amostra mais recente (não para a média das pendentes): nas
    # sessões reais, em que o ruído é correlacionado e o ponto bruto leva
    # alguns quadros para atravessar a tela, isso reduziu o atraso do cursor
    # de 92 para 42 ms (mediana) e de 292 para 110 ms (p90).
    assert arrived <= 6


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
    outs = [st(np.array([x, 500.0]))[0] for x in xs]
    # Segue em degraus: fica no máximo ~1 raio de trava atrás, e não trava de vez.
    assert abs(outs[-1] - 700) < st.lock_radius + 20
    assert outs[-1] > 600


def _wander(n, sigma, rho, rng):
    """Ruído AR(1): passeia devagar, como a estimativa real (autocorrelação ~0,97)."""
    e = np.zeros((n, 2))
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, sigma * np.sqrt(1 - rho ** 2), 2)
    return e


def test_stabilizer_holds_still_when_estimate_wanders():
    """Problema relatado: com a vista parada o cursor se movia um pouco.

    A estimativa real não treme de quadro a quadro, ela passeia (ruído
    correlacionado). A média deslizante acompanhava esse passeio."""
    rng = np.random.default_rng(3)
    moving = []
    for _ in range(20):
        st = FixationStabilizer(saccade_px=134, lock_px=67)
        raw = np.array([900.0, 500.0]) + _wander(60, 20.0, 0.97, rng)
        out = np.array([st(p) for p in raw])
        step = np.linalg.norm(np.diff(out[15:], axis=0), axis=1)
        moving.append(np.mean(step > 0.5))
    assert np.mean(moving) < 0.05   # cursor parado em >95% dos quadros da fixação


def test_head_fine_tuner_directions_deadzone_and_resets():
    from eyegaze.filters import HeadFineTuner
    ft = HeadFineTuner(gain_px_per_deg=25, deadzone_deg=0.3, max_px=100, reset_deg=6, smooth_hz=1000)
    c = np.array([900.0, 500.0])
    t = 0.0
    assert np.allclose(ft(c, 1, (0.0, 0.0), t), c)           # nova fixação: referência
    t += 0.03
    assert np.allclose(ft(c, 1, (0.2, -0.2), t), c)          # dentro da zona morta
    t += 0.03
    out = ft(c, 1, (2.3, 0.0), t)                             # cabeça 2,3° para a direita
    assert out[0] == pytest.approx(900 + 25 * 2.0, abs=1)     # (2,3 - 0,3) x 25
    t += 0.03
    out = ft(c, 1, (0.0, 1.3), t)                             # cabeça 1,3° para cima
    assert out[1] == pytest.approx(500 - 25 * 1.0, abs=1)     # cursor sobe
    t += 0.03
    out = ft(c, 1, (5.5, 5.5), t)
    assert np.linalg.norm(out - c) == pytest.approx(100, abs=1)   # limite (~184 px sem ele)
    t += 0.03
    assert np.allclose(ft(c, 1, (8.0, 0.0), t), c)            # giro grande: mudança de postura
    t += 0.03
    assert np.allclose(ft(c, 2, (9.0, 0.0), t), c)            # nova fixação zera


