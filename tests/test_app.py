"""
Teste de ponta a ponta do app com um "usuário simulado".

O FakeTracker substitui o MediaPipe: ele olha para o alvo que o app está
mostrando (ou para ``look_at``), segue as setas da etapa de movimento da
cabeça e devolve landmarks sintéticos coerentes com isso. O "usuário" tem o
erro típico do MediaPipe (movimento da íris encolhido, mais na vertical) e
ruído nos landmarks.

Exercita: calibração (com etapa de cabeça), cursor estável, cursor com a
cabeça girada/transladada, validação, recentralização, mouse, dwell-click,
perda de rosto e as teclas de visualização.
"""

import os

import cv2
import numpy as np

from eyegaze.app import EyeGazeApp
from eyegaze.config import AppConfig

from synthetic import ScreenGeometry, make_observation

SCREEN = (1920, 1080)
EYE_POS = np.array([0.0, 10.0, 60.0])
# Globo maior que o anatômico (raio 0,5 x largura do olho) e movimento
# vertical da íris a 30%: o raio efetivo de cada eixo absorve isso. A íris
# contra-rotaciona com a cabeça; nos dados reais ela quase não o faz na
# vertical, limitação medida em docs/metodologia.md (não simulada aqui).
USER = dict(rho_true=0.5, v_scale=0.3)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeSource:
    def __init__(self, clock):
        self.clock = clock
        self.frame = np.full((720, 1280, 3), 70, np.uint8)
        self.ts = 0

    def read(self):
        self.clock.t += 1 / 30
        self.ts += 33
        return self.frame, self.ts

    def release(self):
        pass


class FakeTracker:
    def __init__(self):
        self.geom = ScreenGeometry()
        self.app = None
        self.head = (0.0, 0.0, 0.0)
        self.pos = EYE_POS.copy()
        self.look_at = (960.0, 540.0)
        self.face = True
        self.rng = np.random.default_rng(0)
        self.noise_px = 0.25
        self._head_amount = 0.0
        self.mouth = 0.0
        self._mouth_t = 0

    def _head_for_prompt(self, proc):
        """Simula a pessoa girando a cabeça devagar na direção pedida."""
        prompt = getattr(proc, "head_prompt", None)
        if prompt is None:
            self._head_amount = 0.0
            return (0.0, 0.0, 0.0)
        _, (dx, dy) = prompt
        self._head_amount = min(1.0, self._head_amount + 1 / 20)
        # yaw > 0 = direita do usuário; pitch > 0 = para cima (dy = -1 é CIMA).
        return (15.0 * dx * self._head_amount, -10.0 * dy * self._head_amount, 0.0)

    def process(self, frame, ts):
        if not self.face:
            return None
        proc = self.app.procedure
        head, pos = self.head, self.pos
        if proc is not None and not proc.collecting_done:
            # look_at_override: subclasses simulam olhar fora do alvo.
            target = getattr(self, "look_at_override", None) or proc.current_target()
            head = self._head_for_prompt(proc)
            # Posição do rosto durante os procedimentos (padrão: a da calibração).
            pos = getattr(self, "proc_pos", EYE_POS)
        else:
            target = self.look_at
        # Boca: abre e fecha na etapa da boca; fora dela, ``self.mouth``.
        mouth = self.mouth
        if proc is not None and getattr(proc, "in_mouth_stage", False):
            self._mouth_t += 1
            mouth = 0.15 * abs(np.sin(self._mouth_t / 8.0))
        g = self.geom.gaze_vector(*target, pos)
        return make_observation(head=head, head_pos_cm=tuple(pos), with_transform=True,
                                gaze_vec=g, noise_px=self.noise_px, rng=self.rng,
                                timestamp_ms=ts, mouth_open=mouth, **USER)

    def close(self):
        pass


class FakeMouse:
    def __init__(self):
        self.enabled = False
        self.available = True
        self.moves = []
        self.clicks = 0

    def move(self, x, y):
        if self.enabled:
            self.moves.append((x, y))

    def click(self):
        self.clicks += 1


def make_app(tmp_path):
    cfg = AppConfig()
    cfg.calibration.save_path = tmp_path / "cal.json"
    cfg.validation.results_dir = tmp_path / "results"
    cfg.calibration.results_dir = tmp_path / "results"
    cfg.calibration.random_seed = 0   # ordem dos alvos reproduzível nos testes
    cfg.validation.screen_width_cm = 53.0
    cfg.calibration.screen_width_cm = 53.0     # monitor do usuário simulado
    from synthetic import SYNTH_HFOV_DEG
    cfg.camera.hfov_deg = SYNTH_HFOV_DEG        # câmera do usuário simulado
    cfg.validation.viewing_distance_cm = 60.0
    clock = Clock()
    tracker = FakeTracker()
    mouse = FakeMouse()
    app = EyeGazeApp(cfg, tracker, FakeSource(clock), SCREEN, mouse=mouse,
                     headless=True, clock=clock)
    tracker.app = app
    return app, tracker, mouse


def run_procedure(app, max_steps=4000, out_dir=None, snap_head=False):
    snapped = False
    for _ in range(max_steps):
        assert app.step()
        proc = app.procedure
        if (snap_head and out_dir and not snapped and proc is not None
                and getattr(proc, "head_prompt", None) is not None
                and proc.head_prompt[0] == "CIMA"):
            cv2.imwrite(os.path.join(out_dir, "calibracao_etapa_cabeca.png"),
                        app.last_target_view)
            snapped = True
        if proc is None or proc.collecting_done:
            break
    return app.procedure


def settle(app, n=45):
    for _ in range(n):
        app.step()


def err(app, target):
    return float(np.hypot(*(app.screen_point - np.array(target))))


def cursor_jitter(app, n=30):
    pts = []
    for _ in range(n):
        app.step()
        pts.append(app.screen_point.copy())
    return float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def test_end_to_end(tmp_path):
    out_dir = os.environ.get("EYEGAZE_TEST_OUT")
    app, tracker, mouse = make_app(tmp_path)
    settle(app, 5)
    assert not app.calibrated

    # Mouse não liga sem calibração.
    app.handle_key(ord("m"))
    assert not mouse.enabled

    # --- calibração (16 pontos + etapa da cabeça) ----------------------
    app.handle_key(ord(" "))
    proc = run_procedure(app, out_dir=out_dir, snap_head=True)
    assert proc.collecting_done and proc.succeeded
    assert len(proc.head_samples) > 60
    assert proc.stats["usa_etapa_cabeca"]
    assert max(r.error_px for r in proc.results) < 70
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, "calibracao_resultado.png"), app.last_target_view)
    app.handle_key(13)  # qualquer tecla fecha a tela de resultado
    assert app.procedure is None and app.calibrated
    assert app.cfg.calibration.save_path.exists()

    # --- cursor estável e certo, inclusive com a cabeça girada --------
    for target in [(960, 540), (400, 300), (1500, 850)]:
        for head in [(0, 0, 0), (15, 0, 0), (-12, 8, 3), (0, -10, 0)]:
            tracker.look_at, tracker.head = target, head
            settle(app)
            # Cabeça girada até 15°: até ~150 px de resíduo no usuário sintético
            # (compressão vertical da íris x forma tangente do modelo do olho;
            # ver test_pipeline.test_head_rotation_and_translation_are_compensated).
            assert err(app, target) < 160, (target, head, err(app, target))
            assert cursor_jitter(app) < 4.0, (target, head)
    # ... e com a cabeça deslocada (translação).
    tracker.head = (0, 0, 0)
    for pos in [(5.0, 10.0, 60.0), (0.0, 13.0, 52.0)]:
        tracker.pos = np.array(pos)
        tracker.look_at = (700, 450)
        settle(app)
        assert err(app, (700, 450)) < 80, (pos, err(app, (700, 450)))
    tracker.pos = EYE_POS.copy()

    # --- validação ---------------------------------------------------
    app.handle_key(ord("t"))
    proc = run_procedure(app)
    assert proc.summary["acuracia_media_px"] < 45
    assert proc.summary["acuracia_media_graus"] is not None
    assert proc.csv_path.exists()
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, "validacao_resultado.png"), app.last_target_view)
    app.handle_key(13)

    # --- recentralizar corrige um desvio residual ----------------------
    app.model.offset = app.model.offset + np.array([120.0, -90.0])  # desvio artificial
    tracker.look_at = (960, 540)
    settle(app)
    assert err(app, (960, 540)) > 100
    app.handle_key(ord("r"))
    run_procedure(app)
    settle(app, 60)  # a tela de resultado fecha sozinha
    assert app.procedure is None
    tracker.look_at = (960, 540)
    settle(app)
    assert err(app, (960, 540)) < 40

    # --- mouse + dwell -------------------------------------------------
    app.handle_key(ord("m"))
    assert mouse.enabled
    app.handle_key(ord("d"))
    assert app.dwell_enabled
    settle(app, 60)  # 2 s olhando para o mesmo lugar
    assert len(mouse.moves) >= 1
    assert mouse.clicks >= 1
    # Cursor parado durante a fixação: pouquíssimos comandos de movimento.
    n0 = len(mouse.moves)
    settle(app, 30)
    assert len(mouse.moves) - n0 <= 10

    # Um salto de olhar move o mouse.
    tracker.look_at = (300, 900)
    settle(app, 30)
    assert np.hypot(mouse.moves[-1][0] - 300, mouse.moves[-1][1] - 900) < 80

    # --- perda do rosto: cursor para ----------------------------------
    n = len(mouse.moves)
    tracker.face = False
    settle(app, 30)
    assert len(mouse.moves) == n
    assert app.screen_point is None
    tracker.face = True
    tracker.look_at = (1200, 300)
    settle(app, 20)
    assert len(mouse.moves) > n

    # --- teclas de visualização ---------------------------------------
    flags = {"l": "show_landmarks", "g": "show_gaze", "p": "show_head_pose",
             "v": "show_minimap", "h": "show_help", "f": "mirror", "3": "show_scene3d"}
    for key, attr in flags.items():
        before_flag = getattr(app, attr)
        app.handle_key(ord(key))
        assert getattr(app, attr) != before_flag
        app.handle_key(ord(key.upper()))  # maiúscula também funciona
        assert getattr(app, attr) == before_flag

    for attr in flags.values():
        setattr(app, attr, True)
    app.step()
    assert app.last_scene_view is not None and app.last_scene_view.ndim == 3
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, "camera_tudo_ligado.png"), app.last_camera_view)
        cv2.imwrite(os.path.join(out_dir, "vista_3d.png"), app.last_scene_view)
    for attr in flags.values():
        setattr(app, attr, False)
    app.step()
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, "camera_tudo_desligado.png"), app.last_camera_view)

    # --- ESC cancela procedimento; Q sai -------------------------------
    app.handle_key(ord("c"))
    settle(app, 10)
    assert app.procedure is not None
    app.handle_key(27)
    assert app.procedure is None
    assert app.handle_key(ord("q")) is False


def test_calibration_without_head_stage(tmp_path):
    """Usuário que não pode mover a cabeça: etapa desligada, parâmetros padrão."""
    app, tracker, _ = make_app(tmp_path)
    app.cfg.calibration.head_stage = False
    app.handle_key(ord(" "))
    proc = run_procedure(app)
    assert proc.succeeded and not proc.stats["usa_etapa_cabeca"]
    app.handle_key(13)
    tracker.look_at = (700, 400)
    settle(app)
    assert err(app, (700, 400)) < 60


def test_head_tilt_without_head_stage(tmp_path):
    """Sem a etapa da cabeça, o feixe ainda acompanha a rotação da cabeça pela
    geometria (R): inclinar a cabeça mantém o cursor no lugar."""
    app, tracker, _ = make_app(tmp_path)
    app.cfg.calibration.head_stage = False
    app.handle_key(ord(" "))
    proc = run_procedure(app)
    assert proc.succeeded and proc.stats["largura_conhecida"]
    app.handle_key(13)
    for head in [(0, -8, 0), (0, 6, 0)]:
        tracker.head, tracker.look_at = head, (900, 850)
        settle(app)
        assert err(app, (900, 850)) < 90, (head, err(app, (900, 850)))


def test_recenter_rejects_huge_offset(tmp_path):
    """Caso real: R logo depois de calibrar, com a postura muito diferente,
    aplicou -964 px no eixo y e o cursor não alcançava mais a parte de baixo
    da tela. Desvios acima de 25% da tela não são aplicados."""
    app, tracker, _ = make_app(tmp_path)
    app.cfg.calibration.head_stage = False
    app.handle_key(ord(" "))
    run_procedure(app)
    app.handle_key(13)
    app.model.offset = app.model.offset + np.array([0.0, 600.0])   # desvio enorme
    before = app.model.offset.copy()
    tracker.look_at = (960, 540)
    app.handle_key(ord("r"))
    proc = run_procedure(app)
    assert not proc.succeeded
    settle(app, 200)   # a tela de aviso fecha sozinha
    assert app.procedure is None
    assert np.allclose(app.model.offset, before)


def test_mouth_stage_end_to_end(tmp_path):
    """Usuário em que abrir a boca "desce" a íris (artefato do MediaPipe): com
    a etapa da boca, o cursor não se move quando a boca abre."""
    import synthetic
    app, tracker, _ = make_app(tmp_path)
    orig = synthetic.make_points

    def with_artifact(*a, **k):
        k.setdefault("mouth_coupling", 0.6)
        return orig(*a, **k)

    synthetic.make_points = with_artifact
    try:
        app.handle_key(ord(" "))
        proc = run_procedure(app)
        assert proc.succeeded and "boca_px_por_unidade" in proc.stats
        app.handle_key(13)
        tracker.look_at = (900, 600)
        settle(app)
        closed = app.screen_point.copy()
        tracker.mouth = 0.12
        settle(app)
        assert np.linalg.norm(app.screen_point - closed) < 60, (closed, app.screen_point)
    finally:
        synthetic.make_points = orig


def test_head_fine_adjustment_moves_cursor_from_gaze_point(tmp_path):
    """Modo A: olhando um alvo, um pequeno giro da cabeça desloca o cursor
    (precisão da cabeça); um salto do olhar zera o ajuste."""
    app, tracker, _ = make_app(tmp_path)
    app.handle_key(ord(" "))
    run_procedure(app)
    app.handle_key(13)
    tracker.look_at, tracker.head = (900, 500), (0, 0, 0)
    settle(app, 60)
    base = app.screen_point.copy()
    # Sem o modo: girar a cabeça olhando o mesmo alvo não move o cursor.
    tracker.head = (2.0, 0, 0)
    settle(app, 30)
    assert np.linalg.norm(app.screen_point - base) < 30
    tracker.head = (0, 0, 0)
    settle(app, 60)
    app.handle_key(ord("a"))
    assert app.fine_enabled
    settle(app, 30)
    base = app.screen_point.copy()
    tracker.head = (2.0, 0, 0)          # cabeça 2° para a direita
    settle(app, 30)
    dx = app.screen_point[0] - base[0]
    assert 20 < dx < 80, dx             # ~ (2 - 0,3) x 25 = 42 px
    tracker.head = (0, 1.5, 0)          # cabeça para cima
    settle(app, 30)
    assert app.screen_point[1] < base[1] - 10
    # Salto do olhar: o cursor vai ao novo ponto e o ajuste zera.
    tracker.head = (0, 0, 0)
    tracker.look_at = (300, 900)
    settle(app, 45)
    assert err(app, (300, 900)) < 80
    assert np.linalg.norm(app.fine.offset) < 1
    app.handle_key(ord("a"))
    assert not app.fine_enabled
