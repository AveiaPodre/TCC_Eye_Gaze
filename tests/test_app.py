"""
Teste de ponta a ponta do app com um "usuário simulado".

O FakeTracker substitui o MediaPipe: ele olha para o alvo que o app está
mostrando (ou para ``look_at``), segue as setas da etapa de movimento da
cabeça e devolve landmarks sintéticos coerentes com isso. O "usuário" tem os
erros típicos de webcam (ganho do olho diferente do assumido, íris achatada
na vertical, acoplamento com o pitch da cabeça) e ruído nos landmarks.

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
# Íris quase parada na vertical (como nos dados reais do MediaPipe) e
# pálpebras acompanhando o olhar: o sinal vertical vem das pálpebras.
USER = dict(rho_true=0.5, v_scale=0.2, pitch_coupling=0.4, lid_follow=0.7)


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
        g = self.geom.gaze_vector(*target, pos)
        return make_observation(head=head, head_pos_cm=tuple(pos), with_transform=True,
                                gaze_vec=g, noise_px=self.noise_px, rng=self.rng,
                                timestamp_ms=ts, **USER)

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

    # --- calibração (9 pontos + etapa de cabeça) -----------------------
    app.handle_key(ord(" "))
    proc = run_procedure(app, out_dir=out_dir, snap_head=True)
    assert proc.collecting_done and proc.succeeded
    assert len(proc.head_samples) > 60
    assert proc.stats["usa_etapa_cabeca"]
    assert max(r.error_px for r in proc.results) < 45
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
            assert err(app, target) < 80, (target, head, err(app, target))
            assert cursor_jitter(app) < 4.0, (target, head)
    # ... e com a cabeça deslocada (translação). O caso 8 cm mais perto fica
    # em ~72 px: a validação cruzada mede rotação (a etapa da cabeça), não
    # translação, e escolhe o grau 1, que extrapola um pouco pior a
    # distância (o grau 2 fixo dava ~63 px, mas era pior com rotação).
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
             "v": "show_minimap", "h": "show_help", "f": "mirror"}
    for key, attr in flags.items():
        before_flag = getattr(app, attr)
        app.handle_key(ord(key))
        assert getattr(app, attr) != before_flag
        app.handle_key(ord(key.upper()))  # maiúscula também funciona
        assert getattr(app, attr) == before_flag

    for attr in flags.values():
        setattr(app, attr, True)
    app.step()
    if out_dir:
        cv2.imwrite(os.path.join(out_dir, "camera_tudo_ligado.png"), app.last_camera_view)
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
