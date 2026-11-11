"""
Loop principal da aplicação.

    câmera -> FaceTracker (landmarks + pose)
           -> GazeEstimator (rotação da cabeça, posição 3D do rosto, íris)
           -> CalibrationModel (raio do olhar -> plano da tela -> pixel)
           -> FixationStabilizer -> mouse / dwell

Os procedimentos de calibração, validação e recentralização rodam DENTRO
deste mesmo loop (procedures.py), então o rastreamento nunca para.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from . import visualization as vis
from .calibration import CalibrationModel, build_grid, uncalibrated_mapping
from .config import AppConfig
from .filters import FixationStabilizer, OneEuroFilter
from .gaze import GazeEstimator, GazeSample
from .io_devices import DwellClick, MouseController
from .procedures import (CalibrationProcedure, RecenterProcedure, TargetProcedure,
                         ValidationProcedure)

GREEN = (80, 230, 80)
RED = (80, 80, 255)
YELLOW = (0, 220, 255)
GRAY = (200, 200, 200)


class EyeGazeApp:
    def __init__(self, cfg: AppConfig, tracker, source, screen_size: tuple[int, int],
                 mouse: MouseController | None = None, headless: bool = False,
                 clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock  # injetável para testes
        self.tracker = tracker
        self.source = source
        self.screen_size = screen_size
        self.headless = headless
        self.mouse = mouse if mouse is not None else MouseController(screen_size)

        self.estimator = GazeEstimator(cfg.gaze)
        self.filter = OneEuroFilter(cfg.filter.min_cutoff, cfg.filter.beta, cfg.filter.d_cutoff)
        st = cfg.stabilizer
        self.stabilizer = FixationStabilizer(
            saccade_px=st.saccade_ratio * screen_size[0], confirm_frames=st.confirm_frames,
            window=st.window_frames, follow=st.follow, deadzone_px=st.deadzone_px,
            noise_k=st.noise_k)
        self.dwell = DwellClick(cfg.dwell.dwell_time_s, cfg.dwell.tolerance_px,
                                cfg.dwell.cooldown_s)
        self.dwell_enabled = cfg.dwell.enabled_at_start

        d = cfg.display
        self.show_landmarks = d.show_landmarks
        self.show_gaze = d.show_gaze
        self.show_head_pose = d.show_head_pose
        self.show_minimap = d.show_minimap
        self.show_help = d.show_help
        self.mirror = cfg.camera.mirror_display

        self.model: CalibrationModel | None = CalibrationModel.load(
            cfg.calibration.save_path, screen_size)
        if self.model is not None:
            print(f"[Calibração] Carregada ({self.model.created_at}). "
                  "Pressione ESPAÇO para refazer.")
        self.calib_targets = build_grid(*screen_size, cfg.calibration.grid_cols,
                                        cfg.calibration.grid_rows, cfg.calibration.margin_ratio)

        self.procedure: TargetProcedure | None = None
        self.sample: GazeSample | None = None
        self.obs = None
        self.features_filtered: np.ndarray | None = None
        self.screen_point: np.ndarray | None = None
        self.screen_point_raw: np.ndarray | None = None
        self.trail: deque = deque(maxlen=25)
        self.last_face_time = -1e9
        self.fps = 0.0
        self._last_frame_time: float | None = None
        self._toast: tuple[str, float] | None = None
        self.last_camera_view: np.ndarray | None = None
        self.last_target_view: np.ndarray | None = None
        self._target_window_open = False
        self._last_mouse: np.ndarray | None = None

    # ------------------------------------------------------------ helpers
    def toast(self, message: str, seconds: float = 1.6):
        self._toast = (message, self.clock() + seconds)
        print(f"[UI] {message}")

    @property
    def calibrated(self) -> bool:
        return self.model is not None and self.model.is_fitted

    def _map_to_screen(self, sample: GazeSample, feats_filtered: np.ndarray) -> np.ndarray:
        if self.calibrated:
            p = self.model.predict(sample.raw)
        else:
            c = self.cfg.calibration
            p = uncalibrated_mapping(feats_filtered, self.screen_size,
                                     c.uncalibrated_half_range_yaw_deg,
                                     c.uncalibrated_half_range_pitch_deg)
        w, h = self.screen_size
        # Permite sair um pouco da tela antes de estabilizar (não "gruda" na borda).
        return np.array([np.clip(p[0], -0.1 * w, 1.1 * w), np.clip(p[1], -0.1 * h, 1.1 * h)])

    # --------------------------------------------------------- pipeline
    def process_frame(self, frame: np.ndarray, timestamp_ms: int, now: float):
        if self._last_frame_time is not None:
            dt = max(now - self._last_frame_time, 1e-3)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        self._last_frame_time = now

        self.obs = self.tracker.process(frame, timestamp_ms)
        self.sample = self.estimator.estimate(self.obs) if self.obs is not None else None

        if self.obs is not None:
            self.last_face_time = now
        face_recent = now - self.last_face_time <= self.cfg.mouse.face_lost_timeout_s

        if self.sample is not None and self.sample.valid:
            self.features_filtered = self.filter(self.sample.features(), now)
            self.screen_point_raw = self._map_to_screen(self.sample, self.features_filtered)
            w, h = self.screen_size
            stab = self.stabilizer(self.screen_point_raw)
            self.screen_point = np.array([np.clip(stab[0], 0, w - 1), np.clip(stab[1], 0, h - 1)])
            self.trail.append(self.screen_point.copy())
        elif not face_recent:
            # Rosto perdido: zera os filtros para não "puxar" a posição antiga.
            self.filter.reset()
            self.stabilizer.reset()
            self.features_filtered = None
            self.screen_point_raw = None
            self.screen_point = None
        # Durante uma piscada, mantém a última posição.

        if self.procedure is not None:
            self.procedure.update(self.sample, now)
            if self.procedure.finished:
                self._finish_procedure()
            return

        # Controle do mouse: só com calibração e rosto presente NESTE frame.
        # (Se o rosto some, o cursor fica parado onde está.)
        can_control = (self.calibrated and self.obs is not None
                       and self.screen_point is not None)
        if self.mouse.enabled and can_control:
            if (self._last_mouse is None
                    or np.max(np.abs(self.screen_point - self._last_mouse)) >= 1.0):
                self.mouse.move(*self.screen_point)
                self._last_mouse = self.screen_point.copy()
            if self.dwell_enabled and self.dwell.update(self.screen_point, now):
                self.mouse.click()
                self.toast("Clique (dwell)", 0.8)
        else:
            self.dwell.reset()

    # ----------------------------------------------------- procedimentos
    def start_calibration(self):
        self.procedure = CalibrationProcedure(self.screen_size, self.cfg.calibration)
        self.toast("Calibracao iniciada")

    def start_validation(self):
        if not self.calibrated:
            self.toast("Calibre primeiro (ESPACO)")
            return
        self.procedure = ValidationProcedure(self.screen_size, self.model,
                                             self.cfg.validation, self.cfg.calibration)
        self.toast("Validacao iniciada")

    def start_recenter(self):
        if not self.calibrated:
            self.toast("Calibre primeiro (ESPACO)")
            return
        self.procedure = RecenterProcedure(self.screen_size, self.model, self.cfg.calibration)

    def _finish_procedure(self):
        proc = self.procedure
        self.procedure = None
        self._close_target_window()
        if isinstance(proc, CalibrationProcedure) and proc.succeeded:
            self.model = proc.model
            self.stabilizer.reset()
            self.model.save(self.cfg.calibration.save_path)
            self.toast("Calibracao salva")
        elif isinstance(proc, RecenterProcedure) and proc.succeeded:
            self.model.save(self.cfg.calibration.save_path)
            self.toast("Recentralizado")

    def cancel_procedure(self):
        if self.procedure is not None:
            if self.procedure.collecting_done:
                # Já terminou de coletar: ESC só fecha a tela de resultado.
                self._finish_procedure()
                return
            self.procedure = None
            self._close_target_window()
            self.toast("Procedimento cancelado")

    # ------------------------------------------------------------ teclas
    def handle_key(self, key: int) -> bool:
        """Processa uma tecla; devolve False para encerrar a aplicação."""
        if key in (-1, 255):
            return True

        if self.procedure is not None:
            if key == 27:
                self.cancel_procedure()
            elif self.procedure.collecting_done:
                self.procedure.acknowledge()
                self._finish_procedure()
            return True

        ch = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or ch == "q":
            return False
        if ch in (" ", "c"):
            self.start_calibration()
        elif ch == "t":
            self.start_validation()
        elif ch == "r":
            self.start_recenter()
        elif ch == "m":
            if not self.calibrated:
                self.toast("Calibre antes de ligar o mouse")
            elif not self.mouse.available:
                self.toast("Mouse indisponivel (instale pynput)")
            else:
                self.mouse.enabled = not self.mouse.enabled
                self.toast(f"Mouse: {'ON' if self.mouse.enabled else 'OFF'}")
        elif ch == "d":
            self.dwell_enabled = not self.dwell_enabled
            self.dwell.reset()
            self.toast(f"Dwell-click: {'ON' if self.dwell_enabled else 'OFF'}")
        elif ch == "l":
            self.show_landmarks = not self.show_landmarks
            self.toast(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
        elif ch == "g":
            self.show_gaze = not self.show_gaze
            self.toast(f"Visualizacao do olhar: {'ON' if self.show_gaze else 'OFF'}")
        elif ch == "p":
            self.show_head_pose = not self.show_head_pose
            self.toast(f"Pose da cabeca: {'ON' if self.show_head_pose else 'OFF'}")
        elif ch == "v":
            self.show_minimap = not self.show_minimap
            self.toast(f"Minimapa: {'ON' if self.show_minimap else 'OFF'}")
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "f":
            self.mirror = not self.mirror
            self.toast(f"Espelhar imagem: {'ON' if self.mirror else 'OFF'}")
        return True

    # ----------------------------------------------------------- desenho
    def render_camera_view(self, frame: np.ndarray, now: float) -> np.ndarray:
        img = frame.copy()
        if self.obs is not None and self.show_landmarks:
            vis.draw_landmarks(img, self.obs)
        if self.sample is not None:
            if self.show_head_pose:
                vis.draw_head_pose(img, self.sample)
            if self.show_gaze:
                vis.draw_gaze(img, self.sample)
        if self.mirror:
            img = cv2.flip(img, 1)

        face_ok = self.obs is not None
        lines = [(f"FPS {self.fps:4.1f}", GRAY),
                 ("Rosto: detectado" if face_ok else "Rosto: NAO detectado",
                  GREEN if face_ok else RED)]
        if self.sample is not None and self.sample.blink:
            lines.append(("Piscada", YELLOW))
        lines.append(("Calibrado: sim" if self.calibrated else
                      "Calibrado: NAO (ESPACO para calibrar)",
                      GREEN if self.calibrated else YELLOW))
        if self.mouse.enabled:
            face_recent = now - self.last_face_time <= self.cfg.mouse.face_lost_timeout_s
            lines.append(("Mouse: ON" if face_recent else "Mouse: PAUSADO (sem rosto)",
                          GREEN if face_recent else RED))
        else:
            lines.append(("Mouse: OFF (M)", GRAY))
        lines.append((f"Dwell-click: {'ON' if self.dwell_enabled else 'OFF'} (D)",
                      GREEN if self.dwell_enabled else GRAY))
        if self.show_gaze and self.features_filtered is not None:
            yaw, pitch = self.features_filtered
            lines.append((f"Olhar  yaw {yaw:+5.1f}  pitch {pitch:+5.1f} graus", YELLOW))
        if self.show_head_pose and self.sample is not None:
            hy, hp, hr = self.sample.head.euler_deg
            lines.append((f"Cabeca yaw {hy:+5.1f}  pitch {hp:+5.1f}  roll {hr:+5.1f}"
                          f"  [{self.sample.extras.get('pose_source', '?')}]", GRAY))
        if self.screen_point is not None:
            lines.append((f"Tela ({self.screen_point[0]:.0f}, {self.screen_point[1]:.0f})"
                          f"  ruido ~{self.stabilizer.noise_px:.0f}px", GRAY))
        vis.draw_status(img, lines)

        if self.show_minimap:
            vis.draw_minimap(img, self.screen_size, self.screen_point, self.screen_point_raw,
                             list(self.trail), self.calib_targets,
                             self.dwell.progress if self.dwell_enabled else 0.0,
                             self.calibrated)
        if self.show_help:
            vis.draw_help(img)
        if self._toast is not None:
            msg, until = self._toast
            if now < until:
                vis.draw_toast(img, msg)
            else:
                self._toast = None
        return img

    # ------------------------------------------------------------ janelas
    def _open_target_window(self):
        if self.headless or self._target_window_open:
            return
        name = self.cfg.display.target_window_name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        self._target_window_open = True

    def _close_target_window(self):
        if self._target_window_open and not self.headless:
            cv2.destroyWindow(self.cfg.display.target_window_name)
        self._target_window_open = False
        self.last_target_view = None

    # --------------------------------------------------------------- loop
    def step(self) -> bool:
        frame, ts = self.source.read()
        if frame is None:
            return False
        now = self.clock()
        self.process_frame(frame, ts, now)

        self.last_camera_view = self.render_camera_view(frame, now)
        if self.procedure is not None:
            self.last_target_view = self.procedure.render(now)

        if self.headless:
            return True

        cv2.imshow(self.cfg.display.window_name, self.last_camera_view)
        if self.last_target_view is not None:
            self._open_target_window()
            cv2.imshow(self.cfg.display.target_window_name, self.last_target_view)
        key = cv2.waitKey(1)
        return self.handle_key(key if key == -1 else key & 0xFF)

    def run(self):
        cv2.namedWindow(self.cfg.display.window_name, cv2.WINDOW_NORMAL)
        print("Pronto. ESPACO = calibrar | H = ajuda | Q = sair")
        try:
            while self.step():
                pass
        finally:
            self.source.release()
            self.tracker.close()
            cv2.destroyAllWindows()
