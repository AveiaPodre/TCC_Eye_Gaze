"""
Loop principal da aplicação.

    câmera -> FaceTracker                  1. landmarks
           -> GazeEstimator                2. âncora + pupila de cada olho (cm)
              + EyeballModel               2-3. centro do globo, feixes, intersecção
           -> VirtualScreen                4. vetor do olhar -> monitor virtual
           -> CalibrationModel             5. correção monitor virtual -> real
           -> FixationStabilizer -> mouse / dwell

Antes de calibrar, o mesmo pipeline roda com o globo anatômico típico e sem
correção (o ponto do monitor virtual é o cursor); a calibração troca os dois
pelos ajustados.

Os procedimentos de calibração, validação e recentralização rodam DENTRO
deste mesmo loop (procedures.py), então o rastreamento nunca para.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque

import cv2
import numpy as np

from . import scene3d
from . import visualization as vis
from .calibration import CalibrationModel, build_grid
from .config import AppConfig
from .filters import FixationStabilizer, HeadFineTuner, OneEuroFilter
from .gaze import GazeEstimator, GazeSample, direction_to_angles
from .geometry import VirtualScreen
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
                 clock=time.monotonic, recorder=None):
        self.cfg = cfg
        # Gravação da sessão (io_devices.SessionRecorder) ou reprodução de
        # uma sessão gravada (source com keys_for_frame: as teclas gravadas
        # são reaplicadas nos mesmos quadros).
        self.recorder = recorder
        self._calib_count = 0
        self.clock = clock  # injetável para testes
        self.tracker = tracker
        self.source = source
        self.screen_size = screen_size
        self.headless = headless
        self.mouse = mouse if mouse is not None else MouseController(screen_size)

        self.estimator = GazeEstimator(cfg.gaze)
        st = cfg.stabilizer
        self.stabilizer = FixationStabilizer(
            saccade_px=st.saccade_ratio * screen_size[0], confirm_frames=st.confirm_frames,
            window=st.window_frames, follow=st.follow, lock_px=st.lock_ratio * screen_size[0],
            lock_k=st.lock_k, shift_frames=st.shift_frames, settle_frames=st.settle_frames,
            noise_k=st.noise_k, lock_cap=st.lock_cap)
        # Suavização SÓ da exibição do ponto no monitor virtual (vista 3D):
        # a estimativa bruta treme de quadro a quadro; o cursor não usa isto.
        self._display_filter = OneEuroFilter(1.5, 0.01, 1.0)
        self.virtual_point_display: np.ndarray | None = None
        fc = cfg.fine
        self.fine_enabled = fc.enabled
        self.fine = HeadFineTuner(fc.gain_px_per_deg, fc.deadzone_deg, fc.max_px, fc.reset_deg)
        self.dwell = DwellClick(cfg.dwell.dwell_time_s, cfg.dwell.tolerance_px,
                                cfg.dwell.cooldown_s)
        self.dwell_enabled = cfg.dwell.enabled_at_start

        d = cfg.display
        self.show_landmarks = d.show_landmarks
        self.show_gaze = d.show_gaze
        self.show_head_pose = d.show_head_pose
        self.show_minimap = d.show_minimap
        self.show_help = d.show_help
        self.show_scene3d = d.show_scene3d
        self.view3d = scene3d.OrbitView()
        self.last_scene_view: np.ndarray | None = None
        self._scene_window_open = False
        self.mirror = cfg.camera.mirror_display

        # Monitor virtual (etapa 4) e o pipeline SEM calibração: globo
        # anatômico típico e nenhuma correção virtual -> real.
        cc = cfg.calibration
        self.screen_width_cm = cc.screen_width_cm or cc.assumed_screen_width_cm
        self.base_model = CalibrationModel(VirtualScreen(screen_size, self.screen_width_cm))
        self.model: CalibrationModel | None = CalibrationModel.load(cc.save_path, screen_size)
        if self.model is not None:
            self.estimator.eyeball = self.model.eyeball
            print(f"[Calibração] Carregada ({self.model.created_at}). "
                  "Pressione ESPAÇO para refazer.")
        self.calib_targets = build_grid(*screen_size, cfg.calibration.grid_cols,
                                        cfg.calibration.grid_rows, cfg.calibration.margin_ratio)

        self.procedure: TargetProcedure | None = None
        self.sample: GazeSample | None = None
        self.obs = None
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
        # Amostras recentes (vetor bruto) para correções por alvo conhecido.
        self.recent_raw: deque = deque(maxlen=cfg.adapt.window_frames)
        # Aviso de postura fora da calibração.
        self.posture_off = False
        self._posture_off_since: float | None = None

    # ------------------------------------------------------------ helpers
    def toast(self, message: str, seconds: float = 1.6):
        self._toast = (message, self.clock() + seconds)
        print(f"[UI] {message}")

    @property
    def calibrated(self) -> bool:
        return self.model is not None and self.model.is_fitted

    @property
    def active_model(self) -> CalibrationModel:
        """Pipeline em uso: o calibrado, ou o anatômico sem correção."""
        return self.model if self.calibrated else self.base_model

    def _map_to_screen(self, sample: GazeSample) -> np.ndarray | None:
        """Etapas 4-5: pixel do monitor real (None se o olhar não atinge a tela)."""
        p = self.active_model.predict(sample.raw)
        if not np.all(np.isfinite(p)):
            return None
        w, h = self.screen_size
        # Permite sair um pouco da tela antes de estabilizar (não "gruda" na borda).
        return np.array([np.clip(p[0], -0.1 * w, 1.1 * w), np.clip(p[1], -0.1 * h, 1.1 * h)])

    def _update_posture(self, sample: GazeSample, now: float):
        """Liga ``posture_off`` quando a postura fica fora da faixa calibrada
        por mais de ``sustain_s`` (um movimento rápido não dispara o aviso)."""
        if not self.calibrated:
            self.posture_off, self._posture_off_since = False, None
            return
        pc = self.cfg.posture
        dev = self.model.posture_deviation(sample.raw)
        off = bool(np.max(dev["ang"]) > pc.angle_tol_deg or dev["pos_cm"] > pc.pos_tol_cm)
        if not off:
            self.posture_off, self._posture_off_since = False, None
        elif self._posture_off_since is None:
            self._posture_off_since = now
        elif now - self._posture_off_since >= pc.sustain_s:
            self.posture_off = True

    def confirm_target(self, target) -> np.ndarray | None:
        """Informa que o usuário acabou de selecionar algo com centro
        conhecido (ex.: uma tecla do teclado virtual). Corrige a deriva do
        offset com as amostras recentes. Devolve a correção aplicada (px)."""
        if not self.calibrated or len(self.recent_raw) < 3:
            return None
        a = self.cfg.adapt
        step = self.model.correct_with_known_target(np.array(self.recent_raw), target,
                                                    a.gain, a.max_step_px)
        self.recent_raw.clear()
        return step

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

        point = (self._map_to_screen(self.sample)
                 if self.sample is not None and self.sample.valid else None)
        if point is not None:
            self.recent_raw.append(self.sample.raw)
            self._update_posture(self.sample, now)
            self.screen_point_raw = point
            v = self.active_model.virtual_px(self.sample.raw)[0]
            if np.all(np.isfinite(v)):
                self.virtual_point_display = self._display_filter(v, now)
            w, h = self.screen_size
            stab = self.stabilizer(self.screen_point_raw)
            if self.fine_enabled and self.calibrated:
                hy, hp, _ = self.sample.head.euler_deg
                stab = self.fine(stab, self.stabilizer.fixation_id, (hy, hp), now,
                                 settling=self.stabilizer.settling)
            self.screen_point = np.array([np.clip(stab[0], 0, w - 1), np.clip(stab[1], 0, h - 1)])
            self.trail.append(self.screen_point.copy())
        elif not face_recent:
            # Rosto perdido: zera os filtros para não "puxar" a posição antiga.
            self.stabilizer.reset()
            self.fine.reset()
            self.recent_raw.clear()
            self.screen_point_raw = None
            self.virtual_point_display = None
            self._display_filter.reset()
            self.screen_point = None
        # Durante uma piscada (ou olhar fora da tela), mantém a última posição.

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
    def start_calibration(self, manual: bool | None = None):
        cc = self.cfg.calibration
        if manual is not None:
            cc = dataclasses.replace(cc, manual=manual)
        if cc.random_seed is not None:
            # Semente fixa (gravação/reprodução): ordem diferente a cada
            # calibração da sessão, mas reproduzível.
            cc = dataclasses.replace(cc, random_seed=cc.random_seed + self._calib_count)
        self._calib_count += 1
        self.procedure = CalibrationProcedure(self.screen_size, cc)
        self.toast("Calibracao manual: olhe e aperte ESPACO" if cc.manual
                   else "Calibracao iniciada")

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
            # Feixes e vetor do olhar de cada quadro passam a usar o globo ajustado.
            self.estimator.eyeball = self.model.eyeball
            self.stabilizer.reset()
            self.model.save(self.cfg.calibration.save_path)
            self.toast("Calibracao salva")
        elif isinstance(proc, RecenterProcedure) and proc.succeeded:
            # A postura nova passa a ser a referência do aviso.
            self.model.save(self.cfg.calibration.save_path)
            self.toast("Recentralizado")
        elif isinstance(proc, RecenterProcedure) and proc.collecting_done:
            self.toast("Desvio grande demais: recalibre (ESPACO)", 4.0)
        self.posture_off, self._posture_off_since = False, None

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
            elif key in (32, 13) and getattr(self.procedure, "awaiting_confirmation", False):
                # Calibração manual: confirma que está olhando para o alvo.
                self.procedure.confirm(self.clock())
            elif self.procedure.collecting_done:
                self.procedure.acknowledge()
                self._finish_procedure()
            return True

        ch = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27 or ch == "q":
            return False
        if ch == " ":
            self.start_calibration(manual=False)
        elif ch == "c":
            self.start_calibration(manual=True)
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
        elif ch == "3":
            self.show_scene3d = not self.show_scene3d
            if not self.show_scene3d:
                self._close_scene_window()
            self.toast(f"Vista 3D: {'ON' if self.show_scene3d else 'OFF'}")
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "a":
            self.fine_enabled = not self.fine_enabled
            self.fine.reset()
            self.toast(f"Ajuste fino pela cabeca: {'ON' if self.fine_enabled else 'OFF'}")
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
        if self.fine_enabled:
            ox, oy = self.fine.offset
            lines.append((f"Ajuste fino pela cabeca: ON (A)  desvio ({ox:+.0f}, {oy:+.0f}) px", GREEN))
        else:
            lines.append(("Ajuste fino pela cabeca: OFF (A)", GRAY))
        if self.posture_off:
            lines.append(("Postura fora da calibracao: R para recentralizar", YELLOW))
        gaze = self.sample.gaze if self.sample is not None else None
        if self.show_gaze and gaze is not None:
            yaw, pitch = direction_to_angles(gaze.direction)
            tag = "globo calibrado" if self.calibrated else "globo anatomico, sem calibracao"
            lines.append((f"Olhar yaw {yaw:+5.1f}  pitch {pitch:+5.1f} graus ({tag})", YELLOW))
            if gaze.vergence is not None:
                lines.append((f"Interseccao dos feixes a {gaze.vergence[2]:+.0f} cm do plano "
                              "da tela", GRAY))
            else:
                lines.append(("Feixes nao convergem: usando a bissetriz", GRAY))
        if self.sample is not None and (self.show_head_pose or self.calibrated):
            hy, hp, hr = self.sample.head.euler_deg
            if self.calibrated:
                # Em relação à postura da calibração (a referência "de frente
                # para a tela"); o ângulo absoluto do MediaPipe tem um viés
                # de ~30° para cima.
                ry, rp = hy - self.model.head_ref[0], hp - self.model.head_ref[1]
                lines.append((f"Cabeca (rel. calibracao) yaw {ry:+5.1f}  pitch {rp:+5.1f}", GRAY))
            if self.show_head_pose:
                lines.append((f"Cabeca (MediaPipe) yaw {hy:+5.1f}  pitch {hp:+5.1f}  roll {hr:+5.1f}"
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

    def render_scene_view(self) -> np.ndarray:
        return scene3d.render_scene(
            self.view3d, self.active_model.screen, self.sample, self.obs,
            virtual_point=self.virtual_point_display, cursor=self.screen_point,
            raw_point=self.screen_point_raw, trail=list(self.trail),
            targets=self.calib_targets,
            assumed_width=self.cfg.calibration.screen_width_cm is None,
            head_ref=self.model.head_ref if self.calibrated else None,
            calibrated=self.calibrated)

    def _open_scene_window(self):
        if self.headless or self._scene_window_open:
            return
        name = self.cfg.display.scene3d_window_name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(name, self.view3d.on_mouse)
        self._scene_window_open = True

    def _close_scene_window(self):
        if self._scene_window_open and not self.headless:
            cv2.destroyWindow(self.cfg.display.scene3d_window_name)
        self._scene_window_open = False
        self.last_scene_view = None

    # --------------------------------------------------------------- loop
    def step(self) -> bool:
        frame, ts = self.source.read()
        if frame is None:
            return False
        if self.recorder is not None:
            self.recorder.write(frame, ts)
        now = self.clock()
        self.process_frame(frame, ts, now)

        self.last_camera_view = self.render_camera_view(frame, now)
        if self.procedure is not None:
            self.last_target_view = self.procedure.render(now)
        if self.show_scene3d:
            self.last_scene_view = self.render_scene_view()

        keys = []
        if not self.headless:
            cv2.imshow(self.cfg.display.window_name, self.last_camera_view)
            if self.last_scene_view is not None and self.show_scene3d:
                self._open_scene_window()
                cv2.imshow(self.cfg.display.scene3d_window_name, self.last_scene_view)
            if self.last_target_view is not None:
                self._open_target_window()
                cv2.imshow(self.cfg.display.target_window_name, self.last_target_view)
            key = cv2.waitKey(1)
            if key != -1 and (key & 0xFF) != 255:
                keys.append(key & 0xFF)
        if hasattr(self.source, "keys_for_frame"):
            keys.extend(self.source.keys_for_frame(self.source.frame_index))
        return all(self.press(key) for key in keys)

    def press(self, key: int) -> bool:
        """Tecla do usuário: grava (se houver gravação) e executa.
        Devolve False para encerrar a aplicação."""
        if self.recorder is not None:
            self.recorder.log_key(key)
        return self.handle_key(key)

    def run(self):
        if not self.headless:
            cv2.namedWindow(self.cfg.display.window_name, cv2.WINDOW_NORMAL)
        print("Pronto. ESPACO = calibrar | H = ajuda | Q = sair")
        try:
            while self.step():
                pass
        finally:
            self.source.release()
            self.tracker.close()
            if self.recorder is not None:
                self.recorder.close()
            if not self.headless:
                cv2.destroyAllWindows()
