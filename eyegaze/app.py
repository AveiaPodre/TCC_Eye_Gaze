"""
Loop principal da aplicação.

    câmera -> FaceTracker                  1. landmarks
           -> GazeEstimator                2. âncora + pupila de cada olho (cm)
              + EyeballModel               2-3. centro do globo, feixes, intersecção
           -> VirtualScreen                4. vetor do olhar -> monitor virtual
           -> CalibrationModel             5. correção monitor virtual -> real
           -> GazeFollower (filters.py)    estabilizador do cursor
           -> CursorControl (control.py)   mouse, clique, precisão, barra pelo olhar

Antes de calibrar, o mesmo pipeline roda com o globo anatômico típico e sem
correção (o ponto do monitor virtual é o cursor); a calibração troca os dois
pelos ajustados.

Os procedimentos (calibração, conferência, validação, recentralização)
rodam DENTRO deste mesmo loop (procedures.py), então o rastreamento nunca
para.

Uso sem teclado: depois da calibração o mouse liga sozinho e a barra de
controle pelo olhar (gaze_bar.py) aparece na borda esquerda; ela liga e
desliga o mouse, o clique por permanência e o modo de precisão, e
recentraliza ou recalibra. O teclado fica para o operador (calibração
manual, pesquisa, visualizações).
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque

import cv2
import numpy as np

from . import geometry, scene3d
from . import visualization as vis
from .calibration import CalibrationModel, build_grid
from .config import AppConfig
from .control import CursorControl
from .filters import OneEuroFilter, make_stabilizer
from .gaze_bar import make_view
from .gaze import GazeEstimator, GazeSample, direction_to_angles
from .geometry import VirtualScreen
from .io_devices import MouseController
from .procedures import (CalibrationProcedure, HeadStageProcedure, RecenterProcedure,
                         TargetProcedure, ValidationProcedure)

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
        # Revalidação periódica: instante do último procedimento que mediu ou
        # corrigiu o desvio (calibração, recentralização, validação).
        self._last_check: float | None = None
        self.clock = clock  # injetável para testes
        self.tracker = tracker
        self.source = source
        self.screen_size = screen_size
        self.headless = headless
        self.mouse = mouse if mouse is not None else MouseController(screen_size)

        geometry.set_camera_hfov(cfg.camera.hfov_deg)
        self.estimator = GazeEstimator(cfg.gaze)
        st = cfg.stabilizer
        self.stabilizer = make_stabilizer(st, screen_size[0])
        # Suavização SÓ da exibição do ponto no monitor virtual (vista 3D):
        # a estimativa bruta treme de quadro a quadro; o cursor não usa isto.
        self._display_filter = OneEuroFilter(1.5, 0.01, 1.0)
        self.virtual_point_display: np.ndarray | None = None
        # Uso do cursor: mouse, clique por permanência, modo de precisão,
        # correção contínua e barra de controle pelo olhar (control.py).
        self.control = CursorControl(cfg, screen_size, self.mouse, self.toast)
        self.bar_view = None      # janela da barra (aberta em run())

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
        if self.model is not None and cc.recenter_on_start:
            # Calibração de outra sessão: reancora o desvio antes de usar.
            self.procedure = RecenterProcedure(screen_size, self.model, cc, kind="inicio")
            print("[Calibração] Recentralizando: olhe para os 4 pontos.")
        self.sample: GazeSample | None = None
        self.obs = None
        self.screen_point: np.ndarray | None = None
        self.screen_point_raw: np.ndarray | None = None
        self.trail: deque = deque(maxlen=25)
        self.last_face_time = -1e9
        self.fps = 0.0
        self._last_frame_time: float | None = None
        self._toast: tuple[str, float] | None = None
        # Prévia da câmera mais leve durante o uso do mouse (ver step()).
        self._preview_count = 0
        # Tempo de processamento de cada quadro (ms), gravado com a sessão.
        self.proc_ms = 0.0
        self.last_camera_view: np.ndarray | None = None
        self.last_target_view: np.ndarray | None = None
        self._target_window_open = False
        # Aviso de postura fora da calibração.
        self.posture_off = False
        self._posture_off_since: float | None = None

    # ------------------------------------------------------------- uso
    def confirm_target(self, target) -> np.ndarray | None:
        """Informa que o usuário acabou de selecionar algo com centro
        conhecido (ex.: uma tecla do teclado virtual). Corrige a deriva do
        offset com as amostras recentes. Devolve a correção aplicada (px)."""
        return self.control.confirm_target(self.model if self.calibrated else None, target)

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
            self.control.recent_raw.append(self.sample.raw)
            self._update_posture(self.sample, now)
            self.screen_point_raw = point
            v = self.active_model.virtual_px(self.sample.raw)[0]
            if np.all(np.isfinite(v)):
                self.virtual_point_display = self._display_filter(v, now)
            w, h = self.screen_size
            stab = self.stabilizer(self.screen_point_raw)
            if self.control.fine_enabled and self.calibrated:
                stab = self.control.apply_fine(stab, self.sample, self.stabilizer, now)
            self.screen_point = np.array([np.clip(stab[0], 0, w - 1), np.clip(stab[1], 0, h - 1)])
            self.trail.append(self.screen_point.copy())
        elif not face_recent:
            # Rosto perdido: zera os filtros para não "puxar" a posição antiga.
            self.stabilizer.reset()
            self.control.release_fine()
            self.control.recent_raw.clear()
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
        if self._revalidation_due(now):
            self.start_validation("periodica")
            return

        # Controle do mouse: só com calibração e rosto presente NESTE frame.
        # (Se o rosto some, o cursor fica parado onde está.)
        can_control = (self.calibrated and self.obs is not None
                       and self.screen_point is not None)
        action = self.control.act(self.screen_point, self.model if self.calibrated else None,
                                  can_control, now)
        if action == "recentralizar":
            self.start_recenter()
        elif action == "recalibrar":
            self.start_calibration(manual=False)

    # ----------------------------------------------------- procedimentos
    def _revalidation_due(self, now: float) -> bool:
        """Revalidação periódica (ValidationConfig.revalidate_every_min): a
        cada N minutos de uso, mede a deriva e corrige o desvio constante.
        Só com calibração e o rosto visível."""
        every = self.cfg.validation.revalidate_every_min
        if not every or every <= 0 or not self.calibrated:
            return False
        if self._last_check is None:
            self._last_check = now
            return False
        if now - self._last_check < 60.0 * every or self.obs is None:
            return False
        self._last_check = now     # cancelar (ESC) não repete na hora
        return True

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

    def start_validation(self, kind: str = "reteste"):
        """kind: "imediata" (logo após calibrar) ou "reteste" (tecla T)."""
        if not self.calibrated:
            self.toast("Calibre primeiro (ESPACO)")
            return
        self.procedure = ValidationProcedure(self.screen_size, self.model,
                                             self.cfg.validation, self.cfg.calibration, kind)
        self.toast({"imediata": "Validacao imediata",
                    "periodica": "Revalidacao periodica: olhe para os pontos"}.get(kind, "Reteste"))

    def start_head_test(self):
        """Teste da cabeça (tecla K ou ``--cabeca``): erro com a cabeça em
        poses novas, olhando o centro."""
        if not self.calibrated:
            self.toast("Calibre primeiro (ESPACO)")
            return
        self.procedure = HeadStageProcedure(self.screen_size, self.model, self.cfg.calibration)
        self.toast("Teste da cabeca: olhe o ponto e siga as setas com a cabeca")

    def start_recenter(self, kind: str = "manual"):
        if not self.calibrated:
            self.toast("Calibre primeiro (ESPACO)")
            return
        self.procedure = RecenterProcedure(self.screen_size, self.model, self.cfg.calibration,
                                           kind=kind)
        if kind == "pos_calibracao":
            self.toast("Conferindo a calibracao: olhe para os 4 pontos")

    def _finish_procedure(self):
        proc = self.procedure
        self.procedure = None
        if isinstance(proc, (CalibrationProcedure, RecenterProcedure, ValidationProcedure)) \
                and proc.succeeded:
            self._last_check = self.clock()
        self._close_target_window()
        if isinstance(proc, CalibrationProcedure) and proc.succeeded:
            self.model = proc.model
            # Feixes e vetor do olhar de cada quadro passam a usar o globo ajustado.
            self.estimator.eyeball = self.model.eyeball
            self.stabilizer.reset()
            self.model.save(self.cfg.calibration.save_path)
            self.control.online_corrections, self.control.model_dirty = 0, False
            self.toast("Calibracao salva")
            self._auto_mouse()
            # Conferência (como a "drift correction" do EyeLink): 4 pontos
            # logo depois de calibrar medem e corrigem o desvio constante que
            # aparece nos primeiros segundos de uso (docs/metodologia.md,
            # seção 7.1). Depois dela, no modo pesquisa, a validação imediata.
            if self.cfg.calibration.check_after_calibration:
                self.start_recenter("pos_calibracao")
                return
            if self._after_calibration_chain():
                return
        elif isinstance(proc, RecenterProcedure) and proc.kind == "pos_calibracao":
            if proc.succeeded:
                self.model.save(self.cfg.calibration.save_path)
                self.control.model_dirty = False
            elif proc.collecting_done:
                self.toast("Desvio grande demais: recalibre (ESPACO)", 4.0)
            if self._after_calibration_chain():
                return
        elif isinstance(proc, RecenterProcedure) and proc.succeeded:
            # A postura nova passa a ser a referência do aviso.
            self.model.save(self.cfg.calibration.save_path)
            self.control.model_dirty = False
            self.toast("Recentralizado")
            self._auto_mouse()
        elif isinstance(proc, ValidationProcedure) and proc.applied_offset is not None:
            self.model.save(self.cfg.calibration.save_path)
            self.control.model_dirty = False
        elif (isinstance(proc, ValidationProcedure) and proc.kind == "imediata"
              and proc.collecting_done and self.cfg.calibration.head_stage):
            self.start_head_test()
            return
        elif isinstance(proc, HeadStageProcedure) and proc.succeeded:
            self.model.save(self.cfg.calibration.save_path)   # resultado em model.info
        elif isinstance(proc, RecenterProcedure) and proc.collecting_done:
            self.toast("Desvio grande demais: recalibre (ESPACO)", 4.0)
        self.posture_off, self._posture_off_since = False, None

    def _after_calibration_chain(self) -> bool:
        """Modo pesquisa: validação imediata e, se pedido (--cabeca), o teste
        da cabeça DEPOIS dela (o efeito passageiro dos giros contaminava a
        validação; metodologia, seção 5.11). True se começou um procedimento."""
        if self.cfg.validation.after_calibration:
            self.start_validation("imediata")
            return True
        if self.cfg.calibration.head_stage:
            self.start_head_test()
            return True
        return False

    def _auto_mouse(self):
        """Depois de calibrar ou recentralizar, o mouse liga sozinho (o uso
        não depende do teclado; a barra pausa e religa)."""
        if (self.cfg.mouse.enable_after_calibration and self.calibrated
                and not self.mouse.enabled and self.mouse.available):
            self.control.set_mouse(True, True)

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
            elif (key == 32 and self.procedure.collecting_done
                  and isinstance(self.procedure, CalibrationProcedure)
                  and self.procedure.redo_recommended):
                # Calibração abaixo do critério: refaz na hora, sem salvar.
                manual = self.procedure.manual
                self.procedure = None
                self._close_target_window()
                self.start_calibration(manual=manual)
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
        elif ch == "k":
            self.start_head_test()
        elif ch == "m":
            self.control.set_mouse(not self.mouse.enabled, self.calibrated)
        elif ch == "d":
            self.control.set_dwell(not self.control.dwell_enabled)
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
            self.control.set_fine(not self.control.fine_enabled)
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
        cam_fps = getattr(self.source, "measured_fps", None)
        fps_txt = f"FPS {self.fps:4.1f}" + (f" (camera {cam_fps:4.1f})" if cam_fps else "")
        low_fps = cam_fps is not None and cam_fps < 25
        lines = [(fps_txt + ("  BAIXO: luz/exposicao?" if low_fps else ""),
                  YELLOW if low_fps else GRAY),
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
            lines.append(("Mouse: OFF (M ou barra)", GRAY))
        lines.append((f"Clique pelo olhar: {'ON' if self.control.dwell_enabled else 'OFF'} (D ou barra)",
                      GREEN if self.control.dwell_enabled else GRAY))
        if self.control.fine_enabled:
            ox, oy = self.control.fine.offset
            lines.append((f"Modo de precisao: ON (A ou barra)  ajuste ({ox:+.0f}, {oy:+.0f}) px",
                          GREEN))
        else:
            lines.append(("Modo de precisao: OFF (A ou barra)", GRAY))
        if self.posture_off:
            lines.append(("Postura fora da calibracao: R para recentralizar", YELLOW))
        if self.control.online_corrections:
            ox, oy = self.model.offset
            lines.append((f"Correcao continua: {self.control.online_corrections} alvo(s), "
                          f"desvio ({ox:+.0f}, {oy:+.0f}) px", GRAY))
        gaze = self.sample.gaze if self.sample is not None else None
        if self.show_gaze and gaze is not None:
            yaw, pitch = direction_to_angles(gaze.direction)
            tag = "globo calibrado" if self.calibrated else "globo anatomico, sem calibracao"
            lines.append((f"Olhar yaw {yaw:+5.1f}  pitch {pitch:+5.1f} graus ({tag})", YELLOW))
            if gaze.weights is not None and self.calibrated:
                lines.append((f"Peso dos olhos: direito {gaze.weights[0]:.2f}  "
                              f"esquerdo {gaze.weights[1]:.2f}", GRAY))
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
                             self.control.dwell.progress if self.control.dwell_enabled else 0.0,
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
        t0 = time.perf_counter()
        self.process_frame(frame, ts, now)

        # Com o mouse sendo controlado, ninguém olha a janela da câmera: ela
        # é desenhada só a cada ``preview_every_in_use`` quadros. Desenhar e
        # exibir a prévia custava ~6-8 ms por quadro, o que, somado ao resto,
        # deixava o laço perto dos 33 ms entre quadros da câmera.
        every = max(1, self.cfg.display.preview_every_in_use)
        in_use = self.mouse.enabled and self.procedure is None
        self._preview_count = (self._preview_count + 1) % every if in_use else 0
        new_view = not in_use or self._preview_count == 0 or self.last_camera_view is None
        if new_view:
            self.last_camera_view = self.render_camera_view(frame, now)
        if self.procedure is not None:
            self.last_target_view = self.procedure.render(now)
        if self.show_scene3d:
            self.last_scene_view = self.render_scene_view()

        if self.bar_view is not None and self.control.bar is not None:
            # Barra visível durante o uso (some nos procedimentos, que ocupam a tela).
            self.bar_view.show(self.control.bar, self.calibrated and self.procedure is None)

        self.proc_ms = 1000.0 * (time.perf_counter() - t0)
        if self.recorder is not None and hasattr(self.recorder, "log_frame"):
            self.recorder.log_frame(self.proc_ms, self._state_code())

        keys = []
        if not self.headless:
            if new_view:
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

    def _state_code(self) -> int:
        """Estado do uso num inteiro (gravado por quadro): bit 0 mouse,
        1 clique, 2 precisão, 3 procedimento em andamento."""
        return (int(bool(self.mouse.enabled)) | int(self.control.dwell_enabled) << 1
                | int(self.control.fine_enabled) << 2 | int(self.procedure is not None) << 3)

    def press(self, key: int) -> bool:
        """Tecla do usuário: grava (se houver gravação) e executa.
        Devolve False para encerrar a aplicação."""
        if self.recorder is not None:
            self.recorder.log_key(key)
        return self.handle_key(key)

    def run(self):
        if not self.headless:
            cv2.namedWindow(self.cfg.display.window_name, cv2.WINDOW_NORMAL)
            if self.control.bar is not None:
                self.bar_view = make_view(self.control.bar, self.headless)
                if hasattr(self.bar_view, "give_focus_to"):
                    self.bar_view.give_focus_to(self.cfg.display.window_name)
        print("Pronto. ESPACO = calibrar | H = ajuda | Q = sair")
        try:
            while self.step():
                pass
        finally:
            if self.bar_view is not None:
                self.bar_view.close()
            if self.calibrated and self.control.model_dirty:
                # Guarda a correção contínua aprendida durante o uso.
                self.model.save(self.cfg.calibration.save_path)
            self.source.release()
            self.tracker.close()
            if self.recorder is not None:
                self.recorder.close()
            if not self.headless:
                cv2.destroyAllWindows()
