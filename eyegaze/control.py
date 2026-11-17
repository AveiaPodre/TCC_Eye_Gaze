"""
Uso do cursor: tudo o que acontece depois que o ponto do olhar na tela foi
estimado e estabilizado (app.py).

* **Mouse** (``MouseController``): o cursor do sistema segue o ponto; só com
  calibração e com o rosto visível no quadro.
* **Clique por permanência** (``DwellClick``): olhar parado por
  ``dwell_time_s`` clica. No modo de precisão a tolerância é menor (o
  clique espera o cursor parar).
* **Modo de precisão** (ajuste fino pela cabeça, ``HeadFineTuner``): o
  olhar só "teleporta" o cursor (uma âncora); pequenos giros da cabeça o
  ajustam a partir dali (metodologia, seção 4.3).
* **Correção contínua**: alvos conhecidos (``confirm_target``) e cliques
  depois de um ajuste pela cabeça corrigem o desvio do modelo.
* **Barra de controle pelo olhar** (``gaze_bar.py``): liga e desliga cada
  função sem teclado; olhar para a barra nunca gera clique do sistema.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import AppConfig
from .filters import HeadFineTuner
from .gaze_bar import GazeBar
from .io_devices import DwellClick

BAR_KEYS = ("mouse", "clique", "precisao", "recentralizar", "recalibrar")


class CursorControl:
    def __init__(self, cfg: AppConfig, screen_size, mouse, toast=print):
        self.cfg = cfg
        self.screen_size = screen_size
        self.mouse = mouse
        self.toast = toast
        fc = cfg.fine
        self.fine_enabled = fc.enabled
        self.fine = HeadFineTuner(fc.gain_px_per_deg, fc.deadzone_deg, fc.max_px, fc.reset_deg)
        # Modo de precisão: âncora do cursor (só muda num salto grande do
        # olhar, FineConfig.warp_frac) e a fixação que a criou.
        self.anchor: np.ndarray | None = None
        self.anchor_fid: int | None = None
        self.dwell = DwellClick(cfg.dwell.dwell_time_s, cfg.dwell.tolerance_px,
                                cfg.dwell.cooldown_s)
        self.dwell_enabled = cfg.dwell.enabled_at_start
        self.recent_raw: deque = deque(maxlen=cfg.adapt.window_frames)
        self.online_corrections = 0
        self.model_dirty = False
        self._last_mouse: np.ndarray | None = None
        bc = cfg.bar
        self.bar = (GazeBar(screen_size, BAR_KEYS, bc.dwell_s, bc.cooldown_s, bc.height_px,
                            bc.width_px, bc.gap_px) if bc.enabled else None)
        self.refresh_bar()

    # ------------------------------------------------------- liga/desliga
    def set_mouse(self, on: bool, calibrated: bool) -> bool:
        if on and not calibrated:
            self.toast("Calibre antes de ligar o mouse")
            return False
        if on and not self.mouse.available:
            self.toast("Mouse indisponivel (instale pynput)")
            return False
        self.mouse.enabled = on
        self.toast(f"Mouse: {'LIGADO' if on else 'PAUSADO'}")
        self.refresh_bar()
        return True

    def set_dwell(self, on: bool):
        self.dwell_enabled = on
        self.dwell.reset()
        self.toast(f"Clique pelo olhar: {'LIGADO' if on else 'DESLIGADO'}")
        self.refresh_bar()

    def set_fine(self, on: bool):
        self.fine_enabled = on
        self.release_fine()
        self.toast(f"Modo de precisao (cabeca): {'LIGADO' if on else 'DESLIGADO'}")
        self.refresh_bar()

    def refresh_bar(self):
        if self.bar is None:
            return
        state = {"mouse": self.mouse.enabled, "clique": self.dwell_enabled,
                 "precisao": self.fine_enabled}
        labels = {"mouse": ("Mouse\nLIGADO", "Mouse\nPAUSADO"),
                  "clique": ("Clique\nLIGADO", "Clique\nDESLIGADO"),
                  "precisao": ("Precisao\nLIGADA", "Precisao\nDESLIGADA")}
        for b in self.bar.buttons:
            if b.key in state:
                b.on = bool(state[b.key])
                b.label = labels[b.key][0 if b.on else 1]
            else:
                b.on = None
                b.label = {"recentralizar": "Recentralizar", "recalibrar": "Recalibrar"}[b.key]

    # ----------------------------------------------------- modo de precisão
    def apply_fine(self, stab: np.ndarray, sample, stabilizer, now: float) -> np.ndarray:
        """Cursor = âncora + ajuste pela cabeça. O olhar só "teleporta" a
        âncora: numa nova fixação a mais de ``FineConfig.warp_frac`` da
        largura da tela do cursor atual. Mexer a cabeça com os olhos parados
        mexe a estimativa do olhar (~25-28 px/grau) e olhar para o cursor
        durante o ajuste é um salto de 100-200 px; nenhum dos dois move o
        cursor."""
        fc = self.cfg.fine
        hy, hp, _ = sample.head.euler_deg
        fid = stabilizer.fixation_id
        if self.anchor is None:
            self.anchor, self.anchor_fid = np.asarray(stab, float).copy(), fid
        elif fid != self.anchor_fid:
            cursor = self.anchor + self.fine.offset
            if np.linalg.norm(stab - cursor) > fc.warp_frac * self.screen_size[0]:
                self.anchor, self.anchor_fid = np.asarray(stab, float).copy(), fid
        if fid == self.anchor_fid:
            # Na fixação que criou a âncora: ela acompanha a acomodação do
            # estabilizador, e a referência da cabeça também.
            settling = stabilizer.settling
            if settling:
                self.anchor = np.asarray(stab, float).copy()
        else:
            settling = False
        return self.fine(self.anchor, self.anchor_fid, (hy, hp), now, settling=settling)

    def release_fine(self, keep_cursor_at=None):
        """Zera o ajuste. keep_cursor_at: a âncora vai para esse ponto (o
        cursor não se move e a cabeça vira a nova referência)."""
        if keep_cursor_at is not None and self.anchor is not None:
            self.anchor = np.asarray(keep_cursor_at, float).copy()
        else:
            self.anchor = self.anchor_fid = None
        self.fine.reset()

    # ----------------------------------------------------- correção contínua
    def confirm_target(self, model, target) -> np.ndarray | None:
        if model is None or not model.is_fitted or len(self.recent_raw) < 3:
            return None
        a = self.cfg.adapt
        step = model.correct_with_known_target(np.array(self.recent_raw), target,
                                               a.gain, a.max_step_px)
        self.recent_raw.clear()
        self.online_corrections += 1
        self.model_dirty = True
        return step

    def _learn_from_fine_click(self, model, cursor):
        a = self.cfg.adapt
        if not (a.learn_from_fine_clicks and self.fine_enabled):
            return
        if float(np.linalg.norm(self.fine.offset)) < a.fine_min_px:
            return
        self.confirm_target(model, cursor.copy())

    # ---------------------------------------------------------- por quadro
    def act(self, cursor, model, can_control: bool, now: float) -> str | None:
        """Mouse, clique por permanência e barra. Devolve a ação acionada na
        barra (o app executa as que envolvem procedimentos)."""
        action = None
        on_bar = self.bar is not None and self.bar.in_region(cursor)
        if self.bar is not None:
            # A barra funciona pelo olhar mesmo com o mouse pausado.
            action = self.bar.update(cursor if can_control else None, now)
            if action:
                self.dwell.reset()
                self._bar_toggle(action, model)
        if self.mouse.enabled and can_control:
            if (self._last_mouse is None
                    or np.max(np.abs(cursor - self._last_mouse)) >= 1.0):
                self.mouse.move(*cursor)
                self._last_mouse = cursor.copy()
            self.dwell.tolerance_px = (self.cfg.dwell.fine_tolerance_px if self.fine_enabled
                                       else self.cfg.dwell.tolerance_px)
            if on_bar:
                self.dwell.reset()        # olhar para a barra nunca clica
            elif self.dwell_enabled and self.dwell.update(cursor, now):
                self.mouse.click()
                self._learn_from_fine_click(model, cursor)
                if self.fine_enabled:
                    # O cursor fica onde clicou; a cabeça vira a nova referência.
                    self.release_fine(keep_cursor_at=cursor)
                self.toast("Clique", 0.8)
        else:
            self.dwell.reset()
        return action

    def _bar_toggle(self, action: str, model):
        calibrated = model is not None and model.is_fitted
        if action == "mouse":
            self.set_mouse(not self.mouse.enabled, calibrated)
        elif action == "clique":
            self.set_dwell(not self.dwell_enabled)
        elif action == "precisao":
            self.set_fine(not self.fine_enabled)
