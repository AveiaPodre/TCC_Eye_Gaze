"""
Entrada (câmera / arquivo de vídeo), tamanho da tela e controle do mouse.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


class FrameSource:
    """Lê frames de uma webcam (índice) ou de um arquivo de vídeo.

    O arquivo de vídeo é útil para reproduzir experimentos e depurar sem
    câmera: grave uma sessão e rode o pipeline sobre ela.
    """

    def __init__(self, camera_index: int = 0, video_path: str | None = None,
                 width: int = 1280, height: int = 720, fps: int = 30):
        self.is_file = video_path is not None
        self.cap = cv2.VideoCapture(video_path if self.is_file else camera_index)
        if not self.cap.isOpened():
            what = video_path if self.is_file else f"câmera {camera_index}"
            raise RuntimeError(f"Não foi possível abrir {what}. "
                               "Tente outro índice com --camera N.")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._t0 = time.monotonic()

    def read(self) -> tuple[np.ndarray | None, int]:
        ok, frame = self.cap.read()
        if not ok:
            return None, 0
        if self.is_file:
            ts = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            ts = int((time.monotonic() - self._t0) * 1000)
        return frame, ts

    @property
    def resolution(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        self.cap.release()


def detect_screen_size() -> tuple[int, int] | None:
    """Resolução do monitor principal (screeninfo -> tkinter -> None)."""
    try:
        from screeninfo import get_monitors
        mons = get_monitors()
        primary = next((m for m in mons if getattr(m, "is_primary", False)), mons[0])
        return int(primary.width), int(primary.height)
    except Exception:  # noqa: BLE001
        pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:  # noqa: BLE001
        return None


class MouseController:
    """Move/clica o cursor do sistema via pynput (Windows, macOS, Linux/X11)."""

    def __init__(self, screen_size: tuple[int, int]):
        self.screen_w, self.screen_h = screen_size
        self.enabled = False
        self.available = True
        try:
            from pynput.mouse import Button, Controller
            self._mouse = Controller()
            self._button = Button.left
        except Exception as exc:  # noqa: BLE001
            print(f"[Mouse] pynput indisponível ({exc}); controle do mouse desativado.")
            self.available = False

    def move(self, x: float, y: float):
        if not (self.enabled and self.available):
            return
        x = int(min(max(x, 0), self.screen_w - 1))
        y = int(min(max(y, 0), self.screen_h - 1))
        self._mouse.position = (x, y)

    def click(self):
        if self.enabled and self.available:
            self._mouse.click(self._button, 1)


class DwellClick:
    """Clique por permanência: olhar parado dentro de um raio por X segundos."""

    def __init__(self, dwell_time_s: float, tolerance_px: float, cooldown_s: float):
        self.dwell_time_s = dwell_time_s
        self.tolerance_px = tolerance_px
        self.cooldown_s = cooldown_s
        self.reset()
        self._last_click = -1e9

    def reset(self):
        self._anchor: np.ndarray | None = None
        self._since = 0.0
        self.progress = 0.0

    def update(self, point: np.ndarray, now: float) -> bool:
        """Devolve True quando um clique deve ser disparado."""
        if now - self._last_click < self.cooldown_s:
            self.progress = 0.0
            return False
        if self._anchor is None or np.linalg.norm(point - self._anchor) > self.tolerance_px:
            self._anchor = np.array(point, dtype=float)
            self._since = now
            self.progress = 0.0
            return False
        self.progress = min(1.0, (now - self._since) / self.dwell_time_s)
        if self.progress >= 1.0:
            self._last_click = now
            self.reset()
            return True
        return False
