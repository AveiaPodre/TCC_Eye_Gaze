"""
Barra de controle pelo olhar: liga e desliga as funções sem teclado.

Como nas barras do Tobii Dynavox e do Windows Eye Control, uma coluna fixa de
botões grandes fica sempre visível durante o uso, numa borda da tela; olhar
para um botão por ``dwell_s`` o aciona (o botão enche enquanto isso). Fica na
BORDA ESQUERDA, centralizada na altura (longe dos menus e da barra de título
no topo, e da barra de tarefas embaixo), colada na borda: um olhar estimado
"para fora" da tela ainda cai dentro do botão (borda infinita, lei de Fitts).
Os botões são maiores que o erro típico do olhar (~70-120 px). Olhar para a
barra nunca gera clique do sistema.

A lógica (``GazeBar``) é separada do desenho (``CvBarView``), para os testes
rodarem sem janela. A janela é do próprio OpenCV (sem moldura, sempre na
frente, sem roubar o foco): uma janela do tkinter processava, a cada quadro,
todas as mensagens do programa, inclusive as teclas da janela da câmera, e
os atalhos do teclado paravam de funcionar.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Button:
    key: str               # ação (ver control.CursorControl e EyeGazeApp)
    rect: tuple            # (x0, y0, x1, y1) em pixels da tela
    label: str = ""
    on: bool | None = None  # estado mostrado (None = botão de ação)


class GazeBar:
    """Botões da barra (coluna na borda esquerda) e o acionamento pelo olhar."""

    def __init__(self, screen_size, keys, dwell_s: float = 1.2, cooldown_s: float = 1.5,
                 height_px: int = 100, width_px: int = 170, gap_px: int = 12,
                 margin_px: int = 30):
        _, sh = screen_size
        n = len(keys)
        total = n * height_px + (n - 1) * gap_px
        y = (sh - total) // 2
        self.buttons = []
        for k in keys:
            self.buttons.append(Button(k, (0, y, width_px, y + height_px)))
            y += height_px + gap_px
        self.dwell_s, self.cooldown_s = dwell_s, cooldown_s
        # Região "da barra" (com margem): olhar aqui não gera clique do sistema.
        self.region = (0, self.buttons[0].rect[1] - margin_px, width_px + margin_px,
                       self.buttons[-1].rect[3] + margin_px)
        self.active: str | None = None
        self.progress = 0.0
        self._since: float | None = None
        self._last_fire = -1e9

    @staticmethod
    def _inside(rect, p) -> bool:
        return rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]

    def in_region(self, point) -> bool:
        return point is not None and self._inside(self.region, (max(point[0], 0.0), point[1]))

    def button_at(self, point) -> Button | None:
        if point is None:
            return None
        # A barra vai até a borda: um ponto à esquerda da tela (x < 0) também conta.
        p = (max(point[0], 0.0), point[1])
        return next((b for b in self.buttons if self._inside(b.rect, p)), None)

    def update(self, point, now: float) -> str | None:
        """Devolve a ação acionada neste quadro (ou None)."""
        b = self.button_at(point)
        if b is None or now - self._last_fire < self.cooldown_s:
            self.active, self.progress, self._since = (b.key if b else None), 0.0, None
            return None
        if b.key != self.active or self._since is None:
            self.active, self._since = b.key, now
        self.progress = min(1.0, (now - self._since) / self.dwell_s)
        if self.progress >= 1.0:
            self._last_fire = now
            self.active, self.progress, self._since = None, 0.0, None
            return b.key
        return None

    def reset(self):
        self.active, self.progress, self._since = None, 0.0, None


class NullBarView:
    """Sem janela (testes, reprodução sem janelas)."""

    def show(self, bar: GazeBar, visible: bool):
        pass

    def close(self):
        pass


class CvBarView:
    """Janela do OpenCV com os botões, sem moldura, sempre na frente e sem
    roubar o foco (no Windows, estilos ajustados pela API do sistema)."""

    NAME = "Eye Gaze - barra"
    BG, ON, OFF, ACTION, FILL, TEXT = ((32, 32, 32), (50, 125, 46), (85, 85, 85),
                                       (121, 78, 31), (37, 168, 249), (255, 255, 255))

    def __init__(self, bar: GazeBar):
        x0, y0, x1, y1 = bar.buttons[0].rect[0], bar.buttons[0].rect[1], \
            bar.buttons[-1].rect[2], bar.buttons[-1].rect[3]
        self.origin = (x0, y0)
        self.size = (x1 - x0, y1 - y0)
        self._drawn = None
        self._visible = True
        cv2.namedWindow(self.NAME, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.NAME, np.zeros((self.size[1], self.size[0], 3), np.uint8))
        cv2.moveWindow(self.NAME, x0, y0)
        self._hwnd = self._make_overlay()
        if self._hwnd is None and hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(self.NAME, cv2.WND_PROP_TOPMOST, 1)

    def give_focus_to(self, window_name: str):
        """Criar a janela a ativa: devolve o foco à janela principal (onde
        o operador usa o teclado)."""
        if self._hwnd is None:
            return
        hwnd = self._u32.FindWindowW(None, window_name)
        if hwnd:
            self._u32.SetForegroundWindow(hwnd)

    def _make_overlay(self):
        """Windows: sem moldura, sempre na frente, fora da barra de tarefas,
        semitransparente e sem receber o foco quando aparece."""
        if not sys.platform.startswith("win"):
            return None
        try:
            import ctypes
            from ctypes import wintypes
            u32 = ctypes.WinDLL("user32", use_last_error=True)
            u32.FindWindowW.restype = wintypes.HWND
            hwnd = u32.FindWindowW(None, self.NAME)
            if not hwnd:
                return None
            u32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
            u32.SetWindowLongPtrW(hwnd, -16, 0x80000000 | 0x10000000)        # WS_POPUP | VISIBLE
            u32.SetWindowLongPtrW(hwnd, -20, 0x8 | 0x80 | 0x08000000 | 0x80000)
            # (TOPMOST | TOOLWINDOW | NOACTIVATE | LAYERED)
            u32.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.DWORD,
                                                       ctypes.c_ubyte, wintypes.DWORD]
            u32.SetLayeredWindowAttributes(hwnd, 0, 225, 0x2)                 # LWA_ALPHA
            u32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
            u32.SetWindowPos(hwnd, wintypes.HWND(-1), self.origin[0], self.origin[1],
                             self.size[0], self.size[1], 0x20 | 0x10 | 0x40)
            # (FRAMECHANGED | NOACTIVATE | SHOWWINDOW)
            self._u32 = u32
            return hwnd
        except Exception as exc:  # noqa: BLE001
            print(f"[Barra] Sem janela sobreposta: {exc}")
            return None

    def _set_visible(self, visible: bool):
        if visible == self._visible:
            return
        self._visible = visible
        if self._hwnd is not None:
            self._u32.ShowWindow(self._hwnd, 4 if visible else 0)   # SW_SHOWNOACTIVATE / SW_HIDE
        elif not visible:
            cv2.imshow(self.NAME, np.zeros((1, 1, 3), np.uint8))

    def render(self, bar: GazeBar) -> np.ndarray:
        img = np.full((self.size[1], self.size[0], 3), self.BG, np.uint8)
        ox, oy = self.origin
        for b in bar.buttons:
            x0, y0, x1, y1 = b.rect[0] - ox, b.rect[1] - oy, b.rect[2] - ox - 1, b.rect[3] - oy - 1
            color = self.ACTION if b.on is None else (self.ON if b.on else self.OFF)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
            if bar.active == b.key and bar.progress > 0:
                cv2.rectangle(img, (x0, y1 - 12), (x0 + int((x1 - x0) * bar.progress), y1),
                              self.FILL, -1)
            lines = b.label.split("\n")
            for i, line in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ty = (y0 + y1) // 2 + int((i - (len(lines) - 1) / 2) * (th + 10)) + th // 2
                cv2.putText(img, line, ((x0 + x1 - tw) // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            self.TEXT, 2, cv2.LINE_AA)
        return img

    def show(self, bar: GazeBar, visible: bool):
        self._set_visible(visible)
        state = tuple((b.label, b.on) for b in bar.buttons) + (bar.active,
                                                               round(bar.progress, 2))
        if visible and state != self._drawn:      # só redesenha quando algo muda
            self._drawn = state
            cv2.imshow(self.NAME, self.render(bar))

    def close(self):
        try:
            cv2.destroyWindow(self.NAME)
        except Exception:  # noqa: BLE001
            pass


def make_view(bar: GazeBar, headless: bool):
    if headless:
        return NullBarView()
    try:
        return CvBarView(bar)
    except Exception as exc:  # noqa: BLE001
        print(f"[Barra] Não foi possível abrir a barra de controle: {exc}")
        return NullBarView()
