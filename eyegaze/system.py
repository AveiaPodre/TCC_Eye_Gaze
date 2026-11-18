"""
Ajustes do sistema operacional para o rastreamento não perder quadros.

Nas sessões gravadas de 16/11, depois dos primeiros cliques por permanência
o app caía de ~31 para 13-16 quadros por segundo e não voltava, mesmo com o
clique desligado; o processamento de um quadro leva ~15 ms (MediaPipe
~8 ms, desenho ~4 ms), bem abaixo dos 33 ms entre quadros da câmera. O
clique cai em OUTRA janela e o app passa para segundo plano, e o Windows 11
desacelera processos em segundo plano: reduz a prioridade e a frequência
da CPU (EcoQoS) e ignora o pedido de temporizador de 1 ms, de modo que cada
espera curta do laço (``cv2.waitKey(1)``) vira ~15 ms. Menos quadros
processados deixam o olhar estimado mais ruidoso.

``prepare_process`` pede ao Windows, uma vez no início, que não desacelere
este processo. Fora do Windows, não faz nada.
"""

from __future__ import annotations

import sys

_TIMER_SET = False


def prepare_process(verbose: bool = True) -> dict:
    """Prioridade acima do normal, sem desaceleração em segundo plano e com
    temporizador de 1 ms. Devolve o que deu certo."""
    global _TIMER_SET
    done = {}
    if not sys.platform.startswith("win"):
        return done
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        proc = k32.GetCurrentProcess()
        done["prioridade"] = bool(k32.SetPriorityClass(proc, 0x8000))   # ABOVE_NORMAL

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        # EXECUTION_SPEED (0x1) e IGNORE_TIMER_RESOLUTION (0x4) controlados e
        # desligados (StateMask = 0): sem EcoQoS e com o temporizador honrado.
        state = PowerThrottling(1, 0x1 | 0x4, 0)
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        done["sem_desaceleracao"] = bool(k32.SetProcessInformation(
            proc, 4, ctypes.byref(state), ctypes.sizeof(state)))   # ProcessPowerThrottling
        done["temporizador_1ms"] = ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0
        _TIMER_SET = done["temporizador_1ms"]
    except Exception as exc:  # noqa: BLE001 - ajuste opcional
        done["erro"] = str(exc)
    if verbose:
        print(f"[Sistema] {done}")
    return done


def restore_process():
    """Devolve o temporizador do sistema ao normal (fim do app)."""
    global _TIMER_SET
    if _TIMER_SET and sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.WinDLL("winmm").timeEndPeriod(1)
        except Exception:  # noqa: BLE001
            pass
        _TIMER_SET = False
