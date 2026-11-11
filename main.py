"""
Eye gaze com webcam + MediaPipe para controle do mouse.

Uso:
    python main.py                    # webcam 0
    python main.py --camera 1         # outra webcam
    python main.py --video sessao.mp4 # roda sobre um vídeo gravado
    python main.py --help             # todas as opções
"""

from __future__ import annotations

import argparse
import sys

from eyegaze.app import EyeGazeApp
from eyegaze.config import AppConfig
from eyegaze.face_tracker import FaceTracker
from eyegaze.io_devices import FrameSource, detect_screen_size


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Controle do mouse pelo olhar (webcam + MediaPipe)")
    p.add_argument("--camera", type=int, default=0, help="índice da webcam (padrão 0)")
    p.add_argument("--video", type=str, default=None, help="usar arquivo de vídeo em vez da webcam")
    p.add_argument("--width", type=int, default=1280, help="largura pedida à câmera")
    p.add_argument("--height", type=int, default=720, help="altura pedida à câmera")
    p.add_argument("--screen", type=str, default=None,
                   help="resolução da tela, ex.: 1920x1080 (padrão: detecta)")
    p.add_argument("--no-mirror", action="store_true", help="não espelhar a imagem exibida")
    p.add_argument("--reset-calibration", action="store_true",
                   help="ignora a calibração salva")
    p.add_argument("--screen-width-cm", type=float, default=None,
                   help="largura física da tela (para erro em graus na validação)")
    p.add_argument("--distance-cm", type=float, default=None,
                   help="distância olho-tela (para erro em graus na validação)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = AppConfig()
    cfg.camera.index = args.camera
    cfg.camera.width = args.width
    cfg.camera.height = args.height
    if args.no_mirror:
        cfg.camera.mirror_display = False
    cfg.validation.screen_width_cm = args.screen_width_cm
    cfg.validation.viewing_distance_cm = args.distance_cm

    if args.screen:
        w, h = (int(v) for v in args.screen.lower().split("x"))
        screen_size = (w, h)
    else:
        screen_size = detect_screen_size()
        if screen_size is None:
            print("Não foi possível detectar a resolução da tela; use --screen LARGURAxALTURA.")
            return 1
    print(f"[Tela] {screen_size[0]}x{screen_size[1]}")

    if args.reset_calibration and cfg.calibration.save_path.exists():
        cfg.calibration.save_path.unlink()

    source = FrameSource(cfg.camera.index, args.video, cfg.camera.width,
                         cfg.camera.height, cfg.camera.fps)
    print(f"[Câmera] {source.resolution[0]}x{source.resolution[1]}")
    tracker = FaceTracker(cfg.tracker)
    EyeGazeApp(cfg, tracker, source, screen_size).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
