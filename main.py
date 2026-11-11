"""
Eye gaze com webcam + MediaPipe para controle do mouse.

Uso:
    python main.py                    # webcam 0
    python main.py --camera 1         # outra webcam
    python main.py --video sessao.mp4 # roda sobre um vídeo gravado
    python main.py --gravar           # grava vídeo + teclas em results/sessao_*.avi/.json
    python main.py --reproduzir results/sessao_X.json   # refaz a sessão com o código atual
    python main.py --help             # todas as opções
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from eyegaze.app import EyeGazeApp
from eyegaze.config import AppConfig
from eyegaze.face_tracker import FaceTracker
from eyegaze.io_devices import (FrameSource, NullMouse, ReplaySource, SessionRecorder,
                                detect_screen_size)


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
    p.add_argument("--gravar", action="store_true",
                   help="grava o vídeo da câmera e as teclas em results/sessao_*.avi/.json")
    p.add_argument("--reproduzir", type=str, default=None,
                   help="reproduz uma sessão gravada (results/sessao_*.json) com o código "
                        "atual; os resultados vão para results/reproducoes/")
    p.add_argument("--sem-janela", action="store_true",
                   help="com --reproduzir: roda sem abrir janelas (mais rápido)")
    return p.parse_args(argv)


def _replay(args, cfg: AppConfig) -> int:
    source = ReplaySource(Path(args.reproduzir))
    name = Path(args.reproduzir).stem
    out = cfg.calibration.results_dir / "reproducoes" / name
    # Resultados separados das sessões reais, e sem tocar na calibração salva.
    cfg.calibration.results_dir = out
    cfg.validation.results_dir = out
    cfg.calibration.save_path = out / "calibration.json"
    if cfg.calibration.save_path.exists():
        cfg.calibration.save_path.unlink()
    cfg.calibration.random_seed = source.seed
    print(f"[Reprodução] {name}: tela {source.screen_size}, resultados em {out}")
    tracker = FaceTracker(cfg.tracker)
    app = EyeGazeApp(cfg, tracker, source, source.screen_size, mouse=NullMouse(),
                     headless=args.sem_janela, clock=source.now)
    app.run()
    return 0


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

    if args.reproduzir:
        return _replay(args, cfg)

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

    recorder = None
    if args.gravar:
        # Semente da ordem dos alvos gravada junto: a reprodução repete a ordem.
        seed = int(time.time()) % 100000
        cfg.calibration.random_seed = seed
        stamp = time.strftime("%Y%m%d_%H%M%S")
        recorder = SessionRecorder(cfg.calibration.results_dir / f"sessao_{stamp}",
                                   cfg.camera.fps, source.resolution, screen_size, seed)
        print(f"[Gravação] Gravando em {recorder.video_path}")

    tracker = FaceTracker(cfg.tracker)
    EyeGazeApp(cfg, tracker, source, screen_size, recorder=recorder).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
