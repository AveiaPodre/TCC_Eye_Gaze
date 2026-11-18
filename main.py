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

from eyegaze import system
from eyegaze.app import EyeGazeApp
from eyegaze.config import AppConfig
from eyegaze.face_tracker import FaceTracker
from eyegaze.io_devices import (FrameSource, NullMouse, ReplaySource, SessionRecorder,
                                detect_screen_size, detect_screen_width_cm)


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
                   help="largura física da tela em cm (padrão: detecta no Windows). É o "
                        "tamanho do monitor virtual onde o olhar é projetado e dá o erro em "
                        "graus na validação")
    p.add_argument("--camera-fov", type=float, default=None,
                   help="campo de visão HORIZONTAL da câmera em graus (padrão 90, notebook); "
                        "define a escala do rosto em cm")
    p.add_argument("--distance-cm", type=float, default=None,
                   help="distância olho-tela (para erro em graus na validação)")
    p.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto",
                   help="backend de captura no Windows (padrão: o do OpenCV)")
    p.add_argument("--sem-mjpg", action="store_true",
                   help="não pede MJPG à câmera (por padrão pede: em 1080p muitas webcams "
                        "só chegam a 30 fps em MJPG)")
    p.add_argument("--exposicao", type=float, default=None,
                   help="exposição fixa (valor do driver, ex.: -6 no DirectShow); evita "
                        "que a exposição automática derrube o fps com pouca luz")
    p.add_argument("--sem-autofoco", action="store_true",
                   help="desliga o foco automático (evita a nitidez do olho oscilar)")
    p.add_argument("--pesquisa", action="store_true",
                   help="modo pesquisa: validação imediata e revalidação periódica após "
                        "cada calibração. Ligado automaticamente com --gravar")
    p.add_argument("--cabeca", action="store_true",
                   help="modo pesquisa: teste da cabeça depois da validação imediata "
                        "(fora do protocolo padrão desde 17/11)")
    p.add_argument("--perseguicao", choices=("soma", "alternativa"), default=None,
                   help="calibração por perseguição (seguir um ponto em movimento): 'soma' = "
                        "grade + perseguição + etapa 2; 'alternativa' = perseguição + 5 pontos")
    p.add_argument("--revalidar-min", type=float, default=None,
                   help="revalidação periódica a cada N minutos de uso (0 desliga; "
                        "padrão 10 no modo pesquisa, desligada fora dele)")
    p.add_argument("--gravar", action="store_true",
                   help="grava o vídeo da câmera e as teclas em results/sessao_*.avi/.json "
                        "(implica --pesquisa)")
    p.add_argument("--reproduzir", type=str, default=None,
                   help="reproduz uma sessão gravada (results/sessao_*.json) com o código "
                        "atual; os resultados vão para results/reproducoes/")
    p.add_argument("--sem-janela", action="store_true",
                   help="com --reproduzir: roda sem abrir janelas (mais rápido)")
    return p.parse_args(argv)


def _set_screen_width(cfg: AppConfig, width_cm: float | None):
    """A largura física serve à calibração (monitor virtual), à validação
    (erro em graus) e à vista 3D."""
    cfg.calibration.screen_width_cm = width_cm
    cfg.validation.screen_width_cm = width_cm


def _research_mode(cfg: AppConfig):
    """Modo pesquisa: mede mais (validação imediata depois de cada
    calibração e revalidação a cada 10 min de uso)."""
    cfg.validation.after_calibration = True
    cfg.validation.revalidate_every_min = 10.0


def protocol_of(cfg: AppConfig) -> dict:
    """Protocolo da calibração, gravado com a sessão para a reprodução."""
    c = cfg.calibration
    return {"grade": [c.grid_cols, c.grid_rows], "etapa_cabeca": c.head_stage,
            "validacao_imediata": cfg.validation.after_calibration,
            "perseguicao": c.pursuit,
            "revalidar_min": cfg.validation.revalidate_every_min,
            "conferencia": c.check_after_calibration}


def _replay(args, cfg: AppConfig) -> int:
    source = ReplaySource(Path(args.reproduzir))
    proto = source.protocol
    if proto is None:
        print("[Reprodução] Sessão gravada antes do protocolo atual (grade 4x4 com checagens "
              "do centro e etapa da boca): a calibração reproduzida NÃO acompanha os alvos "
              "do vídeo. Para comparar variantes, use tools/benchmark.py.")
    else:
        cfg.calibration.grid_cols, cfg.calibration.grid_rows = proto["grade"]
        cfg.calibration.head_stage = bool(proto["etapa_cabeca"])
        cfg.validation.after_calibration = bool(proto["validacao_imediata"])
        cfg.calibration.pursuit = proto.get("perseguicao", "nenhuma")
        cfg.validation.revalidate_every_min = float(proto.get("revalidar_min", 0.0))
        cfg.calibration.check_after_calibration = bool(proto.get("conferencia", False))
    name = Path(args.reproduzir).stem
    out = cfg.calibration.results_dir / "reproducoes" / name
    # Resultados separados das sessões reais, e sem tocar na calibração salva.
    cfg.calibration.results_dir = out
    cfg.validation.results_dir = out
    cfg.calibration.save_path = out / "calibration.json"
    if cfg.calibration.save_path.exists():
        cfg.calibration.save_path.unlink()
    cfg.calibration.random_seed = source.seed
    # O vídeo gravado não atende a pedidos novos: a repetição de alvos ruins
    # coletaria o alvo de novo num momento em que a pessoa do vídeo olhava
    # para outro lugar (ex.: durante a etapa da cabeça).
    cfg.calibration.retry_bad_targets = False
    _set_screen_width(cfg, args.screen_width_cm or source.screen_width_cm)
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
    cfg.validation.viewing_distance_cm = args.distance_cm
    if args.camera_fov:
        cfg.camera.hfov_deg = args.camera_fov
    cfg.camera.backend = args.backend
    cfg.camera.mjpg = not args.sem_mjpg
    cfg.camera.exposure = args.exposicao
    cfg.camera.autofocus = not args.sem_autofoco
    if args.pesquisa or args.gravar:
        _research_mode(cfg)
    if args.cabeca:
        cfg.calibration.head_stage = True
    if args.perseguicao:
        cfg.calibration.pursuit = args.perseguicao
    if args.revalidar_min is not None:
        cfg.validation.revalidate_every_min = max(args.revalidar_min, 0.0)

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
    width_cm = args.screen_width_cm or detect_screen_width_cm(screen_size)
    _set_screen_width(cfg, width_cm)
    if width_cm:
        origem = "informada" if args.screen_width_cm else "detectada"
        print(f"[Tela] {screen_size[0]}x{screen_size[1]}, {width_cm:.1f} cm de largura ({origem})")
    else:
        print(f"[Tela] {screen_size[0]}x{screen_size[1]}. Largura física desconhecida: o "
              f"monitor virtual usa {cfg.calibration.assumed_screen_width_cm} cm e a calibração "
              "corrige a escala. Para a geometria exata, use --screen-width-cm.")

    if args.reset_calibration and cfg.calibration.save_path.exists():
        cfg.calibration.save_path.unlink()

    source = FrameSource(cfg.camera.index, args.video, cfg.camera.width,
                         cfg.camera.height, cfg.camera.fps, backend=cfg.camera.backend,
                         mjpg=cfg.camera.mjpg, exposure=cfg.camera.exposure,
                         autofocus=cfg.camera.autofocus)
    print(f"[Câmera] {source.resolution[0]}x{source.resolution[1]} ({source.describe()})")
    if args.pesquisa or args.gravar:
        print("[Modo pesquisa] Validação imediata ligada"
              + (" e teste da cabeça." if cfg.calibration.head_stage else "."))
    if cfg.calibration.pursuit != "nenhuma":
        print(f"[Calibração] Perseguição: {cfg.calibration.pursuit}")
    if cfg.validation.revalidate_every_min:
        print(f"[Validação] Revalidação a cada {cfg.validation.revalidate_every_min:g} min")

    recorder = None
    if args.gravar:
        # Semente da ordem dos alvos gravada junto: a reprodução repete a ordem.
        seed = int(time.time()) % 100000
        cfg.calibration.random_seed = seed
        stamp = time.strftime("%Y%m%d_%H%M%S")
        recorder = SessionRecorder(cfg.calibration.results_dir / f"sessao_{stamp}",
                                   cfg.camera.fps, source.resolution, screen_size, seed,
                                   screen_width_cm=width_cm, protocol=protocol_of(cfg))
        print(f"[Gravação] Gravando em {recorder.video_path}")

    tracker = FaceTracker(cfg.tracker)
    # O Windows desacelera processos em segundo plano (o clique cai em outra
    # janela): sem isto, o app caía de ~31 para 13-16 quadros por segundo.
    system.prepare_process()
    try:
        EyeGazeApp(cfg, tracker, source, screen_size, recorder=recorder).run()
    finally:
        system.restore_process()
    return 0


if __name__ == "__main__":
    sys.exit(main())
