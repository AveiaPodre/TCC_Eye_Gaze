"""
Benchmark offline: avalia variantes do pipeline em TODAS as sessões gravadas.

Para cada ``results/calibracao_*.npz`` com landmarks gravados (e a
validação gravada logo depois dela, se houver), as medidas são recalculadas
com o estimador ATUAL (com cache em results/.cache), cada variante é
ajustada e avaliada com três métricas:

* ``LOO``     - erro médio em alvos da grade deixados de fora do ajuste
                (pontos novos, cabeça parada);
* ``cabeça``  - erro nas amostras da etapa da cabeça (que nunca entram no
                ajuste): cabeça em poses novas, até ±15° (yaw) e ±12°
                (pitch) da pose da grade;
* ``valid.``  - erro médio nos alvos da validação (outro momento da sessão)
                e, entre parênteses, o mesmo erro sem o deslocamento médio
                (o que a recentralização R corrige).

Os erros saem em px e em graus (com a distância do rosto medida pelo
próprio pipeline em cada sessão).

Uso:
    python tools/benchmark.py
    python tools/benchmark.py --screen 1920x1080 --screen-width-cm 34.4
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eyegaze.calibration import CalibrationData, fit_from_samples  # noqa: E402
from eyegaze.gaze import RAW_FACE  # noqa: E402
from reavaliar import load, validation_error  # noqa: E402

PAIR_WINDOW_S = 30 * 60         # validação até 30 min depois da calibração

# (nome, argumentos de fit_from_samples). A primeira é a configuração padrão.
VARIANTS: list[tuple[str, dict]] = [
    ("padrão", dict()),
    ("só monitor virtual", dict(corrections=())),
    ("correção só deslocamento", dict(corrections=("deslocamento",))),
    ("afim fixa (alpha 0,01)", dict(correction=("afim", 1e-2))),
    ("quadrática fixa (alpha 0,01)", dict(correction=("quadratica", 1e-2))),
    ("pupila só MediaPipe", dict(pupil_source="mediapipe")),
    ("pupila só refinada", dict(pupil_source="refinada")),
    ("olhos com peso igual", dict(weight_eyes=False)),
]
WIDTH_CM: float | None = None


@dataclass
class Session:
    name: str
    calib: CalibrationData
    valid: CalibrationData | None


def _stamp(path: Path) -> datetime | None:
    m = re.search(r"(\d{8}_\d{6})", path.name)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


def load_sessions(results: Path) -> list[Session]:
    calibs = sorted(results.glob("calibracao_*.npz"))
    valids = sorted(results.glob("validacao_*.npz"))
    sessions = []
    for i, cp in enumerate(calibs):
        t0 = _stamp(cp)
        t1 = _stamp(calibs[i + 1]) if i + 1 < len(calibs) else None
        vp = None
        for v in valids:
            tv = _stamp(v)
            if (t0 and tv and tv > t0 and (tv - t0).total_seconds() < PAIR_WINDOW_S
                    and (t1 is None or tv < t1)):
                vp = v
                break
        c = load(cp)
        if c is None:
            print(f"  {cp.name}: sem landmarks (sessão antiga), ignorada")
            continue
        sessions.append(Session(cp.stem.replace("calibracao_", ""), c,
                                load(vp) if vp else None))
    return sessions


def _fit(data: CalibrationData, screen, kw: dict):
    model, _, stats = fit_from_samples(data, screen, WIDTH_CM, **kw)
    return model, stats


def _distance_cm(data: CalibrationData) -> float:
    raws = [np.atleast_2d(r) for _, _, r in data.grid if len(r)]
    return float(np.median(np.vstack(raws)[:, RAW_FACE][:, 2]))


def _deg(px: float, dist_cm: float, screen_w: int) -> float:
    cm = px * (WIDTH_CM or 34.5) / screen_w
    return float(np.degrees(np.arctan2(cm, dist_cm)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--screen", type=str, default="1920x1080")
    p.add_argument("--screen-width-cm", type=float, default=None,
                   help="largura física da tela (padrão: detecta no Windows)")
    p.add_argument("--distance-cm", type=float, default=None,
                   help="distância olho-tela real, para os graus (padrão: a medida pelo "
                        "pipeline em cada sessão, que depende do campo de visão suposto)")
    args = p.parse_args(argv)
    screen = tuple(int(v) for v in args.screen.lower().split("x"))
    global WIDTH_CM
    from eyegaze.io_devices import detect_screen_width_cm
    WIDTH_CM = args.screen_width_cm or detect_screen_width_cm(screen)
    print(f"Tela: {screen[0]}x{screen[1]}, "
          + (f"{WIDTH_CM:.1f} cm" if WIDTH_CM else "largura suposta (34,5 cm)"))

    sessions = load_sessions(args.results)
    if not sessions:
        print(f"Nenhuma calibração utilizável em {args.results}.")
        return 1
    print(f"{len(sessions)} sessões ({sum(s.valid is not None for s in sessions)} com validação)")
    print("\nErros médios (média das sessões), em px e [graus]. valid. = erro na validação "
          "(entre parênteses, sem o deslocamento constante).")
    print(f"{'variante':30s} {'LOO':>12s} {'cabeça':>12s} {'valid.':>20s}")
    dists = [args.distance_cm or _distance_cm(s.calib) for s in sessions]
    per_session = {}
    for name, kw in VARIANTS:
        loos, heads, vals, vals_o = [], [], [], []
        dl, dh, dv, dvo = [], [], [], []
        for s, dist in zip(sessions, dists):
            model, stats = _fit(s.calib, screen, kw)
            loos.append(stats.get("erro_loo_medio_px", np.nan))
            heads.append(stats.get("erro_cabeca_px", np.nan))
            va, vo, _ = validation_error(model, s.valid) if s.valid else (np.nan, np.nan, [])
            vals.append(va)
            vals_o.append(vo)
            dl.append(_deg(loos[-1], dist, screen[0]))
            dh.append(_deg(heads[-1], dist, screen[0]))
            dv.append(_deg(va, dist, screen[0]))
            dvo.append(_deg(vo, dist, screen[0]))
            per_session.setdefault(s.name, []).append((loos[-1], heads[-1], va))
        print(f"{name:30s} {np.nanmean(loos):5.0f} [{np.nanmean(dl):4.2f}] "
              f"{np.nanmean(heads):5.0f} [{np.nanmean(dh):4.2f}] "
              f"{np.nanmean(vals):5.0f} ({np.nanmean(vals_o):3.0f}) "
              f"[{np.nanmean(dv):4.2f} ({np.nanmean(dvo):4.2f})]", flush=True)
    print(f"\nDistância mediana do rosto: {np.median(dists):.0f} cm")
    print("\nPor sessão (LOO / cabeça / valid.), uma coluna por variante:")
    for name, rows in per_session.items():
        print(f"  {name}: " + " | ".join(
            f"{a:4.0f} {b:4.0f} {'  - ' if np.isnan(c) else f'{c:4.0f}'}" for a, b, c in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
