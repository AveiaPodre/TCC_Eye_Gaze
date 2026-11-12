"""
Benchmark offline: avalia variantes do pipeline em TODAS as sessões gravadas.

Para cada ``results/calibracao_*.npz`` com landmarks gravados (e a
validação gravada logo depois dela, se houver), as medidas são recalculadas
com o estimador ATUAL, cada variante é ajustada e avaliada com três métricas:

* ``LOO``     - erro médio em alvos da grade deixados de fora do ajuste
                (pontos novos, cabeça parada);
* ``cabeça``  - cada direção da etapa da cabeça (direita/esquerda/cima/
                baixo) fica fora do ajuste e o erro é medido nela (poses
                novas da cabeça). Teste: amostras até ±15° (yaw) e ±12°
                (pitch) da pose da grade, iguais para todas as variantes;
* ``valid.``  - erro médio nos alvos da validação (outro momento da sessão)
                e, entre parênteses, o mesmo erro sem o deslocamento médio
                (o que a recentralização R corrige).

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

from eyegaze.calibration import (CalibrationData, fit_from_samples,  # noqa: E402
                                 head_angles, head_direction)
from reavaliar import load, validation_error  # noqa: E402

TEST_RANGE_DEG = (15.0, 12.0)   # faixa das amostras de teste da cabeça
PAIR_WINDOW_S = 30 * 60         # validação até 30 min depois da calibração

# (nome, argumentos de fit_from_samples). A primeira é a configuração padrão.
VARIANTS: list[tuple[str, dict]] = [
    ("padrão", dict()),
    ("só monitor virtual", dict(corrections=())),
    ("correção só deslocamento", dict(corrections=("deslocamento",))),
    ("pupila só MediaPipe", dict(pupil_sources=("mediapipe",))),
    ("sem etapa da cabeça", dict(_no_head=True)),
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
    kw = dict(kw)
    if kw.pop("_no_head", False):
        data = CalibrationData(grid=data.grid)
    model, _, stats = fit_from_samples(data, screen, WIDTH_CM, **kw)
    return model, stats


def head_cv(data: CalibrationData, screen, kw: dict) -> float:
    """Erro médio (px) em direções da cabeça deixadas de fora do ajuste."""
    if data.head_raw is None or len(data.head_raw) < 10:
        return float("nan")
    hr, hf = np.atleast_2d(data.head_raw), np.atleast_2d(data.head_features)
    grid_raw = np.vstack([np.atleast_2d(r) for _, _, r in data.grid if len(r)])
    delta = head_angles(hr) - np.median(head_angles(grid_raw), axis=0)
    direction = head_direction(delta)
    in_test = ((np.abs(delta[:, 0]) <= TEST_RANGE_DEG[0])
               & (np.abs(delta[:, 1]) <= TEST_RANGE_DEG[1])
               & (np.max(np.abs(delta), axis=1) >= 2.0))
    errs = []
    for q in range(4):
        test = in_test & (direction == q)
        if test.sum() < 5:
            continue
        train = direction != q
        d = CalibrationData(grid=data.grid, head_target=data.head_target,
                            head_raw=hr[train], head_features=hf[train])
        model, _ = _fit(d, screen, kw)
        pred = model.predict(hr[test])
        errs.append(float(np.nanmedian(np.hypot(*(pred - np.asarray(data.head_target)).T))))
    return float(np.mean(errs)) if errs else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--screen", type=str, default="1920x1080")
    p.add_argument("--screen-width-cm", type=float, default=None,
                   help="largura física da tela (padrão: detecta no Windows)")
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
    print("\nErros médios em px (média das sessões). valid. = erro na validação "
          "(entre parênteses, sem o deslocamento constante).")
    print(f"{'variante':28s} {'LOO':>6s} {'cabeça':>7s} {'valid.':>13s}")
    per_session = {}
    for name, kw in VARIANTS:
        loos, heads, vals, vals_o = [], [], [], []
        for s in sessions:
            model, stats = _fit(s.calib, screen, kw)
            loos.append(stats.get("erro_loo_medio_px", np.nan))
            heads.append(head_cv(s.calib, screen, kw))
            va, vo, _ = validation_error(model, s.valid) if s.valid else (np.nan, np.nan, [])
            vals.append(va)
            vals_o.append(vo)
            per_session.setdefault(s.name, []).append((loos[-1], heads[-1], va))
        print(f"{name:28s} {np.nanmean(loos):6.0f} {np.nanmean(heads):7.0f} "
              f"{np.nanmean(vals):7.0f} ({np.nanmean(vals_o):3.0f})")
    print("\nPor sessão (LOO / cabeça / valid.), uma coluna por variante:")
    for name, rows in per_session.items():
        print(f"  {name}: " + " | ".join(
            f"{a:4.0f} {b:4.0f} {'  - ' if np.isnan(c) else f'{c:4.0f}'}" for a, b, c in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
