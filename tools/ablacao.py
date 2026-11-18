"""
Ablação da calibração: quantos alvos são necessários.

Para cada sessão gravada (results/calibracao_*.npz com landmarks), sorteia k
alvos da grade gravada, ajusta o pipeline só com eles e mede:

* o erro nos alvos da grade que ficaram FORA do ajuste;
* o erro na validação gravada logo depois (se houver), bruto e sem o
  deslocamento constante.

A validação é a comparação mais justa entre tamanhos de grade: os alvos são
os mesmos para todos os k e foram vistos em outro momento da sessão.

Uso:
    python tools/ablacao.py
    python tools/ablacao.py --k 6 9 12 --sorteios 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eyegaze.calibration import CalibrationData, fit_from_samples, outlier_mask  # noqa: E402
from eyegaze.io_devices import detect_screen_width_cm  # noqa: E402
from benchmark import load_sessions  # noqa: E402
from reavaliar import validation_error  # noqa: E402


def subset(d: CalibrationData, idx) -> CalibrationData:
    s = CalibrationData(grid=[d.grid[i] for i in idx])
    if len(d.grid_times) == len(d.grid):
        s.grid_times = [d.grid_times[i] for i in idx]
    return s


def heldout_error(model, d: CalibrationData, idx) -> float:
    errs = []
    for i in idx:
        t, f, r = d.grid[i]
        r = np.atleast_2d(r)
        if len(r) < 3:
            continue
        p = np.nanmedian(model.predict(r[outlier_mask(f, 3.5)]), axis=0)
        if np.all(np.isfinite(p)):
            errs.append(float(np.hypot(*(p - np.asarray(t)))))
    return float(np.mean(errs)) if errs else np.nan


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--screen", type=str, default="1920x1080")
    p.add_argument("--screen-width-cm", type=float, default=None)
    p.add_argument("--k", type=int, nargs="+", default=[6, 9, 12])
    p.add_argument("--sorteios", type=int, default=5)
    args = p.parse_args(argv)
    screen = tuple(int(v) for v in args.screen.lower().split("x"))
    width = args.screen_width_cm or detect_screen_width_cm(screen)
    rng = np.random.default_rng(0)

    t0 = time.time()
    sessions = load_sessions(args.results)
    print(f"{len(sessions)} sessões ({sum(s.valid is not None for s in sessions)} com validação)")
    held: dict = {}
    val: dict = {}
    for s in sessions:
        d = s.calib
        n = len(d.grid)
        for k in [k for k in args.k if k < n]:
            for _ in range(args.sorteios):
                tr = np.sort(rng.choice(n, k, replace=False))
                te = np.setdiff1d(np.arange(n), tr)
                try:
                    m, _, _ = fit_from_samples(subset(d, tr), screen, width)
                except ValueError:
                    continue
                held.setdefault(k, []).append(heldout_error(m, d, te))
                if s.valid is not None:
                    val.setdefault(k, []).append(validation_error(m, s.valid)[:2])
        m, _, st = fit_from_samples(d, screen, width)
        held.setdefault(f"{n} (LOO)", []).append(st.get("erro_loo_medio_px", np.nan))
        if s.valid is not None:
            val.setdefault(f"{n} (LOO)", []).append(validation_error(m, s.valid)[:2])

    print("\nAlvos no ajuste | erro nos alvos fora do ajuste | validação: bruta (sem desvio)")
    for k in list(held):
        v = np.array(val.get(k, [(np.nan, np.nan)]))
        print(f"  {str(k):>10s} | {np.nanmean(held[k]):6.0f} px | "
              f"{np.nanmean(v[:, 0]):6.0f} ({np.nanmean(v[:, 1]):3.0f}) px   "
              f"[n = {len(held[k])}, {len(val.get(k, []))}]")
    print(f"\n{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
