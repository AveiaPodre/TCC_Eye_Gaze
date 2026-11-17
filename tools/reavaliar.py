"""
Reanálise offline de uma sessão gravada (sem câmera).

Cada calibração grava ``results/calibracao_*.npz`` e cada validação grava
``results/validacao_*.npz`` com as medidas brutas e os landmarks de todos os
quadros. Este script reajusta o pipeline com outras configurações e mede o
erro na validação, para comparar alternativas com os MESMOS dados.

Uso:
    python tools/reavaliar.py                         # arquivos mais recentes
    python tools/reavaliar.py --calib results/calibracao_X.npz --valid results/validacao_Y.npz

As medidas são sempre recalculadas a partir dos landmarks gravados com o
estimador ATUAL (sessões gravadas com outro layout de medidas só servem
assim). A validação só faz sentido com a calibração da mesma sessão.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eyegaze.calibration import (FEATURE_VERSION, CalibrationData,  # noqa: E402
                                 fit_from_samples, outlier_mask)
from eyegaze.config import GazeConfig  # noqa: E402
from eyegaze.gaze import GazeEstimator  # noqa: E402


def latest(pattern: str) -> Path | None:
    files = sorted((ROOT / "results").glob(pattern))
    return files[-1] if files else None


def recompute(data: CalibrationData) -> CalibrationData:
    """Recalcula features e medidas brutas a partir dos landmarks gravados,
    com a versão ATUAL do estimador (globo anatômico para as features)."""
    est = GazeEstimator(GazeConfig())

    def run(observations):
        F, R, O, idx = [], [], [], []
        for i, o in enumerate(observations):
            s = est.estimate(o)
            if s.valid:
                F.append(s.features())
                R.append(s.raw)
                O.append(o)
                idx.append(i)
        return np.array(F).reshape(-1, 2), np.array(R), O, np.array(idx, dtype=int)

    def times(tt, n_obs, idx):
        # Instantes gravados correspondem às amostras válidas na gravação;
        # só servem se a contagem bater com os landmarks.
        tt = np.asarray(tt, dtype=float)
        return tt[idx] if len(tt) == n_obs else None

    out = CalibrationData()
    grid_times = []
    for j, ((t, _, _), obs) in enumerate(zip(data.grid, data.grid_obs)):
        f, r, o, idx = run(obs)
        out.grid.append((t, f, r))
        out.grid_obs.append(o)
        if len(data.grid_times) == len(data.grid):
            grid_times.append(times(data.grid_times[j], len(obs), idx))
    if grid_times and all(g is not None for g in grid_times):
        out.grid_times = grid_times
    if data.head_obs:
        out.head_target = data.head_target
        out.head_features, out.head_raw, out.head_obs, _ = run(data.head_obs)
    out.feature_version = FEATURE_VERSION
    return out


CACHE_DIR = ROOT / "results" / ".cache"


def _source_digest() -> str:
    """Impressão digital do código que gera as medidas: muda o código, o
    cache é refeito."""
    import hashlib
    h = hashlib.sha1()
    for name in ("gaze.py", "head_pose.py", "geometry.py", "landmarks.py", "config.py"):
        h.update((ROOT / "eyegaze" / name).read_bytes())
    return h.hexdigest()[:12]


def load(path: Path, cache: bool = True) -> CalibrationData | None:
    """Carrega e, se houver landmarks, recalcula com o estimador atual. None
    se não der para usar. As medidas recalculadas ficam em cache
    (results/.cache), invalidado quando o código do estimador muda."""
    path = Path(path)
    cached = CACHE_DIR / f"{path.stem}_{_source_digest()}.npz"
    if cache and cached.exists():
        return CalibrationData.load(cached)
    data = CalibrationData.load(path)
    head_only = not data.grid and bool(data.head_obs)   # cabeca_*.npz
    if not (data.has_landmarks or head_only):
        return None if data.stale else data
    out = recompute(data)
    if cache:
        try:
            out.save(cached)
        except OSError:
            pass
    return out


def validation_error(model, valid: CalibrationData, mad: float = 3.5):
    """(erro médio, erro médio sem o deslocamento constante, por alvo)."""
    rows = []
    for target, feats, raw in valid.grid:
        raw = np.atleast_2d(raw)
        if len(raw) < 3:
            continue
        pred = np.nanmedian(model.predict(raw[outlier_mask(feats, mad)]), axis=0)
        rows.append((target, pred, float(np.hypot(*(pred - np.asarray(target))))))
    E = np.array([p for _, p, _ in rows]) - np.array([t for t, _, _ in rows])
    E = E[np.all(np.isfinite(E), axis=1)]
    if not len(E):
        return float("nan"), float("nan"), rows
    return (float(np.hypot(*E.T).mean()), float(np.hypot(*(E - E.mean(0)).T).mean()), rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--valid", type=Path, default=None)
    p.add_argument("--screen", type=str, default="1920x1080")
    p.add_argument("--screen-width-cm", type=float, default=None,
                   help="largura física da tela (padrão: detecta no Windows)")
    p.add_argument("--mad", type=float, default=3.5)
    args = p.parse_args(argv)

    calib_path = args.calib or latest("calibracao_*.npz")
    if calib_path is None:
        print("Nenhum results/calibracao_*.npz encontrado. Calibre o app primeiro.")
        return 1
    valid_path = args.valid or latest("validacao_*.npz")
    screen = tuple(int(v) for v in args.screen.lower().split("x"))
    from eyegaze.io_devices import detect_screen_width_cm
    width_cm = args.screen_width_cm or detect_screen_width_cm(screen)

    data = load(calib_path)
    if data is None:
        print(f"{calib_path.name}: sem landmarks gravados e de outra versão; não dá para usar.")
        return 1
    valid = load(valid_path) if valid_path else None
    print(f"Calibração: {calib_path.name} ({len(data.grid)} alvos, "
          f"etapa da cabeça: {'sim' if data.head_raw is not None else 'não'})")
    print(f"Validação:  {valid_path.name if valid and valid_path else '(nenhuma)'}")
    print(f"Monitor virtual: {width_cm or 34.5:.1f} cm"
          + ("" if width_cm else " (largura suposta)"))

    variants = [
        ("padrão", dict(), data),
        ("só monitor virtual (sem correção)", dict(corrections=()), data),
        ("olhos com peso igual", dict(weight_eyes=False), data),
    ]
    print(f"\n{'variante':36s} {'pupila':>10s} {'correção':>12s} {'ajuste':>7s} {'LOO':>6s} "
          f"{'cabeça':>7s} {'validação':>14s}  (erro médio, px)")
    for name, kw, d in variants:
        model, results, stats = fit_from_samples(d, screen, width_cm, args.mad, **kw)
        fit_err = np.mean([r.error_px for r in results])
        loo = stats.get("erro_loo_medio_px", float("nan"))
        head = stats.get("erro_cabeca_px", float("nan"))
        va, vo, _ = validation_error(model, valid, args.mad) if valid else (np.nan, np.nan, [])
        print(f"{name:36s} {stats['fonte_pupila']:>10s} {stats['correcao']:>12s} {fit_err:7.0f} "
              f"{loo:6.0f} {head:7.0f} {va:7.0f} ({vo:4.0f})")

    model, _, stats = fit_from_samples(data, screen, width_cm, args.mad)
    print("\nGlobo ajustado (cm, referencial da cabeça):")
    for eye, g in stats["globo"].items():
        print(f"  olho {eye:8s}: raio {g['raio_cm']:.2f} cm, pupila em repouso "
              f"{np.round(g['repouso_pupila_cm'], 3)}, ganho da pupila (x, y) "
              f"{np.round(g['ganho_pupila'], 2)}, peso no ponto da tela {g['peso']:.2f}")
    if valid:
        print("\nValidação por alvo (padrão):")
        for t, pred, e in validation_error(model, valid, args.mad)[2]:
            print(f"  alvo ({t[0]:5.0f}, {t[1]:5.0f}) -> ({pred[0]:5.0f}, {pred[1]:5.0f})  "
                  f"erro {e:4.0f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
