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
        F, R, O = [], [], []
        for o in observations:
            s = est.estimate(o)
            if s.valid:
                F.append(s.features())
                R.append(s.raw)
                O.append(o)
        return np.array(F).reshape(-1, 2), np.array(R), O

    out = CalibrationData()
    for (t, _, _), obs in zip(data.grid, data.grid_obs):
        f, r, o = run(obs)
        out.grid.append((t, f, r))
        out.grid_obs.append(o)
    if data.head_obs:
        out.head_target = data.head_target
        out.head_features, out.head_raw, out.head_obs = run(data.head_obs)
    if data.mouth_obs:
        out.mouth_target = data.mouth_target
        out.mouth_features, out.mouth_raw, out.mouth_obs = run(data.mouth_obs)
    out.feature_version = FEATURE_VERSION
    return out


def load(path: Path) -> CalibrationData | None:
    """Carrega e, se houver landmarks, recalcula. None se não der para usar."""
    data = CalibrationData.load(path)
    if data.has_landmarks:
        return recompute(data)
    return None if data.stale else data


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
        ("sem etapa da cabeça", dict(), CalibrationData(grid=data.grid, grid_obs=data.grid_obs)),
    ]
    print(f"\n{'variante':36s} {'pupila':>10s} {'correção':>12s} {'ajuste':>7s} {'LOO':>6s} "
          f"{'cabeça':>7s} {'validação':>14s}  (erro médio, px)")
    for name, kw, d in variants:
        model, results, stats = fit_from_samples(d, screen, width_cm, args.mad, **kw)
        fit_err = np.mean([r.error_px for r in results])
        loo = stats.get("erro_loo_medio_px", float("nan"))
        head = stats.get("erro_cv_cabeca_px", float("nan"))
        va, vo, _ = validation_error(model, valid, args.mad) if valid else (np.nan, np.nan, [])
        print(f"{name:36s} {stats['fonte_pupila']:>10s} {stats['correcao']:>12s} {fit_err:7.0f} "
              f"{loo:6.0f} {head:7.0f} {va:7.0f} ({vo:4.0f})")

    model, _, stats = fit_from_samples(data, screen, width_cm, args.mad)
    print("\nGlobo ajustado (cm, referencial da cabeça):")
    for eye, g in stats["globo"].items():
        print(f"  olho {eye:8s}: pupila em repouso {np.round(g['repouso_pupila_cm'], 3)}, "
              f"raio efetivo (x, y) {np.round(g['raio_efetivo_cm'], 3)}")
    if "interseccao_dist_tela_cm" in stats:
        print(f"Intersecção dos feixes na grade: z mediano {stats['interseccao_dist_tela_cm']:+.1f} cm "
              f"(convergem em {100 * stats['feixes_convergem_frac']:.0f}% das amostras)")
    if valid:
        print("\nValidação por alvo (padrão):")
        for t, pred, e in validation_error(model, valid, args.mad)[2]:
            print(f"  alvo ({t[0]:5.0f}, {t[1]:5.0f}) -> ({pred[0]:5.0f}, {pred[1]:5.0f})  "
                  f"erro {e:4.0f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
