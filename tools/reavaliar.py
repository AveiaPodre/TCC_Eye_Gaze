"""
Reanálise offline de uma sessão gravada (sem câmera).

Cada calibração grava ``results/calibracao_*.npz`` e cada validação grava
``results/validacao_*.npz`` com as medidas brutas de todos os frames. Este
script reajusta o modelo com outras configurações e mede o erro na
validação, para comparar alternativas com os MESMOS dados.

Uso:
    python tools/reavaliar.py                         # arquivos mais recentes
    python tools/reavaliar.py --calib results/calibracao_X.npz --valid results/validacao_Y.npz
    python tools/reavaliar.py --sinal                 # quanto cada medida varia com a tela

A validação só faz sentido com a calibração feita na mesma sessão (mesma
posição, luz e tela).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eyegaze.calibration import (EYE_COLS, CalibrationData, fit_from_samples,  # noqa: E402
                                 outlier_mask)
from eyegaze.config import GazeConfig  # noqa: E402
from eyegaze.gaze import GazeEstimator  # noqa: E402

MEASURES = ("u (iris horiz.)", "v (iris vert.)", "lid_up (palpebra sup.)", "abertura")


def latest(pattern: str) -> Path | None:
    files = sorted((ROOT / "results").glob(pattern))
    return files[-1] if files else None


def validation_error(model, valid: CalibrationData, mad: float) -> tuple[float, list]:
    errs = []
    for target, feats, raw in valid.grid:
        keep = outlier_mask(feats, mad)
        pred = np.median(model.predict(np.atleast_2d(raw)[keep]), axis=0)
        errs.append((target, pred, float(np.hypot(*(pred - np.asarray(target))))))
    return float(np.mean([e for *_, e in errs])), errs


def recompute(data: CalibrationData) -> CalibrationData:
    """Recalcula features e medidas brutas a partir dos landmarks gravados,
    com a versão ATUAL do estimador do olhar."""
    est = GazeEstimator(GazeConfig())

    def run(observations):
        F, R = [], []
        for o in observations:
            s = est.estimate(o)
            if s.valid:
                F.append(s.features())
                R.append(s.raw)
        return np.array(F), np.array(R)

    out = CalibrationData()
    for (t, _, _), obs in zip(data.grid, data.grid_obs):
        out.grid.append((t, *run(obs)))
    if data.head_obs:
        out.head_target = data.head_target
        out.head_features, out.head_raw = run(data.head_obs)
    if data.mouth_obs:
        out.mouth_target = data.mouth_target
        out.mouth_features, out.mouth_raw = run(data.mouth_obs)
    return out


def signal_report(data: CalibrationData, source: str):
    """Para cada medida do olho: variação entre alvos (em desvios do ruído)."""
    cols = EYE_COLS[source]
    targets = np.array([t for t, _, _ in data.grid])
    meds = np.array([np.median(np.atleast_2d(r)[:, cols], axis=0) for _, _, r in data.grid])
    noise = np.array([np.std(np.atleast_2d(r)[:, cols], axis=0) for _, _, r in data.grid])
    noise = np.median(noise, axis=0)
    print(f"\nSinal de cada medida, íris {source} (correlação com a posição do alvo; "
          "sinal/ruído = variação explicada pelo alvo / ruído dentro do alvo):")
    for k, name in enumerate(MEASURES):
        cx = np.corrcoef(meds[:, k], targets[:, 0])[0, 1]
        cy = np.corrcoef(meds[:, k], targets[:, 1])[0, 1]
        A = np.column_stack([np.ones(len(targets)), targets])
        fit = A @ np.linalg.lstsq(A, meds[:, k], rcond=None)[0]
        snr = np.std(fit) / max(noise[k], 1e-9)
        print(f"  {name:24s} corr x {cx:+.2f}  corr y {cy:+.2f}  sinal/ruido {snr:5.1f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--valid", type=Path, default=None)
    p.add_argument("--screen", type=str, default="1920x1080")
    p.add_argument("--sinal", action="store_true", help="mostra o sinal de cada medida")
    p.add_argument("--mad", type=float, default=3.5)
    p.add_argument("--recalcular", action="store_true",
                   help="recalcula as medidas a partir dos landmarks gravados (estimador atual)")
    args = p.parse_args(argv)

    calib_path = args.calib or latest("calibracao_*.npz")
    if calib_path is None:
        print("Nenhum results/calibracao_*.npz encontrado. Calibre o app primeiro.")
        return 1
    valid_path = args.valid or latest("validacao_*.npz")
    screen = tuple(int(v) for v in args.screen.lower().split("x"))

    data = CalibrationData.load(calib_path)
    valid = CalibrationData.load(valid_path) if valid_path else None
    print(f"Calibração: {calib_path.name} ({len(data.grid)} alvos, "
          f"etapa da cabeça: {'sim' if data.head_raw is not None else 'não'})")
    print(f"Validação:  {valid_path.name if valid_path else '(nenhuma)'}")
    if args.recalcular:
        for name, d in (("calibração", data), ("validação", valid)):
            if d is not None and not d.has_landmarks:
                print(f"A {name} não tem landmarks gravados (sessão antiga); "
                      "não dá para recalcular.")
                return 1
        data = recompute(data)
        valid = recompute(valid) if valid else None
        print("Medidas recalculadas com o estimador atual.")
    raw0 = np.atleast_2d(data.grid[0][2])
    has_refined = not np.allclose(raw0[:, EYE_COLS["refinada"]], raw0[:, EYE_COLS["mediapipe"]])
    if not has_refined:
        print("(sessão sem íris refinada: só a fonte MediaPipe)")
    if args.sinal:
        for src in (("mediapipe", "refinada") if has_refined else ("mediapipe",)):
            signal_report(data, src)

    # "grau até 2": os graus 1 e 2 são testados e fica o de menor erro na
    # validação cruzada (alvos + direções da cabeça), como no app.
    variants = [
        ("grau até 2, com etapa da cabeça", dict(degree=2), data),
        ("grau 1 fixo, com etapa da cabeça", dict(degree=1), data),
        ("grau até 2, sem etapa da cabeça", dict(degree=2), CalibrationData(grid=data.grid)),
    ]
    print(f"\n{'modelo':34s} {'fonte':>10s} {'grau':>4s} {'ajuste':>7s} {'LOO':>6s} "
          f"{'cabeça':>7s} {'validação':>10s}  (erro médio, px)")
    for name, kw, d in variants:
        model, results, stats = fit_from_samples(d, screen, kw["degree"], None, args.mad)
        fit_err = np.mean([r.error_px for r in results])
        loo = stats.get("erro_loo_medio_px", float("nan"))
        head = stats.get("erro_cv_cabeca_px", float("nan"))
        val = validation_error(model, valid, args.mad)[0] if valid else float("nan")
        print(f"{name:34s} {model.eye_source:>10s} {model.degree:4d} {fit_err:7.0f} {loo:6.0f} "
              f"{head:7.0f} {val:10.0f}")
        if len(stats.get("candidatos", [])) > 1:
            print("      candidatos: " + ", ".join(
                f"{c['fonte']}/grau {c['grau']}: LOO {c['erro_loo_px']:.0f}"
                + (f", cabeça {c['erro_cv_cabeca_px']:.0f}" if c["erro_cv_cabeca_px"] else "")
                for c in stats["candidatos"] if c["erro_loo_px"] is not None))

    if valid:
        model, *_ = fit_from_samples(data, screen, 2, None, args.mad)
        print(f"\nValidação por alvo (modelo padrão: grau {model.degree}, com etapa da cabeça):")
        for t, pred, e in validation_error(model, valid, args.mad)[1]:
            print(f"  alvo ({t[0]:5.0f}, {t[1]:5.0f}) -> ({pred[0]:5.0f}, {pred[1]:5.0f})  "
                  f"erro {e:4.0f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
