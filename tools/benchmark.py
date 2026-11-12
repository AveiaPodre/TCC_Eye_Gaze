"""
Benchmark offline: compara variantes do modelo em TODAS as sessões gravadas.

Para cada ``results/calibracao_*.npz`` (e a validação gravada logo depois
dela, se houver), cada variante é ajustada e avaliada com três métricas:

* ``LOO``     - erro médio em alvos da grade deixados de fora do ajuste
                (pontos novos, cabeça parada);
* ``cabeça``  - validação cruzada da etapa da cabeça: cada direção
                (direita/esquerda/cima/baixo) fica fora do ajuste e o erro é
                medido nela (poses novas da cabeça). O conjunto de teste é o
                mesmo para todas as variantes: amostras até ±15° (yaw) e
                ±12° (pitch) da pose da grade, a faixa de uso realista;
* ``valid.``  - erro médio nos alvos da validação (outro momento da sessão)
                e, entre parênteses, o mesmo erro depois de remover o
                deslocamento médio (o que a recentralização R corrige).

Uso:
    python tools/benchmark.py
    python tools/benchmark.py --sem-recalcular     # usa as medidas gravadas
    python tools/benchmark.py --screen 1920x1080

Sessões com landmarks gravados são recalculadas com o estimador ATUAL (as
medidas mudam quando gaze.py muda); as antigas, sem landmarks, usam as
medidas gravadas na época.
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
                                 head_angles, head_direction, outlier_mask)
from reavaliar import recompute  # noqa: E402

TEST_RANGE_DEG = (15.0, 12.0)   # faixa das amostras de teste da cabeça
PAIR_WINDOW_S = 30 * 60         # validação até 30 min depois da calibração

# (nome, argumentos de fit_from_samples). A primeira é a configuração padrão.
VARIANTS: list[tuple[str, dict]] = [
    ("padrão", dict()),
    ("ganho da cabeça aprendido", dict(_learn_head=True)),
    ("faixa 10/8", dict(head_max_deg=(10.0, 8.0))),
    ("etapa da cabeça peso 3", dict(head_weight_points=3.0)),
    ("sem etapa da cabeça", dict(_no_head=True)),
]
# px/cm da tela (ganho da cabeça fixado pela geometria); definido em main().
HEAD_PX_PER_CM: float | None = None


@dataclass
class Session:
    name: str
    calib: CalibrationData
    valid: CalibrationData | None
    recalculated: bool = False


def _stamp(path: Path) -> datetime | None:
    m = re.search(r"(\d{8}_\d{6})", path.name)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


def load_sessions(results: Path, recalc: bool) -> list[Session]:
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
        c = CalibrationData.load(cp)
        v = CalibrationData.load(vp) if vp else None
        recalculated = recalc and c.has_landmarks
        if recalculated:
            c = recompute(c)
        if recalc and v is not None and v.has_landmarks:
            v = recompute(v)
        sessions.append(Session(cp.stem.replace("calibracao_", ""), c, v, recalculated))
    return sessions


def _fit(data: CalibrationData, screen, kw: dict):
    kw = dict(kw)
    if not kw.pop("_learn_head", False):
        kw.setdefault("head_px_per_cm", HEAD_PX_PER_CM)
    if kw.pop("_no_head", False):
        data = CalibrationData(grid=data.grid)
    model, _, stats = fit_from_samples(data, screen, 2, None, 3.5, **kw)
    return model, stats


def head_cv(data: CalibrationData, screen, kw: dict) -> float:
    """Erro médio (px) em direções da cabeça deixadas de fora do ajuste.

    Na variante sem etapa da cabeça, ``_fit`` descarta o treino da cabeça,
    então a métrica mede como o modelo só com a grade se sai nessas poses.
    """
    if data.head_raw is None or len(data.head_raw) < 10:
        return float("nan")
    hr = np.atleast_2d(data.head_raw)
    hf = np.atleast_2d(data.head_features)
    grid_raw = np.vstack([np.atleast_2d(r) for _, _, r in data.grid])
    delta = head_angles(hr) - np.median(head_angles(grid_raw), axis=0)
    direction = head_direction(delta)
    in_test_range = ((np.abs(delta[:, 0]) <= TEST_RANGE_DEG[0])
                     & (np.abs(delta[:, 1]) <= TEST_RANGE_DEG[1])
                     & (np.max(np.abs(delta), axis=1) >= 2.0))
    errs = []
    for q in range(4):
        test = in_test_range & (direction == q)
        if test.sum() < 5:
            continue
        train = direction != q
        d = CalibrationData(grid=data.grid, head_target=data.head_target,
                            head_raw=hr[train], head_features=hf[train])
        model, _ = _fit(d, screen, kw)
        pred = model.predict(hr[test])
        errs.append(float(np.median(np.hypot(*(pred - np.asarray(data.head_target)).T))))
    return float(np.mean(errs)) if errs else float("nan")


def validation(model, valid: CalibrationData | None) -> tuple[float, float]:
    if valid is None:
        return float("nan"), float("nan")
    P, T = [], []
    for target, feats, raw in valid.grid:
        raw = np.atleast_2d(raw)
        if len(raw) < 3:
            continue
        P.append(np.median(model.predict(raw[outlier_mask(feats, 3.5)]), axis=0))
        T.append(target)
    E = np.array(P) - np.array(T)
    return float(np.hypot(*E.T).mean()), float(np.hypot(*(E - E.mean(0)).T).mean())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--screen", type=str, default="1920x1080")
    p.add_argument("--screen-width-cm", type=float, default=None,
                   help="largura física da tela (padrão: detecta no Windows); fixa o ganho "
                        "da cabeça. Sem ela, o ganho é aprendido em todas as variantes")
    p.add_argument("--sem-recalcular", action="store_true",
                   help="não recalcula as medidas a partir dos landmarks gravados")
    args = p.parse_args(argv)
    screen = tuple(int(v) for v in args.screen.lower().split("x"))
    global HEAD_PX_PER_CM
    from eyegaze.io_devices import detect_screen_width_cm
    width_cm = args.screen_width_cm or detect_screen_width_cm(screen)
    HEAD_PX_PER_CM = screen[0] / width_cm if width_cm else None
    print(f"Tela: {screen[0]}x{screen[1]}, "
          + (f"{width_cm:.1f} cm ({HEAD_PX_PER_CM:.1f} px/cm)" if width_cm
             else "largura física desconhecida (ganho da cabeça aprendido)"))

    sessions = load_sessions(args.results, recalc=not args.sem_recalcular)
    if not sessions:
        print(f"Nenhuma calibração em {args.results}.")
        return 1
    print("Sessões:")
    for s in sessions:
        print(f"  {s.name}: {len(s.calib.grid)} alvos, etapa da cabeça "
              f"{'sim' if s.calib.head_raw is not None else 'não'}, "
              f"medidas {'recalculadas' if s.recalculated else 'gravadas'}, "
              f"validação {'sim' if s.valid is not None else 'não'}")

    print("\nErros em px. valid. = erro na validação (depois de recentralizar).")
    header = f"{'variante':26s}" + "".join(f" | {s.name:^27s}" for s in sessions) + " | média LOO  cabeça  valid."
    print(header)
    print(f"{'':26s}" + " | LOO  cabeça  valid.       " * len(sessions))
    for name, kw in VARIANTS:
        row, loos, heads, vals = f"{name:26s}", [], [], []
        for s in sessions:
            model, stats = _fit(s.calib, screen, kw)
            loo = stats.get("erro_loo_medio_px", float("nan"))
            hc = head_cv(s.calib, screen, kw)
            va, vo = validation(model, s.valid)
            loos.append(loo)
            heads.append(hc)
            vals.append(va)
            vtxt = "  -  " if np.isnan(va) else f"{va:4.0f}({vo:3.0f})"
            row += f" | {loo:4.0f}  {hc:5.0f}  {vtxt:10s}"
        row += (f" | {np.nanmean(loos):5.0f}  {np.nanmean(heads):5.0f}  "
                f"{np.nanmean(vals) if not np.all(np.isnan(vals)) else float('nan'):5.0f}")
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
