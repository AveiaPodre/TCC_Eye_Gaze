"""
Bancada do estabilizador do cursor com o ruído REAL das sessões gravadas.

Para cada validação gravada (``results/validacao_*.npz``) com a calibração
logo antes, o olhar estimado em cada alvo (ponto bruto no monitor real,
quadro a quadro) dá uma sequência de ruído real de uma fixação, centrada na
própria mediana. Sequências de fixações são encadeadas com degraus
sintéticos de tamanho conhecido (movimentos sutis de 40-100 px e sacadas de
200-400 px) e passadas pelo estabilizador. Métricas:

* ``segue``    - fração dos degraus em que o cursor chega a menos de 30% do
                 degrau (ou 20 px) do destino em até 0,5 s;
* ``latência`` - mediana dos ms até isso acontecer (degraus seguidos);
* ``parado``   - fração dos quadros de fixação (depois de 0,5 s) com o
                 cursor imóvel (< 1 px);
* ``tremor``   - movimento médio por quadro (px) nesses quadros;
* ``erro``     - distância média do cursor ao centro da fixação depois de
                 0,5 s (px).

Uso:
    python tools/bancada_cursor.py            # padrão x variantes
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eyegaze.calibration import fit_from_samples  # noqa: E402
from eyegaze.config import StabilizerConfig  # noqa: E402
from eyegaze.filters import make_stabilizer  # noqa: E402
from reavaliar import load  # noqa: E402

SCREEN = (1920, 1080)
FPS = 30.0
STEPS = (40, 70, 100, 200, 400)


def _stamp(p: Path) -> datetime:
    return datetime.strptime(re.search(r"(\d{8}_\d{6})", p.name).group(1), "%Y%m%d_%H%M%S")


def noise_sequences(results: Path, width_cm: float = 34.4, limit: int | None = None):
    """Sequências (n, 2) de ruído real (ponto bruto - mediana) por alvo."""
    cals = sorted(results.glob("calibracao_*.npz"))
    seqs = []
    for v in sorted(results.glob("validacao_*.npz")):
        prev = [c for c in cals if 0 < (_stamp(v) - _stamp(c)).total_seconds() < 90]
        if not prev:
            continue
        C, V = load(prev[-1]), load(v)
        if C is None or V is None or len(C.grid) < 5:
            continue
        try:
            model, _, _ = fit_from_samples(C, SCREEN, width_cm)
        except Exception:  # noqa: BLE001
            continue
        for _, _, r in V.grid:
            if len(r) < 25:
                continue
            P = model.predict(r)
            P = P[np.all(np.isfinite(P), axis=1)]
            if len(P) >= 25:
                seqs.append(P - np.median(P, axis=0))
        if limit and len(seqs) >= limit:
            break
    return seqs


def run(seqs, cfg: StabilizerConfig, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    follow = {d: [] for d in STEPS}
    lat = {d: [] for d in STEPS}
    still, jitter, err = [], [], []
    half = int(0.5 * FPS)
    for k in range(0, len(seqs) - 1, 1):
        stab = make_stabilizer(cfg, SCREEN[0])
        center = np.array([960.0, 540.0])
        # fixação inicial
        for p in seqs[k]:
            stab(center + p)
        for j, d in enumerate(STEPS):
            ang = rng.uniform(0, 2 * np.pi)
            new = center + d * np.array([np.cos(ang), np.sin(ang)])
            noise = seqs[(k + 1 + j) % len(seqs)]
            path = []
            for p in noise:
                path.append(stab(new + p))
            path = np.array(path)
            dist = np.linalg.norm(path - new, axis=1)
            tol = max(0.3 * d, 20.0)
            hit = np.flatnonzero(dist[:half] < tol)
            follow[d].append(len(hit) > 0)
            if len(hit):
                lat[d].append(1000.0 * (hit[0] + 1) / FPS)
            tail = path[half:]
            if len(tail) > 3:
                mv = np.linalg.norm(np.diff(tail, axis=0), axis=1)
                still.append(np.mean(mv < 1.0))
                jitter.append(np.mean(mv))
                err.append(np.mean(np.linalg.norm(tail - new, axis=1)))
            center = new if np.all((new > 100) & (new < np.array(SCREEN) - 100)) \
                else np.array([960.0, 540.0])
            if not np.all((new > 0) & (new < np.array(SCREEN))):
                stab.reset()
    return {
        "segue": {d: float(np.mean(follow[d])) for d in STEPS},
        "latencia_ms": {d: (float(np.median(lat[d])) if lat[d] else float("nan")) for d in STEPS},
        "parado": float(np.mean(still)),
        "tremor": float(np.mean(jitter)),
        "erro": float(np.mean(err)),
    }


def fmt(name, r):
    s = " ".join(f"{100 * r['segue'][d]:3.0f}%" for d in STEPS)
    l = " ".join(f"{r['latencia_ms'][d]:4.0f}" for d in STEPS)
    return (f"{name:34s} segue {s} | lat ms {l} | parado {100 * r['parado']:3.0f}% "
            f"tremor {r['tremor']:4.1f} erro {r['erro']:4.0f}")


VARIANTS = [
    ("padrão", dict()),
    ("até 17/11 (start_k 4, alpha 0,4, mediana 7)",
     dict(start_k=4.0, alpha=0.4, median_frames=7)),
    ("start_k 3", dict(start_k=3.0)),
    ("start_k 2", dict(start_k=2.0)),
    ("alpha 0,5", dict(alpha=0.5)),
    ("mediana 7", dict(median_frames=7)),
    ("sacada a partir de 5% da tela", dict(saccade_ratio=0.05)),
]


def main(argv=None) -> int:
    seqs = noise_sequences(ROOT / "results")
    print(f"{len(seqs)} fixações reais; degraus {STEPS} px")
    for name, kw in VARIANTS:
        print(fmt(name, run(seqs, StabilizerConfig(**kw))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
