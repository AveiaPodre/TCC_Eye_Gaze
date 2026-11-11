"""
Modelo de calibração: medidas do rosto -> pixel da tela.

Versão 3 (regressão com geometria da cabeça). A versão anterior otimizava
parâmetros do olho (ganho, offset) por Nelder-Mead dentro de um modelo
geométrico de raio do olhar. Com dados reais isso falhou: o sinal vertical
da íris é ~5x menor que o previsto pela esfera e o pitch tem um viés de
~30 graus, então o otimizador ia para os limites e nem os próprios alvos da
calibração eram ajustados (RMSE de ~300 px).

Agora a geometria entra só onde ela é confiável, a CABEÇA:

    P_cabeça = E_xy + E_z * tan(direção da face)     (cm, plano da câmera)

é o ponto para onde a face aponta no plano da tela. Isso já compensa
rotação e translação da cabeça. O olho entra por medidas brutas (íris
horizontal u, íris vertical v, íris entre as pálpebras lid_v e abertura do
olho), multiplicadas pela distância E_z (o mesmo ângulo de olho cobre mais
centímetros quando se está mais longe) e por sec²(ângulo da cabeça), a
derivada de tan(cabeça + olho) em relação ao olho:

    x = c + a·P_cabeça + Σ_k b_k · E_z · sec²(yaw_cabeça)   · φ_k(olho)
    y = c + a·P_cabeça + Σ_k b_k · E_z · sec²(pitch_cabeça) · φ_k(olho)

φ = [u, v, lid_v, abertura] (+ termos quadráticos no grau 2). O modelo é
LINEAR nos coeficientes: mínimos quadrados com ridge, sem otimização
iterativa e sem mínimos locais. A força do ridge é escolhida por validação
cruzada deixando um alvo de fora (leave-one-target-out), e o erro dessa
validação é reportado: ele estima o erro em pontos não vistos, ao contrário
do erro de ajuste, que esconde sobreajuste.

A etapa em que a pessoa mantém o olhar no centro e gira a cabeça é o que
determina o coeficiente de P_cabeça. Sem ela, o modelo continua funcionando
com a cabeça parada.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .face_tracker import FaceObservation

FEATURE_VERSION = 5  # incremente se o formato mudar (invalida calibrações antigas)

RAW_SIZE = 20        # R (9) + E (3) + (u, v, lid_v, abertura) x 2 fontes da íris
# Colunas das medidas do olho para cada fonte da posição da íris.
EYE_COLS = {"mediapipe": slice(12, 16), "refinada": slice(16, 20)}
ALPHAS = (1e-4, 1e-3, 1e-2, 0.1, 1.0)   # candidatos de ridge (features padronizadas)
# Desvio-padrão mínimo das colunas na padronização. Sem isso, uma coluna
# quase constante (ex.: cabeça parada durante a calibração) teria o ruído
# amplificado e o coeficiente "explodiria" quando a cabeça se mexe.
HEAD_STD_FLOOR_CM = 1.0
EYE_STD_FLOOR = 1e-3


# ------------------------------------------------------------------ geometria
def head_screen_point(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ponto para onde a face aponta no plano z = 0 da câmera (cm) e as
    tangentes (tan yaw, tan pitch) da direção da face.

    raw: (n, 20). Devolve ((n, 2), (n, 2)).
    """
    raw = np.atleast_2d(raw)
    R = raw[:, :9].reshape(-1, 3, 3)
    E = raw[:, 9:12]
    fwd = -R[:, :, 2]                         # a face olha para -z da cabeça
    fz = np.minimum(fwd[:, 2], -1e-3)         # voltada para a câmera: fz < 0
    tans = fwd[:, :2] / -fz[:, None]
    return E[:, :2] + E[:, 2:3] * tans, tans


def eye_terms(raw: np.ndarray, degree: int, ref: np.ndarray | None = None,
              source: str = "mediapipe") -> np.ndarray:
    """φ(olho): [u, v, lid_v, abertura] (+ u², u·lid_v, lid_v², u·v no grau 2).

    ref: valores de referência (média da calibração) subtraídos antes. As
    medidas não são zero com o olho "reto" (a abertura, por exemplo, é ~0.3);
    sem centralizar, multiplicar por E_z faria a distância deslocar o cursor
    mesmo com o olho parado.
    """
    raw = np.atleast_2d(raw)
    m = raw[:, EYE_COLS[source]] - (0.0 if ref is None else ref)
    u, v, lv, ap = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    cols = [u, v, lv, ap]
    if degree >= 2:
        cols += [u * u, u * lv, lv * lv, u * v]
    return np.column_stack(cols)


def design_matrices(raw: np.ndarray, degree: int, ref: np.ndarray | None = None,
                    source: str = "mediapipe") -> tuple[np.ndarray, np.ndarray]:
    """Matrizes de regressão (sem intercepto) para x e para y da tela."""
    raw = np.atleast_2d(raw)
    P, tans = head_screen_point(raw)
    Ez = raw[:, 11:12]
    phi = eye_terms(raw, degree, ref, source)
    sec2 = 1.0 + tans ** 2
    Xx = np.hstack([P, Ez * sec2[:, 0:1] * phi])
    Xy = np.hstack([P, Ez * sec2[:, 1:2] * phi])
    return Xx, Xy


def polynomial_terms(X: np.ndarray, degree: int) -> np.ndarray:
    """[1, a, b] (grau 1) ou [1, a, b, a², ab, b²] (grau 2) para X = [a, b]."""
    X = np.atleast_2d(X)
    a, b = X[:, 0], X[:, 1]
    cols = [np.ones_like(a), a, b]
    if degree >= 2:
        cols += [a * a, a * b, b * b]
    return np.column_stack(cols)


def outlier_mask(samples: np.ndarray, threshold: float) -> np.ndarray:
    """Máscara de amostras a manter (escore-Z modificado via MAD < limiar).

    Iglewicz & Hoaglin (1993): z = 0.6745 (x - mediana) / MAD; 3.5 é o
    limiar usual. Remove frames isolados ruins (sacada atrasada, quase-piscada).
    """
    samples = np.atleast_2d(samples)
    if len(samples) < 5:
        return np.ones(len(samples), bool)
    med = np.median(samples, axis=0)
    mad = np.maximum(np.median(np.abs(samples - med), axis=0), 1e-6)
    keep = np.all(0.6745 * np.abs(samples - med) / mad < threshold, axis=1)
    if keep.sum() < max(3, len(samples) // 3):
        return np.ones(len(samples), bool)  # rejeição agressiva demais
    return keep


def reject_outliers(samples: np.ndarray, threshold: float) -> np.ndarray:
    return np.atleast_2d(samples)[outlier_mask(samples, threshold)]


def build_grid(screen_w: int, screen_h: int, cols: int, rows: int,
               margin_ratio: float) -> list[tuple[float, float]]:
    """Alvos de calibração em pixels, percorridos em zigue-zague."""
    xs = np.linspace(margin_ratio, 1 - margin_ratio, cols) * screen_w
    ys = np.linspace(margin_ratio, 1 - margin_ratio, rows) * screen_h
    points = []
    for r, y in enumerate(ys):
        row = xs if r % 2 == 0 else xs[::-1]
        points.extend((float(x), float(y)) for x in row)
    return points


@dataclass
class PointResult:
    target: tuple[float, float]
    prediction: tuple[float, float]   # mediana das previsões das amostras
    n_samples: int

    @property
    def error_px(self) -> float:
        return float(np.hypot(self.prediction[0] - self.target[0],
                              self.prediction[1] - self.target[1]))


# --------------------------------------------------------------------- modelo
def _std_floor(n_cols: int) -> np.ndarray:
    floor = np.full(n_cols, EYE_STD_FLOOR)
    floor[:2] = HEAD_STD_FLOOR_CM   # as duas primeiras colunas são P_cabeça
    return floor


def _ridge(X, y, w, alpha, floor):
    """Ridge ponderado com padronização; intercepto não penalizado."""
    mean = np.average(X, axis=0, weights=w)
    std = np.maximum(np.sqrt(np.average((X - mean) ** 2, axis=0, weights=w)), floor)
    Z = np.column_stack([np.ones(len(X)), (X - mean) / std])
    Zw = Z * w[:, None]
    reg = alpha * w.sum() * np.eye(Z.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Z.T @ Zw + reg, Zw.T @ y)
    return coef, mean, std


def _ridge_predict(X, coef, mean, std):
    return coef[0] + ((X - mean) / std) @ coef[1:]


class CalibrationModel:
    def __init__(self, degree: int = 2, ridge_alpha: float | None = None,
                 eye_source: str = "mediapipe"):
        self.degree = degree
        self.eye_source = eye_source            # fonte da posição da íris (EYE_COLS)
        self.ridge_alpha = ridge_alpha          # None = escolhe por validação cruzada
        self.params: list | None = None         # [(coef, mean, std) de x, de y]
        self.eye_ref = np.zeros(4)              # referencia das medidas do olho
        self.offset = np.zeros(2)               # correção de "recentralizar"
        self.screen_size: tuple[int, int] | None = None
        self.created_at: str | None = None
        self.info: dict = {}

    @property
    def is_fitted(self) -> bool:
        return self.params is not None

    # -------------------------------------------------------------- ajuste
    def _fit_xy(self, raw, Y, w, alpha):
        Xx, Xy = design_matrices(raw, self.degree, self.eye_ref, self.eye_source)
        floor = _std_floor(Xx.shape[1])
        return [_ridge(Xx, Y[:, 0], w, alpha, floor), _ridge(Xy, Y[:, 1], w, alpha, floor)]

    def _predict_with(self, params, raw):
        Xx, Xy = design_matrices(raw, self.degree, self.eye_ref, self.eye_source)
        return np.column_stack([_ridge_predict(Xx, *params[0]),
                                _ridge_predict(Xy, *params[1])])

    def fit(self, raw: np.ndarray, Y: np.ndarray, weights: np.ndarray | None,
            screen_size: tuple[int, int], groups: np.ndarray | None = None):
        """raw: (n, 20); Y: (n, 2) pixels-alvo; weights: (n,) ou None.

        groups: (n,) índice do alvo de cada amostra (-1 = sempre no treino,
        ex.: etapa da cabeça). Usado na validação cruzada que escolhe o ridge
        e estima o erro em pontos não vistos.
        """
        raw = np.asarray(raw, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        w = np.ones(len(raw)) if weights is None else np.asarray(weights, dtype=np.float64)

        self.eye_ref = np.average(raw[:, EYE_COLS[self.eye_source]], axis=0, weights=w)
        loo: dict[float, list[float]] = {}
        ids = [] if groups is None else [g for g in np.unique(groups) if g >= 0]
        if len(ids) >= 5:
            candidates = ALPHAS if self.ridge_alpha is None else (self.ridge_alpha,)
            for alpha in candidates:
                errs = []
                for g in ids:
                    tr, te = groups != g, groups == g
                    params = self._fit_xy(raw[tr], Y[tr], w[tr], alpha)
                    pred = np.median(self._predict_with(params, raw[te]), axis=0)
                    errs.append(float(np.hypot(*(pred - Y[te][0]))))
                loo[alpha] = errs
            alpha = min(loo, key=lambda a: np.mean(loo[a]))
        else:
            alpha = self.ridge_alpha if self.ridge_alpha is not None else 1e-2

        self.params = self._fit_xy(raw, Y, w, alpha)
        resid = self._predict_with(self.params, raw) - Y
        mse = float(np.sum(w * np.sum(resid ** 2, axis=1)) / w.sum())
        self.offset = np.zeros(2)
        self.screen_size = tuple(int(v) for v in screen_size)
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.info = {"rmse_ajuste_px": float(np.sqrt(mse)), "ridge_alpha": float(alpha)}
        if loo:
            self.info["erro_loo_medio_px"] = float(np.mean(loo[alpha]))
            self.info["erro_loo_por_alvo_px"] = [float(e) for e in loo[alpha]]

    def predict(self, raw: np.ndarray) -> np.ndarray:
        """Aceita (20,) ou (n, 20); devolve (2,) ou (n, 2) em pixels."""
        if not self.is_fitted:
            raise RuntimeError("Modelo de calibração ainda não ajustado.")
        raw = np.asarray(raw, dtype=np.float64)
        out = self._predict_with(self.params, raw) + self.offset
        return out[0] if raw.ndim == 1 else out

    # ----------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {
            "feature_version": FEATURE_VERSION,
            "degree": self.degree,
            "ridge_alpha": self.ridge_alpha,
            "eye_source": self.eye_source,
            "params": [[c.tolist(), m.tolist(), s.tolist()] for c, m, s in self.params],
            "eye_ref": self.eye_ref.tolist(),
            "offset": self.offset.tolist(),
            "screen_size": list(self.screen_size),
            "created_at": self.created_at,
            "info": self.info,
        }

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path, screen_size: tuple[int, int]) -> "CalibrationModel | None":
        """Carrega a calibração salva; devolve None se incompatível."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if d.get("feature_version") != FEATURE_VERSION:
            print("[Calibração] Arquivo salvo é de uma versão antiga; recalibre.")
            return None
        if tuple(d.get("screen_size", ())) != tuple(screen_size):
            print("[Calibração] Arquivo salvo é de outra resolução de tela; recalibre.")
            return None
        m = cls(d["degree"], d.get("ridge_alpha"), d.get("eye_source", "mediapipe"))
        m.params = [tuple(np.array(a) for a in p) for p in d["params"]]
        m.eye_ref = np.array(d["eye_ref"])
        m.offset = np.array(d.get("offset", [0.0, 0.0]))
        m.screen_size = tuple(d["screen_size"])
        m.created_at = d.get("created_at")
        m.info = d.get("info", {})
        return m


@dataclass
class CalibrationData:
    """Amostras coletadas: por alvo da grade e na etapa de cabeça em movimento."""

    # lista de (alvo_px, features (n,2), raw (n,20))
    grid: list = field(default_factory=list)
    head_target: tuple[float, float] | None = None
    head_features: np.ndarray | None = None
    head_raw: np.ndarray | None = None
    # Opcional: landmarks de cada amostra (listas de FaceObservation), para
    # recalcular as medidas com outra versão do estimador (tools/reavaliar.py).
    grid_obs: list = field(default_factory=list)
    head_obs: list = field(default_factory=list)

    def save(self, path: Path):
        """Grava as medidas brutas (.npz) para reanálise offline."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"feature_version": np.array(FEATURE_VERSION)}
        for i, (t, f, r) in enumerate(self.grid):
            arrays[f"alvo_{i}"] = np.asarray(t, dtype=float)
            arrays[f"feats_{i}"] = np.atleast_2d(f)
            arrays[f"raw_{i}"] = np.atleast_2d(r)
            if i < len(self.grid_obs):
                _pack_obs(arrays, f"obs_{i}", self.grid_obs[i])
        if self.head_raw is not None:
            arrays["head_target"] = np.asarray(self.head_target, dtype=float)
            arrays["head_feats"] = np.atleast_2d(self.head_features)
            arrays["head_raw"] = np.atleast_2d(self.head_raw)
            _pack_obs(arrays, "obs_head", self.head_obs)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "CalibrationData":
        z = np.load(path)
        data = cls()
        i = 0
        while f"alvo_{i}" in z:
            data.grid.append((tuple(z[f"alvo_{i}"]), z[f"feats_{i}"],
                              _upgrade_raw(z[f"raw_{i}"])))
            data.grid_obs.append(_unpack_obs(z, f"obs_{i}"))
            i += 1
        if "head_raw" in z:
            data.head_target = tuple(z["head_target"])
            data.head_features, data.head_raw = z["head_feats"], _upgrade_raw(z["head_raw"])
            data.head_obs = _unpack_obs(z, "obs_head")
        return data

    @property
    def has_landmarks(self) -> bool:
        return bool(self.grid_obs) and all(len(o) for o in self.grid_obs)


def _upgrade_raw(raw: np.ndarray) -> np.ndarray:
    """Sessões antigas (16 colunas, sem íris refinada): repete as medidas
    do MediaPipe nas colunas da fonte refinada."""
    raw = np.atleast_2d(raw)
    if raw.shape[1] == 16:
        raw = np.hstack([raw, raw[:, 12:16]])
    return raw


def _pack_obs(arrays: dict, key: str, observations: list):
    obs = [o for o in observations if o is not None]
    if not obs or len(obs) != len(observations):
        return
    arrays[f"{key}_pts"] = np.stack([o.points for o in obs]).astype(np.float32)
    arrays[f"{key}_tf"] = np.stack([np.full((4, 4), np.nan) if o.transform is None
                                    else np.asarray(o.transform) for o in obs])
    arrays[f"{key}_meta"] = np.array([[o.image_width, o.image_height, o.timestamp_ms]
                                      for o in obs], dtype=np.int64)
    arrays[f"{key}_iris"] = np.stack([np.full((2, 2), np.nan) if o.iris_refined is None
                                      else o.iris_refined for o in obs])


def _unpack_obs(z, key: str) -> list:
    if f"{key}_pts" not in z:
        return []
    n = len(z[f"{key}_pts"])
    iris = z[f"{key}_iris"] if f"{key}_iris" in z else [None] * n
    out = []
    for pts, tf, (w, h, ts), ir in zip(z[f"{key}_pts"], z[f"{key}_tf"], z[f"{key}_meta"], iris):
        tf = None if np.isnan(tf).any() else tf
        ir = None if ir is None or np.isnan(ir).all() else ir
        out.append(FaceObservation(pts.astype(np.float64), int(w), int(h), int(ts), tf, ir))
    return out


def fit_from_samples(
    data: CalibrationData,
    screen_size: tuple[int, int],
    degree: int,
    ridge_alpha: float | None,
    mad_threshold: float,
    head_weight_points: float = 1.0,
) -> tuple[CalibrationModel, list[PointResult], dict]:
    """Ajusta o modelo. Devolve (modelo, resultados por alvo, estatísticas)."""
    raws, Ys, ws, gs, cleaned = [], [], [], [], []
    for i, (target, feats, raw) in enumerate(data.grid):
        keep = outlier_mask(feats, mad_threshold)
        r = np.atleast_2d(raw)[keep]
        cleaned.append((target, r))
        raws.append(r)
        Ys.append(np.tile(target, (len(r), 1)))
        ws.append(np.full(len(r), 1.0 / len(r)))   # cada alvo pesa 1
        gs.append(np.full(len(r), i))

    has_head = data.head_raw is not None and len(data.head_raw) >= 5
    if has_head:
        hr = np.atleast_2d(data.head_raw)
        raws.append(hr)
        Ys.append(np.tile(data.head_target, (len(hr), 1)))
        # A etapa de cabeça pesa o equivalente a `head_weight_points` alvos.
        ws.append(np.full(len(hr), head_weight_points / len(hr)))
        gs.append(np.full(len(hr), -1))

    if len(data.grid) < 9:
        degree = 1  # poucos alvos para os termos quadráticos

    raw_all, Y_all = np.vstack(raws), np.vstack(Ys)
    w_all, g_all = np.concatenate(ws), np.concatenate(gs)
    # Fontes da íris a testar: a refinada só se ela existir nesta sessão.
    sources = ["mediapipe"]
    if not np.allclose(raw_all[:, EYE_COLS["refinada"]], raw_all[:, EYE_COLS["mediapipe"]]):
        sources.append("refinada")
    candidates = []
    for src in sources:
        m = CalibrationModel(degree, ridge_alpha, src)
        m.fit(raw_all, Y_all, w_all, screen_size, groups=g_all)
        candidates.append(m)
    # Escolhe pelo erro em alvos não vistos (LOO); sem LOO, fica no MediaPipe.
    model = min(candidates, key=lambda m: m.info.get("erro_loo_medio_px", np.inf))
    if "erro_loo_medio_px" not in model.info:
        model = candidates[0]
    model.info["usa_etapa_cabeca"] = bool(has_head)
    model.info["fonte_iris"] = model.eye_source
    for m in candidates:
        if "erro_loo_medio_px" in m.info:
            model.info[f"erro_loo_{m.eye_source}_px"] = m.info["erro_loo_medio_px"]

    results = []
    for target, r in cleaned:
        pred = np.median(model.predict(r), axis=0)
        results.append(PointResult(target, (float(pred[0]), float(pred[1])), len(r)))

    stats = dict(model.info)
    if has_head:
        preds = model.predict(np.atleast_2d(data.head_raw))
        err = np.hypot(*(preds - np.asarray(data.head_target)).T)
        stats["erro_mediano_cabeca_px"] = float(np.median(err))
    return model, results, stats


def uncalibrated_mapping(features: np.ndarray, screen_size: tuple[int, int],
                         half_yaw: float, half_pitch: float) -> np.ndarray:
    """Mapeamento linear fixo usado só para visualizar antes de calibrar."""
    w, h = screen_size
    x = w / 2 * (1 + features[0] / half_yaw)
    y = h / 2 * (1 - features[1] / half_pitch)
    return np.array([x, y])
