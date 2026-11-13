"""
Etapa 5 do pipeline: calibração com vários pontos.

O cursor sai do pipeline geométrico (ver gaze.py e eyeball.py):

    landmarks -> âncora + pupila de cada olho (cm)
              -> centro do globo de cada olho -> feixe centro -> pupila
              -> intersecção dos dois feixes -> vetor do olhar
              -> projeção no MONITOR VIRTUAL (plano da tela, geometria suposta)
              -> correção monitor virtual -> MONITOR REAL (pixels)

A calibração mostra alvos conhecidos (grade 4x4) e uma etapa em que a
pessoa mantém o olhar no centro e gira a cabeça. Com essas amostras ela
ajusta, nesta ordem:

1. O GLOBO de cada olho (``eyeball.EyeballModel.fit``): para cada olho e
   eixo, a posição de repouso da pupila e o raio efetivo, por mínimos
   quadrados lineares. É isto que faz o feixe passar pelo alvo, e a etapa
   da cabeça é o que separa o movimento do olho do movimento da cabeça.
2. A CORREÇÃO monitor virtual -> real: o ponto atingido no monitor virtual
   (etapa 4) ainda tem erros que não são do olho: largura física da tela
   suposta, posição da webcam, campo de visão da câmera, viés da pose do
   MediaPipe. A correção é um polinômio pequeno nas coordenadas do monitor
   virtual, somado ao ponto virtual:

       real = virtual + a₀ (+ a₁·u + a₂·v) (+ a₃·u² + a₄·uv + a₅·v²)

   com (u, v) o ponto virtual normalizado pela tela, ajustada por ridge
   (penalidade em direção à identidade). Os candidatos são: nenhuma
   correção (geometria pura), só deslocamento, afim e quadrática.

A escolha (fonte da pupila: MediaPipe ou refinada na imagem; tipo de
correção; força do ridge) é por validação cruzada: cada alvo fica fora do
ajuste por vez (o globo E a correção são reajustados sem ele), e o erro
nesse alvo estima o erro em pontos novos. Cada direção da etapa da cabeça
também fica fora por vez e o erro nela (cabeça em poses novas) é reportado.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .eyeball import EYE_NAMES, EyeballModel
from .face_tracker import FaceObservation
from .gaze import MOUTH_COL, RAW_FACE, RAW_R, RAW_SIZE, has_refined_pupil
from . import geometry
from .geometry import VirtualScreen

FEATURE_VERSION = 9  # incremente se o layout de ``raw`` mudar (invalida calibrações antigas)

CORRECTIONS = ("nenhuma", "deslocamento", "afim", "quadratica")
ALPHAS = (1e-3, 1e-2, 1e-1)       # ridge da correção (coordenadas normalizadas)

# Grupos da validação cruzada (``groups``): >= 0 é o índice do alvo;
# ALWAYS_TRAIN entra sempre no treino; as direções da etapa da cabeça são
# HEAD_GROUP_BASE - k (k = 0..3, ver head_direction).
ALWAYS_TRAIN = -1
HEAD_GROUP_BASE = -2
HEAD_DIRECTIONS = ("direita", "esquerda", "cima", "baixo")
HEAD_STILL_DEG = 2.0       # desvio abaixo disto não é "pose nova" (sempre no treino)
MIN_HEAD_FOLD = 5          # amostras mínimas para uma direção virar dobra da validação


# ------------------------------------------------------------------ cabeça
def head_angles(raw: np.ndarray) -> np.ndarray:
    """(n, 2) yaw e pitch da cabeça em graus (mesma convenção de HeadPose.euler_deg)."""
    raw = np.atleast_2d(raw)
    fwd = -raw[:, RAW_R].reshape(-1, 3, 3)[:, :, 2]
    yaw = np.degrees(np.arctan2(-fwd[:, 0], -fwd[:, 2]))
    pitch = np.degrees(np.arctan2(-fwd[:, 1], np.hypot(fwd[:, 0], fwd[:, 2])))
    return np.column_stack([yaw, pitch])


def head_direction(delta: np.ndarray) -> np.ndarray:
    """Direção dominante de cada desvio de pose (n, 2) -> índice em HEAD_DIRECTIONS."""
    delta = np.atleast_2d(delta)
    horiz = np.abs(delta[:, 0]) >= np.abs(delta[:, 1])
    return np.where(horiz, np.where(delta[:, 0] > 0, 0, 1),
                    np.where(delta[:, 1] > 0, 2, 3))


def split_head_stage(head_raw: np.ndarray, grid_raw: np.ndarray,
                     max_deg: tuple[float, float] | None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separa as amostras da etapa da cabeça.

    Devolve (máscara das amostras usadas, grupo de validação de cada amostra,
    desvio de pose (n, 2) em relação à pose mediana da grade, em graus).
    Amostras com desvio acima de ``max_deg`` (yaw, pitch) são descartadas:
    nesses ângulos os landmarks do olho deixam de ser confiáveis.
    """
    base = np.median(head_angles(grid_raw), axis=0)
    delta = head_angles(head_raw) - base
    keep = np.ones(len(delta), bool)
    if max_deg is not None:
        keep = (np.abs(delta[:, 0]) <= max_deg[0]) & (np.abs(delta[:, 1]) <= max_deg[1])
    groups = HEAD_GROUP_BASE - head_direction(delta)
    groups[np.max(np.abs(delta), axis=1) < HEAD_STILL_DEG] = ALWAYS_TRAIN
    return keep, groups, delta


# ------------------------------------------------------------- utilidades
def outlier_mask(samples: np.ndarray, threshold: float) -> np.ndarray:
    """Máscara de amostras a manter (escore-Z modificado via MAD < limiar).

    Iglewicz & Hoaglin (1993): z = 0.6745 (x - mediana) / MAD; 3.5 é o
    limiar usual. Remove quadros isolados ruins (sacada atrasada, quase-piscada).
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
    """Alvos de calibração em pixels, em zigue-zague (a calibração sorteia a ordem de visita)."""
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


# ------------------------------------------- correção monitor virtual -> real
def correction_basis(V: np.ndarray, screen_size, kind: str) -> np.ndarray:
    """Colunas do polinômio da correção no ponto virtual V (n, 2) em px."""
    V = np.atleast_2d(V)
    u = V[:, 0] / screen_size[0] - 0.5
    v = V[:, 1] / screen_size[1] - 0.5
    cols = [np.ones(len(V))]
    if kind in ("afim", "quadratica"):
        cols += [u, v]
    if kind == "quadratica":
        cols += [u * u, u * v, v * v]
    return np.column_stack(cols)


def fit_correction(V, Y, w, kind: str, alpha: float, screen_size) -> np.ndarray:
    """Coeficientes (k, 2) da correção: Y ≈ V + base(V) @ coef. Ridge nos
    termos não constantes (encolhe em direção à identidade)."""
    if kind == "nenhuma":
        return np.zeros((0, 2))
    ok = np.all(np.isfinite(V), axis=1)
    B = correction_basis(V[ok], screen_size, kind)
    r = (Y - V)[ok]
    ww = w[ok]
    reg = alpha * ww.sum() * np.eye(B.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(B.T @ (B * ww[:, None]) + reg, (B * ww[:, None]).T @ r)


def apply_correction(V, coef, kind: str, screen_size) -> np.ndarray:
    V = np.atleast_2d(V)
    if kind == "nenhuma":
        return V.copy()
    return V + correction_basis(V, screen_size, kind) @ coef


# ------------------------------------------------------------------ modelo
class CalibrationModel:
    """Pipeline completo de uma amostra bruta até o pixel do monitor real."""

    def __init__(self, screen: VirtualScreen, eyeball: EyeballModel | None = None,
                 correction: str = "nenhuma", coef: np.ndarray | None = None,
                 fitted: bool = False):
        self.screen = screen
        self.eyeball = eyeball if eyeball is not None else EyeballModel()
        self.correction = correction
        self.coef = np.zeros((0, 2)) if coef is None else np.asarray(coef, dtype=np.float64)
        self.fitted = fitted
        self.offset = np.zeros(2)               # recentralização (R) e correções por alvo
        # Correção da boca (px por unidade de abertura acima de mouth_ref).
        self.mouth_coef = np.zeros(2)
        self.mouth_ref = 0.0
        self.created_at: str | None = None
        self.info: dict = {}
        # Postura da calibração: pose mediana da cabeça na grade (yaw, pitch),
        # posição mediana do rosto (cm) e a faixa coberta (graus). Usados no
        # aviso de postura fora da calibração.
        self.head_ref = np.zeros(2)
        self.pos_ref = np.zeros(3)
        self.head_span = np.zeros(2)

    @property
    def screen_size(self) -> tuple[int, int]:
        return tuple(self.screen.size_px)

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    # ------------------------------------------------------------ pipeline
    def virtual_px(self, raw) -> np.ndarray:
        """Etapas 2-4: pixel (n, 2) do MONITOR VIRTUAL atingido pelo olhar."""
        origin, direction, *_ = self.eyeball.gaze(np.atleast_2d(raw))
        return self.screen.project(origin, direction)

    def predict(self, raw: np.ndarray) -> np.ndarray:
        """Aceita (RAW_SIZE,) ou (n, RAW_SIZE); devolve (2,) ou (n, 2) pixels do
        monitor real (NaN se o olhar não atinge o plano da tela)."""
        raw = np.asarray(raw, dtype=np.float64)
        r2 = np.atleast_2d(raw)
        out = apply_correction(self.virtual_px(r2), self.coef, self.correction,
                               self.screen_size) + self.offset
        if np.any(self.mouth_coef):
            out = out + np.outer(np.maximum(r2[:, MOUTH_COL] - self.mouth_ref, 0.0),
                                 self.mouth_coef)
        return out[0] if raw.ndim == 1 else out

    # -------------------------------------------------------------- postura
    def posture_deviation(self, raw: np.ndarray) -> dict:
        """Quanto a postura atual se afasta da calibração: ``ang`` (yaw, pitch:
        graus além da faixa calibrada, >= 0) e ``pos_cm`` (deslocamento do rosto)."""
        raw = np.atleast_2d(raw)
        delta = np.abs(head_angles(raw)[0] - self.head_ref)
        beyond = np.maximum(delta - self.head_span, 0.0)
        return {"ang": beyond,
                "pos_cm": float(np.linalg.norm(raw[0, RAW_FACE] - self.pos_ref))}

    def correct_with_known_target(self, raw: np.ndarray, target, gain: float = 0.3,
                                  max_step_px: float = 80.0) -> np.ndarray:
        """Corrige o offset com amostras de um momento em que o usuário olhava
        para um ponto CONHECIDO (ex.: centro da tecla selecionada). Move o
        offset ``gain`` do caminho até o erro medido, limitado a ``max_step_px``.
        Devolve a correção aplicada (px)."""
        pred = np.nanmedian(np.atleast_2d(self.predict(np.atleast_2d(raw))), axis=0)
        step = gain * (np.asarray(target, dtype=float) - pred)
        if not np.all(np.isfinite(step)):
            return np.zeros(2)
        norm = float(np.linalg.norm(step))
        if norm > max_step_px:
            step *= max_step_px / norm
        self.offset = self.offset + step
        return step

    @property
    def cv_score(self) -> float:
        return float(self.info.get("erro_loo_medio_px", np.inf))

    # ----------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {
            "feature_version": FEATURE_VERSION,
            "camera_hfov_deg": geometry.ASSUMED_HFOV_DEG,
            "screen": {"size_px": list(self.screen.size_px), "width_cm": self.screen.width_cm,
                       "gap_cm": self.screen.gap_cm},
            "eyeball": self.eyeball.to_dict(),
            "correction": self.correction,
            "coef": self.coef.tolist(),
            "offset": self.offset.tolist(),
            "mouth_coef": self.mouth_coef.tolist(),
            "mouth_ref": self.mouth_ref,
            "head_ref": self.head_ref.tolist(),
            "pos_ref": self.pos_ref.tolist(),
            "head_span": self.head_span.tolist(),
            "screen_size": list(self.screen.size_px),
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
        if abs(float(d.get("camera_hfov_deg", -1)) - geometry.ASSUMED_HFOV_DEG) > 1e-6:
            print("[Calibração] Arquivo salvo é de outro campo de visão da câmera; recalibre.")
            return None
        if tuple(d.get("screen_size", ())) != tuple(screen_size):
            print("[Calibração] Arquivo salvo é de outra resolução de tela; recalibre.")
            return None
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationModel":
        s = d["screen"]
        m = cls(VirtualScreen(tuple(s["size_px"]), float(s["width_cm"]), float(s["gap_cm"])),
                EyeballModel.from_dict(d["eyeball"]), d["correction"],
                np.array(d["coef"], dtype=np.float64).reshape(-1, 2), fitted=True)
        m.offset = np.array(d.get("offset", [0.0, 0.0]))
        m.mouth_coef = np.array(d.get("mouth_coef", [0.0, 0.0]))
        m.mouth_ref = float(d.get("mouth_ref", 0.0))
        m.head_ref = np.array(d.get("head_ref", [0.0, 0.0]))
        m.pos_ref = np.array(d.get("pos_ref", [0.0, 0.0, 0.0]))
        m.head_span = np.array(d.get("head_span", [0.0, 0.0]))
        m.created_at = d.get("created_at")
        m.info = d.get("info", {})
        return m


# ------------------------------------------------------------------- dados
@dataclass
class CalibrationData:
    """Amostras coletadas: por alvo da grade, na etapa da cabeça e na da boca."""

    # lista de (alvo_px, features (n, 2), raw (n, RAW_SIZE))
    grid: list = field(default_factory=list)
    head_target: tuple[float, float] | None = None
    head_features: np.ndarray | None = None
    head_raw: np.ndarray | None = None
    # Landmarks de cada amostra (listas de FaceObservation), para recalcular
    # as medidas com outra versão do estimador (tools/reavaliar.py).
    grid_obs: list = field(default_factory=list)
    head_obs: list = field(default_factory=list)
    # Etapa da boca: olhar fixo no centro, boca abrindo e fechando.
    mouth_target: tuple[float, float] | None = None
    mouth_features: np.ndarray | None = None
    mouth_raw: np.ndarray | None = None
    mouth_obs: list = field(default_factory=list)
    # Instante (s) de cada amostra da grade (mesma ordem de ``grid``), para a
    # correção de deriva.
    grid_times: list = field(default_factory=list)
    # Checagens do ponto central intercaladas na grade, para medir a deriva:
    # lista de (alvo_px, features (n, 2), raw (n, RAW_SIZE), instantes (n,)).
    checks: list = field(default_factory=list)
    check_obs: list = field(default_factory=list)     # landmarks das checagens
    # Versão do layout das medidas gravadas (``raw``). Diferente da atual:
    # só serve depois de recalcular a partir dos landmarks.
    feature_version: int = FEATURE_VERSION

    @property
    def stale(self) -> bool:
        return self.feature_version != FEATURE_VERSION

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
            if i < len(self.grid_times):
                arrays[f"t_{i}"] = np.asarray(self.grid_times[i], dtype=float)
        for k, (t, f, r, tt) in enumerate(self.checks):
            arrays[f"chk_alvo_{k}"] = np.asarray(t, dtype=float)
            arrays[f"chk_feats_{k}"] = np.atleast_2d(f)
            arrays[f"chk_raw_{k}"] = np.atleast_2d(r)
            arrays[f"chk_t_{k}"] = np.asarray(tt, dtype=float)
            if k < len(self.check_obs):
                _pack_obs(arrays, f"chk_obs_{k}", self.check_obs[k])
        if self.head_raw is not None:
            arrays["head_target"] = np.asarray(self.head_target, dtype=float)
            arrays["head_feats"] = np.atleast_2d(self.head_features)
            arrays["head_raw"] = np.atleast_2d(self.head_raw)
            _pack_obs(arrays, "obs_head", self.head_obs)
        if self.mouth_raw is not None:
            arrays["mouth_target"] = np.asarray(self.mouth_target, dtype=float)
            arrays["mouth_feats"] = np.atleast_2d(self.mouth_features)
            arrays["mouth_raw"] = np.atleast_2d(self.mouth_raw)
            _pack_obs(arrays, "obs_mouth", self.mouth_obs)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "CalibrationData":
        z = np.load(path)
        data = cls(feature_version=int(z["feature_version"]) if "feature_version" in z else 0)
        i = 0
        while f"alvo_{i}" in z:
            data.grid.append((tuple(z[f"alvo_{i}"]), z[f"feats_{i}"], z[f"raw_{i}"]))
            data.grid_obs.append(_unpack_obs(z, f"obs_{i}"))
            if f"t_{i}" in z:
                data.grid_times.append(z[f"t_{i}"])
            i += 1
        if len(data.grid_times) != len(data.grid):
            data.grid_times = []
        k = 0
        while f"chk_alvo_{k}" in z:
            data.checks.append((tuple(z[f"chk_alvo_{k}"]), z[f"chk_feats_{k}"],
                                z[f"chk_raw_{k}"], z[f"chk_t_{k}"]))
            data.check_obs.append(_unpack_obs(z, f"chk_obs_{k}"))
            k += 1
        if "head_raw" in z:
            data.head_target = tuple(z["head_target"])
            data.head_features, data.head_raw = z["head_feats"], z["head_raw"]
            data.head_obs = _unpack_obs(z, "obs_head")
        if "mouth_raw" in z:
            data.mouth_target = tuple(z["mouth_target"])
            data.mouth_features, data.mouth_raw = z["mouth_feats"], z["mouth_raw"]
            data.mouth_obs = _unpack_obs(z, "obs_mouth")
        return data

    @property
    def has_landmarks(self) -> bool:
        return bool(self.grid_obs) and all(len(o) for o in self.grid_obs)


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


# --------------------------------------------------------------- ajuste
def cv_errors_from(preds: dict, Y: np.ndarray, groups: np.ndarray, axis: int | None = None):
    """(erros dos alvos, erros das direções da cabeça, índices dos alvos) a
    partir das previsões de cada dobra da validação cruzada.

    Alvo: distância da mediana das previsões ao alvo. Direção da cabeça:
    mediana das distâncias de cada amostra ao alvo (a cabeça se move durante
    a etapa, então cada amostra conta). axis: só o erro nesse eixo."""
    def dist(d):
        d = np.atleast_2d(d)
        return np.hypot(d[:, 0], d[:, 1]) if axis is None else np.abs(d[:, axis])
    target_errs, head_errs, target_ids = [], [], []
    for g, pred in preds.items():
        te = groups == g
        if g >= 0:
            target_errs.append(float(dist(np.nanmedian(pred, axis=0) - Y[te][0])[0]))
            target_ids.append(int(g))
        else:
            head_errs.append(float(np.nanmedian(dist(pred - Y[te]))))
    return target_errs, head_errs, target_ids


def _candidates(n_targets: int, max_degree: int) -> list[tuple[str, float]]:
    out = [("nenhuma", 0.0), ("deslocamento", 0.0)]
    if n_targets >= 4:
        out += [("afim", a) for a in ALPHAS]
    if n_targets >= 9 and max_degree >= 2:
        out += [("quadratica", a) for a in ALPHAS]
    return out


def fit_from_samples(
    data: CalibrationData,
    screen_size: tuple[int, int],
    screen_width_cm: float | None = None,
    mad_threshold: float = 3.5,
    head_weight_points: float = 1.0,
    head_max_deg: tuple[float, float] | None = (15.0, 12.0),
    max_degree: int = 2,
    assumed_width_cm: float = 34.5,
    corrections: tuple[str, ...] | None = None,
    pupil_sources: tuple[str, ...] | None = None,
    anchors: tuple[str, ...] = ("misto",),
    drift_correction: bool = True,
) -> tuple[CalibrationModel, list[PointResult], dict]:
    """Ajusta o pipeline. Devolve (modelo, resultados por alvo, estatísticas).

    screen_width_cm: largura física da tela (cm). Sem ela, o monitor virtual
    usa ``assumed_width_cm`` e a correção absorve a diferença de escala.
    max_degree: 1 = correções até afim; 2 = também a quadrática.
    corrections / pupil_sources: restringem os candidatos (análises offline).
    drift_correction: com as checagens do centro gravadas (``data.checks``),
    desconta a deriva da pupila ao longo da grade (ver correct_drift).
    """
    if data.stale:
        raise ValueError("Medidas gravadas com outra versão do estimador; recalcule "
                         "a partir dos landmarks (tools/reavaliar.py).")
    screen = VirtualScreen(tuple(int(v) for v in screen_size),
                           float(screen_width_cm or assumed_width_cm))
    has_times = len(data.grid_times) == len(data.grid) and len(data.grid) > 0
    kept, cleaned = [], []    # (índice, raw, instantes) de cada alvo utilizável
    for i, (target, feats, raw) in enumerate(data.grid):
        raw = np.atleast_2d(raw)
        if len(raw) < 3 or raw.shape[1] != RAW_SIZE:
            continue   # alvo sem amostras suficientes (rosto perdido): ignora
        keep = outlier_mask(feats, mad_threshold)
        times = np.asarray(data.grid_times[i], dtype=float)[keep] if has_times else None
        kept.append((i, raw[keep], times))
    if not kept:
        raise ValueError("Nenhum alvo com amostras suficientes para calibrar.")
    checks = []
    for t, f, r, tt in data.checks:
        r = np.atleast_2d(r)
        if len(r) >= 3 and r.shape[1] == RAW_SIZE:
            keep = outlier_mask(f, mad_threshold)
            checks.append((t, r[keep], np.asarray(tt, dtype=float)[keep]))
    checks_orig = [(t, r.copy(), tt) for t, r, tt in checks]
    drift_info: dict = {}
    if drift_correction and has_times and len(checks) >= 2:
        targets_px = [data.grid[i][0] for i, _, _ in kept]
        new_raws, checks, drift_info = correct_drift(
            [r for _, r, _ in kept], [tt for _, _, tt in kept], targets_px, checks, screen,
            anchors[0])
        kept = [(i, r, tt) for (i, _, tt), r in zip(kept, new_raws)]

    raws, Ys, ws, gs = [], [], [], []
    for i, r, _ in kept:
        target = data.grid[i][0]
        cleaned.append((target, r))
        raws.append(r)
        Ys.append(np.tile(target, (len(r), 1)))
        ws.append(np.full(len(r), 1.0 / len(r)))   # cada alvo pesa 1
        gs.append(np.full(len(r), i))
    grid_raw = np.vstack(raws)
    # As checagens do centro também ensinam o globo: entram sempre no treino
    # e, juntas, pesam como 1 alvo.
    for t, r, _ in checks:
        raws.append(r)
        Ys.append(np.tile(t, (len(r), 1)))
        ws.append(np.full(len(r), 1.0 / (len(r) * len(checks))))
        gs.append(np.full(len(r), ALWAYS_TRAIN))

    stats_head: dict = {}
    hr, head_span = None, np.zeros(2)
    if data.head_raw is not None and len(data.head_raw) >= 5:
        hr_all = np.atleast_2d(data.head_raw)
        keep, hgroups, delta = split_head_stage(hr_all, grid_raw, head_max_deg)
        stats_head = {"n_cabeca_total": int(len(hr_all)), "n_cabeca_usadas": int(keep.sum())}
        if keep.sum() >= 5:
            hr = hr_all[keep]
            raws.append(hr)
            Ys.append(np.tile(data.head_target, (len(hr), 1)))
            # A etapa da cabeça pesa o equivalente a ``head_weight_points`` alvos.
            ws.append(np.full(len(hr), head_weight_points / len(hr)))
            gs.append(hgroups[keep])
            head_span = np.percentile(np.abs(delta[keep]), 95, axis=0)
    has_head = hr is not None

    raw_all, Y_all = np.vstack(raws), np.vstack(Ys)
    w_all, g_all = np.concatenate(ws), np.concatenate(gs)
    T_all = screen.px_to_cm(Y_all)
    # Fonte da pupila: com o refinamento na imagem disponível, a MÉDIA das duas
    # medidas (MediaPipe e refinada), cujos erros são em parte independentes;
    # nas sessões gravadas, ela errou menos que cada uma sozinha (seção 5.6).
    sources = ["media"] if has_refined_pupil(raw_all) else ["mediapipe"]
    if pupil_sources is not None:
        sources = list(pupil_sources)
    cands = _candidates(len(cleaned), max_degree)
    if corrections is not None:
        cands = [c for c in cands if c[0] in corrections or c[0] == "nenhuma"]

    # Validação cruzada: para cada dobra, o globo é reajustado sem ela; as
    # correções candidatas são ajustadas sobre esse globo.
    folds = [g for g in np.unique(g_all) if g != ALWAYS_TRAIN
             and not (g <= HEAD_GROUP_BASE and np.sum(g_all == g) < MIN_HEAD_FOLD)]
    scores: dict = {}
    if len(cleaned) >= 5:
        for src, anc in [(s_, a_) for s_ in sources for a_ in anchors]:
            preds = {c: {} for c in cands}
            for g in folds:
                tr, te = g_all != g, g_all == g
                eb = EyeballModel.fit(raw_all[tr], T_all[tr], w_all[tr], src, anchor=anc)
                tmp = CalibrationModel(screen, eb)
                V_tr, V_te = tmp.virtual_px(raw_all[tr]), tmp.virtual_px(raw_all[te])
                for kind, alpha in cands:
                    coef = fit_correction(V_tr, Y_all[tr], w_all[tr], kind, alpha, screen_size)
                    preds[(kind, alpha)][int(g)] = apply_correction(V_te, coef, kind, screen_size)
            for c in cands:
                scores[(src, anc, *c)] = cv_errors_from(preds[c], Y_all, g_all)
        kinds = set(corrections or ("nenhuma",)) if corrections is not None else None
        allowed = [k for k in scores if kinds is None or k[2] in kinds]
        best = min(allowed, key=lambda k: np.mean(scores[k][0]))
    else:
        best = (sources[0], anchors[0], "deslocamento" if corrections is None
                or "deslocamento" in corrections else "nenhuma", 0.0)
    src, anc, kind, alpha = best

    eyeball = EyeballModel.fit(raw_all, T_all, w_all, src, anchor=anc)
    model = CalibrationModel(screen, eyeball, kind, fitted=True)
    V_all = model.virtual_px(raw_all)
    model.coef = fit_correction(V_all, Y_all, w_all, kind, alpha, screen_size)
    model.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    resid = model.predict(raw_all) - Y_all
    ok = np.all(np.isfinite(resid), axis=1)
    info = {
        "fonte_pupila": src, "ancora": anc, "correcao": kind, "ridge_alpha": float(alpha),
        "largura_tela_cm": screen.width_cm, "largura_conhecida": bool(screen_width_cm),
        "usa_etapa_cabeca": bool(has_head),
        "rmse_ajuste_px": float(np.sqrt(np.sum(w_all[ok] * np.sum(resid[ok] ** 2, 1))
                                        / w_all[ok].sum())),
        "globo": {name: {"raio_cm": float(eyeball.r[k]),
                         "repouso_pupila_cm": eyeball.c[k].tolist(),
                         "ganho_pupila": eyeball.gain[k].tolist()}
                  for k, name in enumerate(EYE_NAMES)},
        "erro_angular_ajuste_graus": eyeball.info.get("erro_angular_ajuste_graus"),
    }
    # Intersecção dos feixes na grade: a que distância da tela ela cai.
    _, _, X, _, _ = eyeball.gaze(grid_raw)
    conv = np.all(np.isfinite(X), axis=1)
    info["feixes_convergem_frac"] = float(conv.mean())
    if conv.any():
        info["interseccao_dist_tela_cm"] = float(np.median(X[conv, 2]))
    if scores:
        target_errs, head_errs, target_ids = scores[best]
        info["erro_loo_medio_px"] = float(np.mean(target_errs))
        info["erro_loo_por_alvo_px"] = [float(e) for e in target_errs]
        info["erro_loo_alvos"] = target_ids
        if head_errs:
            info["erro_cv_cabeca_px"] = float(np.mean(head_errs))
        geo = scores[(src, anc, "nenhuma", 0.0)]
        info["erro_loo_monitor_virtual_px"] = float(np.mean(geo[0]))
        info["candidatos"] = [
            {"fonte": k[0], "ancora": k[1], "correcao": k[2], "alpha": k[3],
             "erro_loo_px": float(np.mean(v[0])),
             "erro_cv_cabeca_px": float(np.mean(v[1])) if v[1] else None}
            for k, v in scores.items()]
    info.update(stats_head)
    model.info = info

    # Postura de referência (aviso de postura fora da calibração).
    angles = head_angles(grid_raw)
    model.head_ref = np.median(angles, axis=0)
    span = np.percentile(np.abs(angles - model.head_ref), 95, axis=0)
    model.head_span = np.maximum(span, head_span) if has_head else span
    model.pos_ref = np.median(grid_raw[:, RAW_FACE], axis=0)

    results = []
    for target, r in cleaned:
        pred = np.nanmedian(model.predict(r), axis=0)
        results.append(PointResult(target, (float(pred[0]), float(pred[1])), len(r)))

    if checks_orig:
        # Onde o centro foi visto em cada checagem, pelo modelo final e SEM a
        # correção de deriva: a variação entre elas é a deriva da calibração.
        seen = np.array([np.nanmedian(model.predict(r), axis=0) - np.asarray(t, dtype=float)
                         for t, r, _ in checks_orig])
        model.info["checagens_centro_px"] = seen.tolist()
        model.info["deriva_centro_px"] = float(np.nanmax(np.hypot(*(seen - seen[-1]).T)))
        model.info["deriva_corrigida"] = bool(drift_info)
        model.info.update(drift_info)
    fit_mouth_correction(model, data, grid_raw, mad_threshold)
    stats = dict(model.info)
    if has_head:
        err = np.hypot(*(model.predict(hr) - np.asarray(data.head_target)).T)
        stats["erro_mediano_cabeca_px"] = float(np.nanmedian(err))
    return model, results, stats


PUPIL_COLS = {"mediapipe": ((15, 18), (21, 24)), "refinada": ((24, 27), (27, 30))}


def correct_drift(grid_raws, grid_times, grid_targets, checks, screen, anchor="misto",
                  iterations=2):
    """Desconta a deriva da pupila ao longo da calibração.

    Durante a grade, a posição medida da pupila em relação à âncora deriva
    devagar (cansaço, olho ressecando, postura escorregando): nos dados
    reais, a abertura do olho caiu de 0,27 para 0,16 em ~35 s. As checagens
    do centro, intercaladas na grade, olham sempre para o mesmo ponto: a
    diferença entre a pupila medida e a prevista pelo modelo em cada
    checagem é a deriva naquele instante (por olho e eixo, no referencial
    da cabeça, separando o efeito da cabeça pela geometria). Ajusta-se uma
    tendência LINEAR no tempo a essas medidas (a deriva é lenta, e a reta
    filtra o ruído de cada checagem), subtraída de cada amostra em relação
    ao FIM da calibração: o modelo fica ajustado ao estado do fim da
    calibração, o mais próximo do uso. O ajuste é repetido ``iterations``
    vezes (o modelo usado para medir a deriva melhora com os dados
    corrigidos). Cada fonte da pupila é corrigida à parte.

    Devolve (raws da grade corrigidos, checagens corrigidas, informações)."""
    grid_raws = [np.array(r, dtype=float) for r in grid_raws]
    checks = [(t, np.array(r, dtype=float), np.asarray(tt, dtype=float)) for t, r, tt in checks]
    T_grid = [np.repeat(screen.px_to_cm([t]), len(r), axis=0)
              for t, r in zip(grid_targets, grid_raws)]
    T_chk = [np.repeat(screen.px_to_cm([t]), len(r), axis=0) for t, r, _ in checks]
    info = {}

    def shift(raw, times, delta_fn, cols):
        d = delta_fn(times)                                  # (n, 2 olhos, 2 eixos)
        R = raw[:, RAW_R].reshape(-1, 3, 3)
        for k, (a, b) in enumerate(cols):
            dh = np.column_stack([d[:, k], np.zeros(len(raw))])
            raw[:, a:b] -= np.einsum("nij,nj->ni", R, dh)

    for src, cols in PUPIL_COLS.items():
        total = None
        for _ in range(iterations):
            R_all = np.vstack(grid_raws + [r for _, r, _ in checks])
            T_all = np.vstack(T_grid + T_chk)
            w_all = np.concatenate([np.full(len(r), 1.0 / len(r)) for r in grid_raws]
                                   + [np.full(len(r), 1.0 / (len(r) * len(checks)))
                                      for _, r, _ in checks])
            eb = EyeballModel.fit(R_all, T_all, w_all, src, anchor=anchor)
            tk = np.array([np.median(tt) for _, _, tt in checks])
            res = [eb.pupil_residual(r, T) for (_, r, _), T in zip(checks, T_chk)]
            dk = np.nan_to_num(np.array([np.nanmedian(x, axis=0) for x in res]))
            # Erro-padrão da mediana de cada checagem (dispersão robusta).
            se = np.array([1.2533 * 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x, axis=0)),
                                                          axis=0)
                           / np.sqrt(max(np.sum(np.isfinite(x[:, :, 0]), axis=0).min(), 1))
                           for x in res])
            se = np.nan_to_num(se, nan=0.0)
            # Tendência LINEAR no tempo (a deriva é lenta): com 5 checagens, a
            # reta filtra o ruído de cada uma, que a interpolação ponto a ponto
            # aplicaria como se fosse deriva. Só a inclinação importa: a
            # correção é inclinação × (t − fim).
            t_end = tk.max()
            tc = tk - tk.mean()
            ss = max(float(tc @ tc), 1e-9)
            slope = (tc @ dk.reshape(len(tk), -1)).reshape(2, 2) / ss
            # Só a deriva distinguível do ruído: a inclinação é encolhida por
            # s²/(s² + 4·EP²), EP = erro-padrão da inclinação. Sem isso, num
            # usuário sintético SEM deriva e com ruído realista nos landmarks,
            # a "deriva" estimada do ruído dobrava o erro (26 -> 60 px).
            var = ((tc ** 2) @ (se ** 2).reshape(len(tk), -1)).reshape(2, 2) / ss ** 2
            # O filtro da pupila correlaciona amostras seguidas, e a dispersão
            # DENTRO de cada checagem subestima o erro. Estimativa ENTRE as
            # checagens: resíduo em torno da reta (K − 2 graus de liberdade).
            if len(tk) > 2:
                fit_line = dk.mean(axis=0) + tc[:, None, None] * slope
                s2 = np.sum((dk - fit_line) ** 2, axis=0) / (len(tk) - 2)
                var = np.maximum(var, s2 / ss)
            slope = slope * slope ** 2 / np.maximum(slope ** 2 + 4.0 * var, 1e-18)

            def delta_fn(times, slope=slope, t_end=t_end):
                return (np.asarray(times, dtype=float) - t_end)[:, None, None] * slope

            for r, tt in zip(grid_raws, grid_times):
                shift(r, tt, delta_fn, cols)
            for _, r, tt in checks:
                shift(r, tt, delta_fn, cols)
            step = float(np.abs(slope).max() * (t_end - tk.min()))
            total = step if total is None else total + step
        info[f"deriva_pupila_{src}_cm"] = float(total)
    return grid_raws, checks, info


MOUTH_MIN_RANGE = 0.04   # variação mínima de abertura na etapa da boca


def fit_mouth_correction(model: CalibrationModel, data: CalibrationData, grid_raw: np.ndarray,
                         mad_threshold: float = 3.5):
    """Correção da boca a partir da etapa da boca (olhar fixo no centro).

    O MediaPipe estima a malha inteira de uma vez, e abrir a boca desloca a
    pose da cabeça e os pontos do olho: com o olhar fixo, o ponto estimado
    descia ~50 px. Todo deslocamento da previsão que acompanha a abertura
    da boca é artefato: ajusta-se o deslocamento por unidade de abertura
    (mínimos quadrados pela origem, com rejeição de outliers), descontado
    na previsão. Sem a etapa, ou se a boca quase não abriu, fica em zero.
    """
    model.mouth_coef = np.zeros(2)
    model.mouth_ref = float(np.median(np.atleast_2d(grid_raw)[:, MOUTH_COL]))
    mr = None if data.mouth_raw is None else np.atleast_2d(data.mouth_raw)
    if mr is None or len(mr) < 10 or mr.shape[1] != RAW_SIZE:
        return
    m = np.maximum(mr[:, MOUTH_COL] - model.mouth_ref, 0.0)
    model.info["boca_variacao"] = float(np.ptp(mr[:, MOUTH_COL]))
    if np.ptp(mr[:, MOUTH_COL]) < MOUTH_MIN_RANGE or np.sum(m > 0.5 * MOUTH_MIN_RANGE) < 5:
        return
    resid = np.asarray(data.mouth_target, dtype=float) - model.predict(mr)
    fin = np.all(np.isfinite(resid), axis=1)
    m, resid = m[fin], resid[fin]
    # Desvio com a boca FECHADA (postura, deriva) não é efeito da boca.
    closed = m < 0.25 * MOUTH_MIN_RANGE
    base = np.median(resid[closed], axis=0) if closed.sum() >= 5 else np.zeros(2)
    resid = resid - base
    keep = outlier_mask(resid, mad_threshold)
    mm, rr = m[keep], resid[keep]
    model.mouth_coef = (mm @ rr) / max(float(mm @ mm), 1e-9)
    model.info["boca_px_por_unidade"] = [float(c) for c in model.mouth_coef]
