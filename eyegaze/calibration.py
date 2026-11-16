"""
Etapa 5 do pipeline: calibração com vários pontos.

O cursor sai do pipeline geométrico (ver gaze.py e eyeball.py):

    landmarks -> âncora + pupila de cada olho (cm)
              -> centro do globo de cada olho -> feixe centro -> pupila
              -> projeção de cada feixe no MONITOR VIRTUAL (plano da tela,
                 geometria suposta) -> média ponderada dos dois olhos
              -> correção monitor virtual -> MONITOR REAL (pixels)

A calibração mostra alvos conhecidos (grade) e, com essas amostras, ajusta,
nesta ordem:

1. O GLOBO de cada olho (``eyeball.EyeballModel.fit``): para cada olho e
   eixo, a posição de repouso da pupila e o raio efetivo, por mínimos
   quadrados robustos; e o peso de cada olho no ponto da tela. É isto que
   faz o feixe passar pelo alvo.
2. A CORREÇÃO monitor virtual -> real: o ponto atingido no monitor virtual
   (etapa 4) ainda tem erros que não são do olho: largura física da tela
   suposta, posição da webcam, campo de visão da câmera, viés da pose do
   MediaPipe. A correção é um polinômio pequeno nas coordenadas do monitor
   virtual, somado ao ponto virtual:

       real = virtual + a₀ (+ a₁·u + a₂·v) (+ a₃·u² + a₄·uv + a₅·v²)

   com (u, v) o ponto virtual normalizado pela tela, ajustada por ridge
   (penalidade em direção à identidade). Os candidatos são: nenhuma
   correção (geometria pura), só deslocamento, afim e quadrática.

A escolha do tipo de correção e da força do ridge é por validação cruzada:
cada alvo fica fora do ajuste por vez (o globo E a correção são reajustados
sem ele), e o erro nesse alvo estima o erro em pontos novos.

A etapa da cabeça (opcional, modo pesquisa) NÃO entra no ajuste: é usada só
como TESTE (cabeça em poses novas). Nas sessões gravadas, treinar com ela
melhorava a validação em ~3 px, e ela é inadequada para quem não consegue
mover a cabeça.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .eyeball import EYE_NAMES, EyeballModel
from .face_tracker import FaceObservation
from .gaze import RAW_FACE, RAW_R, RAW_SIZE, has_refined_pupil
from . import geometry
from .geometry import VirtualScreen

FEATURE_VERSION = 10  # incremente se o layout de ``raw`` mudar (invalida calibrações antigas)

CORRECTIONS = ("nenhuma", "deslocamento", "afim", "quadratica")
ALPHAS = (1e-3, 1e-2, 1e-1)       # ridge da correção (coordenadas normalizadas)

HEAD_DIRECTIONS = ("direita", "esquerda", "cima", "baixo")
HEAD_STILL_DEG = 2.0       # desvio abaixo disto não é "pose nova"
MIN_HEAD_TEST = 5          # amostras mínimas de uma direção para entrar no teste


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


def head_test_error(model: "CalibrationModel", head_raw: np.ndarray, head_target,
                    grid_raw: np.ndarray, max_deg: tuple[float, float] | None = (15.0, 12.0)
                    ) -> dict:
    """Erro (px) com a cabeça em poses novas, nas amostras da etapa da cabeça
    (que não entram no ajuste). Para cada direção (direita, esquerda, cima,
    baixo), a mediana da distância de cada amostra ao alvo; o resultado é a
    média das direções. Amostras além de ``max_deg`` (yaw, pitch) da pose da
    grade são descartadas: nesses ângulos os landmarks do olho deixam de ser
    confiáveis."""
    hr = np.atleast_2d(head_raw)
    delta = head_angles(hr) - np.median(head_angles(grid_raw), axis=0)
    use = np.max(np.abs(delta), axis=1) >= HEAD_STILL_DEG
    if max_deg is not None:
        use &= (np.abs(delta[:, 0]) <= max_deg[0]) & (np.abs(delta[:, 1]) <= max_deg[1])
    direction = head_direction(delta)
    err = np.hypot(*(model.predict(hr) - np.asarray(head_target, dtype=float)).T)
    by_dir = {}
    for q, name in enumerate(HEAD_DIRECTIONS):
        sel = use & (direction == q) & np.isfinite(err)
        if sel.sum() >= MIN_HEAD_TEST:
            by_dir[name] = float(np.median(err[sel]))
    out = {"n_cabeca_total": int(len(hr)), "n_cabeca_usadas": int(use.sum())}
    if by_dir:
        out["erro_cabeca_px"] = float(np.mean(list(by_dir.values())))
        out["erro_cabeca_por_direcao_px"] = by_dir
    return out


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
    def virtual_px_per_eye(self, raw) -> np.ndarray:
        """(n, 2 olhos, 2) pixel do MONITOR VIRTUAL atingido pelo feixe de cada olho."""
        return self.eyeball.screen_points(np.atleast_2d(raw), self.screen)[0]

    def virtual_px(self, raw) -> np.ndarray:
        """Etapas 2-4: pixel (n, 2) do MONITOR VIRTUAL (média ponderada dos olhos)."""
        return self.eyeball.screen_points(np.atleast_2d(raw), self.screen)[1]

    def predict(self, raw: np.ndarray) -> np.ndarray:
        """Aceita (RAW_SIZE,) ou (n, RAW_SIZE); devolve (2,) ou (n, 2) pixels do
        monitor real (NaN se o olhar não atinge o plano da tela)."""
        raw = np.asarray(raw, dtype=np.float64)
        out = apply_correction(self.virtual_px(raw), self.coef, self.correction,
                               self.screen_size) + self.offset
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
        m.head_ref = np.array(d.get("head_ref", [0.0, 0.0]))
        m.pos_ref = np.array(d.get("pos_ref", [0.0, 0.0, 0.0]))
        m.head_span = np.array(d.get("head_span", [0.0, 0.0]))
        m.created_at = d.get("created_at")
        m.info = d.get("info", {})
        return m


# ------------------------------------------------------------------- dados
@dataclass
class CalibrationData:
    """Amostras coletadas: por alvo da grade e na etapa da cabeça (opcional)."""

    # lista de (alvo_px, features (n, 2), raw (n, RAW_SIZE))
    grid: list = field(default_factory=list)
    head_target: tuple[float, float] | None = None
    head_features: np.ndarray | None = None
    head_raw: np.ndarray | None = None
    # Landmarks de cada amostra (listas de FaceObservation), para recalcular
    # as medidas com outra versão do estimador (tools/reavaliar.py).
    grid_obs: list = field(default_factory=list)
    head_obs: list = field(default_factory=list)
    # Instante (s) de cada amostra da grade (mesma ordem de ``grid``).
    grid_times: list = field(default_factory=list)
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
        if self.head_raw is not None:
            arrays["head_target"] = np.asarray(self.head_target, dtype=float)
            arrays["head_feats"] = np.atleast_2d(self.head_features)
            arrays["head_raw"] = np.atleast_2d(self.head_raw)
            _pack_obs(arrays, "obs_head", self.head_obs)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "CalibrationData":
        """Carrega um .npz gravado. Dados de etapas removidas (boca, checagens
        de deriva) em gravações antigas são ignorados."""
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
        if "head_raw" in z:
            data.head_target = tuple(z["head_target"])
            data.head_features, data.head_raw = z["head_feats"], z["head_raw"]
            data.head_obs = _unpack_obs(z, "obs_head")
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
    head_max_deg: tuple[float, float] | None = (15.0, 12.0),
    max_degree: int = 2,
    assumed_width_cm: float = 34.5,
    corrections: tuple[str, ...] | None = None,
    pupil_source: str | None = None,
    correction: tuple[str, float] | None = None,
    weight_eyes: bool = True,
) -> tuple[CalibrationModel, list[PointResult], dict]:
    """Ajusta o pipeline. Devolve (modelo, resultados por alvo, estatísticas).

    screen_width_cm: largura física da tela (cm). Sem ela, o monitor virtual
    usa ``assumed_width_cm`` e a correção absorve a diferença de escala.
    max_degree: 1 = correções até afim; 2 = também a quadrática.
    corrections: restringe os tipos de correção candidatos (análises offline).
    correction: (tipo, alpha) fixo; a validação cruzada só mede o erro.
    pupil_source: fonte da pupila (padrão: média do MediaPipe com a refinada,
    se houver; nas sessões gravadas, errou menos que cada uma sozinha).
    weight_eyes: peso de cada olho pelo seu erro (False: pesos iguais).
    """
    if data.stale:
        raise ValueError("Medidas gravadas com outra versão do estimador; recalcule "
                         "a partir dos landmarks (tools/reavaliar.py).")
    screen = VirtualScreen(tuple(int(v) for v in screen_size),
                           float(screen_width_cm or assumed_width_cm))
    cleaned = []    # (índice, alvo, raw) de cada alvo utilizável
    for i, (target, feats, raw) in enumerate(data.grid):
        raw = np.atleast_2d(raw)
        if len(raw) < 3 or raw.shape[1] != RAW_SIZE:
            continue   # alvo sem amostras suficientes (rosto perdido): ignora
        cleaned.append((i, target, raw[outlier_mask(feats, mad_threshold)]))
    if not cleaned:
        raise ValueError("Nenhum alvo com amostras suficientes para calibrar.")

    raw_all = np.vstack([r for _, _, r in cleaned])
    Y_all = np.vstack([np.tile(t, (len(r), 1)) for _, t, r in cleaned])
    w_all = np.concatenate([np.full(len(r), 1.0 / len(r)) for _, _, r in cleaned])  # alvo pesa 1
    g_all = np.concatenate([np.full(len(r), i) for i, _, r in cleaned])
    T_all = screen.px_to_cm(Y_all)
    src = pupil_source or ("media" if has_refined_pupil(raw_all) else "mediapipe")

    if correction is not None:
        cands = [("nenhuma", 0.0), (correction[0], float(correction[1]))]
    else:
        cands = _candidates(len(cleaned), max_degree)
        if corrections is not None:
            cands = [c for c in cands if c[0] in corrections or c[0] == "nenhuma"]

    # Validação cruzada (um alvo fora por vez): o globo é reajustado sem ele e
    # as correções candidatas são ajustadas sobre esse globo.
    scores: dict = {}
    if len(cleaned) >= 5:
        preds = {c: {} for c in cands}
        for g in np.unique(g_all):
            tr, te = g_all != g, g_all == g
            eb = EyeballModel.fit(raw_all[tr], T_all[tr], w_all[tr], src,
                                  weight_eyes=weight_eyes)
            tmp = CalibrationModel(screen, eb)
            V_tr, V_te = tmp.virtual_px(raw_all[tr]), tmp.virtual_px(raw_all[te])
            for kind, alpha in cands:
                coef = fit_correction(V_tr, Y_all[tr], w_all[tr], kind, alpha, screen_size)
                preds[(kind, alpha)][int(g)] = apply_correction(V_te, coef, kind, screen_size)
        for c in cands:
            ids = sorted(preds[c])
            errs = [float(np.hypot(*(np.nanmedian(preds[c][g], axis=0)
                                     - Y_all[g_all == g][0]))) for g in ids]
            scores[c] = (errs, ids)
        if correction is not None:
            best = cands[-1]
        else:
            kinds = None if corrections is None else (tuple(corrections) or ("nenhuma",))
            allowed = [c for c in scores if kinds is None or c[0] in kinds]
            best = min(allowed, key=lambda c: np.nanmean(scores[c][0]))
    elif correction is not None:
        best = cands[-1]
    else:
        best = ("deslocamento", 0.0) if corrections is None or "deslocamento" in corrections \
            else ("nenhuma", 0.0)
    kind, alpha = best

    eyeball = EyeballModel.fit(raw_all, T_all, w_all, src, weight_eyes=weight_eyes)
    model = CalibrationModel(screen, eyeball, kind, fitted=True)
    V_all = model.virtual_px(raw_all)
    model.coef = fit_correction(V_all, Y_all, w_all, kind, alpha, screen_size)
    model.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    resid = model.predict(raw_all) - Y_all
    ok = np.all(np.isfinite(resid), axis=1)
    info = {
        "fonte_pupila": src, "correcao": kind, "ridge_alpha": float(alpha),
        "largura_tela_cm": screen.width_cm, "largura_conhecida": bool(screen_width_cm),
        "rmse_ajuste_px": float(np.sqrt(np.sum(w_all[ok] * np.sum(resid[ok] ** 2, 1))
                                        / w_all[ok].sum())),
        "globo": {name: {"raio_cm": float(eyeball.r[k]),
                         "repouso_pupila_cm": eyeball.c[k].tolist(),
                         "ganho_pupila": eyeball.gain[k].tolist(),
                         "peso": float(eyeball.eye_weights[k])}
                  for k, name in enumerate(EYE_NAMES)},
        "erro_angular_ajuste_graus": eyeball.info.get("erro_angular_ajuste_graus"),
    }
    if scores:
        errs, ids = scores[best]
        info["erro_loo_medio_px"] = float(np.nanmean(errs))
        info["erro_loo_por_alvo_px"] = [float(e) for e in errs]
        info["erro_loo_alvos"] = [int(i) for i in ids]
        info["erro_loo_monitor_virtual_px"] = float(np.nanmean(scores[("nenhuma", 0.0)][0]))
        info["candidatos"] = [{"correcao": c[0], "alpha": c[1],
                               "erro_loo_px": float(np.nanmean(v[0]))}
                              for c, v in scores.items()]
    model.info = info

    # Postura de referência (aviso de postura fora da calibração).
    angles = head_angles(raw_all)
    model.head_ref = np.median(angles, axis=0)
    model.head_span = np.percentile(np.abs(angles - model.head_ref), 95, axis=0)
    model.pos_ref = np.median(raw_all[:, RAW_FACE], axis=0)

    results = []
    for _, target, r in cleaned:
        pred = np.nanmedian(model.predict(r), axis=0)
        results.append(PointResult(target, (float(pred[0]), float(pred[1])), len(r)))

    stats = dict(model.info)
    if data.head_raw is not None and len(np.atleast_2d(data.head_raw)) >= MIN_HEAD_TEST \
            and np.atleast_2d(data.head_raw).shape[1] == RAW_SIZE:
        head = head_test_error(model, data.head_raw, data.head_target, raw_all, head_max_deg)
        model.info.update(head)
        stats.update(head)
    return model, results, stats
