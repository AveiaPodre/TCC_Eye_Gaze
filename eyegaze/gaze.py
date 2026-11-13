"""
Etapas 1 e 2 do pipeline: dos landmarks às medidas 3D de cada olho.

Pipeline completo (ver docs/metodologia.md):

    1. landmarks do rosto (face_tracker.py);
    2. centro do globo ocular e centro da pupila de cada olho (aqui e em
       eyeball.py);
    3. feixe de cada olho (centro do globo -> pupila) e a intersecção dos
       dois feixes (eyeball.py);
    4. o vetor até a intersecção é projetado no monitor virtual
       (geometry.VirtualScreen);
    5. a calibração com vários pontos ajusta o globo de cada olho e a
       correção monitor virtual -> monitor real (calibration.py).

Este módulo mede, a cada quadro, tudo em cm no referencial da câmera
(geometry.py):

* a rotação da cabeça R (head_pose.py) e a posição do rosto E;
* a ÂNCORA de cada olho: o ponto médio dos cantos;
* um ponto RÍGIDO na ponte do nariz (entre os olhos), que não se move com
  as pálpebras (os cantos se movem um pouco com o olhar vertical). O modelo
  do globo usa a ponte para a coordenada vertical da âncora (eyeball.py);
* a PUPILA de cada olho: o centro da íris na imagem (média do anel de 4
  pontos e do centro do MediaPipe, ou o centro refinado na imagem,
  iris_refine.py), levado a 3D pela linha de visada da câmera até o plano
  do olho (plano que passa pela âncora, perpendicular ao eixo z da cabeça).
  O z da íris do MediaPipe não é usado: é a coordenada mais ruidosa e, ao
  girar a cabeça, contaminava as outras. A posição da pupila em relação à
  âncora, no referencial da cabeça, passa por um filtro One-Euro
  (GazeConfig.pupil_filter): o ruído dela vira graus no feixe, e filtrá-la
  aqui estabiliza os feixes e a intersecção sem atrasar a rotação da
  cabeça, que entra depois;
* se cada olho está aberto (EAR relativo à linha de base de cada pessoa).

O CENTRO DO GLOBO não aparece na imagem: ele fica num deslocamento fixo no
referencial da cabeça a partir da âncora, estimado na calibração
(eyeball.EyeballModel). Antes de calibrar, usa-se o valor anatômico típico.

Tudo isso vai para o vetor bruto ``GazeSample.raw`` (layout abaixo), a
entrada da calibração. Ele é gravado nas sessões para reanálise offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GazeConfig
from .face_tracker import FaceObservation
from .geometry import focal_px, pixel_ray, px3d_to_cm, ray_plane
from .filters import OneEuroFilter
from .head_pose import HeadPose, estimate_head_pose, mouth_ratio
from .landmarks import LEFT_EYE, RIGHT_EYE, RIGID_NOSE, EyeIndices

# --------------------------------------------------------- layout de ``raw``
# Pontos em cm: cada landmark é convertido antes de tirar médias (a
# perspectiva não comuta com a média em pixels). As larguras dos olhos
# (cantos externo-interno, cm) dão o tamanho anatômico de cada globo.
# [R (9, linha a linha) | E (3) | âncora D (3), pupila D (3), âncora E (3),
#  pupila E (3) | pupila refinada D (3), pupila refinada E (3) |
#  aberto D, aberto E | boca | ponte do nariz (3) | largura do olho D, E]
#  (D = olho direito, E = esquerdo)
RAW_R = slice(0, 9)
RAW_FACE = slice(9, 12)
RAW_OPEN = slice(30, 32)
MOUTH_COL = 32
RAW_NOSE = slice(33, 36)
RAW_EYE_WIDTH = slice(36, 38)
RAW_SIZE = 38
# Fontes do centro da pupila (ver raw_parts).
PUPIL_SOURCES = ("mediapipe", "refinada", "media")
EYES: tuple[EyeIndices, ...] = (RIGHT_EYE, LEFT_EYE)


def raw_parts(raw: np.ndarray, source: str = "mediapipe"):
    """Separa o vetor bruto: R (n, 3, 3), âncoras (n, 2, 3), pupilas (n, 2, 3)
    da fonte pedida e olhos abertos (n, 2).

    Fontes do centro da pupila: ``mediapipe`` (centro da íris dos landmarks),
    ``refinada`` (centro da íris medido na imagem, iris_refine.py) e ``media``
    (média das duas; padrão da calibração quando a refinada existe).
    """
    raw = np.atleast_2d(raw)
    R = raw[:, RAW_R].reshape(-1, 3, 3)
    A = np.stack([raw[:, 12:15], raw[:, 18:21]], axis=1)
    P_mp = np.stack([raw[:, 15:18], raw[:, 21:24]], axis=1)
    if source == "refinada":
        P = np.stack([raw[:, 24:27], raw[:, 27:30]], axis=1)
    elif source == "media":   # média das duas fontes
        P = 0.5 * (P_mp + np.stack([raw[:, 24:27], raw[:, 27:30]], axis=1))
    else:
        P = P_mp
    return R, A, P, raw[:, RAW_OPEN] > 0.5


def has_refined_pupil(raw: np.ndarray) -> bool:
    """True se a fonte refinada difere da do MediaPipe em alguma amostra."""
    raw = np.atleast_2d(raw)
    return not np.allclose(raw[:, 24:30], raw[:, [15, 16, 17, 21, 22, 23]])


@dataclass
class EyeMeasure:
    name: str
    iris_px: np.ndarray       # (2,) centro da íris na imagem (desenho)
    anchor_cm: np.ndarray     # (3,) ponto médio dos cantos
    pupil_cm: np.ndarray      # (3,) pupila (MediaPipe)
    pupil_ref_cm: np.ndarray  # (3,) pupila refinada (= MediaPipe se indisponível)
    width_cm: float           # distância entre os cantos (cm)
    ear: float                # eye aspect ratio
    open: bool


@dataclass
class GazeSample:
    timestamp_ms: int
    head: HeadPose
    eyes: list[EyeMeasure]
    raw: np.ndarray                    # vetor bruto (layout acima)
    gaze: "object | None" = None       # eyeball.GazeRay (feixes, intersecção, vetor)
    blink: bool = False                # os dois olhos fechados
    extras: dict = field(default_factory=dict)
    obs: FaceObservation | None = None  # landmarks de origem (reanálise offline)

    @property
    def direction_cam(self) -> np.ndarray | None:
        return None if self.gaze is None else self.gaze.direction

    @property
    def valid(self) -> bool:
        return self.gaze is not None and not self.blink

    @property
    def yaw_deg(self) -> float | None:
        return None if self.gaze is None else direction_to_angles(self.gaze.direction)[0]

    @property
    def pitch_deg(self) -> float | None:
        return None if self.gaze is None else direction_to_angles(self.gaze.direction)[1]

    def features(self) -> np.ndarray:
        """(yaw, pitch) do olhar em graus: detecção de fixação e de outliers."""
        return np.array(direction_to_angles(self.gaze.direction), dtype=np.float64)


def direction_to_angles(direction) -> tuple[float, float]:
    """Vetor no referencial da câmera -> (yaw, pitch) em graus.

    yaw > 0: para a DIREITA do usuário; pitch > 0: para CIMA."""
    gx, gy, gz = np.asarray(direction, dtype=np.float64)
    return (float(np.degrees(np.arctan2(-gx, -gz))),
            float(np.degrees(np.arctan2(-gy, np.hypot(gx, gz)))))


def eye_position_cm(obs: FaceObservation, head: HeadPose,
                    outer_canthi_cm: float = 9.0, source: str = "pinhole") -> np.ndarray:
    """Posição 3D do rosto (cm, referencial da câmera): ponte do nariz.

    Distância pela câmera pinhole: z = f · 9 cm / distância 3D entre os
    cantos externos em px (o z do MediaPipe está na escala de x, então a
    distância 3D quase não muda com a rotação da cabeça). A translação da
    matriz do MediaPipe não é usada: a escala dela não é coerente com a
    câmera suposta (ver GazeConfig.outer_canthi_cm)."""
    if source == "mediapipe" and obs.transform is not None:
        return np.diag([1.0, -1.0, -1.0]) @ np.asarray(obs.transform, dtype=np.float64)[:3, 3]
    f = focal_px(obs.image_width)
    z = f * outer_canthi_cm / max(head.scale_px, 1e-6)
    cx, cy = head.origin[0], head.origin[1]
    return np.array([(cx - obs.image_width / 2) * z / f,
                     (cy - obs.image_height / 2) * z / f, z])


def pupil_on_eye_plane(xy, image_size, anchor_cm, normal) -> np.ndarray:
    """Pupila em 3D: linha de visada da câmera pelo pixel ``xy`` cruzada com o
    plano do olho (passa pela âncora, normal = eixo z da cabeça)."""
    P, _ = ray_plane(np.zeros(3), pixel_ray(np.asarray(xy, dtype=np.float64), image_size),
                     anchor_cm, normal)
    return P


class GazeEstimator:
    """Landmarks -> GazeSample. ``eyeball`` (eyeball.EyeballModel) dá os
    feixes, a intersecção e o vetor do olhar de cada quadro; o app troca o
    modelo anatômico padrão pelo calibrado."""

    def __init__(self, cfg: GazeConfig, eyeball=None):
        from .eyeball import EyeballModel
        self.cfg = cfg
        self.eyeball = eyeball if eyeball is not None else EyeballModel()
        self._ear_baseline = {e.name: cfg.ear_open_initial for e in EYES}
        # Filtro da pupila por olho e por fonte (ver GazeConfig.pupil_filter).
        self._pupil_filters = {(k, src): OneEuroFilter(cfg.pupil_min_cutoff, cfg.pupil_beta, 1.0)
                               for k in range(2) for src in range(2)}
        self._last_t: float | None = None

    def _filter_pupil(self, key, pupil, anchor, R, t: float) -> np.ndarray:
        """Filtra a pupila no referencial da cabeça, em relação à âncora: só o
        movimento do olho é suavizado; a rotação e a translação da cabeça
        passam direto."""
        q = R.T @ (pupil - anchor)
        q[:2] = self._pupil_filters[key](q[:2], t)
        return anchor + R @ q

    def _eye(self, obs: FaceObservation, head: HeadPose, depth_cm: float, idx: EyeIndices,
             refined_xy) -> EyeMeasure:
        pts = obs.points
        size = (obs.image_width, obs.image_height)
        outer, inner = pts[idx.outer_corner], pts[idx.inner_corner]
        width = max(float(np.linalg.norm(outer[:2] - inner[:2])), 1e-6)
        # Cada ponto vai para cm e só depois se tira a média: a perspectiva
        # não comuta com a média em pixels.
        corners = px3d_to_cm(np.stack([outer, inner]), size, depth_cm)
        anchor = corners.mean(axis=0)
        width_cm = float(np.linalg.norm(corners[0] - corners[1]))
        normal = head.R[:, 2]

        iris = np.vstack([pts[list(idx.iris_ring)], pts[idx.iris_center]]).mean(axis=0)[:2]
        pupil = pupil_on_eye_plane(iris, size, anchor, normal)
        ok_ref = refined_xy is not None and bool(np.all(np.isfinite(refined_xy)))
        pupil_ref = pupil_on_eye_plane(refined_xy, size, anchor, normal) if ok_ref else pupil

        # EAR: abertura vertical / largura; limiar relativo à linha de base.
        ear = float(np.linalg.norm(pts[idx.upper_lid][:2] - pts[idx.lower_lid][:2]) / width)
        baseline = self._ear_baseline[idx.name]
        is_open = ear >= self.cfg.blink_ratio * baseline
        if is_open:
            a = self.cfg.ear_baseline_alpha
            self._ear_baseline[idx.name] = (1 - a) * baseline + a * ear
        return EyeMeasure(idx.name, iris, anchor, pupil, pupil_ref, width_cm, ear,
                          bool(is_open))

    def estimate(self, obs: FaceObservation) -> GazeSample:
        head, pose_source = estimate_head_pose(obs.points, obs.transform,
                                               self.cfg.head_pose_source)
        face = eye_position_cm(obs, head, self.cfg.outer_canthi_cm, self.cfg.depth_source)
        ref = obs.iris_refined
        eyes = [self._eye(obs, head, float(face[2]), idx, None if ref is None else ref[k])
                for k, idx in enumerate(EYES)]
        if self.cfg.pupil_filter:
            t = obs.timestamp_ms / 1000.0
            gap = self._last_t is None or not 0 < t - self._last_t <= self.cfg.pupil_filter_reset_s
            self._last_t = t
            for k, e in enumerate(eyes):
                if gap or not e.open:
                    for src in range(2):
                        self._pupil_filters[(k, src)].reset()
                if e.open:
                    e.pupil_cm = self._filter_pupil((k, 0), e.pupil_cm, e.anchor_cm, head.R, t)
                    e.pupil_ref_cm = self._filter_pupil((k, 1), e.pupil_ref_cm, e.anchor_cm,
                                                        head.R, t)
        nose = px3d_to_cm(obs.points[list(RIGID_NOSE)], (obs.image_width, obs.image_height),
                          float(face[2])).mean(axis=0)
        mouth = mouth_ratio(obs.points)
        raw = np.concatenate([head.R.reshape(-1), face,
                              eyes[0].anchor_cm, eyes[0].pupil_cm,
                              eyes[1].anchor_cm, eyes[1].pupil_cm,
                              eyes[0].pupil_ref_cm, eyes[1].pupil_ref_cm,
                              [float(eyes[0].open), float(eyes[1].open), mouth], nose,
                              [eyes[0].width_cm, eyes[1].width_cm]])
        extras = {"pose_source": pose_source, "boca": mouth,
                  "iris_refinada": ref is not None and bool(np.all(np.isfinite(ref)))}
        blink = not any(e.open for e in eyes)
        sample = GazeSample(obs.timestamp_ms, head, eyes, raw, blink=blink, extras=extras,
                            obs=obs)
        if not blink:
            sample.gaze = self.eyeball.ray(raw)
        return sample
