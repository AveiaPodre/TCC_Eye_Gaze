"""
Estimação da direção do olhar (modelo geométrico de globo ocular).

Ideia herdada do Webcam3DTracker: cada olho é uma esfera que gira dentro da
cabeça; a direção do olhar é o vetor do centro da esfera até a íris. As
diferenças em relação à referência:

1. O centro do olho é ancorado nos CANTOS do olho a cada frame (pontos
   presos ao crânio), em vez de ser "travado" uma única vez na calibração.
   Assim não acumula erro se a pessoa se mexe.
2. O deslocamento da íris é medido no referencial da CABEÇA (head_pose.py)
   e normalizado pela largura do olho. Com isso ele independe da distância
   à câmera e da rotação da cabeça.
3. Não usamos o z da íris (a coordenada mais ruidosa do MediaPipe): a
   profundidade da íris sai da própria restrição de estar na esfera.

Para cada olho:
    m  = ponto médio dos cantos;  w = distância entre os cantos
    (dx, dy, _) = R^T (íris - m) / w          (referencial da cabeça)
    pitch_olho = asin(-dy / rho)
    yaw_olho   = asin( dx / (rho cos pitch_olho))
onde rho = raio do globo ocular / largura do olho (~0.4).

A direção do olho na cabeça é convertida para a câmera (g = R g_cabeça) e os
dois olhos são combinados. O resultado final são dois ângulos no
referencial da câmera, que combinam rotação da cabeça + rotação dos olhos:

    yaw   > 0  -> olhando para a DIREITA do usuário
    pitch > 0  -> olhando para CIMA

Esses dois ângulos servem para o HUD e para rejeitar outliers. A
calibração usa as medidas brutas (``GazeSample.raw``), incluindo as
medidas verticais pelas PÁLPEBRAS:

    lid_v   = (íris_y - linha média das pálpebras) / largura do olho
    abertura = (pálpebra inferior - superior) / largura do olho

Nos dados reais, o centro da íris do MediaPipe quase não se move na
vertical em relação aos cantos do olho (cerca de 3 graus de variação para a
tela inteira, contra os ~15-18 esperados). Já as pálpebras acompanham o
olhar vertical (a superior desce quando se olha para baixo), então essas
duas medidas trazem a maior parte do sinal vertical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GazeConfig
from .face_tracker import FaceObservation
from .head_pose import HeadPose, estimate_head_pose
from .landmarks import LEFT_EYE, RIGHT_EYE, EyeIndices


@dataclass
class EyeGaze:
    name: str
    iris_px: np.ndarray            # (3,) centro da íris
    corners_mid_px: np.ndarray     # (3,) ponto médio dos cantos
    eyeball_center_px: np.ndarray  # (3,) centro estimado do globo ocular
    eyeball_radius_px: float
    direction_cam: np.ndarray      # (3,) vetor unitário no referencial da câmera
    yaw_in_head_deg: float
    pitch_in_head_deg: float
    ear: float                     # eye aspect ratio
    open: bool
    # Deslocamento da íris em relação ao ponto médio dos cantos, no
    # referencial da cabeça, em larguras de olho (entrada da calibração).
    offset_u: float = 0.0
    offset_v: float = 0.0
    # Medidas verticais pelas pálpebras (ver docstring do módulo).
    lid_v: float = 0.0
    aperture: float = 0.0
    # (u, v, lid_v) com o centro da íris refinado na imagem (iris_refine.py);
    # igual ao do MediaPipe quando o refinamento não está disponível.
    refined: tuple[float, float, float] = (0.0, 0.0, 0.0)
    refined_ok: bool = False


@dataclass
class GazeSample:
    timestamp_ms: int
    head: HeadPose
    eyes: list[EyeGaze]
    direction_cam: np.ndarray | None  # combinação dos olhos abertos
    yaw_deg: float | None
    pitch_deg: float | None
    blink: bool                       # ambos os olhos fechados
    extras: dict = field(default_factory=dict)
    # Vetor bruto usado pela calibração (ver calibration.RAW_LAYOUT):
    # [R (9, linha a linha), posição do olho em cm (3), u, v, lid_v, abertura
    #  (íris do MediaPipe), u, v, lid_v, abertura (íris refinada)]
    raw: np.ndarray | None = None
    # Observacao de origem (landmarks), gravada para reanalise offline.
    obs: FaceObservation | None = None

    @property
    def valid(self) -> bool:
        return self.direction_cam is not None and not self.blink

    def features(self) -> np.ndarray:
        """Vetor de entrada do modelo de calibração: [yaw, pitch] em graus."""
        return np.array([self.yaw_deg, self.pitch_deg], dtype=np.float64)


def direction_to_angles(direction: np.ndarray) -> tuple[float, float]:
    """Converte vetor (referencial da câmera, pixels 3D) em (yaw, pitch) graus."""
    gx, gy, gz = direction
    yaw = np.degrees(np.arctan2(-gx, -gz))
    pitch = np.degrees(np.arctan2(-gy, np.hypot(gx, gz)))
    return float(yaw), float(pitch)


def angles_to_head_direction(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """Direção no referencial da cabeça para ângulos olho-na-cabeça.

    yaw > 0 move a íris para +x da cabeça (lado esquerdo do usuário na
    convenção de landmarks), pitch > 0 move para cima (-y). Olhar para a
    frente é -z.
    """
    cp = np.cos(pitch_rad)
    return np.array([
        np.sin(yaw_rad) * cp,
        -np.sin(pitch_rad),
        -np.cos(yaw_rad) * cp,
    ])


_C = np.diag([1.0, -1.0, -1.0])


def eye_position_cm(obs: FaceObservation, head: HeadPose, assumed_hfov_deg: float = 60.0,
                    outer_canthi_cm: float = 9.0) -> np.ndarray:
    """Posição 3D do rosto em relação à câmera, em cm.

    Eixos como em pixels 3D: x para a direita da imagem, y para baixo, z
    para longe da câmera (z > 0 = distância).

    Usa a translação da matriz de pose do FaceLandmarker (espaço métrico,
    em cm). Sem ela, estima por câmera pinhole: distância = f * 9 cm /
    distância entre cantos externos em px (9 cm é a média adulta).
    """
    if obs.transform is not None:
        return _C @ np.asarray(obs.transform, dtype=np.float64)[:3, 3]
    f = 0.5 * obs.image_width / np.tan(np.radians(assumed_hfov_deg / 2))
    z = f * outer_canthi_cm / max(head.scale_px, 1e-6)
    cx, cy = head.origin[0], head.origin[1]
    return np.array([(cx - obs.image_width / 2) * z / f,
                     (cy - obs.image_height / 2) * z / f,
                     z])


def on_eyeball(point: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Substitui o z de ``point`` pelo z da superfície FRONTAL do globo.

    Mantém x e y da imagem (bem medidos) e resolve a profundidade pela
    interseção da linha de visada da câmera (paralela a z) com a esfera.
    Se a linha não cruza a esfera, usa a profundidade do centro.
    """
    d2 = (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2
    z = center[2] - np.sqrt(max(radius ** 2 - d2, 0.0))
    return np.array([point[0], point[1], z])


class GazeEstimator:
    def __init__(self, cfg: GazeConfig):
        self.cfg = cfg
        self._ear_baseline = {RIGHT_EYE.name: cfg.ear_open_initial,
                              LEFT_EYE.name: cfg.ear_open_initial}

    def _eye(self, pts: np.ndarray, head: HeadPose, idx: EyeIndices,
             refined_xy: np.ndarray | None = None) -> EyeGaze:
        outer = pts[idx.outer_corner]
        inner = pts[idx.inner_corner]
        mid = (outer + inner) / 2.0
        width = max(float(np.linalg.norm(outer - inner)), 1e-6)

        # Centro da íris: média do anel (4 pontos) + ponto central.
        iris = np.vstack([pts[list(idx.iris_ring)], pts[idx.iris_center]]).mean(axis=0)

        rho = self.cfg.eyeball_radius_ratio
        radius_px = rho * width
        # Centro do globo: atrás do ponto médio dos cantos (para a frente
        # é -z, então "atrás" é +z no referencial da cabeça).
        eyeball_center = mid + head.R @ np.array([0.0, 0.0, radius_px])

        # O z da íris do MediaPipe NÃO é usado: ele é ruidoso e, ao girar
        # o vetor para o referencial da cabeça (R^T), "vazava" para u e v
        # quando a cabeça girava (nos dados reais, u variava 4,5x mais por
        # grau de cabeça do que por grau de olho, e v ~14x). A profundidade
        # sai da restrição de a íris estar na superfície do globo.
        def to_eye(p: np.ndarray) -> np.ndarray:
            return head.R.T @ (on_eyeball(p, eyeball_center, radius_px) - mid) / width

        local = to_eye(iris)
        sin_pitch = np.clip(-local[1] / rho, -0.95, 0.95)
        pitch = float(np.arcsin(sin_pitch))
        sin_yaw = np.clip(local[0] / (rho * np.cos(pitch)), -0.95, 0.95)
        yaw = float(np.arcsin(sin_yaw))

        dir_head = angles_to_head_direction(yaw, pitch)
        dir_cam = head.R @ dir_head

        # Pálpebras (média de vários pontos): também apoiadas no globo, com a
        # mesma correção de profundidade.
        up = pts[list(idx.upper_lid_ring or (idx.upper_lid,))].mean(axis=0)
        lo = pts[list(idx.lower_lid_ring or (idx.lower_lid,))].mean(axis=0)
        up_l = to_eye(up)
        lo_l = to_eye(lo)
        lid_mid = (up_l[1] + lo_l[1]) / 2
        lid_v = float(local[1] - lid_mid)
        aperture = float(lo_l[1] - up_l[1])

        refined_ok = refined_xy is not None and bool(np.all(np.isfinite(refined_xy)))
        if refined_ok:
            # z provisório: on_eyeball recalcula a profundidade.
            loc_r = to_eye(np.array([refined_xy[0], refined_xy[1], mid[2]]))
            refined = (float(loc_r[0]), float(loc_r[1]), float(loc_r[1] - lid_mid))
        else:
            refined = (float(local[0]), float(local[1]), lid_v)

        # EAR: abertura vertical / largura.
        ear = float(np.linalg.norm(pts[idx.upper_lid] - pts[idx.lower_lid]) / width)
        baseline = self._ear_baseline[idx.name]
        is_open = ear >= self.cfg.blink_ratio * baseline
        if is_open:
            a = self.cfg.ear_baseline_alpha
            self._ear_baseline[idx.name] = (1 - a) * baseline + a * ear

        return EyeGaze(
            name=idx.name,
            iris_px=iris,
            corners_mid_px=mid,
            eyeball_center_px=eyeball_center,
            eyeball_radius_px=radius_px,
            direction_cam=dir_cam,
            # Sinal invertido para seguir a convenção da saída
            # (yaw > 0 = direita do usuário).
            yaw_in_head_deg=float(-np.degrees(yaw)),
            pitch_in_head_deg=float(np.degrees(pitch)),
            ear=ear,
            open=bool(is_open),
            offset_u=float(local[0]),
            offset_v=float(local[1]),
            lid_v=lid_v,
            aperture=aperture,
            refined=refined,
            refined_ok=refined_ok,
        )

    def estimate(self, obs: FaceObservation) -> GazeSample:
        pts = obs.points
        head, pose_source = estimate_head_pose(pts, obs.transform, self.cfg.head_pose_source)
        extras = {"pose_source": pose_source}
        ref = obs.iris_refined
        eyes = [self._eye(pts, head, RIGHT_EYE, None if ref is None else ref[0]),
                self._eye(pts, head, LEFT_EYE, None if ref is None else ref[1])]

        open_eyes = [e for e in eyes if e.open]
        if not open_eyes:
            return GazeSample(obs.timestamp_ms, head, eyes, None, None, None, blink=True,
                              extras=extras)

        combined = np.mean([e.direction_cam for e in open_eyes], axis=0)
        combined /= np.linalg.norm(combined)
        yaw, pitch = direction_to_angles(combined)

        u = float(np.mean([e.offset_u for e in open_eyes]))
        v = float(np.mean([e.offset_v for e in open_eyes]))
        lv = float(np.mean([e.lid_v for e in open_eyes]))
        ap = float(np.mean([e.aperture for e in open_eyes]))
        ur, vr, lvr = np.mean([e.refined for e in open_eyes], axis=0)
        extras["iris_refinada"] = all(e.refined_ok for e in open_eyes)
        raw = np.concatenate([head.R.reshape(-1), eye_position_cm(obs, head),
                              [u, v, lv, ap, ur, vr, lvr, ap]])
        return GazeSample(obs.timestamp_ms, head, eyes, combined, yaw, pitch, blink=False,
                          extras=extras, raw=raw, obs=obs)
