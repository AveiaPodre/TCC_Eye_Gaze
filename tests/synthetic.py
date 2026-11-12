"""
Rosto 3D sintético para testar o pipeline sem câmera nem modelo.

Gera um array (478, 3) de landmarks em "pixels 3D" (mesma convenção do
FaceTracker) a partir de: rotação/posição/escala da cabeça e direção do
olhar no referencial da câmera. Só os landmarks usados pelo pipeline têm
posição anatômica plausível; os demais ficam espalhados no rosto.
"""

from __future__ import annotations

import math

import numpy as np

from eyegaze.face_tracker import FaceObservation
from eyegaze.landmarks import LEFT_EYE, RIGHT_EYE

RHO = 0.40  # raio do globo / largura do olho (igual ao padrão do estimador)

# Referencial da cabeça: x = esquerda do usuário, y = baixo, z = para trás.
BASE = {
    33: (-4.5, 0.0, 1.0), 133: (-1.5, 0.0, 0.3),
    263: (4.5, 0.0, 1.0), 362: (1.5, 0.0, 0.3),
    127: (-7.0, 0.5, 4.0), 356: (7.0, 0.5, 4.0),
    151: (0.0, -6.0, 1.5), 9: (0.0, -3.5, 0.5), 8: (0.0, -2.5, 0.3),
    2: (0.0, 5.0, -1.5), 164: (0.0, 6.0, -0.8), 168: (0.0, -1.0, 0.0),
    13: (0.0, 7.0, -1.2), 14: (0.0, 7.05, -1.2),   # lábios internos (boca fechada)
}


def rot(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0) -> np.ndarray:
    """Rotação da cabeça (yaw > 0 = vira para a direita do usuário)."""
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    # Virar para a direita do usuário = face aponta para -x da imagem.
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    # Pitch > 0 = olhar para cima = frente (-z) ganha componente -y.
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def gaze_dir(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Inverso de gaze.direction_to_angles."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([-math.sin(y) * math.cos(p), -math.sin(p), -math.cos(y) * math.cos(p)])


def make_points(gaze_yaw=0.0, gaze_pitch=0.0, head=(0.0, 0.0, 0.0),
                center=(640.0, 360.0, 0.0), scale=12.0, blink=False,
                noise_px=0.0, rng: np.random.Generator | None = None,
                gaze_vec=None, rho_true: float = RHO, v_scale: float = 1.0,
                pitch_coupling: float = 0.0, lid_follow: float = 0.0,
                mouth_open: float = 0.0, mouth_coupling: float = 0.0,
                perspective_cm=None) -> np.ndarray:
    """Landmarks sintéticos.

    gaze_vec: direção do olhar na câmera (substitui gaze_yaw/gaze_pitch).
    rho_true: raio real do globo (em larguras de olho) - pode diferir do
        valor assumido pelo estimador, como acontece em pessoas reais.
    v_scale: fator no deslocamento VERTICAL da íris (pálpebras cobrem a
        íris e o MediaPipe "achata" o movimento vertical).
    pitch_coupling: deslocamento vertical espúrio da íris proporcional ao
        pitch da cabeça (perspectiva/pálpebra), em raios por radiano.
    lid_follow: quanto as pálpebras acompanham o deslocamento vertical
        verdadeiro do olho (a superior desce ao olhar para baixo).
    mouth_open: abertura da boca (lábios / distância entre os cantos externos,
        como head_pose.mouth_ratio).
    mouth_coupling: artefato do MediaPipe observado nos dados reais: com a
        boca aberta, a íris e as pálpebras "descem" (mouth_coupling x
        abertura x largura do olho), e o olhar estimado desce sem o olho
        se mover.
    perspective_cm: posição da cabeça (cm, referencial da câmera). Com ela,
        x e y são a projeção em PERSPECTIVA de cada ponto (como os landmarks
        reais do MediaPipe) e z é a profundidade relativa ao rosto na escala
        de x; sem ela, perspectiva fraca (escala única ``scale``).
    """
    rng = rng or np.random.default_rng(0)
    R = rot(*head)
    local = {k: np.array(v, dtype=float) for k, v in BASE.items()}

    g = gaze_dir(gaze_yaw, gaze_pitch) if gaze_vec is None else np.asarray(gaze_vec, float)
    d_head = R.T @ (g / np.linalg.norm(g))
    for eye in (RIGHT_EYE, LEFT_EYE):
        outer, inner = local[eye.outer_corner], local[eye.inner_corner]
        mid = (outer + inner) / 2
        w = np.linalg.norm(outer - inner)
        r = rho_true * w
        center_eye = mid + np.array([0, 0, r])
        iris = center_eye + r * d_head
        off = iris - mid
        gaze_v = off[1]   # deslocamento vertical "verdadeiro" da íris
        off[1] = v_scale * off[1] - pitch_coupling * r * math.radians(head[1])
        iris = mid + off
        local[eye.iris_center] = iris
        for k, (dx, dy) in zip(eye.iris_ring, [(0.6, 0), (0, -0.6), (-0.6, 0), (0, 0.6)]):
            local[k] = iris + np.array([dx, dy, 0])
        open_h = 0.05 if blink else 0.5
        # Pálpebras acompanham o olhar vertical (a superior mais que a inferior).
        # Pálpebras apoiadas na superfície frontal do globo.
        def on_ball(dx, dy):
            dz = math.sqrt(max(r * r - dx * dx - dy * dy, 0.0))
            return center_eye + np.array([dx, dy, -dz])

        up_y = -open_h + lid_follow * gaze_v
        lo_y = open_h + 0.3 * lid_follow * gaze_v
        for k, dx in zip((eye.upper_lid,) + eye.upper_lid_ring, (0, -0.4, 0, 0.4)):
            local[k] = on_ball(dx, up_y)
        for k, dx in zip((eye.lower_lid,) + eye.lower_lid_ring, (0, -0.4, 0, 0.4)):
            local[k] = on_ball(dx, lo_y)

    # Boca: lábio inferior desce com a abertura (largura entre cantos = 9).
    local[14] = local[14] + np.array([0.0, 9.0 * mouth_open, 0.0])
    if mouth_coupling and mouth_open:
        shift = np.array([0.0, mouth_coupling * mouth_open * 3.0, 0.0])
        for eye in (RIGHT_EYE, LEFT_EYE):
            for k in ((eye.iris_center,) + eye.iris_ring + (eye.upper_lid, eye.lower_lid)
                      + eye.upper_lid_ring + eye.lower_lid_ring):
                local[k] = local[k] + shift
    pts = rng.uniform([-6, -6, -1], [6, 7, 3], size=(478, 3))
    for k, v in local.items():
        pts[k] = v
    if perspective_cm is not None:
        P = np.asarray(perspective_cm, dtype=float) + pts @ R.T
        z0 = float(perspective_cm[2])
        out = np.column_stack([IMG_W / 2 + FOCAL * P[:, 0] / P[:, 2],
                               IMG_H / 2 + FOCAL * P[:, 1] / P[:, 2],
                               (P[:, 2] - z0) * FOCAL / z0])
    else:
        out = scale * (pts @ R.T) + np.asarray(center)
    if noise_px:
        out += rng.normal(0, noise_px, out.shape)
    return out


IMG_W, IMG_H = 1280, 720
FOCAL = 0.5 * IMG_W / math.tan(math.radians(30))   # câmera com FOV horizontal de 60°


def make_transform(head=(0.0, 0.0, 0.0), head_pos_cm=(0.0, 0.0, 50.0)) -> np.ndarray:
    """Matriz 4x4 como a do FaceLandmarker (espaço métrico: y p/ cima, z p/ fora)."""
    C = np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = C @ rot(*head) @ C
    T[:3, 3] = C @ np.asarray(head_pos_cm, dtype=float)
    return T


def make_observation(*args, timestamp_ms=0, with_transform=False, head_pos_cm=None,
                     **kwargs) -> FaceObservation:
    """Observação sintética. Com head_pos_cm (x, y, z em cm, eixos da imagem),
    cada landmark é projetado por uma câmera pinhole (perspectiva verdadeira),
    a menos que ``center``/``scale`` sejam dados."""
    if head_pos_cm is not None:
        x, y, z = head_pos_cm
        if "center" not in kwargs and "scale" not in kwargs:
            kwargs["perspective_cm"] = tuple(head_pos_cm)
        kwargs.setdefault("center", (IMG_W / 2 + FOCAL * x / z, IMG_H / 2 + FOCAL * y / z, 0.0))
        kwargs.setdefault("scale", FOCAL / z)
    pts = make_points(*args, **kwargs)
    tf = None
    if with_transform:
        tf = make_transform(kwargs.get("head", (0.0, 0.0, 0.0)),
                            head_pos_cm if head_pos_cm is not None else (0.0, 0.0, 50.0))
    return FaceObservation(pts, IMG_W, IMG_H, timestamp_ms, tf)


class ScreenGeometry:
    """Monitor 53x30 cm, 1920x1080, logo abaixo da câmera (plano z = 0).

    Coordenadas da câmera em cm: x para a direita da imagem, y para baixo,
    z para longe da câmera. Um ponto (x_px, y_px) da tela fica em
    X = (x_px/W - 0.5) * 53, Y = 1 + y_px/H * 30, Z = 0.

    Atenção: "direita da imagem" é a ESQUERDA do usuário (imagem não
    espelhada), então a tela aparece espelhada nesse referencial.
    """

    def __init__(self, w_px=1920, h_px=1080, w_cm=53.0, h_cm=30.0, top_cm=1.0,
                 eye_cm=(0.0, 10.0, 60.0)):
        self.w_px, self.h_px, self.w_cm, self.h_cm = w_px, h_px, w_cm, h_cm
        self.top = top_cm
        self.eye = np.array(eye_cm, dtype=float)
        self.dist = eye_cm[2]

    def px_to_cm(self, x_px, y_px):
        # Direita da tela (para o usuário) = esquerda da imagem = -x.
        return np.array([-(x_px / self.w_px - 0.5) * self.w_cm,
                         self.top + y_px / self.h_px * self.h_cm, 0.0])

    def gaze_vector(self, x_px, y_px, eye_cm=None):
        e = self.eye if eye_cm is None else np.asarray(eye_cm, dtype=float)
        d = self.px_to_cm(x_px, y_px) - e
        return d / np.linalg.norm(d)

    def px_to_angles(self, x_px, y_px, head_shift_cm=(0.0, 0.0)):
        """Ângulos (yaw, pitch) do olhar para o pixel (convenção do estimador)."""
        e = self.eye + np.array([-head_shift_cm[0], head_shift_cm[1], 0.0])
        g = self.gaze_vector(x_px, y_px, e)
        yaw = math.degrees(math.atan2(-g[0], -g[2]))
        pitch = math.degrees(math.atan2(-g[1], math.hypot(g[0], g[2])))
        return yaw, pitch
