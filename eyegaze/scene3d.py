"""
Vista 3D da cena (inspirada na janela "Head/Eye Debug" do Webcam3DTracker).

Mostra, numa câmera virtual que orbita a cena: o rosto (landmarks), os
globos oculares, o raio do olhar de cada olho, o plano da tela com o ponto
atingido pelo olhar e a webcam. Serve para entender o que o modelo está
vendo: se a pose da cabeça está coerente, para onde cada olho aponta e
quanto a calibração corrige o raio puramente geométrico.

Dois "olhares" são desenhados até a tela:

* AMARELO: o cursor (saída da calibração + estabilizador), que é o que
  move o mouse;
* MAGENTA: o raio geométrico do modelo de globo ocular (gaze.py), sem
  calibração, intersectado com o plano da tela. A distância entre os dois
  mostra o quanto a calibração está corrigindo.

Referencial (cm, o mesmo de ``gaze.eye_position_cm``): origem na webcam,
x para a direita da imagem, y para baixo, z para longe da câmera. A tela é
o plano z = 0, com a webcam centralizada logo acima da borda superior
(notebook ou webcam presa no monitor). A largura física da tela vem de
``--screen-width-cm``; sem ela, é suposta (``DisplayConfig``).

Como a imagem da câmera NÃO é espelhada, o lado direito do usuário é -x:
o pixel x = 0 da tela (borda esquerda, do ponto de vista do usuário) fica
em x = +largura/2.

Controles (com a janela 3D em foco): arrastar com o botão esquerdo gira,
a roda do mouse aproxima/afasta e o botão direito volta à vista inicial.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .face_tracker import FaceObservation
from .calibration import SCREEN_GAP_CM, head_screen_point
from .gaze import GazeSample, direction_to_angles, eye_position_cm

FONT = cv2.FONT_HERSHEY_SIMPLEX
ASSUMED_HFOV_DEG = 60.0   # mesmo valor de gaze.eye_position_cm

C_FACE = (150, 150, 150)
C_SCREEN = (0, 200, 255)
C_EYE = (255, 255, 25)
C_EYE_RAY = (255, 200, 60)
C_GEOM = (255, 60, 255)
C_CURSOR = (0, 255, 255)
C_RAW = (170, 170, 170)
C_TARGET = (110, 110, 110)
C_CAMERA = (80, 230, 80)
C_FACE_DIR = (90, 255, 90)


@dataclass
class OrbitView:
    """Câmera virtual que orbita um ponto da cena."""

    yaw_deg: float = 40.0      # 0 = atrás do usuário, olhando para a tela
    pitch_deg: float = 20.0    # > 0 = de cima
    radius_cm: float = 85.0
    fov_deg: float = 50.0
    _drag: tuple[int, int] | None = None

    DEFAULTS = (40.0, 20.0, 85.0)

    def reset(self):
        self.yaw_deg, self.pitch_deg, self.radius_cm = self.DEFAULTS

    def on_mouse(self, event, x, y, flags, _param=None):
        """Callback para ``cv2.setMouseCallback``."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            dx, dy = x - self._drag[0], y - self._drag[1]
            self._drag = (x, y)
            self.yaw_deg = (self.yaw_deg + 0.4 * dx + 180.0) % 360.0 - 180.0
            self.pitch_deg = float(np.clip(self.pitch_deg + 0.4 * dy, -85.0, 85.0))
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            self.radius_cm = float(np.clip(self.radius_cm * (0.9 if delta > 0 else 1.1),
                                           30.0, 600.0))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset()


# ------------------------------------------------------------- geometria
def screen_rect_cm(screen_size, screen_width_cm: float) -> tuple[float, float]:
    sw, sh = screen_size
    return screen_width_cm, screen_width_cm * sh / sw


def screen_px_to_cm(p, screen_size, screen_width_cm: float) -> np.ndarray:
    """Pixel da tela -> ponto 3D (cm) no plano z = 0."""
    sw, sh = screen_size
    wcm, hcm = screen_rect_cm(screen_size, screen_width_cm)
    return np.array([(0.5 - p[0] / sw) * wcm, SCREEN_GAP_CM + p[1] / sh * hcm, 0.0])


def landmarks_cm(obs: FaceObservation, depth_cm: float) -> tuple[np.ndarray, float]:
    """Landmarks em cm por câmera pinhole, com o rosto na profundidade dada.

    Devolve (pontos (n, 3), cm por pixel na profundidade do rosto). O z do
    MediaPipe está na escala de x em pixels, relativo ao centro do rosto.
    """
    w, h = obs.image_width, obs.image_height
    f = 0.5 * w / np.tan(np.radians(ASSUMED_HFOV_DEG / 2))
    pts = obs.points
    z = depth_cm + pts[:, 2] * depth_cm / f
    out = np.column_stack([(pts[:, 0] - w / 2) * z / f, (pts[:, 1] - h / 2) * z / f, z])
    return out, depth_cm / f


def px3d_to_cm(p, obs: FaceObservation, depth_cm: float) -> np.ndarray:
    w, h = obs.image_width, obs.image_height
    f = 0.5 * w / np.tan(np.radians(ASSUMED_HFOV_DEG / 2))
    z = depth_cm + p[2] * depth_cm / f
    return np.array([(p[0] - w / 2) * z / f, (p[1] - h / 2) * z / f, z])


def hit_screen(origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    """Interseção do raio com o plano da tela (z = 0), ou None."""
    if direction[2] >= -1e-6:
        return None   # apontando para longe da tela
    t = -origin[2] / direction[2]
    return origin + t * direction if t > 0 else None


# ------------------------------------------------------------ projeção
class _Projector:
    def __init__(self, view: OrbitView, pivot: np.ndarray, size: tuple[int, int]):
        self.w, self.h = size
        yaw, pitch = np.radians(view.yaw_deg), np.radians(view.pitch_deg)
        offset = view.radius_cm * np.array([np.sin(yaw) * np.cos(pitch),
                                            -np.sin(pitch),
                                            np.cos(yaw) * np.cos(pitch)])
        self.cam = pivot + offset
        fwd = pivot - self.cam
        fwd /= np.linalg.norm(fwd)
        up_world = np.array([0.0, -1.0, 0.0])
        right = np.cross(fwd, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([-1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.V = np.stack([right, up, fwd])
        self.f = 0.5 * self.w / np.tan(np.radians(view.fov_deg / 2))

    def __call__(self, P) -> tuple[int, int] | None:
        pc = self.V @ (np.asarray(P, dtype=np.float64) - self.cam)
        if pc[2] <= 1.0:
            return None
        x = self.w / 2 + self.f * pc[0] / pc[2]
        y = self.h / 2 - self.f * pc[1] / pc[2]
        if not (np.isfinite(x) and np.isfinite(y)) or abs(x) > 1e5 or abs(y) > 1e5:
            return None
        return int(round(x)), int(round(y))

    def depth(self, P) -> float:
        return float((self.V @ (np.asarray(P, dtype=np.float64) - self.cam))[2])

    def line(self, img, a, b, color, thick=1):
        pa, pb = self(a), self(b)
        if pa is not None and pb is not None:
            cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)

    def poly(self, img, pts, color, thick=1, closed=True):
        proj = [self(p) for p in pts]
        if any(p is None for p in proj):
            return
        cv2.polylines(img, [np.array(proj, np.int32)], closed, color, thick, cv2.LINE_AA)

    def circle_3d(self, img, center, radius_cm, color, thick=1):
        """Círculo de raio constante em cm (encolhe com a distância)."""
        c = self(center)
        if c is None:
            return
        r = max(1, int(round(self.f * radius_cm / max(self.depth(center), 1e-3))))
        cv2.circle(img, c, r, color, thick, cv2.LINE_AA)


def _label(img, s, org, color=(230, 230, 230), scale=0.45):
    cv2.putText(img, s, org, FONT, scale, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


# -------------------------------------------------------------- desenho
def render_scene(view: OrbitView, screen_size: tuple[int, int], screen_width_cm: float,
                 sample: GazeSample | None, obs: FaceObservation | None,
                 cursor=None, raw_point=None, trail=(), targets=(),
                 assumed_width: bool = True, size: tuple[int, int] = (760, 560),
                 head_ref=None, face_bias=None, calibrated_gaze=None) -> np.ndarray:
    """head_ref: pose da cabeça na calibração (yaw, pitch); com ela, o
    texto mostra a cabeça em relação à calibração. face_bias: viés da
    direção da face (calibration.face_bias); com ele, desenha em verde o
    raio do rosto até a tela já corrigido. calibrated_gaze: (direção (3,),
    ponto atingido na tela em cm (3,)) do olhar geométrico CALIBRADO
    (o raio dos olhos até o ponto estimado pelo modelo do cursor, antes do
    estabilizador); com ele, o raio magenta é o calibrado."""
    """Imagem da cena 3D vista pela câmera virtual ``view``."""
    W, H = size
    img = np.full((H, W, 3), (28, 24, 22), np.uint8)
    to_cm = lambda p: screen_px_to_cm(p, screen_size, screen_width_cm)  # noqa: E731
    sw, sh = screen_size
    corners = [to_cm(p) for p in ((0, 0), (sw, 0), (sw, sh), (0, sh))]
    screen_center = to_cm((sw / 2, sh / 2))

    E = None
    if sample is not None and obs is not None:
        E = (sample.raw[9:12] if sample.raw is not None
             else eye_position_cm(obs, sample.head))
    head_c = E if E is not None else np.array([0.0, 8.0, 60.0])
    pivot = 0.5 * (head_c + screen_center)
    pj = _Projector(view, pivot, size)

    # --- chão de referência (grade no plano y da base da tela) ----------
    floor_y = corners[2][1] + 4.0
    for k in np.linspace(-40, 40, 9):
        pj.line(img, (k, floor_y, -5), (k, floor_y, 90), (50, 46, 44))
    for k in np.linspace(-5, 90, 8):
        pj.line(img, (-40, floor_y, k), (40, floor_y, k), (50, 46, 44))

    # --- tela ---------------------------------------------------------
    pj.poly(img, corners, C_SCREEN, 2)
    pj.line(img, corners[0], corners[2], (0, 110, 150))
    pj.line(img, corners[1], corners[3], (0, 110, 150))
    for t in targets:
        p = pj(to_cm(t))
        if p is not None:
            cv2.circle(img, p, 2, C_TARGET, -1, cv2.LINE_AA)
    tl = pj(corners[0])
    if tl is not None:
        txt = f"tela {screen_width_cm:.0f} cm" + (" (suposta)" if assumed_width else "")
        _label(img, txt, (tl[0] + 4, tl[1] - 6), C_SCREEN)

    # --- webcam -------------------------------------------------------
    s = 1.2
    cam_pts = [np.array(v) * s for v in ((-1, -0.7, 1.6), (1, -0.7, 1.6), (1, 0.7, 1.6),
                                         (-1, 0.7, 1.6))]
    pj.poly(img, cam_pts, C_CAMERA, 1)
    for v in cam_pts:
        pj.line(img, (0, 0, 0), v, C_CAMERA)

    # --- trilha e cursor na tela --------------------------------------
    tr = [pj(to_cm(p)) for p in trail]
    for a, b in zip(tr, tr[1:]):
        if a is not None and b is not None:
            cv2.line(img, a, b, (0, 140, 200), 1, cv2.LINE_AA)
    if raw_point is not None:
        p = pj(to_cm(raw_point))
        if p is not None:
            cv2.circle(img, p, 3, C_RAW, -1, cv2.LINE_AA)

    lines = []
    if sample is None or obs is None:
        _label(img, "Rosto nao detectado", (12, 24), (80, 80, 255), 0.6)
    else:
        pts_cm, cm_per_px = landmarks_cm(obs, E[2])
        for P in pts_cm[:468:2]:
            p = pj(P)
            if p is not None:
                cv2.circle(img, p, 1, C_FACE, -1)

        # Eixos da cabeça a partir do ponto entre os olhos.
        o = px3d_to_cm(sample.head.origin, obs, E[2])
        for vec, color in ((sample.head.R[:, 0], (0, 0, 255)), (sample.head.R[:, 1], (0, 200, 0)),
                           (-sample.head.R[:, 2], (255, 80, 0))):
            pj.line(img, o, o + 5.0 * vec, color, 2)

        # Globos oculares e raio de cada olho.
        centers = []
        for eye in sample.eyes:
            c = px3d_to_cm(eye.eyeball_center_px, obs, E[2])
            centers.append(c)
            pj.circle_3d(img, c, eye.eyeball_radius_px * cm_per_px, C_EYE if eye.open else C_RAW)
            if eye.open and calibrated_gaze is None:   # raios crus só antes de calibrar
                pj.line(img, c, c + 12.0 * eye.direction_cam, C_EYE_RAY)
        mid = np.mean(centers, axis=0)

        # Raio geométrico até a tela: calibrado, se houver calibração; senão
        # o do modelo do globo ocular puro (~40° para cima do olhar real).
        if calibrated_gaze is not None:
            gdir, ghit = calibrated_gaze
            pj.line(img, mid, ghit, C_GEOM, 1)
            p = pj(ghit)
            if p is not None:
                cv2.drawMarker(img, p, C_GEOM, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
            yaw, pitch = direction_to_angles(gdir)
            lines.append((f"olhar calibrado: yaw {yaw:+.1f}  pitch {pitch:+.1f} graus",
                          C_GEOM))
        elif sample.direction_cam is not None:
            hit = hit_screen(mid, sample.direction_cam)
            if hit is not None:
                pj.line(img, mid, hit, C_GEOM, 1)
                p = pj(hit)
                if p is not None:
                    cv2.drawMarker(img, p, C_GEOM, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
            else:
                pj.line(img, mid, mid + 80.0 * sample.direction_cam, C_GEOM, 1)
            yaw, pitch = sample.yaw_deg, sample.pitch_deg
            lines.append((f"olhar geometrico (sem calibracao): yaw {yaw:+.1f}  pitch {pitch:+.1f}",
                          C_GEOM))
        # Olhar calibrado: do meio dos olhos até o cursor.
        if cursor is not None:
            c = to_cm(cursor)
            pj.line(img, mid, c, C_CURSOR, 2)
            pj.circle_3d(img, c, 1.2, C_CURSOR, 2)
        # Direção do rosto corrigida pelo viés da calibração.
        if face_bias is not None and sample.raw is not None:
            P, _ = head_screen_point(sample.raw, np.asarray(face_bias))
            hit = np.array([P[0, 0], P[0, 1], 0.0])
            pj.line(img, E, hit, C_FACE_DIR, 1)
            pf = pj(hit)
            if pf is not None:
                cv2.drawMarker(img, pf, C_FACE_DIR, cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
        hy, hp, _ = sample.head.euler_deg
        if head_ref is not None:
            lines.append((f"cabeca (rel. calibracao): yaw {hy - head_ref[0]:+.1f}  "
                          f"pitch {hp - head_ref[1]:+.1f} graus   distancia {E[2]:.0f} cm",
                          (220, 220, 220)))
        else:
            lines.append((f"cabeca: yaw {hy:+.1f}  pitch {hp:+.1f} graus   distancia {E[2]:.0f} cm",
                          (220, 220, 220)))

    legend = [("amarelo: cursor (calibrado)", C_CURSOR),
              ("verde: direcao do rosto (corrigida na calibracao)", C_FACE_DIR),
              ("magenta: olhar" + (" calibrado (mesmo modelo do cursor)" if calibrated_gaze is not None
                                              else " (sem calibracao)"), C_GEOM),
              ] + ([] if calibrated_gaze is not None else [("azul claro: raio de cada olho", C_EYE_RAY)])
    for i, (s, color) in enumerate(lines + legend):
        _label(img, s, (12, 22 + 20 * i), color)
    _label(img, "arrastar: girar | roda: zoom | botao direito: vista inicial",
           (12, H - 12), (180, 180, 180), 0.42)
    return img
