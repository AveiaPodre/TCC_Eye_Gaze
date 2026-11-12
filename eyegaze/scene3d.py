"""
Vista 3D da cena: o pipeline do olhar, etapa por etapa.

Numa câmera virtual que orbita a cena, desenha:

* o rosto (landmarks) e os eixos da cabeça (etapa 1);
* para cada olho, o CENTRO DO GLOBO (círculo) e a PUPILA (ponto) (etapa 2);
* o FEIXE de cada olho, do centro do globo passando pela pupila, até o
  plano da tela (azul claro) e a INTERSECÇÃO dos dois feixes (etapa 3);
* o VETOR DO OLHAR, do ponto entre os olhos até a intersecção, prolongado
  até o MONITOR VIRTUAL (magenta, com o ponto atingido; etapa 4);
* o CURSOR no monitor real, depois da correção da calibração e do
  estabilizador (amarelo; etapa 5).

Referencial (cm): o da câmera (geometry.py). A tela é o plano z = 0, com a
webcam centralizada logo acima da borda superior. Como a imagem da câmera
NÃO é espelhada, a borda esquerda da tela (para o usuário) fica em +x.

Controles (com a janela 3D em foco): arrastar com o botão esquerdo gira,
a roda do mouse aproxima/afasta e o botão direito volta à vista inicial.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .face_tracker import FaceObservation
from .gaze import RAW_FACE, GazeSample, direction_to_angles
from .geometry import VirtualScreen, px3d_to_cm

FONT = cv2.FONT_HERSHEY_SIMPLEX

C_FACE = (150, 150, 150)
C_SCREEN = (0, 200, 255)
C_EYE = (255, 255, 25)
C_BEAM = (255, 200, 60)
C_VERGENCE = (255, 255, 255)
C_VIRTUAL = (255, 60, 255)
C_CURSOR = (0, 255, 255)
C_RAW = (170, 170, 170)
C_TARGET = (110, 110, 110)
C_CAMERA = (80, 230, 80)


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


def landmarks_cm(obs: FaceObservation, depth_cm: float) -> np.ndarray:
    """Landmarks (n, 3) em cm pela câmera pinhole, com o rosto na profundidade dada."""
    return px3d_to_cm(obs.points, (obs.image_width, obs.image_height), depth_cm)


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
        P = np.asarray(P, dtype=np.float64)
        if not np.all(np.isfinite(P)):
            return None
        pc = self.V @ (P - self.cam)
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

    def marker(self, img, P, color, kind=cv2.MARKER_TILTED_CROSS, size=14):
        p = self(P)
        if p is not None:
            cv2.drawMarker(img, p, color, kind, size, 2, cv2.LINE_AA)


def _label(img, s, org, color=(230, 230, 230), scale=0.45):
    cv2.putText(img, s, org, FONT, scale, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


# -------------------------------------------------------------- desenho
def render_scene(view: OrbitView, screen: VirtualScreen, sample: GazeSample | None,
                 obs: FaceObservation | None, virtual_point=None, cursor=None,
                 raw_point=None, trail=(), targets=(), assumed_width: bool = True,
                 size: tuple[int, int] = (760, 560), head_ref=None,
                 calibrated: bool = False) -> np.ndarray:
    """Imagem da cena 3D vista pela câmera virtual ``view``.

    virtual_point: pixel do MONITOR VIRTUAL atingido pelo vetor do olhar
    (etapa 4); cursor / raw_point: pixel do monitor real depois da correção,
    com e sem o estabilizador (etapa 5); head_ref: pose da cabeça na
    calibração (yaw, pitch), para o texto."""
    W, H = size
    img = np.full((H, W, 3), (28, 24, 22), np.uint8)
    to_cm = lambda p: screen.px_to_cm(p)[0]  # noqa: E731
    sw, sh = screen.size_px
    corners = [to_cm(p) for p in ((0, 0), (sw, 0), (sw, sh), (0, sh))]
    screen_center = to_cm((sw / 2, sh / 2))

    gaze = sample.gaze if sample is not None else None
    E = sample.raw[RAW_FACE] if sample is not None else None
    head_c = E if E is not None else np.array([0.0, 8.0, 60.0])
    pj = _Projector(view, 0.5 * (head_c + screen_center), size)

    # --- chão de referência (grade no plano y da base da tela) ----------
    floor_y = corners[2][1] + 4.0
    for k in np.linspace(-40, 40, 9):
        pj.line(img, (k, floor_y, -5), (k, floor_y, 90), (50, 46, 44))
    for k in np.linspace(-5, 90, 8):
        pj.line(img, (-40, floor_y, k), (40, floor_y, k), (50, 46, 44))

    # --- monitor (virtual = real na geometria suposta) ------------------
    pj.poly(img, corners, C_SCREEN, 2)
    pj.line(img, corners[0], corners[2], (0, 110, 150))
    pj.line(img, corners[1], corners[3], (0, 110, 150))
    for t in targets:
        p = pj(to_cm(t))
        if p is not None:
            cv2.circle(img, p, 2, C_TARGET, -1, cv2.LINE_AA)
    tl = pj(corners[0])
    if tl is not None:
        txt = f"monitor {screen.width_cm:.1f} cm" + (" (largura suposta)" if assumed_width else "")
        _label(img, txt, (tl[0] + 4, tl[1] - 6), C_SCREEN)

    # --- webcam -------------------------------------------------------
    cam_pts = [1.2 * np.array(v) for v in ((-1, -0.7, 1.6), (1, -0.7, 1.6), (1, 0.7, 1.6),
                                           (-1, 0.7, 1.6))]
    pj.poly(img, cam_pts, C_CAMERA, 1)
    for v in cam_pts:
        pj.line(img, (0, 0, 0), v, C_CAMERA)

    # --- trilha e cursor no monitor real --------------------------------
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
        for P in landmarks_cm(obs, float(E[2]))[:468:2]:
            p = pj(P)
            if p is not None:
                cv2.circle(img, p, 1, C_FACE, -1)
        # Eixos da cabeça a partir do ponto entre os olhos.
        anchors = np.array([e.anchor_cm for e in sample.eyes])
        o = anchors.mean(axis=0)
        R = sample.head.R
        for vec, color in ((R[:, 0], (0, 0, 255)), (R[:, 1], (0, 200, 0)),
                           (-R[:, 2], (255, 80, 0))):
            pj.line(img, o, o + 4.0 * vec, color, 2)

        if gaze is not None:
            # Etapas 2-3: globo, pupila e feixe de cada olho.
            for k in range(2):
                c, p, d = gaze.centers[k], gaze.pupils[k], gaze.directions[k]
                color = C_EYE if gaze.open[k] else C_RAW
                pj.circle_3d(img, c, float(np.linalg.norm(p - c)), color)
                q = pj(p)
                if q is not None:
                    cv2.circle(img, q, 2, C_BEAM, -1, cv2.LINE_AA)
                if gaze.open[k]:
                    hit = screen.hit(c, d)[0]
                    end = hit if np.all(np.isfinite(hit)) else c + 80.0 * d
                    pj.line(img, c, end, C_BEAM, 1)
            if gaze.vergence is not None:
                pj.line(img, gaze.origin, gaze.vergence, C_VERGENCE, 1)
                pj.marker(img, gaze.vergence, C_VERGENCE, cv2.MARKER_DIAMOND, 10)
                lines.append((f"interseccao dos feixes: z = {gaze.vergence[2]:+.1f} cm "
                              f"(distancia entre feixes {gaze.vergence_gap_cm:.2f} cm)",
                              C_VERGENCE))
            else:
                lines.append(("feixes nao convergem a frente: bissetriz", C_VERGENCE))
            # Etapa 4: vetor do olhar até o monitor virtual.
            hit = screen.hit(gaze.origin, gaze.direction)[0]
            if np.all(np.isfinite(hit)):
                pj.line(img, gaze.origin, hit, C_VIRTUAL, 2)
                pj.marker(img, hit, C_VIRTUAL)
            else:
                pj.line(img, gaze.origin, gaze.origin + 80.0 * gaze.direction, C_VIRTUAL, 2)
            yaw, pitch = direction_to_angles(gaze.direction)
            lines.append((f"vetor do olhar: yaw {yaw:+.1f}  pitch {pitch:+.1f} graus", C_VIRTUAL))
        if virtual_point is not None and np.all(np.isfinite(virtual_point)):
            pj.circle_3d(img, to_cm(virtual_point), 0.8, C_VIRTUAL, 1)
        # Etapa 5: cursor no monitor real (correção + estabilizador).
        if cursor is not None and gaze is not None:
            c = to_cm(cursor)
            pj.line(img, gaze.origin, c, C_CURSOR, 1)
            pj.circle_3d(img, c, 1.2, C_CURSOR, 2)
        hy, hp, _ = sample.head.euler_deg
        if head_ref is not None:
            hy, hp = hy - head_ref[0], hp - head_ref[1]
        lines.append((f"cabeca{' (rel. calibracao)' if head_ref is not None else ''}: "
                      f"yaw {hy:+.1f}  pitch {hp:+.1f} graus   distancia {E[2]:.0f} cm",
                      (220, 220, 220)))

    legend = [("circulo/ponto: centro do globo / pupila", C_EYE),
              ("azul claro: feixe centro -> pupila de cada olho", C_BEAM),
              ("branco: interseccao dos feixes", C_VERGENCE),
              ("magenta: vetor do olhar -> monitor virtual", C_VIRTUAL),
              ("amarelo: cursor no monitor real" + ("" if calibrated else " (sem calibracao)"),
               C_CURSOR)]
    for i, (s, color) in enumerate(lines + legend):
        _label(img, s, (12, 22 + 20 * i), color)
    _label(img, "arrastar: girar | roda: zoom | botao direito: vista inicial",
           (12, H - 12), (180, 180, 180), 0.42)
    return img
