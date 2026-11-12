"""
Desenho das visualizações em tempo real (cada uma pode ser ligada/desligada).

As camadas "geométricas" (landmarks, olhar, pose) são desenhadas sobre o
frame ORIGINAL, antes de um eventual espelhamento da exibição; textos,
minimapa e ajuda são desenhados depois, para não ficarem espelhados.
"""

from __future__ import annotations

import cv2
import numpy as np

from .face_tracker import FaceObservation
from .gaze import GazeSample
from .landmarks import LEFT_EYE, RIGHT_EYE

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Contornos dos olhos (mesmos índices de FaceLandmarksConnections).
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                    398, 384, 385, 386, 387, 388, 466]


def _p(v) -> tuple[int, int]:
    return int(round(v[0])), int(round(v[1]))


def text(img, s, org, scale=0.55, color=(240, 240, 240), thick=1):
    """Texto com fundo escuro (legível sobre qualquer imagem)."""
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), (15, 15, 15), -1)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


# ------------------------------------------------------------- camadas 3D
def draw_landmarks(img, obs: FaceObservation):
    pts = obs.points
    for x, y, _ in pts[:468]:
        cv2.circle(img, (int(x), int(y)), 1, (170, 170, 170), -1)
    for contour in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR):
        poly = np.array([_p(pts[i]) for i in contour], dtype=np.int32)
        cv2.polylines(img, [poly], True, (0, 220, 255), 1, cv2.LINE_AA)
    for eye in (RIGHT_EYE, LEFT_EYE):
        ring = np.array([_p(pts[i]) for i in eye.iris_ring], dtype=np.int32)
        cv2.polylines(img, [ring], True, (255, 120, 0), 1, cv2.LINE_AA)
        cv2.circle(img, _p(pts[eye.iris_center]), 2, (255, 120, 0), -1)
        for idx in (eye.outer_corner, eye.inner_corner):
            cv2.circle(img, _p(pts[idx]), 3, (0, 255, 0), -1)


def draw_gaze(img, sample: GazeSample):
    ray_len = 2.5 * sample.head.scale_px
    for eye in sample.eyes:
        color = (255, 255, 0) if eye.open else (120, 120, 120)
        cv2.circle(img, _p(eye.eyeball_center_px), max(2, int(eye.eyeball_radius_px)),
                   color, 1, cv2.LINE_AA)
        cv2.circle(img, _p(eye.iris_px), 3, color, -1, cv2.LINE_AA)
        if eye.open:
            end = eye.iris_px + eye.direction_cam * ray_len
            cv2.line(img, _p(eye.iris_px), _p(end), (0, 255, 120), 1, cv2.LINE_AA)
    if sample.direction_cam is not None:
        origin = np.mean([e.iris_px for e in sample.eyes], axis=0)
        end = origin + sample.direction_cam * ray_len * 1.4
        cv2.arrowedLine(img, _p(origin), _p(end), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.15)


def draw_head_pose(img, sample: GazeSample):
    head = sample.head
    o = head.origin
    L = 0.6 * head.scale_px
    axes = [(head.R[:, 0], (0, 0, 255), "x"),
            (head.R[:, 1], (0, 200, 0), "y"),
            (-head.R[:, 2], (255, 80, 0), "frente")]
    for vec, color, _ in axes:
        cv2.arrowedLine(img, _p(o), _p(o + vec * L), color, 2, cv2.LINE_AA, tipLength=0.2)


# ------------------------------------------------------------------- HUD
def draw_minimap(img, screen_size, point, raw_point, trail, targets, dwell_progress,
                 calibrated: bool):
    sw, sh = screen_size
    h, w = img.shape[:2]
    mw = min(260, w // 3)
    mh = int(mw * sh / sw)
    x0, y0 = w - mw - 12, h - mh - 12

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + mw, y0 + mh), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + mh), (200, 200, 200), 1)

    def to_map(p):
        return (int(x0 + np.clip(p[0], 0, sw) / sw * mw),
                int(y0 + np.clip(p[1], 0, sh) / sh * mh))

    for t in targets:
        cv2.circle(img, to_map(t), 2, (110, 110, 110), -1)
    for a, b in zip(trail, trail[1:]):
        cv2.line(img, to_map(a), to_map(b), (0, 140, 200), 1, cv2.LINE_AA)
    if raw_point is not None:
        cv2.circle(img, to_map(raw_point), 2, (160, 160, 160), -1)
    if point is not None:
        c = to_map(point)
        cv2.circle(img, c, 6, (0, 255, 255) if calibrated else (0, 160, 255), -1, cv2.LINE_AA)
        if dwell_progress > 0:
            cv2.ellipse(img, c, (11, 11), -90, 0, 360 * dwell_progress, (0, 255, 0), 2,
                        cv2.LINE_AA)
    label = "tela (estimativa)" if calibrated else "tela (NAO calibrado)"
    text(img, label, (x0 + 4, y0 - 6), 0.45)


def draw_status(img, lines: list[tuple[str, tuple[int, int, int]]]):
    y = 22
    for s, color in lines:
        text(img, s, (10, y), 0.55, color)
        y += 22


HELP_LINES = [
    "ESPACO  calibrar (automatica)",
    "C  calibrar (manual: olhe + ESPACO)",
    "T  validar (mede erro)",
    "R  recentralizar (1 ponto)",
    "M  liga/desliga mouse",
    "D  liga/desliga dwell-click",
    "L  landmarks   G  olhar",
    "P  pose cabeca V  minimapa",
    "3  vista 3D    F  espelhar",
    "H  ajuda",
    "Q/ESC  sair",
]


def draw_help(img):
    h, w = img.shape[:2]
    bw, bh = 280, 22 * len(HELP_LINES) + 14
    x0, y0 = 10, h - bh - 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, s in enumerate(HELP_LINES):
        text(img, s, (x0 + 10, y0 + 24 + 22 * i), 0.5)


def draw_toast(img, message: str):
    h, w = img.shape[:2]
    (tw, _), _ = cv2.getTextSize(message, FONT, 0.8, 2)
    text(img, message, ((w - tw) // 2, int(h * 0.12)), 0.8, (0, 255, 255), 2)
