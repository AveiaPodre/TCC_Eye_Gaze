"""
Detecção de landmarks faciais com a API atual do MediaPipe (Tasks API).

O script de referência (Webcam3DTracker) usa ``mp.solutions.face_mesh``,
que foi removido das versões recentes do pacote ``mediapipe``. Aqui usamos
``mediapipe.tasks.vision.FaceLandmarker`` em modo VIDEO: a chamada é
síncrona (o resultado corresponde exatamente ao frame enviado, sem atraso
de callback) e o modelo reaproveita o rastreamento entre frames.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MODEL_URL, TrackerConfig
from .iris_refine import refine_irises
from .landmarks import NUM_LANDMARKS


@dataclass
class FaceObservation:
    """Landmarks de um rosto em um frame.

    ``points`` tem forma (478, 3) em "pixels 3D": x e y em pixels da imagem
    e z na mesma escala de x (z_px = z_normalizado * largura_da_imagem).
    z menor = mais perto da câmera. Essa é a mesma convenção do script de
    referência (x*w, y*h, z*w).
    """

    points: np.ndarray
    image_width: int
    image_height: int
    timestamp_ms: int
    # Matriz 4x4 do modelo canônico -> rosto detectado (espaço métrico do
    # MediaPipe). Usada para a pose da cabeça; pode ser None.
    transform: np.ndarray | None = None
    # (2, 2): centro da íris refinado na imagem (olho direito, esquerdo),
    # em pixels; NaN onde falhou. None = refinamento desligado.
    iris_refined: np.ndarray | None = None


def ensure_model(model_path: Path) -> Path:
    """Garante que o arquivo .task do modelo existe (baixa se necessário)."""
    model_path = Path(model_path)
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Modelo] Baixando face_landmarker.task para {model_path} ...")
    try:
        tmp = model_path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(model_path)
    except Exception as exc:  # noqa: BLE001 - mensagem amigável ao usuário
        raise RuntimeError(
            "Não foi possível baixar o modelo automaticamente.\n"
            f"Baixe manualmente de:\n  {MODEL_URL}\n"
            f"e salve em:\n  {model_path}"
        ) from exc
    print("[Modelo] Download concluído.")
    return model_path


def landmarks_to_points(landmarks, width: int, height: int) -> np.ndarray:
    """Converte a lista de NormalizedLandmark em array (N, 3) de pixels 3D."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    pts[:, 0] *= width
    pts[:, 1] *= height
    pts[:, 2] *= width
    return pts


class FaceTracker:
    def __init__(self, cfg: TrackerConfig):
        # Import local: permite usar o resto do pacote (e os testes) sem
        # carregar o MediaPipe.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = ensure_model(cfg.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=cfg.min_face_detection_confidence,
            min_face_presence_confidence=cfg.min_face_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1
        self._refine = cfg.refine_iris

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceObservation | None:
        # O modo VIDEO exige timestamps estritamente crescentes.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        # cvtColor devolve um array contíguo, exigido por mp.Image.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        if len(landmarks) < NUM_LANDMARKS:
            # Modelo sem os pontos da íris: não dá para estimar o olhar.
            return None

        transform = None
        if result.facial_transformation_matrixes:
            transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)

        h, w = frame_bgr.shape[:2]
        points = landmarks_to_points(landmarks, w, h)
        return FaceObservation(
            points=points,
            image_width=w,
            image_height=h,
            timestamp_ms=timestamp_ms,
            transform=transform,
            iris_refined=refine_irises(frame_bgr, points) if self._refine else None,
        )

    def close(self):
        self._landmarker.close()
