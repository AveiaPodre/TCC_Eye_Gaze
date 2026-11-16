"""
Referencial (pose) da cabeça a partir dos landmarks.

O script de referência estima a orientação da cabeça com PCA sobre pontos do
nariz. PCA não tem sinal definido nos autovetores (daí o "hack" de
estabilização por matriz de referência) e a ordem dos eixos troca quando
duas variâncias ficam parecidas. Aqui os eixos são definidos
geometricamente, sempre com o mesmo significado:

    x  -> do lado direito para o lado esquerdo do usuário
          (para a direita na imagem não espelhada)
    y  -> da testa para a base do nariz (para baixo na imagem)
    z  =  x × y -> para trás da cabeça (afastando-se da câmera)

A face "olha" para -z. A matriz R = [x y z] (colunas) leva coordenadas do
referencial da cabeça para o referencial da câmera em pixels 3D.

Duas fontes de rotação são suportadas:

* ``matrix`` (padrão): a matriz de transformação facial devolvida pelo
  próprio FaceLandmarker (alinhamento do modelo canônico de rosto a todos os
  468 pontos). É a mais estável e o modelo canônico é "de pé", então o eixo
  -z aponta exatamente para a frente do rosto.
* ``landmarks``: eixos construídos a partir de poucos pontos (cantos dos
  olhos, têmporas, linha média). Não depende de nada além dos landmarks,
  mas o eixo y fica levemente inclinado (testa e base do nariz não estão na
  mesma profundidade). Essa inclinação é constante e a calibração a
  absorve; é usada como reserva e como verificação de consistência.

O espaço "métrico" do MediaPipe tem y para cima e z saindo da tela (câmera
olhando para -z). O espaço de pixels 3D tem y para baixo e z entrando na
cena. A conversão é R_pix = C R_met C com C = diag(1, -1, -1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmarks import HEAD_X_PAIRS, HEAD_Y_BOTTOM, HEAD_Y_TOP, NOSE_BRIDGE


@dataclass
class HeadPose:
    R: np.ndarray          # (3, 3) rotação cabeça -> câmera
    origin: np.ndarray     # (3,) ponto entre os olhos (pixels 3D)
    scale_px: float        # distância média entre cantos externos dos olhos (px)

    @property
    def euler_deg(self) -> tuple[float, float, float]:
        """(yaw, pitch, roll) em graus, só para exibição.

        yaw > 0: cabeça virada para a direita do usuário;
        pitch > 0: cabeça inclinada para cima;
        roll > 0: cabeça inclinada no sentido horário na imagem.
        """
        fwd = -self.R[:, 2]  # direção para onde a face aponta
        yaw = np.degrees(np.arctan2(-fwd[0], -fwd[2]))
        pitch = np.degrees(np.arctan2(-fwd[1], np.hypot(fwd[0], fwd[2])))
        x_axis = self.R[:, 0]
        roll = np.degrees(np.arctan2(x_axis[1], x_axis[0]))
        return float(yaw), float(pitch), float(roll)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


_C = np.diag([1.0, -1.0, -1.0])


def rotation_from_transform(transform: np.ndarray) -> np.ndarray:
    """Rotação (pixels 3D) a partir da matriz 4x4 do FaceLandmarker."""
    M = np.asarray(transform, dtype=np.float64)[:3, :3]
    # Remove eventual escala/cisalhamento: rotação mais próxima (SVD).
    U, _, Vt = np.linalg.svd(M)
    R_met = U @ Vt
    if np.linalg.det(R_met) < 0:
        U[:, -1] *= -1
        R_met = U @ Vt
    return _C @ R_met @ _C


def _landmark_rotation(points: np.ndarray) -> np.ndarray:
    x_vec = np.zeros(3)
    for right_idx, left_idx in HEAD_X_PAIRS:
        x_vec += points[left_idx] - points[right_idx]
    x_axis = _normalize(x_vec)

    top = points[list(HEAD_Y_TOP)].mean(axis=0)
    bottom = points[list(HEAD_Y_BOTTOM)].mean(axis=0)
    y_raw = bottom - top
    # Gram-Schmidt: remove a componente de y paralela a x.
    y_axis = _normalize(y_raw - np.dot(y_raw, x_axis) * x_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def rotations_agree(R_a: np.ndarray, R_b: np.ndarray, max_deg: float = 35.0) -> bool:
    """Verifica se duas estimativas de rotação concordam nos eixos x e z.

    O eixo y é ignorado de propósito: a versão por landmarks tem uma
    inclinação fixa nele (ver docstring do módulo).
    """
    cos_max = np.cos(np.radians(max_deg))
    return (np.dot(R_a[:, 0], R_b[:, 0]) > cos_max and
            np.dot(R_a[:, 2], R_b[:, 2]) > np.cos(np.radians(max_deg + 15)))


def estimate_head_pose(points: np.ndarray, transform: np.ndarray | None = None,
                       source: str = "matrix") -> tuple[HeadPose, str]:
    """Devolve (pose, fonte_usada). Usa a matriz do modelo se disponível e
    coerente com os landmarks; senão, os eixos geométricos."""
    R_lm = _landmark_rotation(points)
    R, used = R_lm, "landmarks"
    if source == "matrix" and transform is not None:
        R_mx = rotation_from_transform(transform)
        if rotations_agree(R_mx, R_lm):
            R, used = R_mx, "matrix"
        else:
            used = "landmarks (matriz incoerente)"

    outer_r, outer_l = HEAD_X_PAIRS[0]
    scale = float(np.linalg.norm(points[outer_l] - points[outer_r]))
    return HeadPose(R=R, origin=points[NOSE_BRIDGE].copy(), scale_px=scale), used
