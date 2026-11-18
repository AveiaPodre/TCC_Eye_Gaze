"""
Índices dos landmarks do FaceLandmarker (malha de 478 pontos).

Convenção do MediaPipe: "LEFT"/"RIGHT" referem-se ao lado do PRÓPRIO
USUÁRIO. Numa imagem de webcam não espelhada, o olho esquerdo do usuário
aparece do lado DIREITO da imagem.

Cada olho é descrito com os cantos, pálpebras e íris do MESMO olho
(conferido com mediapipe.tasks.vision.FaceLandmarksConnections:
FACE_LANDMARKS_LEFT_EYE contém 263/362/386/374 e FACE_LANDMARKS_LEFT_IRIS
contém 474-477).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EyeIndices:
    name: str
    outer_corner: int
    inner_corner: int
    upper_lid: int
    lower_lid: int
    iris_center: int
    iris_ring: tuple[int, int, int, int]
    # Vários pontos de cada pálpebra (centro e vizinhos): a média é bem
    # menos ruidosa que um ponto só. Usados nas medidas verticais do olhar.
    upper_lid_ring: tuple[int, ...] = ()
    lower_lid_ring: tuple[int, ...] = ()
    # Contorno completo da abertura do olho, em ordem (polígono). Usado no
    # refinamento da íris na imagem (iris_refine.py).
    contour: tuple[int, ...] = ()


RIGHT_EYE = EyeIndices(
    name="direito",
    outer_corner=33,
    inner_corner=133,
    upper_lid=159,
    lower_lid=145,
    iris_center=468,
    iris_ring=(469, 470, 471, 472),
    upper_lid_ring=(158, 159, 160),
    lower_lid_ring=(144, 145, 153),
    contour=(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
)

LEFT_EYE = EyeIndices(
    name="esquerdo",
    outer_corner=263,
    inner_corner=362,
    upper_lid=386,
    lower_lid=374,
    iris_center=473,
    iris_ring=(474, 475, 476, 477),
    upper_lid_ring=(385, 386, 387),
    lower_lid_ring=(373, 374, 380),
    contour=(263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466),
)

# --- Pontos usados para o referencial da cabeça -------------------------
# Eixo horizontal: pares (lado direito do usuário, lado esquerdo do usuário).
# Cantos dos olhos e têmporas: são pontos presos ao crânio e quase não se
# mexem com expressões faciais.
HEAD_X_PAIRS: tuple[tuple[int, int], ...] = (
    (33, 263),    # cantos externos dos olhos
    (133, 362),   # cantos internos dos olhos
    (127, 356),   # têmporas
)
# Eixo vertical: linha média da face, da testa/glabela até a base do nariz.
# Evitamos o queixo (152) porque ele se move ao abrir a boca.
HEAD_Y_TOP: tuple[int, ...] = (151, 9, 8)
HEAD_Y_BOTTOM: tuple[int, ...] = (2, 164)
# Ponto usado como origem para desenhar os eixos da cabeça.
NOSE_BRIDGE = 168
# Ponte do nariz (entre os olhos): pontos rígidos que NÃO se movem com as
# pálpebras, na altura e quase na profundidade dos olhos. Âncora vertical do
# globo ocular (eyeball.EyeballModel, âncora "misto"). Os cantos do olho
# fazem parte do contorno das pálpebras e sobem/descem com o olhar, o que
# "come" o sinal vertical da pupila. O projeto de referência (Webcam3DTracker)
# ancora o globo na região do nariz inteira (inclui a ponta, ~4 cm abaixo e à
# frente dos olhos); nas sessões gravadas, esse braço de alavanca amplificava
# os erros da pose da cabeça (ver docs/metodologia.md, seção 5.6).
RIGID_NOSE: tuple[int, ...] = (168, 6, 197, 8, 9)

NUM_LANDMARKS = 478
