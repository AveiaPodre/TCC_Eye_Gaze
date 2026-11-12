"""
Modelo de calibração: medidas do rosto -> pixel da tela.

Versão 3 (regressão com geometria da cabeça). A versão anterior otimizava
parâmetros do olho (ganho, offset) por Nelder-Mead dentro de um modelo
geométrico de raio do olhar. Com dados reais isso falhou: o sinal vertical
da íris é ~5x menor que o previsto pela esfera e o pitch tem um viés de
~30 graus, então o otimizador ia para os limites e nem os próprios alvos da
calibração eram ajustados (RMSE de ~300 px).

Agora a geometria entra só onde ela é confiável, a CABEÇA:

    P_cabeça = E_xy + E_z * tan(direção da face)     (cm, plano da câmera)

é o ponto para onde a face aponta no plano da tela. Isso já compensa
rotação e translação da cabeça. O olho entra por medidas brutas (íris
horizontal u, íris vertical v, altura da pálpebra superior lid_up e abertura do
olho), multiplicadas pela distância E_z (o mesmo ângulo de olho cobre mais
centímetros quando se está mais longe) e por sec²(ângulo da cabeça), a
derivada de tan(cabeça + olho) em relação ao olho:

    x = c + a·P_cabeça + Σ_k b_k · E_z · sec²(yaw_cabeça)   · φ_k(olho)
    y = c + a·P_cabeça + Σ_k b_k · E_z · sec²(pitch_cabeça) · φ_k(olho)

φ = [u, v, lid_up, abertura] (+ termos quadráticos no grau 2). O modelo é
LINEAR nos coeficientes: mínimos quadrados com ridge, sem otimização
iterativa e sem mínimos locais. A força do ridge é escolhida por validação
cruzada deixando um alvo de fora (leave-one-target-out), e o erro dessa
validação é reportado: ele estima o erro em pontos não vistos, ao contrário
do erro de ajuste, que esconde sobreajuste.

A etapa em que a pessoa mantém o olhar no centro e gira a cabeça é o que
determina o coeficiente de P_cabeça. Sem ela, o modelo continua funcionando
com a cabeça parada.

Versão 4 (robustez com dados reais). A análise das sessões gravadas mostrou:

* Na etapa da cabeça as pessoas giravam até ±30° (yaw) e 40° (pitch). Nesses
  ângulos os landmarks da íris e das pálpebras deixam de ser confiáveis e o
  modelo linear não consegue ajustá-los (erro mediano ~210 px na própria
  etapa). Como cada alvo e a etapa inteira pesam o mesmo, esses pontos
  distorciam os coeficientes justamente na faixa pequena (2-8°) que ocorre
  no uso. Agora só entram amostras dentro de uma faixa útil em torno da pose
  média da grade (``head_max_deg``).
* Escolher o modelo só pelo LOO dos alvos ignora o problema principal, a
  invariância à cabeça: na grade a cabeça fica quase parada. A etapa da
  cabeça é dividida em 4 direções (direita, esquerda, cima, baixo) e cada
  uma é deixada de fora por vez (validação cruzada da cabeça). A escolha de
  ridge, grau e fonte da íris minimiza a média dos dois erros.

Versão 5 (medida vertical coerente + modelo físico). A medida ``lid_v``
(íris entre as pálpebras) foi trocada por ``lid_up``, a altura da pálpebra
superior com o z do MediaPipe (ver gaze.py): é a única medida vertical que
responde do mesmo jeito à rotação do olho na grade e à inclinação da
cabeça. Além da regressão livre acima, há um candidato FÍSICO:

    x = c_x - a·P_x + b_x · E_z · sec²(yaw_cabeça)   · u
    y = c_y + a·P_y + b_y · E_z · sec²(pitch_cabeça) · lid_up

com o MESMO a (px/cm) nos dois eixos, porque os pixels da tela são
quadrados (o sinal de x é negativo porque a imagem não é espelhada: a
direita do usuário é -x na câmera). São 3 coeficientes em vez de ~20. Ele
é aprendido com o giro horizontal da etapa da cabeça, que é coerente, e
vale também para o eixo vertical. A validação cruzada escolhe entre os
candidatos em cada sessão.

Versão 6 (ganho da cabeça fixado pela geometria). Se a cabeça gira θ com o
olho parado dentro dela, o olhar gira θ: o cursor tem de andar
a·ΔP_cabeça, com a = px/cm da tela, que é conhecido quando se sabe a
largura física da tela (detectada no Windows pelo EDID ou dada por
--screen-width-cm). Os modelos acima APRENDIAM a, e com uma etapa da
cabeça curta e medidas imperfeitas aprendiam ~1/3 a 2/3 do valor físico
(12-27 px/grau na vertical, contra ~38): ao inclinar a cabeça para olhar
para baixo o cursor não descia o suficiente, como se a pessoa estivesse
sempre de frente para a câmera. Com a conhecido (``head_px_per_cm``), o
termo da cabeça vira um deslocamento fixo e só o olho é ajustado, no
modelo COMPACTO (x: u; y: lid_up, lid_up²; o termo quadrático cobre a
saturação da pálpebra ao olhar para baixo). Com o ganho fixo, só entram
medidas do olho COERENTES com a cabeça (que respondem do mesmo jeito à
rotação do olho na grade e na etapa da cabeça; ver gaze.py): a íris
vertical v, a abertura e a íris refinada na vertical respondem ~0 quando a
cabeça gira com o olhar fixo, e com o ganho fixo elas deslocariam o cursor
nessa situação.

O ganho geométrico é usado só no eixo VERTICAL (``y_model``). Nas sessões
gravadas ele corrigiu o alcance vertical e o erro vertical com a cabeça em
poses novas (203/200/371 px -> 102/148/349 px), mas no horizontal piorava
o erro em pontos novos (até 82 -> 140 px): a regressão livre usa mais
medidas do olho (inclusive a íris refinada, que às vezes tem bem mais
sinal horizontal) e já funcionava bem nesse eixo. Então x vem do melhor
modelo aprendido (como antes) e y do compacto com a cabeça pela geometria;
a validação cruzada é refeita para a combinação.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .face_tracker import FaceObservation

FEATURE_VERSION = 6  # incremente se o formato mudar (invalida calibrações antigas)

RAW_SIZE = 21        # R (9) + E (3) + (u, v, lid_up, abertura) x 2 fontes da íris + boca
MOUTH_COL = 20       # abertura da boca (head_pose.mouth_ratio)
# Colunas das medidas do olho para cada fonte da posição da íris.
EYE_COLS = {"mediapipe": slice(12, 16), "refinada": slice(16, 20)}
ALPHAS = (1e-4, 1e-3, 1e-2, 0.1, 1.0)   # candidatos de ridge (features padronizadas)
# Desvio-padrão mínimo das colunas na padronização. Sem isso, uma coluna
# quase constante (ex.: cabeça parada durante a calibração) teria o ruído
# amplificado e o coeficiente "explodiria" quando a cabeça se mexe.
HEAD_STD_FLOOR_CM = 1.0
EYE_STD_FLOOR = 1e-3

# Grupos especiais na validação cruzada (``groups`` em CalibrationModel.fit):
# >= 0 é o índice do alvo; ALWAYS_TRAIN entra sempre no treino; as direções
# da etapa da cabeça são HEAD_GROUP_BASE - k (k = 0..3, ver head_direction).
ALWAYS_TRAIN = -1
HEAD_GROUP_BASE = -2
HEAD_DIRECTIONS = ("direita", "esquerda", "cima", "baixo")
# Distância da webcam à borda superior da tela (cm): webcam centralizada logo
# acima da tela (notebook ou webcam presa no monitor). Usada para localizar
# o centro da tela no referencial da câmera (face_bias) e na vista 3D.
SCREEN_GAP_CM = 1.0
# Modelos candidatos do eixo y com o ganho da cabeça pela geometria
# (fit_from_samples); a escolha é pelo erro só no eixo y. A regressão livre
# (todas as medidas do olho) com a cabeça fixa: na VALIDAÇÃO (alvos de outro
# momento da sessão) os modelos baseados só na pálpebra (compacto,
# geométrico) comprimiam o eixo y (escala 0,61-0,81) quando a postura mudava
# entre calibrar e usar; a livre manteve a escala (1,01) com erro de 115 px,
# contra 125 do geométrico. Ver docs/metodologia.md.
Y_MODEL_KINDS: tuple = (("livre", 2), ("livre", 1))
# A escolha do modelo principal (e do x) usa o erro na etapa da cabeça? Na
# etapa o alvo é sempre o CENTRO, então esse erro premia modelos que encolhem
# a previsão para o centro: com ele, em 5 das 15 calibrações o x ficava com
# escala 0,44-0,89 (o cursor não alcançava as laterais). Ver metodologia.
X_SCORE_HEAD: bool = False
# Eixo x com o ganho da cabeça pela geometria (como o y), quando a largura
# da tela é conhecida e há etapa da cabeça. Nas 15 calibrações: erro x em
# alvos novos 78 -> 75 px e, com a cabeça em poses novas, 184 -> 147 px
# (o modelo aprendido escolhido pela grade perdia a compensação da cabeça).
X_MODEL_GEOMETRIC: bool = True
HEAD_STILL_DEG = 2.0       # desvio abaixo disto não é "pose nova" (sempre no treino)
MIN_HEAD_FOLD = 5          # amostras mínimas para uma direção virar dobra da validação


# ------------------------------------------------------------------ geometria
def head_angles(raw: np.ndarray) -> np.ndarray:
    """(n, 2) yaw e pitch da cabeça em graus (mesma convenção de HeadPose.euler_deg)."""
    raw = np.atleast_2d(raw)
    fwd = -raw[:, :9].reshape(-1, 3, 3)[:, :, 2]
    yaw = np.degrees(np.arctan2(-fwd[:, 0], -fwd[:, 2]))
    pitch = np.degrees(np.arctan2(-fwd[:, 1], np.hypot(fwd[:, 0], fwd[:, 2])))
    return np.column_stack([yaw, pitch])


def head_direction(delta: np.ndarray) -> np.ndarray:
    """Direção dominante de cada desvio de pose (n, 2) -> índice em HEAD_DIRECTIONS."""
    delta = np.atleast_2d(delta)
    horiz = np.abs(delta[:, 0]) >= np.abs(delta[:, 1])
    return np.where(horiz, np.where(delta[:, 0] > 0, 0, 1),
                    np.where(delta[:, 1] > 0, 2, 3))


def head_screen_point(raw: np.ndarray, bias: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Ponto para onde a face aponta no plano z = 0 da câmera (cm) e as
    tangentes (tan yaw, tan pitch) da direção da face.

    bias: (2,) viés da direção da face em radianos (no espaço dos ângulos
    atan(tans)), subtraído antes de calcular o ponto. Ver ``face_bias``.
    raw: (n, 20). Devolve ((n, 2), (n, 2)).
    """
    raw = np.atleast_2d(raw)
    R = raw[:, :9].reshape(-1, 3, 3)
    E = raw[:, 9:12]
    fwd = -R[:, :, 2]                         # a face olha para -z da cabeça
    fz = np.minimum(fwd[:, 2], -1e-3)         # voltada para a câmera: fz < 0
    tans = fwd[:, :2] / -fz[:, None]
    if bias is not None:
        tans = np.tan(np.clip(np.arctan(tans) - bias, -1.4, 1.4))
    return E[:, :2] + E[:, 2:3] * tans, tans


def face_bias(grid_raw: np.ndarray, screen_center_cm: np.ndarray) -> np.ndarray:
    """Viés da direção da face medida pelo MediaPipe (radianos, (2,)).

    O eixo "para a frente" do modelo canônico do MediaPipe não coincide com
    a direção para onde a pessoa vira o rosto: nas sessões gravadas, com a
    pessoa olhando para a tela (abaixo da câmera), a face aparecia
    inclinada ~13° para CIMA, quando a geometria pede ~20° para baixo. Com o
    ganho da cabeça fixado pela geometria, esse viés distorce o termo
    E_z·tan(direção da face): por exemplo, aproximar-se da tela subia o
    cursor em vez de descê-lo.

    Supõe-se que, na postura da calibração (grade de alvos), o rosto aponta
    em média para o CENTRO da tela: o viés é a diferença entre a direção
    medida (mediana da grade) e a direção do rosto até o centro da tela.
    Assim a postura da calibração vira a referência "de frente para a tela".
    """
    _, tans = head_screen_point(grid_raw)
    E = np.median(np.atleast_2d(grid_raw)[:, 9:12], axis=0)
    d = np.asarray(screen_center_cm, dtype=float) - E
    target = d[:2] / max(-d[2], 1e-3)
    return np.median(np.arctan(tans), axis=0) - np.arctan(target)


def eye_terms(raw: np.ndarray, degree: int, ref: np.ndarray | None = None,
              source: str = "mediapipe") -> np.ndarray:
    """φ(olho): [u, v, lid_up, abertura] (+ u², u·lid_up, lid_up², u·v no grau 2).

    ref: valores de referência (média da calibração) subtraídos antes. As
    medidas não são zero com o olho "reto" (a abertura, por exemplo, é ~0.3);
    sem centralizar, multiplicar por E_z faria a distância deslocar o cursor
    mesmo com o olho parado.
    """
    raw = np.atleast_2d(raw)
    m = raw[:, EYE_COLS[source]] - (0.0 if ref is None else ref)
    u, v, lv, ap = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    cols = [u, v, lv, ap]
    if degree >= 2:
        cols += [u * u, u * lv, lv * lv, u * v]
    return np.column_stack(cols)


def design_matrices(raw: np.ndarray, degree: int, ref: np.ndarray | None = None,
                    source: str = "mediapipe", bias: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Matrizes de regressão (sem intercepto) para x e para y da tela."""
    raw = np.atleast_2d(raw)
    P, tans = head_screen_point(raw, bias)
    Ez = raw[:, 11:12]
    phi = eye_terms(raw, degree, ref, source)
    sec2 = 1.0 + tans ** 2
    Xx = np.hstack([P, Ez * sec2[:, 0:1] * phi])
    Xy = np.hstack([P, Ez * sec2[:, 1:2] * phi])
    return Xx, Xy


def polynomial_terms(X: np.ndarray, degree: int) -> np.ndarray:
    """[1, a, b] (grau 1) ou [1, a, b, a², ab, b²] (grau 2) para X = [a, b]."""
    X = np.atleast_2d(X)
    a, b = X[:, 0], X[:, 1]
    cols = [np.ones_like(a), a, b]
    if degree >= 2:
        cols += [a * a, a * b, b * b]
    return np.column_stack(cols)


def outlier_mask(samples: np.ndarray, threshold: float) -> np.ndarray:
    """Máscara de amostras a manter (escore-Z modificado via MAD < limiar).

    Iglewicz & Hoaglin (1993): z = 0.6745 (x - mediana) / MAD; 3.5 é o
    limiar usual. Remove frames isolados ruins (sacada atrasada, quase-piscada).
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


# --------------------------------------------------------------------- modelo
def head_offset(raw: np.ndarray, px_per_cm: float, bias: np.ndarray | None = None) -> np.ndarray:
    """Termo FIXO da cabeça (n, 2) em px: a·P_cabeça com a = px/cm da tela.

    O sinal de x é negativo porque a imagem não é espelhada (a direita do
    usuário é -x na câmera); y da câmera e da tela apontam para baixo."""
    P, _ = head_screen_point(raw, bias)
    return np.column_stack([-px_per_cm * P[:, 0], px_per_cm * P[:, 1]])


def compact_design(raw: np.ndarray, ref: np.ndarray | None = None,
                   source: str = "mediapipe", bias: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Medidas do olho do modelo compacto, já escaladas por E_z·sec².

    x: [u]; y: [lid_up, lid_up²]: só medidas coerentes com a cabeça."""
    raw = np.atleast_2d(raw)
    _, tans = head_screen_point(raw, bias)
    Ez = raw[:, 11:12]
    sec2 = 1.0 + tans ** 2
    m = raw[:, EYE_COLS[source]] - (0.0 if ref is None else ref)
    u, lid = m[:, 0], m[:, 2]
    Xx = Ez * sec2[:, 0:1] * u[:, None]
    Xy = Ez * sec2[:, 1:2] * np.column_stack([lid, lid * lid])
    return Xx, Xy


# ------------------------------------------------- olhar geométrico calibrado
# O "olhar geométrico" de gaze.py (globo ocular + rotação da cabeça) aponta,
# sem calibração, ~40° para CIMA do olhar real, com 18-29% da variação
# vertical e 34-47% da horizontal (nove sessões gravadas). ~30° vêm do viés
# da direção da face do MediaPipe (face_bias) e ~10° da íris, que em repouso
# fica acima da linha dos cantos do olho. O modelo "geometrico" calibra esse
# raio: ângulos (no espaço de atan(tans), como head_screen_point)
#     φ = direção da face corrigida (face_bias) + ângulo do olho na cabeça,
# com o ângulo do olho = c + b·medidas (x: u; y: lid_up, lid_up², v),
# ajustado para que o raio acerte os alvos. A cabeça e a distância entram
# pela geometria exata (tan, posição do rosto); só o olho é ajustado.

def screen_cm(Y: np.ndarray, screen_size, px_per_cm: float) -> np.ndarray:
    """Pixels da tela -> (x, y) em cm no referencial da câmera (plano z = 0)."""
    Y = np.atleast_2d(Y)
    return np.column_stack([(0.5 * screen_size[0] - Y[:, 0]) / px_per_cm,
                            SCREEN_GAP_CM + Y[:, 1] / px_per_cm])


def screen_to_angles(Y, raw, screen_size, px_per_cm) -> np.ndarray:
    """Ângulos (atan(tans)) do rosto até os pontos Y da tela."""
    raw = np.atleast_2d(raw)
    E = raw[:, 9:12]
    T = screen_cm(Y, screen_size, px_per_cm)
    return np.arctan((T - E[:, :2]) / np.maximum(E[:, 2:3], 1.0))


def angles_to_screen(phi, raw, screen_size, px_per_cm) -> np.ndarray:
    raw = np.atleast_2d(raw)
    E = raw[:, 9:12]
    hit = E[:, :2] + E[:, 2:3] * np.tan(np.clip(phi, -1.3, 1.3))
    return np.column_stack([0.5 * screen_size[0] - hit[:, 0] * px_per_cm,
                            (hit[:, 1] - SCREEN_GAP_CM) * px_per_cm])


def geometric_head_angles(raw, bias) -> np.ndarray:
    _, tans = head_screen_point(raw, bias)
    return np.arctan(tans)


def geometric_eye_features(raw, ref, source, use_v: bool = True
                           ) -> tuple[np.ndarray, np.ndarray]:
    """x: [u]; y: [lid_up, lid_up², v] (v só com ``use_v``).

    v (íris vertical) não responde à inclinação da cabeça (razão ~0 nos dados
    reais), mas traz sinal na grade: sem ela, o erro y nas nove sessões subia
    de 91 para 113 px (alvos novos) e de 261 para 289 px (cabeça em poses
    novas). Ela só é usada com a etapa da cabeça, cujas amostras (olhar fixo,
    cabeça inclinando) ensinam ao ajuste quanto confiar em v. Sem a etapa,
    nada impede o modelo de depender de v, e inclinar a cabeça deslocaria o
    cursor; aí entram só as medidas coerentes com a cabeça."""
    raw = np.atleast_2d(raw)
    m = raw[:, EYE_COLS[source]] - (0.0 if ref is None else ref)
    u, v, lid = m[:, 0], m[:, 1], m[:, 2]
    cols = [lid, lid * lid] + ([v] if use_v else [])
    return u[:, None], np.column_stack(cols)


def _std_floor(n_cols: int) -> np.ndarray:
    floor = np.full(n_cols, EYE_STD_FLOOR)
    floor[:2] = HEAD_STD_FLOOR_CM   # as duas primeiras colunas são P_cabeça
    return floor


def _ridge(X, y, w, alpha, floor):
    """Ridge ponderado com padronização; intercepto não penalizado."""
    mean = np.average(X, axis=0, weights=w)
    std = np.maximum(np.sqrt(np.average((X - mean) ** 2, axis=0, weights=w)), floor)
    Z = np.column_stack([np.ones(len(X)), (X - mean) / std])
    Zw = Z * w[:, None]
    reg = alpha * w.sum() * np.eye(Z.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Z.T @ Zw + reg, Zw.T @ y)
    return coef, mean, std


def _ridge_predict(X, coef, mean, std):
    return coef[0] + ((X - mean) / std) @ coef[1:]


def physical_design(raw: np.ndarray, ref: np.ndarray | None = None,
                    source: str = "mediapipe", bias: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Modelo físico: colunas [a, b_x, b_y] das equações de x e de y.

    x: -P_x (a) e E_z·sec²(yaw)·u (b_x); y: +P_y (a) e E_z·sec²(pitch)·lid_up (b_y).
    """
    raw = np.atleast_2d(raw)
    P, tans = head_screen_point(raw, bias)
    Ez = raw[:, 11]
    sec2 = 1.0 + tans ** 2
    m = raw[:, EYE_COLS[source]] - (0.0 if ref is None else ref)
    z = np.zeros(len(raw))
    Xx = np.column_stack([-P[:, 0], Ez * sec2[:, 0] * m[:, 0], z])
    Xy = np.column_stack([P[:, 1], z, Ez * sec2[:, 1] * m[:, 2]])
    return Xx, Xy


PHYSICAL_STD_FLOOR = np.array([HEAD_STD_FLOOR_CM, EYE_STD_FLOOR, EYE_STD_FLOOR])


def _ridge_joint(Xx, Xy, Y, w, alpha):
    """Ridge com coeficientes COMPARTILHADOS entre as equações de x e de y.

    Cada eixo tem seu intercepto (não penalizado, via centralização). As
    colunas são padronizadas pelo desvio conjunto, com piso."""
    mx = np.average(Xx, axis=0, weights=w)
    my = np.average(Xy, axis=0, weights=w)
    ybar = np.array([np.average(Y[:, 0], weights=w), np.average(Y[:, 1], weights=w)])
    Z = np.vstack([Xx - mx, Xy - my])
    t = np.concatenate([Y[:, 0] - ybar[0], Y[:, 1] - ybar[1]])
    ww = np.concatenate([w, w])
    std = np.maximum(np.sqrt(np.average(Z ** 2, axis=0, weights=ww)), PHYSICAL_STD_FLOOR)
    Zs = Z / std
    A = Zs.T @ (Zs * ww[:, None]) + alpha * ww.sum() * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, (Zs * ww[:, None]).T @ t) / std
    return beta, mx, my, ybar


def _ridge_joint_predict(Xx, Xy, beta, mx, my, ybar):
    return np.column_stack([ybar[0] + (Xx - mx) @ beta, ybar[1] + (Xy - my) @ beta])


def cv_score(target_errs: list[float], head_errs: list[float]) -> float:
    """Erro em pontos novos e, se houver etapa da cabeça, em poses novas
    (média simples dos dois: os dois tipos de erro importam no uso)."""
    if not target_errs:
        return float("inf")
    loo = float(np.mean(target_errs))
    return loo if not head_errs else 0.5 * (loo + float(np.mean(head_errs)))


class CalibrationModel:
    def __init__(self, degree: int = 2, ridge_alpha: float | None = None,
                 eye_source: str = "mediapipe", kind: str = "livre",
                 head_px_per_cm: float | None = None,
                 face_bias: np.ndarray | None = None):
        # "livre": regressão por eixo com todas as medidas do olho;
        # "fisico": 3 coeficientes, px/cm compartilhado e aprendido
        #           (physical_design; só sem head_px_per_cm);
        # "compacto": poucas medidas do olho, cabeça fixa (compact_design;
        #           exige head_px_per_cm).
        # "geometrico": olhar geométrico calibrado (geometric_*; exige
        #           head_px_per_cm): direção da face corrigida + ângulo do
        #           olho na cabeça, intersectados com o plano da tela.
        self.kind = kind
        # px/cm da tela: com ele, o termo da cabeça é fixo (geometria) e só o
        # olho é ajustado. None = termo da cabeça aprendido.
        self.head_px_per_cm = head_px_per_cm
        # Modelo opcional que substitui o eixo y da previsão (y pelo modelo
        # compacto com a cabeça pela geometria; ver docstring do módulo).
        self.y_model: "CalibrationModel | None" = None
        # Idem para o eixo x (cabeça pela geometria; ver fit_from_samples).
        self.x_model: "CalibrationModel | None" = None
        # Olhar geométrico calibrado (modelo "geometrico") para a vista 3D e
        # o HUD; o cursor usa y_model.
        self.geo: "CalibrationModel | None" = None
        # Modelo "geometrico": usa a íris vertical v (só com etapa da cabeça).
        self.geo_use_v = True
        # Viés da direção da face (radianos, (2,)); ver face_bias().
        self.face_bias: np.ndarray | None = face_bias
        self.degree = 1 if kind == "fisico" else degree
        self.eye_source = eye_source            # fonte da posição da íris (EYE_COLS)
        self.ridge_alpha = ridge_alpha          # None = escolhe por validação cruzada
        self.params: list | None = None         # [(coef, mean, std) de x, de y]
        self.eye_ref = np.zeros(4)              # referencia das medidas do olho
        self.offset = np.zeros(2)               # correção de "recentralizar"
        # Correção da boca (px por unidade de abertura acima de mouth_ref);
        # ver fit_mouth_correction.
        self.mouth_coef = np.zeros(2)
        self.mouth_ref = 0.0
        self.screen_size: tuple[int, int] | None = None
        self.created_at: str | None = None
        self.info: dict = {}
        # Postura da calibração: pose média da cabeça na grade (yaw, pitch em
        # graus), posição média do rosto (cm) e o quanto a cabeça se afastou
        # dela na etapa da cabeça (graus). Usados para avisar quando a
        # postura sai da faixa em que o modelo foi ajustado.
        self.head_ref = np.zeros(2)
        self.pos_ref = np.zeros(3)
        self.head_span = np.zeros(2)

    @property
    def is_fitted(self) -> bool:
        return self.params is not None

    def posture_deviation(self, raw: np.ndarray) -> dict:
        """Quanto a postura atual se afasta da calibração.

        Devolve ``ang`` (yaw, pitch: graus além da faixa calibrada, >= 0) e
        ``pos_cm`` (deslocamento do rosto em cm em relação à calibração).
        """
        raw = np.atleast_2d(raw)
        delta = np.abs(head_angles(raw)[0] - self.head_ref)
        beyond = np.maximum(delta - self.head_span, 0.0)
        return {"ang": beyond, "pos_cm": float(np.linalg.norm(raw[0, 9:12] - self.pos_ref))}

    def correct_with_known_target(self, raw: np.ndarray, target, gain: float = 0.3,
                                  max_step_px: float = 80.0) -> np.ndarray:
        """Corrige o offset com amostras de um momento em que o usuário
        olhava para um ponto CONHECIDO (ex.: centro da tecla selecionada).

        Move o offset ``gain`` do caminho até o erro medido (mediana das
        amostras), limitado a ``max_step_px``: uma seleção errada desloca
        pouco o cursor, e seleções corretas seguidas corrigem a deriva.
        Devolve a correção aplicada (px).
        """
        pred = np.median(np.atleast_2d(self.predict(np.atleast_2d(raw))), axis=0)
        step = gain * (np.asarray(target, dtype=float) - pred)
        norm = float(np.linalg.norm(step))
        if norm > max_step_px:
            step *= max_step_px / norm
        self.offset = self.offset + step
        return step

    # -------------------------------------------------------------- ajuste
    def _geometric_angles(self, params, raw):
        raw = np.atleast_2d(raw)
        Xx, Xy = geometric_eye_features(raw, self.eye_ref, self.eye_source, self.geo_use_v)
        eye = np.column_stack([_ridge_predict(Xx, *params[0]), _ridge_predict(Xy, *params[1])])
        return geometric_head_angles(raw, self.face_bias) + eye

    def gaze_direction(self, raw) -> np.ndarray:
        """Direção do olhar calibrada (n, 3) no referencial da câmera (só no
        modelo "geometrico"): para a vista 3D e o HUD."""
        phi = self._geometric_angles(self.params, raw)
        d = np.column_stack([np.tan(phi[:, 0]), np.tan(phi[:, 1]), -np.ones(len(phi))])
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def gaze_hit_cm(self, raw) -> np.ndarray:
        """Ponto (n, 3) em cm onde o olhar geométrico calibrado atinge a tela."""
        raw = np.atleast_2d(raw)
        phi = np.clip(self._geometric_angles(self.params, raw), -1.3, 1.3)
        hit = raw[:, 9:11] + raw[:, 11:12] * np.tan(phi)
        return np.column_stack([hit, np.zeros(len(raw))])

    def _eye_design(self, raw):
        """(Xx, Xy, deslocamento fixo da cabeça (n, 2)) com a cabeça fixa."""
        if self.kind == "compacto":
            Xx, Xy = compact_design(raw, self.eye_ref, self.eye_source, self.face_bias)
        else:
            Xx, Xy = design_matrices(raw, self.degree, self.eye_ref, self.eye_source,
                                     self.face_bias)
            Xx, Xy = Xx[:, 2:], Xy[:, 2:]   # sem as colunas de P_cabeça
        return Xx, Xy, head_offset(raw, self.head_px_per_cm, self.face_bias)

    def _fit_xy(self, raw, Y, w, alpha):
        if self.kind == "geometrico":
            head = geometric_head_angles(raw, self.face_bias)
            eye = (screen_to_angles(Y, raw, self.screen_size, self.head_px_per_cm) - head)
            Xx, Xy = geometric_eye_features(raw, self.eye_ref, self.eye_source, self.geo_use_v)
            return [_ridge(Xx, eye[:, 0], w, alpha, np.full(Xx.shape[1], EYE_STD_FLOOR)),
                    _ridge(Xy, eye[:, 1], w, alpha, np.full(Xy.shape[1], EYE_STD_FLOOR))]
        if self.head_px_per_cm is not None:
            Xx, Xy, off = self._eye_design(raw)
            floor = np.full(Xx.shape[1], EYE_STD_FLOOR), np.full(Xy.shape[1], EYE_STD_FLOOR)
            return [_ridge(Xx, Y[:, 0] - off[:, 0], w, alpha, floor[0]),
                    _ridge(Xy, Y[:, 1] - off[:, 1], w, alpha, floor[1])]
        if self.kind == "fisico":
            return _ridge_joint(*physical_design(raw, self.eye_ref, self.eye_source,
                                                 self.face_bias), Y, w, alpha)
        Xx, Xy = design_matrices(raw, self.degree, self.eye_ref, self.eye_source,
                                 self.face_bias)
        floor = _std_floor(Xx.shape[1])
        return [_ridge(Xx, Y[:, 0], w, alpha, floor), _ridge(Xy, Y[:, 1], w, alpha, floor)]

    def _predict_with(self, params, raw):
        if self.kind == "geometrico":
            return angles_to_screen(self._geometric_angles(params, raw), raw,
                                    self.screen_size, self.head_px_per_cm)
        if self.head_px_per_cm is not None:
            Xx, Xy, off = self._eye_design(raw)
            return np.column_stack([_ridge_predict(Xx, *params[0]),
                                    _ridge_predict(Xy, *params[1])]) + off
        if self.kind == "fisico":
            return _ridge_joint_predict(*physical_design(raw, self.eye_ref, self.eye_source,
                                                         self.face_bias), *params)
        Xx, Xy = design_matrices(raw, self.degree, self.eye_ref, self.eye_source,
                                 self.face_bias)
        return np.column_stack([_ridge_predict(Xx, *params[0]),
                                _ridge_predict(Xy, *params[1])])

    def _cv_errors(self, raw, Y, w, groups, alpha) -> tuple[list[float], list[float], list[int]]:
        """Erros da validação cruzada para um valor de ridge.

        * alvos (groups >= 0): cada alvo fica fora do ajuste por vez; erro =
          distância da mediana das previsões ao alvo;
        * direções da cabeça (groups <= HEAD_GROUP_BASE): cada direção fica
          fora por vez; erro = mediana das distâncias de cada amostra ao alvo
          (a cabeça se move durante a etapa, então cada amostra conta).

        Devolve (erros dos alvos, erros das direções, índices dos alvos).
        """
        return cv_errors_from(self._cv_predictions(raw, Y, w, groups, alpha), Y, groups)

    def _cv_predictions(self, raw, Y, w, groups, alpha) -> dict[int, np.ndarray]:
        """Previsões de cada dobra da validação cruzada (grupo -> (n, 2))."""
        preds = {}
        for g in np.unique(groups):
            if g == ALWAYS_TRAIN:
                continue
            tr, te = groups != g, groups == g
            if g <= HEAD_GROUP_BASE and te.sum() < MIN_HEAD_FOLD:
                continue
            params = self._fit_xy(raw[tr], Y[tr], w[tr], alpha)
            preds[int(g)] = self._predict_with(params, raw[te])
        return preds

    def fit(self, raw: np.ndarray, Y: np.ndarray, weights: np.ndarray | None,
            screen_size: tuple[int, int], groups: np.ndarray | None = None,
            axis_score: int | None = None, score_head: bool = True):
        """raw: (n, 20); Y: (n, 2) pixels-alvo; weights: (n,) ou None.

        groups: (n,) grupo de cada amostra para a validação cruzada que
        escolhe o ridge e estima o erro em pontos e poses não vistos:
        índice do alvo (>= 0), ALWAYS_TRAIN, ou HEAD_GROUP_BASE - direção.
        axis_score: 0 ou 1 = a validação cruzada mede só o erro nesse eixo
        (para modelos usados num eixo só, como o ``y_model``).
        score_head: False = a escolha do ridge (e ``cv_score``) usa só os
        alvos da grade; o erro na etapa da cabeça continua sendo medido e
        reportado. Na etapa da cabeça o alvo é sempre o CENTRO, então esse
        erro favorece modelos que encolhem a previsão para o centro.
        """
        raw = np.asarray(raw, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        w = np.ones(len(raw)) if weights is None else np.asarray(weights, dtype=np.float64)
        groups = (np.full(len(raw), ALWAYS_TRAIN) if groups is None
                  else np.asarray(groups, dtype=int))

        self.eye_ref = np.average(raw[:, EYE_COLS[self.eye_source]], axis=0, weights=w)
        self.screen_size = tuple(int(v) for v in screen_size)   # usado pelo "geometrico"
        n_targets = int(np.sum(np.unique(groups) >= 0))
        cv: dict[float, tuple[list[float], list[float], list[int]]] = {}
        if n_targets >= 5:
            candidates = ALPHAS if self.ridge_alpha is None else (self.ridge_alpha,)
            for alpha in candidates:
                preds = self._cv_predictions(raw, Y, w, groups, alpha)
                cv[alpha] = cv_errors_from(preds, Y, groups, axis_score)
            alpha = min(cv, key=lambda a: cv_score(cv[a][0], cv[a][1] if score_head else []))
        else:
            alpha = self.ridge_alpha if self.ridge_alpha is not None else 1e-2

        self.params = self._fit_xy(raw, Y, w, alpha)
        resid = self._predict_with(self.params, raw) - Y
        mse = float(np.sum(w * np.sum(resid ** 2, axis=1)) / w.sum())
        self.offset = np.zeros(2)
        self.screen_size = tuple(int(v) for v in screen_size)
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.info = {"rmse_ajuste_px": float(np.sqrt(mse)), "ridge_alpha": float(alpha)}
        if cv:
            target_errs, head_errs, target_ids = cv[alpha]
            self.info["erro_loo_medio_px"] = float(np.mean(target_errs))
            self.info["erro_loo_por_alvo_px"] = [float(e) for e in target_errs]
            self.info["erro_loo_alvos"] = target_ids   # índice do alvo de cada erro acima
            if head_errs:
                self.info["erro_cv_cabeca_px"] = float(np.mean(head_errs))
            self.info["pontuacao_cv_px"] = cv_score(target_errs, head_errs if score_head else [])

    @property
    def cv_score(self) -> float:
        """Critério de escolha entre modelos (inf se não houve validação cruzada)."""
        return float(self.info.get("pontuacao_cv_px", np.inf))

    def predict(self, raw: np.ndarray) -> np.ndarray:
        """Aceita (20,) ou (n, 20); devolve (2,) ou (n, 2) em pixels."""
        if not self.is_fitted:
            raise RuntimeError("Modelo de calibração ainda não ajustado.")
        raw = np.asarray(raw, dtype=np.float64)
        r2 = np.atleast_2d(raw)
        out = self._predict_raw(r2) + self.offset
        if r2.shape[1] > MOUTH_COL and np.any(self.mouth_coef):
            out = out + np.outer(np.maximum(r2[:, MOUTH_COL] - self.mouth_ref, 0.0), self.mouth_coef)
        return out[0] if raw.ndim == 1 else out

    def _predict_raw(self, raw: np.ndarray) -> np.ndarray:
        """Previsão sem o offset de recentralização (com o y composto)."""
        out = self._predict_with(self.params, raw)
        if self.y_model is not None:
            out[:, 1] = self.y_model._predict_with(self.y_model.params, raw)[:, 1]
        if self.x_model is not None:
            out[:, 0] = self.x_model._predict_with(self.x_model.params, raw)[:, 0]
        return out

    # ----------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {
            "feature_version": FEATURE_VERSION,
            "kind": self.kind,
            "head_px_per_cm": self.head_px_per_cm,
            "face_bias": None if self.face_bias is None else np.asarray(self.face_bias).tolist(),
            "degree": self.degree,
            "ridge_alpha": self.ridge_alpha,
            "eye_source": self.eye_source,
            "params": ([np.asarray(a).tolist() for a in self.params]
                       if self.kind == "fisico" and self.head_px_per_cm is None
                       else [[c.tolist(), m.tolist(), s.tolist()] for c, m, s in self.params]),
            "eye_ref": self.eye_ref.tolist(),
            "offset": self.offset.tolist(),
            "mouth_coef": self.mouth_coef.tolist(),
            "mouth_ref": self.mouth_ref,
            "head_ref": self.head_ref.tolist(),
            "pos_ref": self.pos_ref.tolist(),
            "head_span": self.head_span.tolist(),
            "screen_size": list(self.screen_size),
            "created_at": self.created_at,
            "info": self.info,
            "y_model": None if self.y_model is None else self.y_model.to_dict(),
            "x_model": None if self.x_model is None else self.x_model.to_dict(),
            "geo": None if self.geo is None else self.geo.to_dict(),
            "geo_use_v": self.geo_use_v,
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
        if tuple(d.get("screen_size", ())) != tuple(screen_size):
            print("[Calibração] Arquivo salvo é de outra resolução de tela; recalibre.")
            return None
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationModel":
        m = cls(d["degree"], d.get("ridge_alpha"), d.get("eye_source", "mediapipe"),
                d.get("kind", "livre"), d.get("head_px_per_cm"),
                None if d.get("face_bias") is None else np.array(d["face_bias"]))
        if m.kind == "fisico" and m.head_px_per_cm is None:
            m.params = tuple(np.array(a) for a in d["params"])
        else:
            m.params = [tuple(np.array(a) for a in p) for p in d["params"]]
        m.eye_ref = np.array(d["eye_ref"])
        m.offset = np.array(d.get("offset", [0.0, 0.0]))
        m.mouth_coef = np.array(d.get("mouth_coef", [0.0, 0.0]))
        m.mouth_ref = float(d.get("mouth_ref", 0.0))
        m.head_ref = np.array(d.get("head_ref", [0.0, 0.0]))
        m.pos_ref = np.array(d.get("pos_ref", [0.0, 0.0, 0.0]))
        m.head_span = np.array(d.get("head_span", [0.0, 0.0]))
        m.screen_size = tuple(d["screen_size"])
        m.created_at = d.get("created_at")
        m.info = d.get("info", {})
        if d.get("y_model"):
            m.y_model = cls.from_dict(d["y_model"])
        if d.get("x_model"):
            m.x_model = cls.from_dict(d["x_model"])
        if d.get("geo"):
            m.geo = cls.from_dict(d["geo"])
        m.geo_use_v = bool(d.get("geo_use_v", True))
        return m


@dataclass
class CalibrationData:
    """Amostras coletadas: por alvo da grade e na etapa de cabeça em movimento."""

    # lista de (alvo_px, features (n,2), raw (n,20))
    grid: list = field(default_factory=list)
    head_target: tuple[float, float] | None = None
    head_features: np.ndarray | None = None
    head_raw: np.ndarray | None = None
    # Opcional: landmarks de cada amostra (listas de FaceObservation), para
    # recalcular as medidas com outra versão do estimador (tools/reavaliar.py).
    grid_obs: list = field(default_factory=list)
    head_obs: list = field(default_factory=list)
    # Etapa da boca: olhar fixo no centro, boca abrindo e fechando.
    mouth_target: tuple[float, float] | None = None
    mouth_features: np.ndarray | None = None
    mouth_raw: np.ndarray | None = None
    mouth_obs: list = field(default_factory=list)

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
        if self.head_raw is not None:
            arrays["head_target"] = np.asarray(self.head_target, dtype=float)
            arrays["head_feats"] = np.atleast_2d(self.head_features)
            arrays["head_raw"] = np.atleast_2d(self.head_raw)
            _pack_obs(arrays, "obs_head", self.head_obs)
        if self.mouth_raw is not None:
            arrays["mouth_target"] = np.asarray(self.mouth_target, dtype=float)
            arrays["mouth_feats"] = np.atleast_2d(self.mouth_features)
            arrays["mouth_raw"] = np.atleast_2d(self.mouth_raw)
            _pack_obs(arrays, "obs_mouth", self.mouth_obs)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "CalibrationData":
        z = np.load(path)
        data = cls()
        i = 0
        while f"alvo_{i}" in z:
            data.grid.append((tuple(z[f"alvo_{i}"]), z[f"feats_{i}"],
                              _upgrade_raw(z[f"raw_{i}"])))
            data.grid_obs.append(_unpack_obs(z, f"obs_{i}"))
            i += 1
        if "head_raw" in z:
            data.head_target = tuple(z["head_target"])
            data.head_features, data.head_raw = z["head_feats"], _upgrade_raw(z["head_raw"])
            data.head_obs = _unpack_obs(z, "obs_head")
        if "mouth_raw" in z:
            data.mouth_target = tuple(z["mouth_target"])
            data.mouth_features, data.mouth_raw = z["mouth_feats"], _upgrade_raw(z["mouth_raw"])
            data.mouth_obs = _unpack_obs(z, "obs_mouth")
        return data

    @property
    def has_landmarks(self) -> bool:
        return bool(self.grid_obs) and all(len(o) for o in self.grid_obs)


def _upgrade_raw(raw: np.ndarray) -> np.ndarray:
    """Sessões antigas: 16 colunas (sem íris refinada) repetem as medidas do
    MediaPipe nas colunas da fonte refinada; 20 colunas (sem a abertura da
    boca) ganham a boca fechada (0)."""
    raw = np.atleast_2d(raw)
    if raw.shape[1] == 16:
        raw = np.hstack([raw, raw[:, 12:16]])
    if raw.shape[1] == 20:
        raw = np.hstack([raw, np.zeros((len(raw), 1))])
    return raw


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


def split_head_stage(head_raw: np.ndarray, grid_raw: np.ndarray,
                     max_deg: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separa as amostras da etapa da cabeça.

    Devolve (máscara das amostras usadas, grupo de validação de cada amostra,
    desvio de pose (n, 2) em relação à pose média da grade, em graus).
    Amostras com desvio acima de ``max_deg`` (yaw, pitch) são descartadas:
    nesses ângulos os landmarks do olho deixam de ser confiáveis.
    """
    base = np.median(head_angles(grid_raw), axis=0)
    delta = head_angles(head_raw) - base
    keep = np.ones(len(delta), bool)
    if max_deg is not None:
        keep = (np.abs(delta[:, 0]) <= max_deg[0]) & (np.abs(delta[:, 1]) <= max_deg[1])
    groups = HEAD_GROUP_BASE - head_direction(delta)
    groups[np.max(np.abs(delta), axis=1) < HEAD_STILL_DEG] = ALWAYS_TRAIN
    return keep, groups, delta


def cv_errors_from(preds: dict, Y: np.ndarray, groups: np.ndarray, axis: int | None = None):
    """(erros dos alvos, erros das direções da cabeça, índices dos alvos)
    a partir das previsões de cada dobra (ver CalibrationModel._cv_errors).
    axis: None = distância na tela; 0 ou 1 = só o erro nesse eixo."""
    def dist(d):
        return np.hypot(*np.atleast_2d(d).T) if axis is None else np.abs(np.atleast_2d(d)[:, axis])
    target_errs, head_errs, target_ids = [], [], []
    for g, pred in preds.items():
        te = groups == g
        if g >= 0:
            target_errs.append(float(dist(np.median(pred, axis=0) - Y[te][0])[0]))
            target_ids.append(int(g))
        else:
            head_errs.append(float(np.median(dist(pred - Y[te]))))
    return target_errs, head_errs, target_ids


def _compose_y(model: CalibrationModel, ym: CalibrationModel, raw, Y, w, groups,
               xm: CalibrationModel | None = None):
    """Usa ``ym`` no eixo y (e ``xm``, se dado, no eixo x) de ``model`` e
    refaz a validação cruzada e o erro de ajuste da combinação (os números
    da tela de resultado)."""
    model.y_model = ym
    model.x_model = xm
    model.info["modelo_y"] = (f"{ym.kind} grau {ym.degree} (cabeça pela geometria, "
                              f"íris {ym.eye_source})")
    if xm is not None:
        model.info["modelo_x"] = (f"{xm.kind} grau {xm.degree} (cabeça pela geometria, "
                                  f"íris {xm.eye_source})")
    resid = model._predict_raw(raw) - Y
    model.info["rmse_ajuste_px"] = float(np.sqrt(np.sum(w * np.sum(resid ** 2, 1)) / w.sum()))
    if not np.isfinite(model.cv_score):
        return
    xsrc = model if xm is None else xm
    px = xsrc._cv_predictions(raw, Y, w, groups, xsrc.info["ridge_alpha"])
    py = ym._cv_predictions(raw, Y, w, groups, ym.info["ridge_alpha"])
    preds = {g: np.column_stack([px[g][:, 0], py[g][:, 1]]) for g in px if g in py}
    target_errs, head_errs, target_ids = cv_errors_from(preds, Y, groups)
    model.info["erro_loo_medio_px"] = float(np.mean(target_errs))
    model.info["erro_loo_por_alvo_px"] = [float(e) for e in target_errs]
    model.info["erro_loo_alvos"] = target_ids
    if head_errs:
        model.info["erro_cv_cabeca_px"] = float(np.mean(head_errs))
    model.info["pontuacao_cv_px"] = cv_score(target_errs, head_errs)


def fit_from_samples(
    data: CalibrationData,
    screen_size: tuple[int, int],
    degree: int,
    ridge_alpha: float | None,
    mad_threshold: float,
    head_weight_points: float = 1.0,
    head_max_deg: tuple[float, float] | None = (15.0, 12.0),
    head_px_per_cm: float | None = None,
) -> tuple[CalibrationModel, list[PointResult], dict]:
    """Ajusta o modelo. Devolve (modelo, resultados por alvo, estatísticas).

    degree: grau MÁXIMO dos termos do olho; graus 1..degree são testados
    (com cada fonte da íris disponível) e fica o de menor erro na validação
    cruzada (alvos + direções da cabeça).
    head_max_deg: faixa (yaw, pitch) da etapa da cabeça aproveitada, em
    graus em torno da pose média da grade; None usa todas as amostras.
    head_px_per_cm: px/cm da tela. Se dado, o termo da cabeça é fixo pela
    geometria (candidatos livre e compacto); se None, é aprendido
    (candidatos livre e físico).
    """
    raws, Ys, ws, gs, cleaned = [], [], [], [], []
    for i, (target, feats, raw) in enumerate(data.grid):
        raw = np.atleast_2d(raw)
        if len(raw) < 3:
            continue   # alvo sem amostras suficientes (rosto perdido): ignora
        keep = outlier_mask(feats, mad_threshold)
        r = raw[keep]
        cleaned.append((target, r))
        raws.append(r)
        Ys.append(np.tile(target, (len(r), 1)))
        ws.append(np.full(len(r), 1.0 / len(r)))   # cada alvo pesa 1
        gs.append(np.full(len(r), i))
    if not raws:
        raise ValueError("Nenhum alvo com amostras suficientes para calibrar.")
    grid_raw = np.vstack(raws)

    stats_head: dict = {}
    hr = None
    if data.head_raw is not None and len(data.head_raw) >= 5:
        hr_all = np.atleast_2d(data.head_raw)
        keep, hgroups, delta = split_head_stage(hr_all, grid_raw, head_max_deg)
        stats_head = {"n_cabeca_total": int(len(hr_all)), "n_cabeca_usadas": int(keep.sum())}
        if keep.sum() >= 5:
            hr = hr_all[keep]
            raws.append(hr)
            Ys.append(np.tile(data.head_target, (len(hr), 1)))
            # A etapa de cabeça pesa o equivalente a `head_weight_points` alvos.
            ws.append(np.full(len(hr), head_weight_points / len(hr)))
            gs.append(hgroups[keep])
            head_span = np.percentile(np.abs(delta[keep]), 95, axis=0)
        else:
            head_span = np.zeros(2)
    has_head = hr is not None

    if len(cleaned) < 9:
        degree = 1  # poucos alvos para os termos quadráticos

    raw_all, Y_all = np.vstack(raws), np.vstack(Ys)
    w_all, g_all = np.concatenate(ws), np.concatenate(gs)
    # Fontes da íris a testar: a refinada só se ela existir nesta sessão.
    sources = ["mediapipe"]
    if not np.allclose(raw_all[:, EYE_COLS["refinada"]], raw_all[:, EYE_COLS["mediapipe"]]):
        sources.append("refinada")
    candidates, y_candidates, x_candidates, geo_candidates = [], [], [], []
    bias = None
    if head_px_per_cm is not None:
        # Postura da calibração = rosto voltado para o centro dos alvos.
        targets_px = np.array([t for t, _ in cleaned], dtype=float).mean(axis=0)
        center_cm = np.array([(0.5 - targets_px[0] / screen_size[0]) * screen_size[0] / head_px_per_cm,
                              SCREEN_GAP_CM + targets_px[1] / head_px_per_cm, 0.0])
        bias = face_bias(grid_raw, center_cm)
    for src in sources:
        if head_px_per_cm is not None:
            # Eixo y: cabeça pela geometria (Y_MODEL_KINDS). O olhar
            # geométrico calibrado é ajustado à parte, para a vista 3D e o HUD.
            # Sem a etapa da cabeça, nada ensina ao ajuste que a íris
            # vertical e a abertura não acompanham a inclinação da cabeça:
            # o eixo y usa só medidas coerentes com ela (geométrico com a
            # pálpebra). No usuário sintético, a regressão livre deslocava o
            # cursor 132 px a 10° de inclinação nesse caso.
            y_kinds = list(Y_MODEL_KINDS) if has_head else [("geometrico", 1)]
            # Eixo x: regressão livre com a cabeça pela geometria, escolhida
            # pelo erro só no eixo x nos alvos da grade (X_MODEL_GEOMETRIC).
            if X_MODEL_GEOMETRIC and has_head:
                for deg in range(degree, 0, -1):
                    m = CalibrationModel(deg, ridge_alpha, src, kind="livre",
                                         head_px_per_cm=head_px_per_cm, face_bias=bias)
                    m.fit(raw_all, Y_all, w_all, screen_size, groups=g_all, axis_score=0,
                          score_head=False)
                    x_candidates.append(m)
            for kind, deg in y_kinds:
                m = CalibrationModel(deg, ridge_alpha, src, kind=kind,
                                     head_px_per_cm=head_px_per_cm, face_bias=bias)
                m.geo_use_v = has_head
                # Escolha só pelos alvos da grade (score_head=False): pelo
                # erro na etapa da cabeça (alvo sempre no centro), modelos
                # que encolhem o eixo y para o centro pareciam melhores e o
                # cursor não descia o suficiente (sessão de 12/11 09:07).
                m.fit(raw_all, Y_all, w_all, screen_size, groups=g_all, axis_score=1,
                      score_head=False)
                if (kind, deg) in y_kinds:
                    y_candidates.append(m)
                if kind == "geometrico":
                    geo_candidates.append(m)
        for deg in range(degree, 0, -1):
            m = CalibrationModel(deg, ridge_alpha, src)
            m.fit(raw_all, Y_all, w_all, screen_size, groups=g_all, score_head=X_SCORE_HEAD)
            candidates.append(m)
        if has_head:
            # O modelo físico só faz sentido com a etapa da cabeça: sem ela,
            # o px/cm compartilhado fica indeterminado (cabeça parada).
            m = CalibrationModel(1, ridge_alpha, src, kind="fisico")
            m.fit(raw_all, Y_all, w_all, screen_size, groups=g_all, score_head=X_SCORE_HEAD)
            candidates.append(m)
    # Escolhe pelo erro da validação cruzada; sem ela, o primeiro candidato
    # (MediaPipe, grau máximo).
    model = min(candidates, key=lambda m: m.cv_score)
    if not np.isfinite(model.cv_score):
        model = candidates[0]
    model.info["usa_etapa_cabeca"] = bool(has_head)
    model.info["fonte_iris"] = model.eye_source
    model.info["grau"] = model.degree
    model.info["modelo"] = model.kind
    model.info["px_por_cm_cabeca"] = head_px_per_cm
    model.info["candidatos"] = [
        {"modelo": m.kind, "fonte": m.eye_source, "grau": m.degree,
         "erro_loo_px": m.info.get("erro_loo_medio_px"),
         "erro_cv_cabeca_px": m.info.get("erro_cv_cabeca_px")}
        for m in candidates + y_candidates]
    if geo_candidates:
        # Olhar geométrico calibrado (vista 3D e HUD).
        model.geo = min(geo_candidates, key=lambda m: m.cv_score)
    if y_candidates:
        ym = min(y_candidates, key=lambda m: m.cv_score)
        if not np.isfinite(ym.cv_score):
            ym = y_candidates[0]
        xm = min(x_candidates, key=lambda m: m.cv_score) if x_candidates else None
        if xm is not None and not np.isfinite(xm.cv_score):
            xm = x_candidates[0]
        _compose_y(model, ym, raw_all, Y_all, w_all, g_all, xm)
    model.info.update(stats_head)

    # Postura de referência (aviso de postura fora da calibração).
    grid_delta = head_angles(grid_raw) - np.median(head_angles(grid_raw), axis=0)
    span = np.percentile(np.abs(grid_delta), 95, axis=0)
    if has_head:
        span = np.maximum(span, head_span)
    model.head_ref = np.median(head_angles(grid_raw), axis=0)
    model.pos_ref = np.median(grid_raw[:, 9:12], axis=0)
    model.head_span = span

    results = []
    for target, r in cleaned:
        pred = np.median(model.predict(r), axis=0)
        results.append(PointResult(target, (float(pred[0]), float(pred[1])), len(r)))

    fit_mouth_correction(model, data, grid_raw, mad_threshold)

    stats = dict(model.info)
    if has_head:
        err = np.hypot(*(model.predict(hr) - np.asarray(data.head_target)).T)
        stats["erro_mediano_cabeca_px"] = float(np.median(err))
    return model, results, stats


MOUTH_MIN_RANGE = 0.04   # variação mínima de abertura na etapa da boca


def fit_mouth_correction(model: CalibrationModel, data: CalibrationData, grid_raw: np.ndarray,
                         mad_threshold: float = 3.5):
    """Correção da boca a partir da etapa da boca (olhar fixo no centro).

    O MediaPipe estima a malha inteira de uma vez, e abrir a boca desloca a
    pose da cabeça e os pontos do olho: com o olhar fixo num alvo, o ponto
    estimado descia ~50 px (sessão de 12/11 09:47). Com o olhar fixo, todo
    deslocamento da previsão que acompanha a abertura da boca é artefato.
    Ajusta-se o deslocamento por unidade de abertura acima da abertura
    típica da grade (mínimos quadrados pela origem, com rejeição de
    outliers), e a previsão passa a descontá-lo. Sem a etapa, ou se a boca
    quase não abriu, a correção fica em zero.
    """
    model.mouth_coef = np.zeros(2)
    model.mouth_ref = float(np.median(np.atleast_2d(grid_raw)[:, MOUTH_COL]))
    mr = None if data.mouth_raw is None else np.atleast_2d(data.mouth_raw)
    if mr is None or len(mr) < 10 or mr.shape[1] <= MOUTH_COL:
        return
    m = np.maximum(mr[:, MOUTH_COL] - model.mouth_ref, 0.0)
    model.info["boca_variacao"] = float(np.ptp(mr[:, MOUTH_COL]))
    if np.ptp(mr[:, MOUTH_COL]) < MOUTH_MIN_RANGE or np.sum(m > 0.5 * MOUTH_MIN_RANGE) < 5:
        return
    resid = np.asarray(data.mouth_target, dtype=float) - model.predict(mr)
    # Desvio da previsão com a boca FECHADA (postura, deriva) não é efeito
    # da boca: subtrai-se a mediana dos quadros de boca fechada.
    closed = m < 0.25 * MOUTH_MIN_RANGE
    base = np.median(resid[closed], axis=0) if closed.sum() >= 5 else np.zeros(2)
    resid = resid - base
    keep = outlier_mask(resid, mad_threshold)
    mm, rr = m[keep], resid[keep]
    coef = (mm @ rr) / max(float(mm @ mm), 1e-9)
    model.mouth_coef = coef
    model.info["boca_px_por_unidade"] = [float(c) for c in coef]


def uncalibrated_mapping(features: np.ndarray, screen_size: tuple[int, int],
                         half_yaw: float, half_pitch: float) -> np.ndarray:
    """Mapeamento linear fixo usado só para visualizar antes de calibrar."""
    w, h = screen_size
    x = w / 2 * (1 + features[0] / half_yaw)
    y = h / 2 * (1 - features[1] / half_pitch)
    return np.array([x, y])
