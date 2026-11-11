"""
Configuração central da aplicação.

Todos os parâmetros ajustáveis ficam aqui, agrupados por módulo. Os valores
padrão foram escolhidos para uma webcam comum (720p, 30 fps) com o usuário
sentado a ~50-70 cm da tela.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Espelha apenas a IMAGEM EXIBIDA (efeito "espelho"). O processamento é
    # sempre feito sobre o frame original, então isso não afeta o olhar.
    mirror_display: bool = True


@dataclass
class TrackerConfig:
    model_path: Path = PROJECT_ROOT / "models" / "face_landmarker.task"
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Refina o centro da íris nos pixels da imagem (iris_refine.py). A
    # calibração usa essa fonte só se ela tiver erro menor que a do MediaPipe.
    refine_iris: bool = True


@dataclass
class GazeConfig:
    # Fonte da rotação da cabeça: "matrix" (matriz do FaceLandmarker, com
    # verificação de coerência) ou "landmarks" (eixos geométricos).
    head_pose_source: str = "matrix"

    # Raio do globo ocular expresso em "larguras de olho" (distância entre os
    # cantos do olho). Anatomicamente: raio ~12 mm / largura da fenda
    # palpebral ~30 mm = 0.4. Erros neste valor são absorvidos em grande
    # parte pela calibração (afetam só a escala do ângulo).
    eyeball_radius_ratio: float = 0.40

    # Detecção de piscada pelo EAR (eye aspect ratio = altura/largura do
    # olho). O limiar é RELATIVO ao EAR típico do usuário com o olho aberto
    # (estimado de forma contínua), então não precisa ser ajustado por pessoa.
    # Não mais alto que 0.5: olhar para baixo já fecha bastante o olho.
    blink_ratio: float = 0.5
    ear_open_initial: float = 0.28   # chute inicial do EAR com olho aberto
    ear_baseline_alpha: float = 0.02  # velocidade de adaptação da linha de base


@dataclass
class FilterConfig:
    # One-Euro Filter (Casiez et al., 2012) aplicado aos ângulos de olhar
    # exibidos no HUD e ao mapeamento usado antes de calibrar.
    min_cutoff: float = 0.5   # menor = mais suave quando o olhar está parado
    beta: float = 0.005       # maior = menos atraso em movimentos rápidos
    d_cutoff: float = 1.0


@dataclass
class StabilizerConfig:
    # Estabilizador de fixação aplicado ao ponto na tela (cursor e dwell).
    # Ver filters.FixationStabilizer.
    saccade_ratio: float = 0.07   # salto mínimo (fração da largura da tela) p/ sacada
    confirm_frames: int = 3       # amostras seguidas fora da fixação p/ confirmar salto
    window_frames: int = 20       # janela da média durante a fixação (~0,7 s a 30 fps)
    follow: float = 0.3           # fração do caminho até a média percorrida por frame
    deadzone_px: float = 6.0      # movimentos menores que isto são ignorados
    noise_k: float = 3.0          # limiar adaptativo = max(base, k * ruído medido)


@dataclass
class CalibrationConfig:
    # 4x4: o eixo vertical tem o sinal mais fraco e precisa de mais linhas
    # de alvos do que o 3x3 oferecia (~35 s no total).
    grid_cols: int = 4
    grid_rows: int = 4
    margin_ratio: float = 0.08     # distância dos alvos à borda (fração da tela)
    # Ordem aleatória dos alvos. Na ordem linha a linha, o tempo (cansaço,
    # olho ressecando, postura escorregando) andava junto com a posição
    # vertical: nos dados reais a abertura do olho caiu de 0,27 para 0,16 ao
    # longo da grade e o modelo aprendia isso como "olhar para baixo".
    shuffle_targets: bool = True
    random_seed: int | None = None
    # Coleta por fixação: depois de ``settle_s`` (o olho chega ao alvo), a
    # coleta começa quando o olhar estabiliza (``fixation_window`` quadros
    # seguidos a menos de ``fixation_max_deg`` da mediana; o ruído típico
    # medido é ~0,2°) ou, no máximo, em ``settle_max_s``.
    settle_s: float = 0.6
    settle_max_s: float = 1.8
    fixation_window: int = 6
    fixation_max_deg: float = 0.8
    collect_s: float = 1.2         # tempo de coleta de amostras por alvo
    min_samples_per_point: int = 8
    # Repetição automática de alvos ruins: depois do ajuste, alvos cujo erro
    # na validação cruzada passa de ``retry_ratio`` x a mediana (e de
    # ``retry_min_frac`` da largura da tela) são coletados de novo, uma vez
    # (piscada longa, distração, olhar fora do alvo).
    retry_bad_targets: bool = True
    retry_max_targets: int = 3
    retry_ratio: float = 2.5
    retry_min_frac: float = 0.08
    polynomial_degree: int = 2     # 1 = só termos lineares do olho, 2 = + quadráticos
    # Regularização ridge (features padronizadas). None = escolhida por
    # validação cruzada deixando um alvo de fora.
    ridge_alpha: float | None = None
    outlier_mad_threshold: float = 3.5
    save_path: Path = PROJECT_ROOT / "models" / "calibration.json"
    # Medidas brutas de cada calibração (.npz), para reanálise offline
    # (tools/reavaliar.py).
    results_dir: Path = PROJECT_ROOT / "results"

    # Etapa de movimento da cabeça (após os alvos): o usuário mantém o
    # olhar no alvo central e gira a cabeça seguindo as setas. É o que
    # permite separar movimento de olho de movimento de cabeça.
    head_stage: bool = True
    # Quanto girar em cada direção (yaw, pitch em graus, em relação à pose
    # da grade). A seta enche conforme a cabeça gira e fica verde ao chegar
    # a ``head_reach_frac`` do alvo; depois de ``head_hold_s`` ali, passa
    # para a próxima direção. Quem não alcança segue após o tempo limite.
    # Nos dados reais, sem guia, as pessoas giravam até 30-40°, faixa em
    # que os landmarks do olho não são confiáveis.
    head_target_deg: tuple = (10.0, 8.0)
    head_reach_frac: float = 0.8
    head_hold_s: float = 0.6
    head_step_timeout_s: float = 4.0
    head_center_s: float = 1.0        # etapa final com a cabeça de volta ao centro
    head_motion_settle_s: float = 0.8
    # Faixa aproveitada no ajuste (amostras além dela são descartadas).
    head_max_deg: tuple = (15.0, 12.0)
    head_motion_weight: float = 1.0   # peso da etapa, em "número de alvos"

    # Mapeamento usado ANTES de calibrar (só para visualização): ângulo de
    # olhar que corresponderia à borda da tela.
    uncalibrated_half_range_yaw_deg: float = 20.0
    uncalibrated_half_range_pitch_deg: float = 12.0


@dataclass
class ValidationConfig:
    # Alvos de validação (frações da tela), diferentes dos de calibração.
    points: tuple = (
        (0.5, 0.5),
        (0.3, 0.3), (0.7, 0.3),
        (0.3, 0.7), (0.7, 0.7),
        (0.5, 0.2), (0.5, 0.8),
        (0.15, 0.5), (0.85, 0.5),
    )
    settle_s: float = 0.9
    collect_s: float = 1.2
    results_dir: Path = PROJECT_ROOT / "results"
    # Opcional: dimensões físicas para converter erro de px para graus de
    # ângulo visual. Deixe em None se não souber.
    screen_width_cm: float | None = None
    viewing_distance_cm: float | None = None


@dataclass
class MouseConfig:
    # Por segurança, o mouse só é movido se houver calibração E rosto
    # detectado no frame atual (sem rosto, o cursor fica parado). Se o rosto
    # sumir por mais que isto, o filtro é reiniciado e o HUD mostra
    # "PAUSADO" (uma piscada ou um frame perdido não contam).
    face_lost_timeout_s: float = 0.5


@dataclass
class PostureConfig:
    # Aviso de postura fora da calibração. O modelo só é confiável perto da
    # postura em que foi ajustado; nas sessões reais, a pessoa mudava de
    # posição entre calibrar e validar e a maior parte do erro (até 220 px)
    # era um desvio constante que a recentralização (R) corrige.
    angle_tol_deg: float = 4.0     # graus ALÉM da faixa coberta na calibração
    pos_tol_cm: float = 4.0        # deslocamento do rosto
    sustain_s: float = 1.5         # tempo fora da faixa antes de avisar


@dataclass
class AdaptConfig:
    # Correção contínua por alvos conhecidos (ex.: tecla do teclado virtual
    # selecionada): o offset anda ``gain`` do caminho até o erro medido,
    # limitado a ``max_step_px`` por correção.
    gain: float = 0.3
    max_step_px: float = 80.0
    window_frames: int = 15        # amostras recentes usadas (~0,5 s)


@dataclass
class DwellClickConfig:
    enabled_at_start: bool = False
    dwell_time_s: float = 1.0
    tolerance_px: int = 60
    cooldown_s: float = 1.0


@dataclass
class DisplayConfig:
    show_landmarks: bool = False
    show_gaze: bool = True
    show_head_pose: bool = False
    show_minimap: bool = True
    show_help: bool = True
    window_name: str = "Eye Gaze - camera"
    target_window_name: str = "Eye Gaze - alvos"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    dwell: DwellClickConfig = field(default_factory=DwellClickConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
