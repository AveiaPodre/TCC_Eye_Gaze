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
    # Campo de visão HORIZONTAL suposto da câmera (graus). Define a escala
    # métrica do rosto (geometry.py). Webcams de notebook: ~80-95°; webcams
    # externas comuns: ~65-80°. Mudá-lo invalida a calibração salva.
    hfov_deg: float = 90.0
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

    # Filtro One-Euro (Casiez et al., 2012) na posição da pupila em relação
    # à âncora, no referencial da cabeça, SÓ para o feixe exibido (desenho,
    # HUD, detecção de fixação na coleta): estabiliza os feixes e a
    # intersecção. O cursor e a calibração usam a pupila sem filtro: nas
    # sessões gravadas, o filtro atrasava o cursor (47 ms de mediana) e até
    # aumentava o tremor depois do estabilizador (metodologia, seção 4.1).
    # Corte em Hz; beta em 1/cm.
    pupil_filter: bool = True
    pupil_min_cutoff: float = 0.7
    pupil_beta: float = 10.0
    pupil_filter_reset_s: float = 0.3   # intervalo sem amostra que reinicia o filtro

    # Escala métrica do rosto: distância entre os cantos externos dos olhos
    # (média adulta ~9 cm). A distância do rosto à câmera sai da câmera
    # pinhole com essa medida. A translação da matriz do MediaPipe NÃO é
    # usada: nas sessões gravadas ela dava ~36 cm com a pessoa a ~64 cm (a
    # escala dela não é coerente com o campo de visão suposto), e o rosto
    # inteiro em cm ficava a 56% do tamanho real.
    outer_canthi_cm: float = 9.0
    # "pinhole" (acima) ou "mediapipe" (translação da matriz do modelo).
    depth_source: str = "pinhole"
    # Razão raio do globo / largura do olho (cantos): ~12 mm / ~30 mm.
    eyeball_radius_ratio: float = 0.40

    # Detecção de piscada pelo EAR (eye aspect ratio = altura/largura do
    # olho). O limiar é RELATIVO ao EAR típico do usuário com o olho aberto
    # (estimado de forma contínua), então não precisa ser ajustado por pessoa.
    # Não mais alto que 0.5: olhar para baixo já fecha bastante o olho.
    blink_ratio: float = 0.5
    ear_open_initial: float = 0.28   # chute inicial do EAR com olho aberto
    ear_baseline_alpha: float = 0.02  # velocidade de adaptação da linha de base


@dataclass
class StabilizerConfig:
    # Estabilizador de fixação aplicado ao ponto na tela (cursor e dwell).
    # Ver filters.FixationStabilizer.
    saccade_ratio: float = 0.07   # salto mínimo (fração da largura da tela) p/ sacada
    confirm_frames: int = 2       # amostras seguidas fora da fixação p/ confirmar salto
    window_frames: int = 30       # amostras guardadas da fixação atual (~1 s a 30 fps)
    # Fração do caminho até o destino percorrida por quadro. 0,85 (era 0,6):
    # chega mais rápido e se mexe menos na fixação (2,7% contra 3,8% dos
    # quadros), sem mudar o tremor (metodologia, seção 4.1).
    follow: float = 0.85
    # Trava da fixação: depois de ``settle_frames`` quadros o cursor para e
    # só volta a andar se ``shift_frames`` amostras seguidas ficarem fora do
    # raio de trava = max(lock_ratio * largura, lock_k * passeio medido).
    lock_ratio: float = 0.035     # ~67 px numa tela de 1920 px
    # Passeio da estimativa com o olho parado: ~50 px de desvio (medido nos
    # vídeos gravados). Com lock_k 2,5 e teto de 0,6 x limiar de sacada, o
    # cursor "destravava" ~18 vezes por minuto seguindo esse passeio; com
    # 3,5 e teto 0,9, ~6,5 vezes, sem mudar o atraso após sacadas.
    lock_k: float = 3.5
    lock_cap: float = 0.9         # raio de trava <= lock_cap x limiar de sacada
    # Quadros seguidos fora da trava para seguir um movimento PEQUENO do
    # olhar (menor que o limiar de sacada). 5 (~0,17 s; era 8): degraus de
    # 60-100 px passam a ser seguidos em ~130 ms em vez de ~230 ms.
    shift_frames: int = 5
    settle_frames: int = 8
    noise_k: float = 3.0          # limiar de sacada = max(base, k * ruído medido)


@dataclass
class CalibrationConfig:
    # 4x4: o eixo vertical tem o sinal mais fraco e precisa de mais linhas
    # de alvos do que o 3x3 oferecia (~35 s no total).
    grid_cols: int = 4
    grid_rows: int = 4
    # Distância dos alvos à borda (fração da tela). Era 0,08: a última linha
    # ficava a 92% da altura, e abaixo dela (barra de tarefas) o modelo
    # extrapolava medidas que saturam ao olhar para baixo.
    margin_ratio: float = 0.05
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
    # Controle de deriva: um ponto de checagem no CENTRO no início e a cada
    # ``drift_check_every`` alvos da grade. Como a pessoa olha sempre para o
    # mesmo ponto conhecido, a mudança da pupila entre as checagens é deriva
    # (cansaço, olho ressecando, postura escorregando); a calibração a
    # desconta e fica ajustada ao estado do fim (calibration.correct_drift).
    drift_checks: bool = True
    drift_check_every: int = 4
    # Ao abrir o app com uma calibração salva, recentraliza (1 ponto) antes
    # de usar: reancora o desvio da nova sessão.
    recenter_on_start: bool = True
    retry_bad_targets: bool = True
    retry_max_targets: int = 3
    retry_ratio: float = 2.5
    retry_min_frac: float = 0.08
    # Grau máximo da correção monitor virtual -> real (calibration.py):
    # 1 = até afim; 2 = também quadrática. A validação cruzada escolhe.
    polynomial_degree: int = 2
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

    # Etapa da boca (depois da etapa da cabeça): olhar fixo no centro e a boca
    # abrindo e fechando. Ensina quanto a abertura da boca desloca a
    # estimativa (o MediaPipe deforma a malha inteira) para descontá-la.
    mouth_stage: bool = True
    mouth_stage_s: float = 4.0       # duração mínima
    mouth_timeout_s: float = 7.0     # segue mesmo se a boca não abrir
    mouth_min_open: float = 0.08     # abertura pedida (lábios / largura entre olhos)

    # Calibração manual (tecla C): cada alvo só avança quando o usuário,
    # olhando para ele, aperta ESPAÇO/ENTER. As etapas da cabeça e da boca
    # continuam automáticas.
    manual: bool = False

    # Recentralização (R): desvios maiores que esta fração da tela (em cada
    # eixo) não são aplicados; a mensagem pede para recalibrar.
    recenter_max_frac: float = 0.25

    # Largura física da tela (cm): tamanho do MONITOR VIRTUAL onde o olhar é
    # projetado (geometry.VirtualScreen). Preenchida por --screen-width-cm ou
    # detectada no Windows. Sem ela, usa-se ``assumed_screen_width_cm``
    # (34,5 cm = notebook de 15,6" 16:9) e a correção monitor virtual -> real
    # absorve a diferença de escala.
    screen_width_cm: float | None = None
    assumed_screen_width_cm: float = 34.5


@dataclass
class ValidationConfig:
    # Validação IMEDIATA, logo depois da calibração (mede o erro do modelo,
    # sem tempo para a postura mudar). A tecla T faz o RETESTE, mais tarde:
    # o primeiro ponto (centro) mede a deriva desde a calibração e reancora.
    after_calibration: bool = True
    # Alvos de validação (frações da tela), diferentes dos de calibração.
    # O PRIMEIRO precisa ser o centro (ponto de reancoragem do reteste).
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
class FineConfig:
    # Ajuste fino pela cabeça (tecla A; filters.HeadFineTuner): durante uma
    # fixação, pequenos giros da cabeça deslocam o cursor a partir do ponto
    # do olhar. A cabeça é medida com ~10x mais precisão que o olhar.
    enabled: bool = False
    gain_px_per_deg: float = 25.0   # o "físico" seria ~38 px/grau; menor = mais preciso
    deadzone_deg: float = 0.3       # balanço natural da cabeça não move o cursor
    max_px: float = 250.0           # ajuste máximo a partir do ponto do olhar
    reset_deg: float = 6.0          # giro maior que isto = mudança de postura (zera)


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
    # Vista 3D (scene3d.py): rosto, globos oculares, feixes, intersecção e tela.
    show_scene3d: bool = False
    window_name: str = "Eye Gaze - camera"
    target_window_name: str = "Eye Gaze - alvos"
    scene3d_window_name: str = "Eye Gaze - 3D"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    dwell: DwellClickConfig = field(default_factory=DwellClickConfig)
    fine: FineConfig = field(default_factory=FineConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
