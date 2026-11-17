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
    # Captura. "auto" usa o backend padrão do OpenCV; no Windows, "dshow" e
    # "msmf" diferem em fps, formatos e controles de exposição.
    backend: str = "auto"
    # MJPG: em 1080p, muitas webcams só chegam a 30 fps nesse formato (em
    # YUY2 caem para 5-15 fps).
    mjpg: bool = True
    # Exposição fixa (valor do driver; None = automática). Com pouca luz, a
    # exposição automática alonga o tempo de cada quadro: o fps cai e a íris
    # borra nas sacadas.
    exposure: float | None = None
    autofocus: bool = True


@dataclass
class TrackerConfig:
    model_path: Path = PROJECT_ROOT / "models" / "face_landmarker.task"
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Refina o centro da íris nos pixels da imagem (iris_refine.py). A
    # calibração usa a MÉDIA das duas fontes (MediaPipe e refinada): no
    # benchmark (44 validações), 110 px, contra 136 só com o MediaPipe e 115
    # só com a refinada.
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

    # Detecção de piscada pelo EAR (eye aspect ratio = altura/largura do
    # olho). O limiar é RELATIVO ao EAR típico do usuário com o olho aberto
    # (estimado de forma contínua), então não precisa ser ajustado por pessoa.
    # Não mais alto que 0.5: olhar para baixo já fecha bastante o olho.
    blink_ratio: float = 0.5
    ear_open_initial: float = 0.28   # chute inicial do EAR com olho aberto
    ear_baseline_alpha: float = 0.02  # velocidade de adaptação da linha de base


@dataclass
class StabilizerConfig:
    # Estabilizador do cursor (filters.GazeFollower): salta nas sacadas e, na
    # fixação, fica parado até o olhar se mover de verdade (histerese).
    saccade_ratio: float = 0.07   # salto mínimo (fração da largura da tela) p/ sacada
    confirm_frames: int = 2       # amostras seguidas fora da fixação p/ confirmar salto
    noise_k: float = 3.0          # limiar de sacada = max(base, k * ruído medido)
    median_frames: int = 5        # destino = mediana das últimas N amostras
    # Histerese: parado, o cursor só anda quando o destino se afasta mais que
    # start_k x ruído; desliza (alpha por quadro) até ficar a menos de
    # stop_k x ruído e para. Escolhidos na bancada com o ruído real das
    # validações gravadas (tools/bancada_cursor.py, 428 fixações; metodologia,
    # seção 4.0.1): segue 53%/58%/76% dos movimentos sutis de 40/70/100 px em
    # até 0,5 s (antes 33%/43%/62% com start_k 4, alpha 0,4, mediana 7), com
    # o cursor parado em 87% dos quadros de olho fixo (antes 85%).
    start_k: float = 2.5
    stop_k: float = 1.0
    alpha: float = 0.7
    noise_floor_px: float = 15.0
    settle_frames: int = 8        # quadros de acomodação depois de uma sacada


@dataclass
class CalibrationConfig:
    # 3 colunas x 4 linhas (12 alvos, ~25-30 s). Nas 40 sessões gravadas
    # (grade 4x4), tirar uma COLUNA deu a mesma validação que os 16 alvos
    # (148 px; 79-80 sem o desvio constante); tirar uma LINHA piorou ~5%
    # (155-157 px): o eixo vertical tem o sinal mais fraco e precisa das 4
    # linhas (docs/metodologia.md, seção 5.4).
    grid_cols: int = 3
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
    retry_bad_targets: bool = True
    retry_max_targets: int = 3
    retry_ratio: float = 2.5
    retry_min_frac: float = 0.08
    # Ao abrir o app com uma calibração salva, recentraliza (4 pontos) antes
    # de usar: reancora o desvio da nova sessão.
    recenter_on_start: bool = True
    # Conferência logo depois de calibrar (4 pontos, ~9 s): mede e corrige o
    # desvio constante que aparece nos primeiros segundos (encolhido pela
    # incerteza; procedures.shrink_offset). Nas 48 validações gravadas, o erro
    # nos alvos seguintes caiu de 105 para 84 px (metodologia, seção 7.1).
    check_after_calibration: bool = True
    # Grau máximo da correção monitor virtual -> real (calibration.py):
    # 1 = até afim; 2 = também quadrática. A validação cruzada escolhe.
    polynomial_degree: int = 2
    outlier_mad_threshold: float = 3.5
    save_path: Path = PROJECT_ROOT / "models" / "calibration.json"
    # Medidas brutas de cada calibração (.npz), para reanálise offline
    # (tools/reavaliar.py).
    results_dir: Path = PROJECT_ROOT / "results"

    # Calibração manual (tecla C): cada alvo só avança quando o usuário,
    # olhando para ele, aperta ESPAÇO/ENTER (útil quando um operador conduz
    # a calibração).
    manual: bool = False

    # Recentralização (R): desvios maiores que esta fração da tela (em cada
    # eixo) não são aplicados; a mensagem pede para recalibrar.
    recenter_max_frac: float = 0.25
    # Critério de aceitação da calibração (como no Tobii e no EyeLink): com o
    # erro em pontos novos (LOO) acima desta fração da largura da tela, o
    # resultado recomenda refazer (ESPAÇO refaz na hora, sem salvar). Nas 35
    # sessões com validação gravada, calibrações com LOO <= 90 px (4,7%)
    # erraram 76 px na validação; acima disso, 123 px. Repetir só os pontos
    # ruins não mudava a validação (108 -> 107 px), e juntar duas calibrações
    # seguidas piorava (90 -> 108 px): refazer substitui a anterior.
    redo_loo_frac: float = 0.047
    # Alvos da recentralização (frações da tela). O desvio é a MÉDIA dos
    # erros nesses alvos: com 1 ponto a correção piorava o erro nas
    # validações gravadas (138 -> 147 px); com 3, 96 px; com 5, 91 px.
    recenter_points: tuple = ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))
    # Etapa 2 da calibração (refinamento): depois da grade, estes alvos
    # internos (frações da tela) e o ajuste usa TODOS os pontos juntos, como
    # o "calibrar, validar e recalibrar" do EyeLink/Tobii. Nas 33 sessões com
    # validação gravada, somar 4 alvos internos à grade reduziu o erro nos
    # demais alvos de 105 para 85 px (−19%); com 8, para 86 px. Vazio = só
    # a grade.
    refine_points: tuple = ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))

    # Calibração por perseguição (Pfeuffer et al., 2013; --perseguicao): um
    # ponto percorre a tela e a pessoa o segue com os olhos; cada quadro vira
    # uma amostra com o seu alvo. "nenhuma" (padrão até haver dados que
    # mostrem ganho), "soma" (grade -> perseguição pelo perímetro -> etapa 2)
    # ou "alternativa" (perseguição pelo perímetro e pelas linhas do meio +
    # os 5 ``pursuit_points`` fixos, no lugar da grade).
    pursuit: str = "nenhuma"
    pursuit_points: tuple = ((0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))
    pursuit_speed_px_s: float = 420.0   # ~7 graus/s a 60 cm: o olho acompanha sem sacadas
    pursuit_hold_s: float = 1.0         # parado no início (o olho chega)
    pursuit_skip_s: float = 0.4         # descarta o começo do movimento
    pursuit_corner_skip_s: float = 0.25  # e o trecho logo depois de cada canto
    pursuit_delay_s: float = 0.1        # atraso do olho em relação ao ponto
    pursuit_jump_deg: float = 2.0       # salto entre quadros = sacada (descarta)
    # Peso TOTAL da perseguição no ajuste, em "alvos fixos" (cada um pesa 1).
    pursuit_weight: float = 6.0

    # Teste da cabeça (tecla K, ou depois da validação imediata com
    # ``--cabeca``): o olhar fica no centro e a pessoa gira a cabeça seguindo
    # as setas. As amostras NÃO entram no ajuste: medem o erro com a cabeça
    # em poses novas. Saiu do modo pesquisa em 17/11: alongava a sessão em
    # ~10 s, o efeito passageiro dos giros contaminava o que vinha depois, a
    # etapa é inadequada para quem não move a cabeça, e treinar com ela
    # melhorava a validação em só ~3 px.
    head_stage: bool = False
    # Quanto girar em cada direção (yaw, pitch em graus, em relação à pose
    # da grade). A seta enche conforme a cabeça gira e fica verde ao chegar
    # a ``head_reach_frac`` do alvo; depois de ``head_hold_s`` ali, passa
    # para a próxima direção. Quem não alcança segue após o tempo limite.
    head_target_deg: tuple = (10.0, 8.0)
    head_reach_frac: float = 0.8
    head_hold_s: float = 0.6
    head_step_timeout_s: float = 4.0
    head_center_s: float = 1.0        # etapa final com a cabeça de volta ao centro
    head_motion_settle_s: float = 0.8
    # Faixa usada no teste (amostras além dela são descartadas).
    head_max_deg: tuple = (15.0, 12.0)

    # Largura física da tela (cm): tamanho do MONITOR VIRTUAL onde o olhar é
    # projetado (geometry.VirtualScreen). Preenchida por --screen-width-cm ou
    # detectada no Windows. Sem ela, usa-se ``assumed_screen_width_cm``
    # (34,5 cm = notebook de 15,6" 16:9) e a correção monitor virtual -> real
    # absorve a diferença de escala.
    screen_width_cm: float | None = None
    assumed_screen_width_cm: float = 34.5


@dataclass
class ValidationConfig:
    # Validação IMEDIATA, depois da calibração e da conferência (mede o erro
    # no começo do uso); ligada no modo pesquisa. A tecla T faz o RETESTE,
    # mais tarde: mede o erro e corrige o desvio médio dos alvos (encolhido
    # pela incerteza, procedures.shrink_offset).
    after_calibration: bool = False
    # Alvos de validação (frações da tela), diferentes dos de calibração
    # (inclusive os da etapa 2, em 30/70%); visitados em ordem aleatória
    # (procedures.ValidationProcedure). Até 16/11 os 4 diagonais ficavam em
    # 30/70%; agora em 20/80%, mais perto das bordas.
    points: tuple = (
        (0.5, 0.5),
        (0.2, 0.2), (0.8, 0.2),
        (0.2, 0.8), (0.8, 0.8),
        (0.5, 0.2), (0.5, 0.8),
        (0.15, 0.5), (0.85, 0.5),
    )
    settle_s: float = 0.9
    collect_s: float = 1.2
    # Revalidação periódica: a cada N minutos de uso, os mesmos alvos da
    # validação medem a deriva e corrigem o desvio constante (0 = desligada;
    # --revalidar-min; 10 no modo pesquisa).
    revalidate_every_min: float = 0.0
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
    # Liga o controle do mouse sozinho depois de uma calibração (ou da
    # recentralização ao abrir o app): o uso não depende do teclado. A barra
    # de controle pelo olhar pausa e religa.
    enable_after_calibration: bool = True


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
    # Correção contínua por alvos conhecidos: o offset anda ``gain`` do
    # caminho até o erro medido, limitado a ``max_step_px`` por correção.
    # Nas validações gravadas, com os alvos chegando um a um, esse passo
    # levou o erro de 138 para 101 px (94 px depois de 7 alvos); usar só o
    # último alvo dava 116 px.
    gain: float = 0.3
    max_step_px: float = 80.0
    window_frames: int = 15        # amostras recentes usadas (~0,5 s)
    # Fontes de alvos conhecidos durante o uso. Num clique pelo olhar, o
    # clique cai onde o cursor (a própria estimativa) está e não ensina nada;
    # mas, com o ajuste fino pela cabeça (tecla A), o cursor foi corrigido
    # pela cabeça antes do clique: o ponto final é o alvo, e a correção feita
    # é o erro do olhar naquele momento.
    learn_from_fine_clicks: bool = True
    fine_min_px: float = 15.0      # ajuste menor que isto não conta (balanço da cabeça)


@dataclass
class FineConfig:
    # Ajuste fino pela cabeça (tecla A; filters.HeadFineTuner): durante uma
    # fixação, pequenos giros da cabeça deslocam o cursor a partir do ponto
    # do olhar. A cabeça é medida com ~10x mais precisão que o olhar.
    enabled: bool = False
    gain_px_per_deg: float = 25.0   # o "físico" seria ~38 px/grau; menor = mais preciso
    deadzone_deg: float = 0.3       # balanço natural da cabeça não move o cursor
    max_px: float = 250.0           # ajuste máximo a partir do ponto do olhar
    # Giro maior que isto = mudança de postura (zera). Era 6°: a 25 px/grau,
    # limitava o ajuste a ~140 px, menos que o erro do olhar (na sessão de
    # 16/11, ajustes de até 183 px); 10° cobre o ajuste máximo (max_px).
    reset_deg: float = 10.0
    # Modo de precisão: cursor = âncora + ajuste da cabeça. Sem ajuste da
    # cabeça em curso, a âncora acompanha o olhar (cursor igual ao modo
    # normal). Com ajuste em curso, o olhar só move a âncora quando está a
    # mais de ``warp_frac`` da largura da tela do CURSOR: na hora, numa nova
    # fixação; ou depois de ``warp_hold_s`` longe, na mesma fixação.
    # Até 17/11 o olhar só movia a âncora em saltos > 20% (384 px): na
    # sessão de 17/11 15:45, ele ficou 13 vezes >= 0,5 s a 150-360 px do
    # cursor sem que o cursor se movesse (nenhuma vez fora do modo de
    # precisão); agora, 1 vez (162 px). 10% (192 px) fica acima do erro do
    # olhar (mediana 48 px, 90% abaixo de 158 px no modo normal).
    warp_frac: float = 0.1
    warp_hold_s: float = 0.3


@dataclass
class DwellClickConfig:
    enabled_at_start: bool = False
    dwell_time_s: float = 1.0
    tolerance_px: int = 60
    # No ajuste fino (tecla A), o clique só sai com o cursor PARADO: com 60 px
    # de tolerância, os 24 cliques feitos com ajuste fino na sessão de 16/11
    # saíram com o ajuste ainda em andamento.
    fine_tolerance_px: int = 12
    cooldown_s: float = 1.0


@dataclass
class BarConfig:
    # Barra de controle pelo olhar (gaze_bar.py): coluna de botões na borda
    # ESQUERDA, centralizada na altura, durante o uso; liga/desliga o mouse,
    # o clique, o modo de precisão, e recentraliza ou recalibra, sem teclado.
    # Botões maiores que o erro típico do olhar (~70-120 px), colados na borda.
    enabled: bool = True
    dwell_s: float = 1.2           # olhar para um botão por este tempo o aciona
    cooldown_s: float = 1.5
    height_px: int = 100
    width_px: int = 170
    gap_px: int = 12


@dataclass
class DisplayConfig:
    # Com o mouse sendo controlado (e nenhum procedimento), a janela da
    # câmera é redesenhada a cada N quadros (10 fps): libera ~6 ms por quadro.
    preview_every_in_use: int = 3
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
    bar: BarConfig = field(default_factory=BarConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
