# Eye Gaze por webcam: controle do mouse pelo olhar

Programa em Python para pessoas com deficiência motora controlarem o cursor
com os olhos, usando só uma webcam (interna ou externa). Os landmarks do rosto
e das íris vêm do **MediaPipe FaceLandmarker** (Tasks API). O olhar é modelado
geometricamente: um raio sai da posição 3D do olho, na direção dada pela
rotação da cabeça somada à rotação do globo ocular, e é intersectado com o
plano da tela. A calibração tem 16 pontos (grade 4×4) e uma etapa curta de movimento da
cabeça. Com isso, a pessoa pode girar ou deslocar a cabeça para uma posição
confortável sem perder a pontaria. Um estabilizador de fixação mantém o
cursor parado enquanto o olhar está parado.

A geometria do olho parte do projeto *Webcam3DTracker* (esfera do olho +
vetor centro→íris), reescrita para a API atual do MediaPipe. O
`mp.solutions.face_mesh` usado por ele foi removido das versões recentes do
pacote.

## Instalação

Requer Python 3.10 ou mais recente.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

O modelo `face_landmarker.task` é baixado sozinho para `models/`
na primeira execução. Sem internet, baixe-o de
<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
e salve em `models/face_landmarker.task`.

> **Linux:** o `pynput` move o mouse em sessões X11. No Wayland, entre numa
> sessão "Xorg" ou rode via XWayland.
> **macOS:** dê permissão de *Acessibilidade* e de *Câmera* ao terminal.

## Uso

```bash
python main.py                      # webcam 0
python main.py --camera 1           # outra webcam
python main.py --video sessao.mp4   # roda sobre um vídeo gravado
python main.py --screen 1920x1080   # se a resolução não for detectada
python main.py --screen-width-cm 53 --distance-cm 60   # largura física da tela (fora do Windows) e erro em graus
python main.py --gravar             # grava vídeo + teclas (results/sessao_*.avi/.json)
python main.py --reproduzir results/sessao_X.json --sem-janela   # refaz a sessão offline
python main.py --help
```

Fluxo recomendado:

1. Sente-se como vai usar o computador, com o rosto bem iluminado e de frente
   para a câmera.
2. Pressione **ESPAÇO** e olhe para cada um dos 16 alvos, que aparecem em
   ordem aleatória. O círculo laranja encolhe enquanto o olho chega ao alvo.
   A coleta (ponto verde com anel de progresso) só começa quando o olhar
   estabiliza no alvo. Na **calibração manual** (tecla **C**), cada alvo só
   avança quando você, olhando para ele, aperta **ESPAÇO** (ou ENTER). São
   usadas as amostras do segundo **antes** da tecla, porque ao apertar é
   comum piscar ou olhar para o teclado. Se o olhar ainda estava se movendo,
   a tela pede para tentar de novo.
3. **Etapa da cabeça (~10 s):** um ponto verde fica no centro. Sem tirar os
   olhos dele, gire a cabeça na direção da seta (direita, esquerda, cima,
   baixo) até a seta ficar **verde** (~10° para os lados, ~8° para cima e
   para baixo) e depois volte ao centro. Não precisa girar mais que isso:
   se a seta ficar vermelha, você girou demais. É essa etapa que ensina ao
   modelo quanto do movimento vem dos olhos e quanto vem da cabeça. Quem não
   consegue mover a cabeça pode só continuar olhando o ponto (cada direção
   avança sozinha em 4 s).
4. **Etapa da boca (~4 s):** continue olhando o ponto central e abra e feche
   a boca devagar algumas vezes (a barra mostra a abertura). O MediaPipe
   deforma a malha inteira do rosto quando a boca abre, e sem esta etapa o
   cursor descia com a boca aberta. A etapa ensina ao modelo quanto
   descontar. Quem não puder abrir a boca pode só continuar olhando (segue
   sozinha em 7 s).
5. Alvos com erro muito alto (piscada longa, distração) são repetidos
   automaticamente, uma vez.
6. A tela de resultado mostra o erro em cada alvo, o erro estimado em pontos
   novos e em poses novas da cabeça (validação cruzada) e uma classificação
   da qualidade (boa / razoável / ruim).
7. Pressione **T** para validar: 9 alvos novos, que não entram na
   calibração, medem a acurácia e a precisão. O resultado vai para
   `results/` e mostra também quanto do erro é um desvio constante, que o
   **R** corrige.
8. Pressione **M** para ligar o mouse e **D** para ligar o clique por
   permanência (*dwell-click*).
9. Se o HUD mostrar **"Postura fora da calibração"** (você se mexeu ou
   girou a cabeça além do que foi calibrado) ou o cursor ficar
   sistematicamente deslocado, pressione **R** e olhe para o centro
   (recentralização de 1 ponto).

### Coletando dados para melhorar o modelo

Para cada sessão de teste, use `python main.py --gravar`, calibre (ESPAÇO),
valide (T) e saia (Q). Depois:

```bash
python tools/benchmark.py            # compara variantes do modelo em todas as sessões
python main.py --reproduzir results/sessao_X.json --sem-janela   # refaz com o código atual
python tools/benchmark.py --results results/reproducoes/sessao_X
```

O benchmark mede o erro em alvos novos, em poses novas da cabeça e na
validação de cada sessão, com o código atual. A reprodução refaz a sessão
inteira a partir do vídeo, o que permite testar mudanças no estimador do
olhar (ex.: `TrackerConfig.refine_iris`) com dados reais.

A calibração é salva em `models/calibration.json` e recarregada
automaticamente. Ela é descartada se a resolução da tela mudar.

### Teclas

| Tecla | Ação |
|---|---|
| `ESPAÇO` | calibrar, automática (16 pontos + etapas da cabeça e da boca) |
| `C` | calibrar, manual: cada alvo avança com ESPAÇO/ENTER enquanto você olha para ele |
| `T` | validar (mede o erro em alvos novos e salva CSV/JSON em `results/`) |
| `R` | recentralizar (corrige deslocamento com 1 ponto) |
| `M` | liga/desliga o controle do mouse (exige calibração) |
| `D` | liga/desliga o dwell-click |
| `A` | liga/desliga o **ajuste fino pela cabeça**: o olhar leva o cursor à região e, enquanto você continua olhando, pequenos giros da cabeça (direita/esquerda/cima/baixo) movem o cursor a partir dali com precisão de poucos pixels. Um novo salto do olhar zera o ajuste. Parâmetros em `FineConfig` |
| `L` | liga/desliga a visualização dos **landmarks** (malha, contorno dos olhos, íris, cantos) |
| `G` | liga/desliga a visualização do **olhar** (globos oculares, raio de cada olho, raio combinado, ângulos) |
| `P` | liga/desliga os **eixos da cabeça** e os ângulos de pose |
| `V` | liga/desliga o **minimapa** da tela com o ponto estimado em tempo real |
| `3` | liga/desliga a **vista 3D** (janela própria): rosto, globos oculares, raio de cada olho, plano da tela com o cursor (amarelo), o raio geométrico (magenta; calibrado depois da calibração, cru antes) e a direção do rosto corrigida (verde). Arraste para girar, use a roda para zoom e o botão direito para voltar à vista inicial. Informe `--screen-width-cm` para a tela ficar em escala |
| `F` | espelha ou não a imagem exibida (o processamento não muda) |
| `H` | mostra/oculta a ajuda |
| `ESC` | cancela calibração/validação em andamento; fora delas, sai |
| `Q` | sai |

As teclas funcionam com a janela da câmera (ou a de alvos) em foco. Os
valores iniciais de cada visualização ficam em `DisplayConfig`
(`eyegaze/config.py`).

### Segurança do controle do mouse

- O mouse só liga depois de uma calibração.
- Se o rosto não for detectado num frame, o cursor fica parado. Depois de
  0,5 s sem rosto o HUD mostra "PAUSADO" e os filtros são reiniciados.
- Durante piscadas (os dois olhos fechados) o cursor mantém a última posição.
- Com `--reproduzir`, o mouse nunca é movido.

## Como funciona (resumo)

```
webcam -> FaceLandmarker (478 landmarks + matriz de pose)
       -> rotação da cabeça R e posição 3D do rosto E (cm)
       -> íris (u, v) e pálpebras (altura da pálpebra superior lid_up,
          abertura) no referencial da cabeça
       -> calibração: ponto para onde a face aponta (E + E_z·tan(cabeça))
                      + medidas do olho escaladas pela distância
                      -> regressão ridge livre ou modelo físico (px/cm
                         compartilhado), escolhidos por validação cruzada
                         em alvos e em direções da cabeça -> (x, y) em pixels
       -> aviso de postura fora da calibração (R recentraliza)
       -> estabilizador de fixação (cursor parado na fixação, salta na sacada)
       -> mouse + dwell-click
```

A explicação completa, com as equações e as decisões de projeto, está em
[`docs/metodologia.md`](docs/metodologia.md).

## Estrutura

```
main.py                    # ponto de entrada (argumentos de linha de comando)
eyegaze/
  config.py                # todos os parâmetros ajustáveis
  landmarks.py             # índices dos landmarks usados
  face_tracker.py          # MediaPipe FaceLandmarker (modo VIDEO) + download do modelo
  head_pose.py             # rotação da cabeça (matriz do modelo, com reserva por landmarks)
  gaze.py                  # modelo do globo ocular, posição 3D do rosto, piscadas
  filters.py               # One-Euro Filter e estabilizador de fixação com trava
  calibration.py           # regressão: geometria da cabeça + medidas do olho/pálpebras -> pixel
  procedures.py            # calibração (com etapa da cabeça), validação e recentralização
  visualization.py         # landmarks, olhar, pose, minimapa, HUD, ajuda
  scene3d.py               # vista 3D: rosto, globos oculares, raios do olhar, tela
  io_devices.py            # câmera/vídeo, gravação/reprodução de sessões, tela, mouse, dwell-click
  app.py                   # loop principal e teclas
tests/
  synthetic.py             # rosto 3D sintético (testes sem câmera)
  test_pipeline.py         # geometria, pose, olhar, calibração, invariância à cabeça, tremor
  test_app.py              # ponta a ponta com "usuário simulado"
  test_procedures.py       # protocolo de coleta, repetição de alvos, aviso de postura
  test_recording.py        # gravação e reprodução de sessões
docs/metodologia.md        # método detalhado (base para o texto do TCC)
models/                    # face_landmarker.task e calibration.json (gerados)
results/                   # validações (CSV/JSON), medidas brutas (.npz) e sessões gravadas
tools/benchmark.py         # compara variantes do modelo em TODAS as sessões gravadas
tools/reavaliar.py         # detalha uma sessão (sinal de cada medida, erro por alvo)
```

## Testes

```bash
pip install pytest
python -m pytest -q tests
```

Os testes não usam câmera nem o modelo. Eles geram rostos 3D sintéticos com
pose e olhar conhecidos e conferem:

- que os índices de cada olho pertencem ao mesmo olho;
- que, com a etapa da cabeça, o cursor continua no lugar certo com a cabeça
  girada (±15° horizontal, ±10° vertical, roll) ou deslocada. O "usuário"
  sintético tem os erros típicos de webcam: ganho do olho diferente do
  assumido, movimento vertical da íris achatado e acoplamento com a
  inclinação da cabeça;
- que o estabilizador elimina o tremor na fixação sem atrasar as sacadas, e
  mantém o cursor parado quando a estimativa passeia devagar (ruído
  correlacionado, como nos dados reais);
- que a calibração generaliza para pontos fora da grade;
- o fluxo completo do app: calibrar, validar, recentralizar, mouse,
  dwell-click, perda de rosto e todas as teclas de visualização;
- o protocolo de coleta: ordem aleatória, espera pela fixação, etapa da
  cabeça para quem não move a cabeça e para quem gira demais, repetição de
  alvos ruins, aviso de postura e correção por alvos conhecidos;
- que uma sessão gravada é reproduzida com a mesma ordem de alvos.

## Ajustes comuns (`eyegaze/config.py`)

| Sintoma | Ajuste |
|---|---|
| Cursor anda sozinho com o olhar parado | aumentar `StabilizerConfig.lock_ratio` (ex.: 0.05) ou `shift_frames` (ex.: 12) |
| Cursor não acompanha pequenos deslocamentos do olhar | diminuir `StabilizerConfig.lock_ratio` (ex.: 0.025) ou `shift_frames` (ex.: 5) |
| Cursor salta sozinho | aumentar `StabilizerConfig.saccade_ratio` (ex.: 0.10) ou `confirm_frames` (ex.: 4) |
| Cursor demora a chegar após olhar para outro lugar | diminuir `confirm_frames` (mín. 2) ou aumentar `follow` |
| Cursor não desce o suficiente ao inclinar a cabeça | confira se a largura da tela foi detectada (linha `[Tela]` no terminal); se não, informe `--screen-width-cm` |
| Cursor desloca quando gira a cabeça | refazer a calibração caprichando na etapa da cabeça (ver o erro com a cabeça em movimento na tela de resultado) |
| Cliques acidentais | aumentar `DwellClickConfig.dwell_time_s` ou diminuir `tolerance_px` |
| Piscadas não detectadas / falsas | ajustar `GazeConfig.blink_ratio` |
| Pular a etapa da cabeça | `CalibrationConfig.head_stage = False` |
| Etapa da cabeça pede giro demais/de menos | `CalibrationConfig.head_target_deg` (padrão 10°, 8°) |
| Coleta demora a começar em cada alvo | aumentar `CalibrationConfig.fixation_max_deg` ou diminuir `settle_max_s` |
| Aviso de postura aparece demais | aumentar `PostureConfig.angle_tol_deg` / `pos_tol_cm` |
| Alvos muito perto da borda | aumentar `CalibrationConfig.margin_ratio` |
| Webcam errada | `--camera N` |

O HUD da janela da câmera mostra o ruído estimado do ponto na tela
(`ruido ~N px`). No minimapa (`V`), o ponto cinza é a estimativa bruta e o
amarelo é o cursor estabilizado. A diferença entre os dois mostra o tremor
que está sendo removido.

## Limitações conhecidas

- **Posturas extremas.** A compensação de cabeça é aprendida dentro da
  faixa de movimento feita na etapa da cabeça (até ~15°/12°). Posturas
  muito além dela (girar a cabeça 40°, por exemplo) extrapolam o modelo, e
  nesses ângulos os landmarks do olho também deixam de ser confiáveis. O
  HUD avisa quando a postura sai da faixa calibrada; nesse caso, volte à
  postura ou recentralize (R).
- **Posição 3D do rosto.** A distância à câmera vem da estimativa métrica do
  MediaPipe, que assume um campo de visão padrão para a câmera. Webcams com
  lente muito diferente geram um pequeno erro de escala. A calibração
  absorve a maior parte dele, e `R` corrige o resto.
- **Eixo vertical.** O eixo vertical é menos preciso que o horizontal: a íris
  se desloca menos e as pálpebras a cobrem parcialmente. Isso é comum em
  rastreadores por webcam. O ganho da cabeça no eixo vertical é fixado pela
  geometria quando a largura física da tela é conhecida (detectada no
  Windows ou `--screen-width-cm`): abaixar a cabeça para olhar para baixo
  move o cursor o quanto deve. Com a cabeça parada, as medidas do olho
  saturam perto da borda de baixo; abaixo da última linha de alvos (92% da
  altura) o alcance é menor.
- **Óculos e iluminação.** Reflexos nas lentes e contraluz pioram os
  landmarks da íris.
- **Acurácia.** Rastreadores por webcam costumam ter erro de alguns graus de
  ângulo visual, bem acima de rastreadores infravermelhos dedicados. Use
  alvos grandes na interface (botões grandes, teclado virtual com teclas
  grandes). Meça com `T` em cada sessão de testes.

## Próximos passos sugeridos

1. Indicador visual do dwell-click sobre a área de trabalho (janela
   transparente sempre no topo).
2. Perfis por usuário (um `calibration.json` por pessoa).
3. Modos de clique alternativos: piscada longa voluntária ou acionador
   externo.
4. Teclado virtual com teclas grandes, usando `EyeGazeApp.confirm_target`
   (já implementado) a cada tecla selecionada: o centro da tecla é um alvo
   conhecido, e o offset é corrigido continuamente. Com o mouse do sistema
   isso não é possível, porque o app não sabe onde estão os botões.
5. Teste com usuários: aplicar o protocolo de validação (`T`) em várias
   pessoas e condições de luz e reportar acurácia e precisão em graus.
