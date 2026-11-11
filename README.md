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
python main.py --screen-width-cm 53 --distance-cm 60   # erro em graus na validação
python main.py --help
```

Fluxo recomendado:

1. Sente-se como vai usar o computador, com o rosto bem iluminado e de frente
   para a câmera.
2. Pressione **ESPAÇO** e olhe para cada um dos 16 alvos. O círculo laranja
   encolhe enquanto o olho chega ao alvo; o ponto verde com o anel de
   progresso indica que a coleta está em andamento.
3. **Etapa da cabeça (~9 s):** um ponto verde fica no centro. Sem tirar os
   olhos dele, gire a cabeça devagar na direção das setas (esquerda, direita,
   cima, baixo, centro), só o quanto for confortável. É essa etapa que ensina
   ao modelo quanto do movimento vem dos olhos e quanto vem da cabeça. Quem
   não consegue mover a cabeça pode só continuar olhando o ponto; o modelo
   então usa parâmetros padrão para o olho.
4. A tela de resultado mostra o erro em cada alvo, o erro durante a etapa da
   cabeça e o erro estimado em pontos novos (validação cruzada deixando um
   alvo de fora).
5. Pressione **T** para validar: 9 alvos novos, que não entram na
   calibração, medem a acurácia e a precisão. O resultado vai para
   `results/`.
6. Pressione **M** para ligar o mouse e **D** para ligar o clique por
   permanência (*dwell-click*).
7. Se o cursor ficar sistematicamente deslocado (por exemplo, depois de
   uma mudança grande de postura), pressione **R** e olhe para o centro
   (recentralização de 1 ponto).

A calibração é salva em `models/calibration.json` e recarregada
automaticamente. Ela é descartada se a resolução da tela mudar.

### Teclas

| Tecla | Ação |
|---|---|
| `ESPAÇO` ou `C` | calibrar (16 pontos + etapa da cabeça) |
| `T` | validar (mede o erro em alvos novos e salva CSV/JSON em `results/`) |
| `R` | recentralizar (corrige deslocamento com 1 ponto) |
| `M` | liga/desliga o controle do mouse (exige calibração) |
| `D` | liga/desliga o dwell-click |
| `L` | liga/desliga a visualização dos **landmarks** (malha, contorno dos olhos, íris, cantos) |
| `G` | liga/desliga a visualização do **olhar** (globos oculares, raio de cada olho, raio combinado, ângulos) |
| `P` | liga/desliga os **eixos da cabeça** e os ângulos de pose |
| `V` | liga/desliga o **minimapa** da tela com o ponto estimado em tempo real |
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

## Como funciona (resumo)

```
webcam -> FaceLandmarker (478 landmarks + matriz de pose)
       -> rotação da cabeça R e posição 3D do rosto E (cm)
       -> íris (u, v) e pálpebras (lid_v, abertura) no referencial da cabeça
       -> calibração: ponto para onde a face aponta (E + E_z·tan(cabeça))
                      + medidas do olho escaladas pela distância
                      -> regressão ridge (validação cruzada) -> (x, y) em pixels
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
  filters.py               # One-Euro Filter e estabilizador de fixação (anti-tremor)
  calibration.py           # regressão: geometria da cabeça + medidas do olho/pálpebras -> pixel
  procedures.py            # calibração (com etapa da cabeça), validação e recentralização
  visualization.py         # landmarks, olhar, pose, minimapa, HUD, ajuda
  io_devices.py            # câmera/vídeo, resolução da tela, mouse (pynput), dwell-click
  app.py                   # loop principal e teclas
tests/
  synthetic.py             # rosto 3D sintético (testes sem câmera)
  test_pipeline.py         # geometria, pose, olhar, calibração, invariância à cabeça, tremor
  test_app.py              # ponta a ponta com "usuário simulado"
docs/metodologia.md        # método detalhado (base para o texto do TCC)
models/                    # face_landmarker.task e calibration.json (gerados)
results/                   # validações (CSV/JSON) e medidas brutas (.npz) das sessões
tools/reavaliar.py         # reajusta o modelo offline com os .npz e compara variantes
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
- que o estabilizador elimina o tremor na fixação sem atrasar as sacadas;
- que a calibração generaliza para pontos fora da grade;
- o fluxo completo do app: calibrar, validar, recentralizar, mouse,
  dwell-click, perda de rosto e todas as teclas de visualização.

## Ajustes comuns (`eyegaze/config.py`)

| Sintoma | Ajuste |
|---|---|
| Cursor ainda tremendo | aumentar `StabilizerConfig.window_frames` (ex.: 30) ou `deadzone_px` (ex.: 10) |
| Cursor salta sozinho | aumentar `StabilizerConfig.saccade_ratio` (ex.: 0.10) ou `confirm_frames` (ex.: 4) |
| Cursor demora a chegar após olhar para outro lugar | diminuir `confirm_frames` (mín. 2) ou aumentar `follow` |
| Cursor desloca quando gira a cabeça | refazer a calibração caprichando na etapa da cabeça (ver o erro com a cabeça em movimento na tela de resultado) |
| Cliques acidentais | aumentar `DwellClickConfig.dwell_time_s` ou diminuir `tolerance_px` |
| Piscadas não detectadas / falsas | ajustar `GazeConfig.blink_ratio` |
| Pular a etapa da cabeça | `CalibrationConfig.head_motion_step_s = 0` |
| Alvos muito perto da borda | aumentar `CalibrationConfig.margin_ratio` |
| Webcam errada | `--camera N` |

O HUD da janela da câmera mostra o ruído estimado do ponto na tela
(`ruido ~N px`). No minimapa (`V`), o ponto cinza é a estimativa bruta e o
amarelo é o cursor estabilizado. A diferença entre os dois mostra o tremor
que está sendo removido.

## Limitações conhecidas

- **Posturas extremas.** A compensação de cabeça é aprendida dentro da
  faixa de movimento feita na etapa da cabeça. Posturas muito além dela
  (girar a cabeça 40°, por exemplo) extrapolam o modelo. Faça a etapa da
  cabeça cobrindo as posições que a pessoa realmente usa.
- **Posição 3D do rosto.** A distância à câmera vem da estimativa métrica do
  MediaPipe, que assume um campo de visão padrão para a câmera. Webcams com
  lente muito diferente geram um pequeno erro de escala. A calibração
  absorve a maior parte dele, e `R` corrige o resto.
- **Eixo vertical.** O eixo vertical é menos preciso que o horizontal: a íris
  se desloca menos e as pálpebras a cobrem parcialmente. Isso é comum em
  rastreadores por webcam.
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
4. Recalibração contínua: usar os cliques confirmados (o usuário olhou para
   onde clicou) como novas amostras de calibração.
5. Teste com usuários: aplicar o protocolo de validação (`T`) em várias
   pessoas e condições de luz e reportar acurácia e precisão em graus.
#   T C C _ E y e _ G a z e  
 