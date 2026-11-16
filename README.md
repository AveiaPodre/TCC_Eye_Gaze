# Eye Gaze por webcam: controle do mouse pelo olhar

Programa em Python para pessoas com deficiência motora controlarem o cursor
com os olhos, usando só uma webcam (interna ou externa). Os landmarks do rosto
e das íris vêm do **MediaPipe FaceLandmarker** (Tasks API). O olhar é modelado
geometricamente, em cinco etapas:

1. landmarks do rosto;
2. centro do globo ocular e centro da pupila de cada olho;
3. feixe de cada olho (do centro do globo, passando pela pupila);
4. cada feixe é projetado num monitor virtual (o plano da tela, com a
   geometria suposta), e o ponto do olhar é a média dos dois pontos,
   ponderada pela qualidade de cada olho;
5. a calibração com vários pontos (grade 3×4, ~30 s) ajusta o globo de cada
   olho e a correção entre o monitor virtual e o monitor real.

Como a rotação e a posição da cabeça entram pela geometria, a pessoa pode
girar ou deslocar a cabeça sem perder toda a pontaria. Um estabilizador de
fixação mantém o cursor parado enquanto o olhar está parado.

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
python main.py --camera-fov 75      # campo de visão horizontal da webcam (padrão 90°, notebook)
python main.py --pesquisa           # modo pesquisa: etapa da cabeça (teste) + validação imediata
python main.py --gravar             # grava vídeo + teclas (results/sessao_*.avi/.json); implica --pesquisa
python main.py --width 1920 --height 1080 --sem-autofoco --exposicao -6   # captura (ver "Câmera")
python main.py --reproduzir results/sessao_X.json --sem-janela   # refaz a sessão offline
python main.py --help
```

Fluxo recomendado:

1. Sente-se como vai usar o computador, com o rosto bem iluminado e de frente
   para a câmera.
2. Pressione **ESPAÇO** e olhe para cada um dos 12 alvos, que aparecem em
   ordem aleatória. O círculo laranja encolhe enquanto o olho chega ao alvo.
   A coleta (ponto verde com anel de progresso) só começa quando o olhar
   estabiliza no alvo. Na **calibração manual** (tecla **C**, útil quando um
   operador conduz a calibração), cada alvo só avança quando a pessoa,
   olhando para ele, aperta **ESPAÇO** (ou ENTER). São usadas as amostras do
   segundo **antes** da tecla, porque ao apertar é comum piscar ou olhar
   para o teclado. Se o olhar ainda estava se movendo, a tela pede para
   tentar de novo.
3. Alvos com erro muito alto (piscada longa, distração) são repetidos
   automaticamente, uma vez.
4. A tela de resultado mostra o erro em cada alvo, o erro estimado em pontos
   novos (validação cruzada), quanto desse erro sobra só com o monitor
   virtual (sem a correção da etapa 5) e uma classificação da qualidade
   (boa / razoável / ruim).
5. **Modo pesquisa** (`--pesquisa` ou `--gravar`): depois dos alvos vem a
   **etapa da cabeça (~10 s)**: um ponto verde fica no centro e, sem tirar os
   olhos dele, a pessoa gira a cabeça na direção da seta (direita, esquerda,
   cima, baixo) até a seta ficar **verde** (~10° para os lados, ~8° para
   cima e para baixo). Essas amostras **não** entram no ajuste: medem o
   erro com a cabeça em poses novas. Em seguida vem a **validação imediata**
   (9 alvos novos), que mede o erro do modelo, em px e em graus, por região
   da tela.
6. Mais tarde, a tecla **T** faz o **reteste**: mede o erro em 9 alvos e
   corrige no cursor o desvio médio entre eles. Os resultados vão para
   `results/`. Ao abrir o app com uma calibração salva, uma recentralização
   de 4 pontos roda sozinha antes do uso.
7. Pressione **M** para ligar o mouse e **D** para ligar o clique por
   permanência (*dwell-click*).
8. Se o HUD mostrar **"Postura fora da calibração"** (a pessoa se mexeu ou
   girou a cabeça além do que foi calibrado) ou o cursor ficar
   sistematicamente deslocado, pressione **R** e olhe para os 4 pontos
   (recentralização, ~9 s). Com um ponto só a correção piorava o erro; com
   4, ele cai ~30% (`docs/metodologia.md`, seção 7).
9. Com o **ajuste fino pela cabeça** (tecla **A**) e o dwell-click ligados,
   cada clique feito depois de corrigir o cursor com a cabeça ensina o
   desvio do olhar ao modelo (correção contínua, salva ao sair).

### Câmera

Por padrão o app pede **MJPG** à câmera (em 1080p, muitas webcams só chegam
a 30 fps nesse formato) e mostra no terminal o formato, o fps do driver e a
exposição. Com pouca luz, a exposição automática alonga cada quadro, o fps
cai e a íris borra: fixe a exposição com `--exposicao` (o valor depende do
driver; no DirectShow, de -1 a -13, menor = mais curta) e ilumine o rosto de
frente. `--sem-autofoco` evita que a nitidez do olho oscile. `--backend
dshow` ou `--backend msmf` trocam o backend de captura no Windows.

Os landmarks do MediaPipe são calculados num recorte do rosto de tamanho
fixo, então a resolução da câmera quase não muda a precisão deles; ela vale
para o refinamento da íris na imagem (`iris_refine.py`).


### Coletando dados para melhorar o modelo

Para cada sessão de teste, use `python main.py --gravar` (modo pesquisa),
calibre (ESPAÇO; a validação imediata vem em seguida), faça o reteste (T)
mais tarde e saia (Q). Depois:

```bash
python tools/benchmark.py            # compara variantes do modelo em todas as sessões
python tools/ablacao.py              # quantos alvos de calibração são necessários
python main.py --reproduzir results/sessao_X.json --sem-janela   # refaz com o código atual
python tools/benchmark.py --results results/reproducoes/sessao_X
```

As medidas recalculadas a partir dos landmarks ficam em cache em
`results/.cache` (refeito quando o código do estimador muda). A reprodução
só acompanha os alvos do vídeo em sessões gravadas com o protocolo atual
(o protocolo vai junto na gravação); as sessões gravadas até 15/11/2026
usavam a grade 4×4 com checagens do centro e etapa da boca, e servem ao
benchmark, não à reprodução.

O benchmark mede o erro em alvos novos, em poses novas da cabeça e na
validação de cada sessão, com o código atual. A reprodução refaz a sessão
inteira a partir do vídeo, o que permite testar mudanças no estimador do
olhar (ex.: `TrackerConfig.refine_iris`) com dados reais.

A calibração é salva em `models/calibration.json` e recarregada
automaticamente. Ela é descartada se a resolução da tela, o campo de visão
da câmera (`--camera-fov`) ou a versão das medidas mudar.

### Teclas

| Tecla | Ação |
|---|---|
| `ESPAÇO` | calibrar, automática (12 pontos; no modo pesquisa, + etapa da cabeça e validação imediata) |
| `C` | calibrar, manual: cada alvo avança com ESPAÇO/ENTER enquanto a pessoa olha para ele |
| `T` | reteste: mede o erro em 9 alvos novos, corrige o desvio médio no cursor e salva CSV/JSON em `results/` |
| `R` | recentralizar (corrige o desvio constante com 4 pontos) |
| `M` | liga/desliga o controle do mouse (exige calibração) |
| `D` | liga/desliga o dwell-click |
| `A` | liga/desliga o **ajuste fino pela cabeça**: o olhar leva o cursor à região e, enquanto você continua olhando, pequenos giros da cabeça (direita/esquerda/cima/baixo) movem o cursor a partir dali com precisão de poucos pixels. Um novo salto do olhar zera o ajuste. Parâmetros em `FineConfig` |
| `L` | liga/desliga a visualização dos **landmarks** (malha, contorno dos olhos, íris, cantos) |
| `G` | liga/desliga a visualização do **olhar**: para cada olho, o globo (0,4 × largura do olho), a pupila medida (verde), a pupila corrigida pelo ganho da calibração (anel laranja) e o feixe; vetor do olhar, ângulos e peso de cada olho |
| `P` | liga/desliga os **eixos da cabeça** e os ângulos de pose |
| `V` | liga/desliga o **minimapa** da tela com o ponto estimado em tempo real |
| `3` | liga/desliga a **vista 3D** (janela própria), com as etapas do pipeline: rosto, globo e pupila de cada olho, feixes até a tela (azul claro), ponto de cada olho no monitor virtual (branco), ponto do olhar no monitor virtual (magenta) e cursor no monitor real depois da calibração (amarelo). Arraste para girar, use a roda para zoom e o botão direito para voltar à vista inicial. Informe `--screen-width-cm` para a tela ficar em escala |
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
webcam -> 1. FaceLandmarker (478 landmarks + matriz de pose)
              -> rotação da cabeça R e posição 3D do rosto (cm)
       -> 2. por olho: âncora (cantos; vertical pela ponte do nariz) e pupila em cm;
              centro do globo = âncora + R·(c_x, c_y, s)
       -> 3. feixe centro do globo -> pupila de cada olho
       -> 4. cada feixe projetado no monitor virtual (plano da tela);
              média dos dois pontos, ponderada pelo erro de cada olho -> pixel virtual
       -> 5. correção monitor virtual -> real (calibração) -> pixel real
       -> aviso de postura fora da calibração (R recentraliza)
       -> estabilizador de fixação (cursor parado na fixação, salta na sacada)
       -> mouse + dwell-click
```

A calibração ajusta, por pessoa e por olho, a posição de repouso da pupila e
o raio efetivo do globo em cada eixo (o MediaPipe encolhe o movimento da
íris, mais na vertical), e depois a correção virtual -> real (nenhuma,
deslocamento, afim ou quadrática, escolhida por validação cruzada).

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
  gaze.py                  # etapas 1-2: âncora e pupila de cada olho em cm, piscadas
  eyeball.py               # etapas 2-4: centro do globo, feixes, ponto de cada olho, ajuste
  geometry.py              # câmera pinhole, intersecção de retas, monitor virtual (etapa 4)
  filters.py               # One-Euro Filter e estabilizador de fixação com trava
  calibration.py           # etapa 5: ajuste do globo + correção monitor virtual -> real
  procedures.py            # calibração (com etapa da cabeça), validação e recentralização
  visualization.py         # landmarks, olhar, pose, minimapa, HUD, ajuda
  scene3d.py               # vista 3D: globos, feixes, intersecção, monitor virtual, cursor
  io_devices.py            # câmera/vídeo, gravação/reprodução de sessões, tela, mouse, dwell-click
  app.py                   # loop principal e teclas
tests/
  synthetic.py             # rosto 3D sintético (testes sem câmera)
  test_pipeline.py         # geometria, pose, globo/feixes/intersecção, calibração, tremor
  test_scene3d.py          # vista 3D
  test_app.py              # ponta a ponta com "usuário simulado"
  test_procedures.py       # protocolo de coleta, repetição de alvos, aviso de postura
  test_recording.py        # gravação e reprodução de sessões
docs/metodologia.md        # método detalhado (base para o texto do TCC)
models/                    # face_landmarker.task e calibration.json (gerados)
results/                   # validações (CSV/JSON), medidas brutas (.npz) e sessões gravadas
tools/benchmark.py         # compara variantes do pipeline em TODAS as sessões gravadas
tools/reavaliar.py         # detalha uma sessão (globo ajustado, feixes, erro por alvo)
tools/ablacao.py           # quantos alvos de calibração são necessários (sessões gravadas)
```

## Testes

```bash
pip install pytest
python -m pytest -q tests -m "not slow"   # suíte rápida (~6 s)
python -m pytest -q tests                 # completa, com o app simulado (~5 min)
```

Os testes não usam câmera nem o modelo. Eles geram rostos 3D sintéticos com
pose e olhar conhecidos e conferem:

- que os índices de cada olho pertencem ao mesmo olho;
- a geometria: intersecção de retas, monitor virtual, medidas do olho em cm;
- que o ajuste do globo recupera o raio real de um olho sintético, que o
  feixe passa pela pupila e que os dois feixes se cruzam no alvo;
- que o cursor continua perto do lugar certo com a cabeça girada ou
  deslocada. O "usuário" sintético tem o erro típico do MediaPipe:
  movimento da íris encolhido, mais na vertical;
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
| Cursor desloca ao inclinar a cabeça / desvio vertical constante | recentralize (R); confira se a largura da tela foi detectada (linha `[Tela]` no terminal); se não, informe `--screen-width-cm` |
| Cursor desloca quando gira a cabeça | refazer a calibração caprichando na etapa da cabeça (ver o erro com a cabeça em movimento na tela de resultado) |
| Cliques acidentais | aumentar `DwellClickConfig.dwell_time_s` ou diminuir `tolerance_px` |
| Piscadas não detectadas / falsas | ajustar `GazeConfig.blink_ratio` |
| Medir o erro com a cabeça em poses novas | `--pesquisa` (liga a etapa da cabeça, que só serve de teste) |
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

- **Posturas extremas.** A rotação da cabeça entra pela geometria, mas
  posturas muito além da calibrada (girar a cabeça 40°, por exemplo)
  extrapolam o modelo, e nesses ângulos os landmarks do olho deixam de ser
  confiáveis. O HUD avisa quando a postura sai da faixa calibrada; nesse
  caso, volte à postura ou recentralize (R).
- **Posição 3D do rosto.** A distância à câmera vem de uma câmera pinhole
  com campo de visão suposto (90°, `--camera-fov`) e da distância média
  entre os cantos dos olhos (9 cm). Webcams com lente muito diferente geram
  um pequeno erro de escala; a calibração absorve a maior parte dele, e `R`
  corrige o resto.
- **Eixo vertical.** É a principal limitação medida (ver
  `docs/metodologia.md`, seção 5.6). A pupila do MediaPipe anda só ~25–30%
  do que um olho real andaria, então um deslocamento de décimos de
  milímetro da pupila (mudança de postura, abertura do olho) move o cursor
  bastante. A vertical do globo é ancorada na ponte do nariz, e não nos
  cantos do olho, que sobem e descem com as pálpebras: isso aumentou o
  sinal vertical em ~60% e reduziu à metade o erro com a cabeça inclinada
  para cima (470 → 216 px). Inclinar a cabeça para baixo ainda erra ~350 px
  na vertical. Use a recentralização (R) quando o cursor ficar deslocado.
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
4. Correção contínua por alvos conhecidos, usando
   `EyeGazeApp.confirm_target` (já implementado) a cada clique: o centro do
   elemento clicado é um alvo conhecido, e a correção é atualizada
   continuamente. No Windows, a UI Automation informa onde estão os botões
   (ver `docs/roteiro.md`, fase 4).
5. Teste com usuários: aplicar o protocolo de validação (`T`) em várias
   pessoas e condições de luz e reportar acurácia e precisão em graus.
