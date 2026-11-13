# Metodologia

Este documento descreve o método implementado, na ordem do pipeline. Cada
seção indica o arquivo correspondente. Use-o como base para o capítulo de
metodologia do TCC.

**Pipeline.** O cursor é o resultado de cinco etapas:

1. obter os **landmarks** do rosto (seção 1; a pose da cabeça, seção 2, sai
   deles);
2. a partir dos landmarks, obter o **centro do globo ocular** e o **centro da
   pupila** de cada olho (seção 3.1);
3. calcular o **feixe** de cada olho (do centro do globo, passando pela
   pupila) e a **intersecção** dos dois feixes (seção 3.2);
4. **projetar** o vetor do olhar (até a intersecção) no **monitor virtual**,
   o plano da tela com a geometria suposta (seção 3.3);
5. **calibrar** com vários pontos: ajustar o globo de cada olho e a correção
   entre o monitor virtual e o monitor real (seção 5).

Depois disso, o ponto passa pelo estabilizador de fixações (seção 4) e
move o mouse (seção 8).

| Etapa | Arquivo |
|---|---|
| 1. landmarks | `face_tracker.py`, `head_pose.py` |
| 2. globo e pupila | `gaze.py` (medidas), `eyeball.py` (modelo do globo) |
| 3. feixes e intersecção | `eyeball.py`, `geometry.closest_point_between_rays` |
| 4. monitor virtual | `geometry.VirtualScreen` |
| 5. calibração | `calibration.py`, `procedures.py` |

## 1. Captura e landmarks (`face_tracker.py`)

Cada frame da webcam vai para o `FaceLandmarker` do MediaPipe (Tasks API) em
modo `VIDEO`, que é síncrono e reaproveita o rastreamento entre frames. O
modelo devolve:

- **478 landmarks** normalizados: 468 da malha facial e 10 das íris
  (5 por olho: centro + 4 no contorno);
- uma **matriz de transformação facial** 4×4 que alinha o modelo canônico de
  rosto do MediaPipe ao rosto detectado.

Os landmarks são convertidos para "pixels 3D": `(x·L, y·A, z·L)`, onde L e A
são a largura e a altura da imagem. Nesse espaço, x aponta para a direita da
imagem, y para baixo e z para longe da câmera. É a mesma convenção do
Webcam3DTracker.

**Lados.** No MediaPipe, "esquerdo" e "direito" se referem ao lado do próprio
usuário. Os índices usados (`landmarks.py`):

| Olho | canto externo | canto interno | pálpebra sup./inf. | íris (centro; contorno) |
|---|---|---|---|---|
| direito | 33 | 133 | 159 / 145 | 468; 469–472 |
| esquerdo | 263 | 362 | 386 / 374 | 473; 474–477 |

Um teste automático confere essa tabela contra
`FaceLandmarksConnections` do próprio MediaPipe. A versão anterior do projeto
combinava a íris de um olho com os cantos do outro.

## 2. Pose da cabeça (`head_pose.py`)

A pose é uma rotação R (3×3) cujas colunas são os eixos da cabeça no
referencial da câmera:

- x: do lado direito para o lado esquerdo do usuário;
- y: de cima para baixo;
- z: para trás da cabeça. A face aponta para −z.

**Fonte principal: a matriz do modelo.** O espaço métrico do MediaPipe tem y
para cima e z saindo da tela. A conversão para pixels 3D é a conjugação

  R = C · R_met · C, com C = diag(1, −1, −1).

Antes da conversão, a parte 3×3 da matriz é ortonormalizada por SVD, o que
remove escala residual.

**Reserva e verificação: eixos por landmarks.** Os eixos também são
construídos geometricamente:

- x = soma dos vetores (esquerdo − direito) entre pares simétricos: cantos
  externos, cantos internos e têmporas;
- y = da testa/glabela (151, 9, 8) até a base do nariz (2, 164),
  ortogonalizado em relação a x (Gram-Schmidt). O queixo fica de fora porque
  se move ao falar;
- z = x × y.

Se a matriz do modelo e os eixos por landmarks discordarem em mais de 35°
no eixo x (ou 50° no eixo z), o programa usa os landmarks. A fonte usada
aparece no HUD com a tecla `P`. Essa verificação protege contra erros de
convenção de eixos.

O Webcam3DTracker estimava a orientação por PCA sobre pontos do nariz. Os
autovetores do PCA não têm sinal definido e trocam de ordem quando duas
variâncias se aproximam. Por isso aquele código precisava de uma "matriz de
referência" para desfazer inversões.

## 3. Globo ocular, pupila, feixes e monitor virtual (`gaze.py`, `eyeball.py`, `geometry.py`)

Esta seção cobre as etapas 2, 3 e 4 do pipeline. Tudo é calculado em
centímetros no **referencial da câmera**: origem no centro óptico da
webcam, x para a direita da imagem, y para baixo e z para longe da câmera.
Os landmarks viram cm por uma câmera pinhole com campo de visão horizontal
suposto (padrão 90°, `--camera-fov`). A distância do rosto sai da mesma
câmera: z = f · 9 cm / distância 3D entre os cantos externos dos olhos em
pixels (9 cm é a média adulta; o z do MediaPipe está na escala de x, então
essa distância quase não muda com a rotação da cabeça).

**Escala métrica.** Até a versão anterior, a distância vinha da translação
da matriz de pose do MediaPipe e o campo de visão suposto era 60°. Os dois
não são coerentes: o MediaPipe supõe uma câmera com ~63° de campo
*vertical* (~95° na horizontal, em 16:9). Nas sessões gravadas, a matriz
dava o rosto a ~37 cm, e com 60° o rosto inteiro em cm ficava a 56% do
tamanho real (olho com 1,6 cm de largura, olhos a 3,6 cm um do outro). Era
por isso que o globo ocular desenhado parecia grande demais. Agora a escala
sai da própria câmera suposta e é coerente (olho com ~3 cm). O campo de
visão foi escolhido pelos dados (26 sessões; erro em pontos novos / com a
cabeça em poses novas / na validação, sem o desvio constante entre
parênteses):

| Distância | campo de visão | LOO | cabeça | validação | rosto a | olho |
|---|---|---|---|---|---|---|
| matriz do MediaPipe | 60° (incoerente) | 132 | 319 | 142 (86) | 37 cm | 1,6 cm |
| pinhole | 60° | 139 | 413 | 226 (97) | 66 cm | 2,9 cm |
| pinhole | 75° | 135 | 341 | 210 (91) | 50 cm | 3,0 cm |
| **pinhole** | **90°** | **133** | **326** | **190 (89)** | **38 cm** | **3,0 cm** |
| pinhole | 105° | 133 | 327 | 215 (86) | 29 cm | 3,0 cm |

Com 60°, a pessoa ficava a 66 cm e a compensação da cabeça (a cabeça girar
θ desloca o olhar ~distância · tan θ) ficava grande demais. De 90° para
cima, os erros robustos (LOO e cabeça) se estabilizam, e 90° é típico de
webcam de notebook. A combinação incoerente (matriz do MediaPipe com 60°)
teve validação melhor (142 px), mas só há 4 validações, todas dominadas
pelo desvio constante entre calibrar e validar; sem ele, as variantes
ficam entre 86 e 97 px. Ela continua disponível
(`GazeConfig.depth_source = "mediapipe"` com `CameraConfig.hfov_deg = 60`).
O campo de visão fica gravado na calibração: mudá-lo exige recalibrar.

### 3.1 Etapa 2: âncora, pupila e centro do globo

Para cada olho, a cada quadro:

- **Âncora A**: ponto de partida, preso ao crânio, do deslocamento até o
  centro do globo. Na horizontal e na profundidade, é o ponto médio dos
  cantos interno e externo. Na **vertical**, é a **ponte do nariz** (pontos
  168, 6, 197, 8 e 9, entre os olhos) mais um deslocamento fixo de cada olho
  no referencial da cabeça, medido na calibração (mediana das amostras). O
  motivo: os cantos do olho fazem parte do contorno das pálpebras e sobem
  ou descem um pouco quando o olhar muda na vertical. Ancorada neles, a
  âncora acompanhava a pupila e "comia" o sinal vertical. A ideia vem do
  projeto de referência (seção 5.9). Cada landmark é convertido para cm
  antes de tirar a média: a perspectiva não comuta com a média em pixels,
  e fazer a média em pixels fazia o deslocamento ponte → olho variar
  ~0,01 cm com a inclinação da cabeça (~1° no feixe).
- **Pupila P**: centro da íris na imagem (média do anel de 4 pontos e do
  ponto central do MediaPipe, ou o centro refinado na imagem, seção 5.3).
  O z da íris do MediaPipe é a coordenada mais ruidosa e, ao girar a
  cabeça, contaminava as outras. Por isso a pupila 3D é a interseção da
  linha de visada da câmera pelo pixel da íris com o **plano do olho**
  (plano que passa pela âncora, perpendicular ao eixo z da cabeça).
- **Centro do globo C**: não aparece na imagem. É uma esfera com o tamanho
  de um olho real, raio r = 0,4 × largura do olho (cantos; ~12 mm / ~30 mm,
  como na versão original e no projeto de referência), com o centro r atrás
  da pupila, num deslocamento fixo no referencial da cabeça:

  C = A + R · (c_x, c_y, r)

  onde (c_x, c_y) é a posição da pupila no plano do olho quando o olho
  aponta para a frente da cabeça. Antes de calibrar, r vem da largura do
  olho de cada quadro, c = 0 e a âncora é a dos cantos. A calibração fixa r
  (mediana da largura) e ajusta c por pessoa e por olho (seção 5.2).

**Filtro da pupila.** A posição da pupila em relação à âncora, no
referencial da cabeça, p = Rᵀ(P − A), passa por um filtro One-Euro (Casiez
et al., 2012; corte mínimo 0,7 Hz, β = 10 cm⁻¹) antes de virar ângulo. O
raio efetivo é pequeno (~0,3 cm, seção 5.6), então 0,01 cm de ruído da
pupila vira ~2° no feixe: é daí que vinha a instabilidade dos feixes e da
intersecção. Filtrar aqui suaviza só o movimento do olho; a rotação e a
translação da cabeça entram depois (R e A) sem atraso. O filtro reinicia
quando o olho fecha ou há um intervalo de mais de 0,3 s sem quadro. Nas
sessões gravadas, o tremor quadro a quadro do ponto previsto caiu de 16
para 7,7 px, com atraso de 3 quadros (~100 ms) numa sacada e sem mudar o
erro. A média móvel de 10 quadros do projeto de referência tremia menos
(6,3 px), mas atrasava 9 quadros (~300 ms) e piorava o erro com a cabeça
em movimento (338 → 396 px).

O filtro vale só para o **feixe exibido** (desenho, HUD e detecção de
fixação na coleta). O cursor e a calibração usam a pupila sem filtro: o
estabilizador de fixação cancela o ruído rápido sem atraso, e o filtro
atrasava o cursor em ~47 ms (seção 4.1).

### 3.2 Etapa 3: feixe de cada olho e intersecção

O **feixe** de cada olho sai do centro do globo na direção da pupila. No
referencial da cabeça, com p = Rᵀ(P − A):

  direção_cabeça ∝ ( g_x·(p_x − c_x)/r , g_y·(p_y − c_y)/r , −1 )
  direção_câmera = R · direção_cabeça

g = (g_x, g_y) é o **ganho da pupila**, ajustado na calibração. O MediaPipe
"encolhe" o movimento da íris (seção 5.6): a íris medida anda menos que a
pupila de verdade, e mais ainda na vertical. O ganho corrige essa
compressão na própria medida da pupila, antes de somar a rotação da
cabeça, então girar a cabeça continua girando o olhar 1:1. Com g = 1 (olho
medido sem compressão), o feixe passa exatamente pela pupila medida. Com
g > 1, ele passa pela **pupila corrigida**, o ponto da superfície do globo
onde a pupila de verdade estaria. O projeto de referência faz o mesmo no
mapeamento ângulo → tela, com ±15° na horizontal e só ±5° na vertical
(3× mais ganho na vertical).

Na versão anterior, a compressão era um "globo menor" (raio efetivo
~0,3 cm). A direção do feixe é a mesma (r/g faz o papel daquele raio), mas
o centro ficava a ~3 mm da pupila, e não a ~12 mm como num olho real. Agora
o globo tem o tamanho real e a compressão é tratada como o que ela é, um
erro de medida da íris. A forma esférica (pupila = c + s·sen em vez de
tan) foi testada antes e errou mais nos dados reais.

A **intersecção** dos dois feixes é o ponto fixado. Duas retas no espaço
quase nunca se cruzam exatamente, então a intersecção é o ponto médio do
segmento perpendicular comum (`geometry.closest_point_between_rays`). O
**vetor do olhar** sai do ponto médio entre os dois centros (o "olho
ciclópico") e vai até a intersecção. Quando os feixes não convergem à
frente do rosto (entre 5 e 400 cm ao longo de cada feixe), usa-se a
bissetriz das duas direções. Com um olho fechado, usa-se o feixe do outro.
Nas sessões gravadas, os feixes convergiram em 50–100% das amostras da
grade, com a intersecção de 0 a 30 cm além do plano da tela. Usar sempre a
bissetriz deu exatamente o mesmo erro.

(yaw, pitch) do vetor do olhar, com o globo atual (anatômico ou calibrado),
aparecem no HUD e servem para detectar fixações e rejeitar outliers na
coleta:

  yaw   = atan2(−d_x, −d_z)          (> 0: olhando para a direita do usuário)
  pitch = atan2(−d_y, √(d_x² + d_z²)) (> 0: olhando para cima)

### 3.3 Etapa 4: projeção no monitor virtual

O **monitor virtual** (`geometry.VirtualScreen`) é um retângulo no plano
z = 0 da câmera, com a webcam centralizada 1 cm acima da borda superior
(notebook ou webcam presa no monitor). A largura física vem do monitor
(EDID, detectada no Windows) ou de `--screen-width-cm`. Sem ela, supõe-se
34,5 cm (notebook de 15,6" 16:9). Como a imagem não é espelhada, a borda
esquerda da tela (para o usuário) fica em +x.

A reta que sai da origem do olhar na direção do vetor do olhar é cruzada
com esse plano, e o ponto atingido vira um pixel pela geometria pura:

  x_px = L/2 − x_cm · px/cm        y_px = (y_cm − 1 cm) · px/cm

Esse é o "pixel virtual". Antes de calibrar, ele é o próprio cursor. A
etapa 5 aprende a correção entre ele e o pixel real.

### 3.4 Vetor bruto (entrada da calibração)

O estimador entrega, por quadro, um vetor com tudo o que as etapas 2–5
precisam, e é isso que é gravado nas sessões:

  raw = [R (9) | E (3) | A_d, P_d, A_e, P_e (12) | P_d e P_e refinadas (6) |
         olho direito aberto, olho esquerdo aberto | abertura da boca |
         ponte do nariz (3)]

(A = meio dos cantos; as pupilas já filtradas.)

(d = olho direito, e = esquerdo; tudo em cm no referencial da câmera.) O
globo, os feixes, a intersecção e o pixel virtual são recalculados a partir
desse vetor, então as sessões gravadas podem ser reajustadas offline com
outro modelo do globo.

### 3.5 Piscadas

Para cada olho, EAR = distância entre as pálpebras / largura do olho. O olho
conta como fechado quando o EAR cai abaixo de 50% da sua linha de base. A
linha de base é uma média móvel do EAR com o olho aberto, então se adapta a
cada pessoa sem ajuste manual. Com um olho fechado, usa-se o feixe do
outro. Com os dois fechados, o quadro é marcado como piscada e o cursor
mantém a última posição.

## 4. Estabilização do cursor (`filters.py`)

O ruído dos landmarks da íris é pequeno em pixels da imagem (~0,5–1 px), mas
a íris se move poucos pixels por grau de rotação do olho. Esse ruído vira
3–4° de ângulo, ou mais de 100 px na tela. A versão anterior usava um
One-Euro Filter nos ângulos, mas com β em graus/s qualquer ruído desse
tamanho parece "movimento rápido" e quase não é filtrado. O cursor tremia
continuamente.

A solução usa a estrutura do próprio olhar. O olho alterna entre
**fixações** (parado num ponto por 200–600 ms) e **sacadas** (saltos
rápidos). O `FixationStabilizer` foi inspirado nos detectores de fixação por
dispersão (I-DT; Salvucci e Goldberg, 2000):

1. **Sacada:** uma amostra a mais de L px da média da fixação é candidata.
   O cursor só salta quando 2 amostras seguidas estão fora e são coerentes
   entre si (dispersão < max(L, metade do salto)), e salta para a amostra
   mais recente (ver 4.1). Picos de um quadro (quase-piscadas, erros de
   detecção) são ignorados.
2. **Limiar adaptativo:** L = max(7% da largura da tela, 3 × ruído medido na
   fixação atual). Em condições ruins (pouca luz, rosto longe), o ruído
   aumenta e fica mais difícil confundir tremor com sacada.
3. **Acomodação:** nos primeiros 8 quadros de uma fixação, o cursor desliza
   (60% do caminho por quadro) até a média das amostras.
4. **Trava:** depois disso o cursor **para**. Ele só volta a andar quando as
   últimas 8 amostras (~0,27 s) estão **todas** a mais de r px da posição
   travada, ou seja, um deslocamento pequeno porém sustentado. Aí começa uma
   nova fixação no ponto médio dessas amostras. O raio é
   r = max(3,5% da largura da tela, 3,5 × passeio típico), em que o passeio
   típico é a dispersão das fixações anteriores (média móvel), limitado a
   0,9 L (antes 2,5 × e 0,6 L; ver 4.2).

**Por que a trava (versão 2 do estabilizador).** A primeira versão seguia a
média deslizante das últimas 20 amostras, com zona morta de 6 px. Isso
supõe ruído branco (independente de quadro para quadro), e a média de N
amostras reduz o desvio para σ/√N. Nas sessões gravadas, porém, o ruído da
estimativa é fortemente **correlacionado**: a autocorrelação entre quadros
seguidos é 0,97, e a média de 8 amostras tem praticamente o mesmo desvio
que uma amostra isolada (13 px contra 17 px, quando o esperado com ruído
branco seria 6 px). A estimativa não treme, ela **passeia**: em 1,2 s de
fixação, afasta-se 50–130 px do ponto inicial em metade dos alvos (≈1–2,5°,
bem acima do movimento fisiológico do olho na fixação). A média deslizante
acompanhava esse passeio e o cursor "andava" com o olho parado. A trava
troca um pouco de resolução (deslocamentos menores que r não movem o
cursor) por um cursor parado, o que é aceitável porque a acurácia do
sistema (~150 px) já é maior que r.

Resultado nas sequências reais das duas sessões gravadas (16 alvos cada,
reproduzidas na ordem em que foram vistos):

| | quadros com o cursor em movimento na fixação | quadros até chegar após uma sacada (mediana) |
|---|---|---|
| média deslizante (versão 1) | 71% | 2 |
| trava (versão 2) | ~8% (quase todos no deslizamento inicial) | 2 |

E num sinal simulado com ruído branco de 45 px por eixo:

| | movimento do cursor por frame durante a fixação | frames até chegar após uma sacada |
|---|---|---|
| sem filtro | ~82 px | 0 |
| One-Euro | ~46 px | 0 |
| estabilizador de fixação | ~2 px | 2–3 (~100 ms) |

O One-Euro continua sendo usado, com parâmetros mais suaves, só nos ângulos
mostrados no HUD e no mapeamento antes da calibração. O HUD mostra o ruído
estimado (`ruido ~N px`). O minimapa mostra o ponto bruto (cinza) e o cursor
estabilizado (amarelo).

### 4.1 Atraso do cursor

**Medição.** O MediaPipe de cada quadro das duas sessões gravadas em vídeo
foi guardado em cache. O ponto bruto de cada quadro foi calculado com o
modelo da calibração original da sessão, e os arquivos de calibração e
validação dão o alvo em cada instante. Em cada troca de alvo (89 trocas
com salto ≥ 250 px), o **atraso extra do cursor** é o tempo entre o ponto
bruto chegar perto do novo alvo e o cursor chegar.

| Estabilizador | atraso mediano | atraso p90 | quadros com o cursor em movimento na fixação | erro do cursor no alvo |
|---|---|---|---|---|
| anterior (confirmação em 3 quadros, desliza 35%/quadro, salta para a média das pendentes) | 92 ms | 292 ms | 23% | 148 px |
| **atual** (2 quadros, 60%/quadro, salta para a amostra mais recente) | **42 ms** | **110 ms** | **15%** | **138 px** |

O ponto bruto leva alguns quadros para atravessar a tela (o MediaPipe
suaviza os landmarks no tempo). A média das amostras pendentes caía no meio
dessa travessia, o cursor parava ali e ainda tinha de deslizar ou esperar
um "deslocamento pequeno" de 8 quadros. Saltar para a amostra mais recente
leva o cursor direto ao destino. A tolerância de coerência das pendentes
passou a crescer com o tamanho do salto. O cursor também se move menos na
fixação, porque termina de deslizar mais cedo, e fica mais perto do alvo
(138 px contra 135 do ponto bruto).

**Medição de 13/11 (pipeline geométrico).** O MediaPipe foi rodado em
todos os quadros das calibrações de 8 sessões gravadas em vídeo, inclusive
nas transições entre alvos, com o filtro da pupila desligado. O modelo foi
ajustado com a calibração da própria sessão, e o filtro da pupila e o
estabilizador foram simulados quadro a quadro. Em cada troca de alvo com
salto ≥ 250 px (~118 trocas), o atraso é o tempo entre o ponto bruto (sem
filtro algum) chegar perto do destino e o cursor chegar. Para movimentos
**pequenos**, que a calibração não tem, somou-se um degrau de 60, 100 ou
150 px ao ponto bruto real no meio de cada fixação, mantendo o ruído real.

| Configuração | atraso mediano | p90 | cursor em movimento na fixação | tremor RMS na fixação |
|---|---|---|---|---|
| filtro da pupila no cursor (0,7 Hz), estabilizador anterior | 47 ms | 124 ms | 6,5% | 16,5 px |
| sem filtro da pupila no cursor | 0 ms | 53 ms | 3,8% | 9,3 px |
| sem filtro, desliza 85%/quadro | 0 ms | 53 ms | 2,7% | 9,9 px |
| **sem filtro, desliza 85%, deslocamento em 5 quadros (atual)** | **0 ms** | **58 ms** | **3,8%** | **12,4 px** |
| idem, com raio de trava menor | 0 ms | 64–66 ms | 6,7–9,0% | 22 px |
| filtro da pupila 1,0 Hz / β 40 | 31 ms | 75 ms | 5,0% | 10,8 px |
| confirmação de sacada em 1 quadro | 32 ms | 124 ms | 6,1% | 13,1 px |

| Degrau pequeno do olhar | 60 px | 100 px | 150 px |
|---|---|---|---|
| anterior: segue em / atraso | 54% / 231 ms | 79% / 231 ms | 88% / 33 ms |
| **atual** | **60% / 132 ms** | **76% / 132 ms** | **91% / 33 ms** |

Conclusões:

- O **filtro da pupila** (seção 3.1) era a principal fonte de atraso, e nem
  reduzia o tremor final: ele transformava o ruído rápido, que o
  estabilizador cancela pela média da fixação, num passeio lento que o
  estabilizador seguia com atraso e mais trocas de fixação. Agora ele vale
  só para o feixe EXIBIDO (desenho, HUD, detecção de fixação na coleta); o
  cursor e a calibração usam a pupila sem filtro. A precisão da calibração
  não mudou (36 sessões: LOO 121 px, cabeça em poses novas 289 px).
- Deslizar 85% do caminho por quadro (era 60%) e seguir um deslocamento
  pequeno sustentado em 5 quadros (era 8) fazem o cursor seguir o olhar
  ~100 ms mais cedo em movimentos de 60–100 px.
- Reduzir o raio de trava faria o cursor seguir um pouco mais os
  movimentos mínimos, mas quase dobra o tremor; foi mantido.
- Degraus de 60 px ainda são seguidos só em ~60% das vezes: estão no
  limite do passeio da estimativa com o olho parado (~50 px). Para alvos
  pequenos, o ajuste fino pela cabeça (seção 4.3) é mais preciso.

**Câmera.** O driver da webcam guardava 3 quadros (medido). Com leitura
síncrona, quando um quadro levava mais que ~33 ms para ser processado
(MediaPipe 13 ms, desenho 5–10 ms, tela de alvos 10 ms, gravação de vídeo),
o app processava imagens com até ~100 ms de atraso. Agora uma thread lê a
câmera sem parar e o loop pega sempre o quadro mais recente
(`FrameSource`). Com um processamento simulado de 50 ms por quadro, o
quadro processado tem ~7 ms de idade. O fundo da tela de alvos passou a
ser copiado de uma imagem pronta em vez de criado a cada quadro.

### 4.2 Tremor do olhar estimado

O raio do olhar na vista 3D treme porque mostra a estimativa **bruta**,
antes do estabilizador. Nas quatro sessões gravadas em vídeo, com o olho
parado num alvo conhecido, ela anda ~18 px de um quadro para o outro e
passeia com ~50 px de desvio. Origem desse tremor, testada nos dados:

| Teste | Resultado |
|---|---|
| filtrar no tempo a pose da cabeça (One-Euro 0,5–1 Hz) | nada muda (18 px/quadro, 58 px de dispersão) |
| filtrar as medidas do olho (One-Euro 1–2 Hz) | tremor rápido cai à metade, passeio quase igual (50 px), atraso p90 do cursor sobe de 149 para 248–402 ms |
| regularização (ridge) mais forte | dispersão igual (46–50 px) até α = 0,1; com α = 1 cai, mas o erro vai a 356 px (tudo comprimido) |
| quadros com o olho abaixo de 80% da abertura normal | 3,4% das amostras, 46% dos desvios > 150 px (piscadas parciais) |

O tremor vem das medidas do olho do MediaPipe nessa câmera e é o limite de
**precisão** do sistema (~50 px de desvio com o olho parado). Descartar
quadros com queda brusca de abertura do olho (piscada parcial) reduziu
pouco os movimentos do cursor (36 → 32/min) e foi deixado de fora.

O que movia o cursor junto com o tremor era o **destravamento por
deslocamento pequeno**. Classificando os eventos do estabilizador durante
fixações conhecidas: sacadas falsas, 2,6/min; destravamentos (o passeio
fica fora do raio de trava por 8 quadros), ~19/min, de ~119 px cada. O
raio de trava adaptativo estava limitado a 0,6 do limiar de sacada, menos
que o passeio real. Com o fator 3,5 (antes 2,5) e o teto de 0,9:

| | destravamentos/min | atraso p90 após sacada | cursor em movimento na fixação | erro no alvo |
|---|---|---|---|---|
| anterior (2,5; 0,6) | 18,3 | 139 ms | 16,0% | 118 px |
| **atual (3,5; 0,9)** | **6,5** | **139 ms** | **12,3%** | 122 px |

O custo é que deslocamentos do olhar menores que ~130 px não movem mais o
cursor. Com um passeio de ~50 px, eles não se distinguem do ruído em tempo
útil. Na vista 3D, o raio passou a ser desenhado com uma suavização leve
(One-Euro, só na exibição), sem efeito no cursor.

### 4.3 Ajuste fino pela cabeça (modo híbrido, tecla A)

Com o olhar parado num alvo, a estimativa do olhar varia ~41 px (desvio,
272 alvos das calibrações gravadas), mas a pose da cabeça varia só 0,10°,
o que com o ganho de 25 px/grau dá ~2-4 px. A ponta do nariz varia 0,34 px na
imagem. A cabeça é medida com precisão ~10x maior que o olhar.

O modo híbrido (`filters.HeadFineTuner`, desligado por padrão) usa as duas:

1. o olhar leva o cursor à região, e o estabilizador o trava na fixação;
2. durante a acomodação da fixação (primeiros 8 quadros), a pose da cabeça
   vira a referência;
3. depois, cursor = ponto da fixação + 25 px/grau × (pose − referência),
   com zona morta de 0,3° (o balanço natural da cabeça não move o cursor),
   limite de 250 px, girar para a direita move para a direita e inclinar para
   cima move para cima;
4. uma nova fixação (sacada ou deslocamento do olhar) zera o ajuste, e um
   giro maior que 6° também (mudança de postura).

Enquanto a pessoa ajusta pela cabeça e continua olhando o alvo, o olho
gira ao contrário da cabeça. Com a compensação da cabeça do modelo, a
estimativa do olhar fica no lugar, dentro do raio de trava, e o ponto da
fixação não muda. O ganho (25 px/grau) é menor que o físico (~38) para
favorecer a precisão. Serve só para quem tem algum controle da cabeça; o
modo só com o olhar continua sendo o padrão.

## 5. Etapa 5: calibração com vários pontos (`calibration.py`, `procedures.py`)

A pessoa olha para alvos conhecidos (grade 4×4) e depois mantém o olhar no
centro enquanto gira a cabeça. Com essas amostras, a calibração ajusta, em
ordem:

1. o **globo ocular** de cada olho (seção 5.2), que faz o feixe passar pelo
   alvo;
2. a **correção monitor virtual → monitor real** (seção 5.3), que leva o
   pixel virtual da etapa 4 ao pixel onde o alvo de fato está.

As duas etapas são problemas lineares (mínimos quadrados), sem otimização
iterativa e sem mínimos locais.

### 5.1 Alvos no referencial da câmera

Cada alvo da tela (pixel) vira um ponto 3D T no plano do monitor virtual
(seção 3.3). Na etapa da cabeça, o alvo é sempre o centro da tela.

### 5.2 Ajuste do globo ocular

Olhando para T, a direção do centro do globo até o alvo é conhecida. No
referencial da cabeça, τ = Rᵀ(T − C), e a pupila medida deve satisfazer,
em cada eixo j ∈ {x, y}:

  p_j = c_j + (r / g_j) · tan_j(T),   tan_j = τ_j / (−τ_z)

Para cada olho e cada eixo, é uma regressão linear com duas incógnitas:
**c_j** e **r/g_j** (r é fixo, 0,4 × mediana da largura do olho). Ela é
resolvida por mínimos quadrados ponderados **robustos** (Huber, k = 1,345,
por repesagem iterativa em 4 passos): quadros em que a pessoa piscou,
olhou para fora do alvo ou o landmark falhou pesam menos, em vez de puxar o
globo. Cada alvo pesa 1, a etapa da cabeça inteira pesa 1 alvo, e uma
regularização fraca (λ = 0,002 em unidades de tan²) puxa o ganho para 1,
com limites de 0,2 a 25. Como o centro depende de c e a direção até o alvo
depende do centro, o ajuste é repetido duas vezes (a mudança é de
milímetros a dezenas de cm).

Do projeto de referência, foi considerado e descartado o "travamento" do
globo num único quadro, olhando o centro da tela (centro = íris + raio na
direção da câmera). Ele herda todo o ruído daquele quadro, e o ajuste em 16
alvos mais a etapa da cabeça estima o mesmo centro com centenas de
amostras.

A **etapa da cabeça** (olhar fixo, cabeça girando) entra no mesmo ajuste.
É ela que separa o movimento do olho do movimento da cabeça: com o olhar
fixo, a direção até o alvo no referencial da cabeça muda só pela rotação da
cabeça, e o ajuste aprende quanto a pupila contra-rotaciona.

### 5.3 Correção monitor virtual → monitor real

O pixel virtual ainda tem erros que não são do olho: largura física da tela
suposta, posição da webcam em relação à tela, campo de visão da câmera e
viés da pose do MediaPipe. A correção é um polinômio pequeno nas
coordenadas do pixel virtual V, somado a ele:

  real = V + a₀ (+ a₁·u + a₂·v) (+ a₃·u² + a₄·uv + a₅·v²)

com (u, v) = V normalizado pela tela e centralizado. Os candidatos são:
**nenhuma** (geometria pura), **deslocamento**, **afim** e **quadrática**.
O ajuste é ridge nos termos não constantes (α ∈ {10⁻³, 10⁻², 10⁻¹}), o que
encolhe a correção em direção à identidade. A quadrática só é candidata com
9 alvos ou mais.

**Fonte da pupila.** Há duas medidas do centro da íris: a do MediaPipe e
a refinada na imagem (`iris_refine.py`: centróide dos pixels escuros dentro
do contorno das pálpebras, com resolução sub-pixel e sem a regularização do
modelo). A calibração usa a **média das duas**: os erros delas são em parte
independentes, e a média errou menos que cada uma sozinha (seção 5.10). Sem
o refinamento (desligado ou indisponível), usa-se a do MediaPipe.

### 5.4 Coleta

**Etapa 1, grade 4×4 com checagens do centro:** margem de 5% e **ordem
aleatória**. Alvos colados na borda não ajudariam: ali a pálpebra cobre a
íris e o MediaPipe é menos preciso. Na ordem linha a linha, o tempo andava
junto com a posição vertical: nos dados reais, a abertura do olho caiu de
0,27 para 0,16 ao longo dos ~35 s da grade (cansaço, olho ressecando), e o
modelo aprendia essa deriva como "olhar para baixo". Com a ordem aleatória,
a deriva vira ruído em vez de viés, e as **checagens do centro** permitem
medi-la e descontá-la (seção 5.4.1): um ponto no centro da tela aparece no
início e a cada 4 alvos (5 checagens, ~10 s a mais).

Para cada alvo, a coleta começa quando o olhar **fixa** no alvo: depois de
0,6 s de acomodação, espera-se que 6 quadros seguidos fiquem a menos de 0,8°
da mediana deles, com limite de 1,8 s. Assim, as primeiras amostras não são
mais da sacada ou de um olhar "procurando" o alvo (detecção de fixação por
dispersão, Salvucci e Goldberg, 2000). Seguem 1,2 s de coleta das amostras
válidas. Se o rosto sumir, o alvo espera até juntar pelo menos 8 amostras.
Em cada alvo, amostras com escore-Z modificado (MAD) > 3,5 são descartadas
(Iglewicz e Hoaglin, 1993).

**Etapa 2, movimento da cabeça (guiada):** o alvo fica no centro da tela e
uma seta pede que a pessoa gire a cabeça para a direita, a esquerda, para
cima e para baixo, sem tirar o olho do alvo, e depois volte ao centro. A
seta enche conforme a cabeça gira (medida em relação à pose mediana durante
a grade) e fica verde ao chegar a 80% do ângulo pedido (10° de yaw e 8° de
pitch). Depois de 0,6 s ali, passa à próxima direção. Quem não alcança o
ângulo segue depois de 4 s, e quem não pode mover a cabeça só continua
olhando o ponto. Se a cabeça passar da faixa aproveitada (15° de yaw, 12° de
pitch), a seta fica vermelha e aparece "girou demais". Só entram no ajuste
as amostras dentro dessa faixa: sem guia, as pessoas giravam até 30° (yaw)
e 40° (pitch), ângulos em que os landmarks do olho deixam de ser
confiáveis.

**Calibração manual (tecla C).** Cada alvo só avança quando o usuário,
olhando para ele, aperta ESPAÇO ou ENTER. As amostras usadas são as de 1 s
**antes** da tecla, sem os últimos 0,15 s (tempo de reação): ao apertar a
tecla é comum piscar ou olhar para o teclado. A confirmação é recusada (com
aviso) se o olhar ainda se movia nessa janela (80% das amostras a mais de
1,6° da mediana) ou se faltam amostras, e ignorada nos primeiros 0,5 s do
alvo. As etapas da cabeça e da boca continuam automáticas. Ela não
substitui a automática: o público do projeto (pessoas com deficiência
motora) muitas vezes não consegue apertar uma tecla a cada alvo.

**Etapa 3, boca:** o ponto fica no centro e a pessoa abre e fecha a boca
devagar (≥ 4 s; segue em 7 s se a boca não abrir). O MediaPipe estima a
malha inteira de uma vez, e abrir a boca contamina a pose da cabeça e os
pontos do olho: na sessão de 12/11 09:47, com o olhar fixo num alvo
conhecido, o ponto estimado descia ~50 px nos quadros de boca aberta. Na
etapa da boca o olhar está fixo, então todo deslocamento da previsão que
acompanha a abertura é artefato. Ajusta-se esse deslocamento por unidade
de abertura acima da abertura típica da grade (mínimos quadrados pela
origem, com rejeição de outliers e descontando o desvio com a boca
fechada), e a previsão passa a descontá-lo (`fit_mouth_correction`). Sem a
etapa, ou se a boca quase não abriu (variação < 0,04), a correção é zero.

**Repetição de alvos ruins:** depois do ajuste, alvos com erro na validação
cruzada acima de 2,5× a mediana e acima de 8% da largura da tela (até 3
alvos) são coletados de novo, uma vez, e o modelo é reajustado. Isso cobre
piscadas longas, distração ou olhar fora do alvo.

As medidas brutas e os landmarks de cada calibração e validação são
gravados em `results/*.npz`. `tools/benchmark.py` recalcula as medidas com
o estimador atual e avalia variantes do pipeline em todas as sessões
gravadas. `tools/reavaliar.py` detalha uma sessão (globo ajustado,
convergência dos feixes, erro por alvo na validação).

#### 5.4.1 Correção da deriva (`calibration.correct_drift`)

Nas checagens a pessoa olha sempre para o mesmo ponto conhecido. Com o
globo ajustado, a diferença entre a pupila medida e a prevista em cada
checagem (por olho e eixo, no referencial da cabeça, o que separa o efeito
da cabeça pela geometria) é a deriva da medida naquele instante. A
calibração:

1. ajusta uma **tendência linear no tempo** a essas 5 medidas (a deriva é
   lenta; a reta filtra o ruído de cada checagem);
2. **encolhe** a inclinação por s²/(s² + 4·EP²), onde EP é o erro-padrão da
   inclinação (o maior entre a dispersão dentro das checagens e o resíduo
   entre elas, já que o filtro da pupila correlaciona amostras seguidas).
   Uma deriva bem acima de 2 erros-padrão passa quase inteira; o ruído
   puro, quase nada;
3. subtrai de cada amostra da grade inclinação × (t − t_fim): o modelo fica
   ajustado ao **estado do fim da calibração**, o mais próximo do uso;
4. repete duas vezes (o globo usado para medir a deriva melhora com os
   dados corrigidos), para cada fonte da pupila.

A tela de resultado mostra quanto o centro "andou" entre as checagens. No
usuário sintético cuja pupila deriva linearmente ao longo da grade
(mediana de 20 quadros por alvo, mesmas amostras para os dois modelos,
média de 4 sementes; erro no estado do fim, em px):

| Deriva da pupila na grade | sem correção | com correção |
|---|---|---|
| nenhuma, sem ruído | 25 | 25 |
| nenhuma, ruído de 0,3 px nos landmarks | 27 | 27 |
| 0,02 cm, com ruído | 63 | 27 |
| 0,04 cm, com ruído | 102 | 27 |
| 0,08 cm, com ruído | 168 | 27 |

As sessões gravadas até 13/11 não têm checagens intercaladas, então o ganho
com dados reais ainda não foi medido (`tools/benchmark.py` passa a medi-lo
nas calibrações novas). A deriva DEPOIS da calibração é tratada na
validação (reteste) e na recentralização (seções 6 e 7).

### 5.5 Escolha e erro em pontos e poses novos

A fonte da pupila, o tipo de correção e a força do ridge são escolhidos por
**validação cruzada deixando um alvo de fora**: para cada alvo, o globo E a
correção são reajustados sem ele, e o erro é a distância da mediana das
previsões ao alvo (pontos novos, cabeça parada). Fica a combinação de menor
erro médio.

A etapa da cabeça também é dividida pela direção dominante do desvio de
pose (direita, esquerda, cima, baixo). Cada direção fica fora do ajuste por
vez, e o erro é a mediana das distâncias de suas amostras ao alvo central
(poses novas da cabeça). Amostras a menos de 2° da pose da grade ficam
sempre no treino. Esse erro é reportado, mas não entra na escolha: como o
alvo da etapa é sempre o centro, ele favoreceria correções que encolhem a
previsão para o centro da tela.

A tela de resultado mostra o erro em pontos novos, o erro só com o monitor
virtual (sem correção, para ver quanto a etapa 5 corrige), o erro com a
cabeça em poses novas e uma classificação da qualidade (boa até 4,5% da
largura da tela, razoável até 8%, ruim acima disso).

### 5.6 Resultados nas sessões gravadas

`tools/benchmark.py`, 33 calibrações gravadas de 11 a 13/11/2026 (4 delas
com validação logo depois), tela de 1920×1080 com 34,4 cm (detectada pelo
EDID), medidas recalculadas a partir dos landmarks gravados, câmera suposta
com 90° (seção 3). Erros médios em px (média das sessões):

| Variante | pontos novos (LOO) | cabeça em poses novas | validação (sem o desvio constante) |
|---|---|---|---|
| **pipeline (padrão: escala pela câmera, globo anatômico + ganho, âncora mista, média das duas medidas da pupila)** | **123** | **292** | **211 (77)** |
| pupila só do MediaPipe | 148 | 299 | 245 (99) |
| pupila só refinada na imagem | 139 | 328 | 213 (91) |
| âncora nos cantos do olho | 137 | 330 | 162 (87) |
| âncora na ponte do nariz (x e y) | 118 | 303 | 242 (78) |
| só o monitor virtual (sem a correção 5.3) | 152 | 329 | 211 (74) |
| correção só por deslocamento | 152 | 326 | 212 (74) |
| sem a etapa da cabeça | 121 | 304 | 212 (78) |
| versão de 12/11 (pupila escolhida entre as duas; 28 sessões) | 133 | 332 | 190 (88) |
| versão anterior (escala do MediaPipe, "globo menor"; 26 sessões) | 134 | 330 | 146 (87) |
| modelo de regressão (seção 5.7; 25 sessões) | 118 | 374 | 113 (~100) |

Em relação à versão anterior, o erro em pontos novos e com a cabeça em
poses novas é o mesmo, e a validação sem o desvio constante também; a
validação bruta piorou (146 → 190 px) nas 4 sessões com validação, por
causa do desvio constante (ver a escolha do campo de visão, seção 3). A
mudança corrige a geometria (escala real do rosto, globo do tamanho de um
olho) sem ganho de acurácia mensurável com os dados atuais.

Experimentos à parte (25 sessões, mesma metodologia; seção 5.9):

| Variante | LOO | cabeça | validação | raio efetivo vertical | erro y, cabeça para cima |
|---|---|---|---|---|---|
| âncora nos cantos | 143 | 349 | 164 (94) | 0,19 cm | 470 px |
| **âncora mista (y pela ponte do nariz)** | **134** | **327** | **146 (87)** | **0,31 cm** | **216 px** |
| âncora na região do nariz da referência (com a ponta) | 135 | 462 | 288 (102) | 0,29 cm | 441 px |
| forma esférica do globo (sen em vez de tan) | 151 | 363 | 163 (99) | — | — |
| pálpebra superior como vertical da pupila | 162 | 370 | — | — | 605 px |

O que os números mostram, etapa por etapa:

- **Etapa 2 (globo).** O raio efetivo ajustado fica em ~0,3–0,4 cm na
  horizontal e ~0,3 cm na vertical (0,19 cm com a âncora nos cantos),
  contra ~1,2 cm de um globo real: o MediaPipe entrega ~25–30% do movimento
  da pupila. Por isso a pupila precisa de raio por eixo. Com a âncora nos
  cantos, a vertical parecia ainda mais comprimida (~15%), porque os cantos
  acompanham as pálpebras; ancorar a vertical na ponte do nariz recuperou
  ~60% de sinal vertical. O erro angular de cada feixe nas próprias
  amostras de ajuste é de 3–7°.
- **Etapa 3 (feixes e intersecção).** O filtro da pupila (seção 3.1) reduz
  o tremor quadro a quadro do ponto de 16 para 7,7 px, sem mudar o erro.
  Os feixes convergem à frente do rosto em ~85% das amostras da grade, e a
  profundidade da intersecção varia ~4–5 cm (desvio-padrão) com o olhar
  fixo num alvo: com os olhos a ~6 cm um do outro e a tela a ~60 cm, a
  profundidade é a coordenada mais sensível ao ruído. A direção até a
  intersecção é estável, e usar sempre a bissetriz dá o mesmo erro.
- **Etapa 4 → 5.** Só com o monitor virtual, o erro em pontos novos é de
  155 px; a correção (quase sempre quadrática) o reduz para 134 px. A
  correção não muda o erro com a cabeça em poses novas.
- **Cabeça em poses novas.** Separando por direção (mediana do erro
  absoluto em x e y, âncora mista): direita (234, 225), esquerda (137,
  217), cima (96, 216), baixo (144, 353). Com a âncora nos cantos, a
  cabeça para cima dava 470 px em y: com o olhar fixo, a pupila do
  MediaPipe quase não contra-rotaciona na vertical em relação aos cantos.
  Em relação à ponte do nariz, contra-rotaciona mais. Sobra um erro grande
  ao inclinar a cabeça para baixo (353 px em y), onde as pálpebras cobrem
  mais a íris. A recentralização (R), o aviso de postura e o ajuste fino
  pela cabeça (tecla A) são as formas de conviver com isso.
- **Comparação com o modelo anterior.** O pipeline geométrico erra ~14% a
  mais em pontos novos com a cabeça parada (134 contra 118 px), menos com
  a cabeça em poses novas (330 contra 374 px) e o mesmo na validação sem o
  desvio constante (87 contra ~100 px). O desvio constante entre calibrar
  e validar ainda é maior (146 contra 113 px), e a recentralização o
  corrige. As 4 validações são poucas para separar os dois.

### 5.7 Histórico: modelo de regressão (versões 3 a 6)

Antes do pipeline geométrico, o cursor vinha de uma regressão ridge sobre
medidas brutas: ponto para onde a face aponta no plano da tela, íris
horizontal e vertical, altura da pálpebra superior e abertura do olho,
multiplicadas pela distância e por sec²(ângulo da cabeça), com termos
quadráticos e variantes ("livre", "física", "compacta", eixo y com o ganho
da cabeça pela geometria). O feixe centro do globo → pupila existia só para
desenho. Esse modelo foi substituído para que o cursor saia do próprio
pipeline (landmarks → globo e pupila → feixes e intersecção → monitor
virtual → calibração), cujas etapas têm significado físico e podem ser
inspecionadas uma a uma (vista 3D, seção 9). As análises daquela fase
(compressão da íris, coerência da pálpebra com a cabeça, viés da direção da
face do MediaPipe, efeito da boca) estão no histórico do git
(`docs/metodologia.md` até o commit 05dcc11).

### 5.8 Efeito (simulação)

Num usuário sintético com o movimento da íris encolhido como no MediaPipe
(globo de 0,5 larguras de olho e vertical a 30%) e monitor de 53×30 cm a
60 cm (`tests/test_pipeline.py`):

- com a cabeça parada, o erro em pontos fora da grade fica abaixo de 30 px;
- girando ou deslocando a cabeça (até 12° de yaw, 9° de pitch, 8 cm de
  distância), fica abaixo de ~130 px. O resíduo vem de o usuário sintético
  comprimir o seno do ângulo vertical, enquanto o modelo usa a tangente;
- no olho sintético "físico" (sem compressão), o ajuste recupera o raio
  real do globo (1,2 cm), e os dois feixes se cruzam a menos de 4 cm do
  alvo na tela.

Esses testes validam a matemática e o software. A acurácia real é a da
seção 5.6 e deve ser medida com a validação (seção 6).

### 5.9 Comparação com o projeto de referência (Webcam3DTracker)

O projeto de referência (`docs/Webcam3DTracker.zip`, `MonitorTracking.py`)
implementa o mesmo pipeline de forma mais simples. O que foi aproveitado
e o que não foi:

| Referência | Aqui | Decisão |
|---|---|---|
| Centro do globo "travado" num único quadro, olhando o centro da tela (tecla C): centro = íris + 20 px na direção da câmera | Ajustado por mínimos quadrados em 16 alvos + etapa da cabeça (seção 5.2) | Mantido o ajuste: o travamento num quadro herda todo o ruído daquele quadro |
| Centro ancorado na **região do nariz** (24 pontos, PCA), "para estabilizar o rastreamento vertical" | Âncora **mista**: horizontal pelos cantos, vertical pela **ponte do nariz** (5 pontos entre os olhos) | A ideia está certa (os cantos se movem com as pálpebras), mas a região inteira inclui a ponta do nariz, ~4 cm abaixo e à frente dos olhos. Esse braço de alavanca amplificava os erros da pose: com ela, o erro com a cabeça em movimento foi de 349 para 462 px. A ponte do nariz fica na altura dos olhos |
| Raio fixo de 20 px; o ganho fica no mapeamento ângulo → tela (±15° na horizontal, ±5° na vertical, ou seja, 3× mais ganho na vertical) | Raio efetivo por eixo, ajustado (s_x, s_y) | Equivalentes em espírito: os dois compensam a compressão maior na vertical. Aqui o ganho entra antes de somar a rotação da cabeça, que continua valendo 1:1 |
| z da íris do MediaPipe usado no feixe | Pupila na linha de visada da câmera até o plano do olho | Mantido: o z da íris é a coordenada mais ruidosa |
| Média móvel de 10 quadros na direção combinada | Filtro One-Euro na pupila, no referencial da cabeça | One-Euro: tremor de 7,7 px com 3 quadros de atraso, contra 6,3 px com 9 quadros de atraso e mais erro com a cabeça em movimento (seção 3.1) |
| Direções dos dois olhos combinadas pela média; monitor virtual a 50 cm à frente do rosto, na direção do olhar no momento da calibração | Intersecção dos feixes; monitor virtual no plano da tela, com a geometria real (webcam acima da tela) e correção virtual → real em 16 pontos | A referência usa só um ponto de calibração (o centro); a correção em vários pontos é a etapa 5 |

### 5.10 Precisão do ponto no monitor: de onde vem o erro

Decomposição do erro nas 33 calibrações gravadas até 13/11 (px):

- erro em pontos novos (LOO): 132; nos próprios alvos do ajuste, ~65 por
  eixo. A diferença é erro sistemático: a relação pupila medida → ângulo do
  olho não se repete bem de um alvo para outro;
- ruído dentro de cada alvo (olhar parado): 55 RMS. O LOO usa a mediana
  de cada alvo e quase não depende dele; no cursor, o filtro da pupila e o
  estabilizador o tratam;
- cada olho sozinho (só o monitor virtual): 175 (direito) e 198
  (esquerdo). Combinar os dois feixes ajuda pouco, e em 7 das 33 sessões
  um olho era mais de 1,5× pior que o outro.

O que foi testado para reduzir o erro (LOO / cabeça em poses novas /
validação, sem o desvio constante entre parênteses):

| Variante | LOO | cabeça | validação |
|---|---|---|---|
| base (pupila refinada ou do MediaPipe, escolhida por validação cruzada) | 132 | 324 | 190 (88) |
| termo quadrático no olho (compressão não linear da íris) | 131 | 332 | 228 (91) |
| intersecção ponderada pelo erro de cada olho | 130 | 333 | 186 (91) |
| compensação da cabeça aprendida na etapa da cabeça | 128 | 441 | 231 (87) |
| **pupila = média das duas medidas (adotada)** | **123** | **297** | 211 (**77**) |

Num segundo experimento, a íris foi medida de novo nos quadros originais
dos vídeos das sessões (18 sessões, 25 calibrações, landmarks idênticos),
trocando o centróide por um **ajuste de círculo à borda íris/esclera**
(raios a partir do centro do MediaPipe procurando a transição escuro →
claro dentro da abertura do olho, círculo por RANSAC):

| Medida da íris | fonte | LOO | cabeça | validação |
|---|---|---|---|---|
| centróide (atual) | refinada | 144 | 322 | 193 (91) |
| centróide (atual) | **média com o MediaPipe** | **125** | **292** | 192 (**78**) |
| borda (círculo) | refinada | 289 | 394 | 330 (193) |
| borda (círculo) | média com o MediaPipe | 193 | 356 | 297 (143) |

O ajuste de borda falhou em 8,6% dos olhos e foi muito pior: a 720p, a íris
tem ~5 px de raio na imagem, e um círculo ajustado a um arco tão pequeno,
cortado pelas pálpebras, é instável. O centróide usa a área inteira da
íris e é mais robusto nessa resolução.

Conclusões:

- O limite é a **medida da pupila**, não a geometria nem a forma da
  calibração: modelos mais flexíveis (quadrático, compensação da cabeça)
  não generalizam, e só medidas melhores (a média das duas) reduziram o
  erro.
- Na validação, a maior parte do erro bruto é um **desvio constante**
  entre calibrar e usar (190 px brutos contra 77–88 px sem ele), que a
  recentralização (`R`) e a correção por alvos conhecidos tratam.
- O próximo ganho provável é de **resolução**: a 1080p a íris tem 1,5× mais
  pixels que a 720p padrão (`--width 1920 --height 1080`, se a webcam
  suportar). Não há sessões gravadas em 1080p para medir.

## 6. Validação (`procedures.py` → `ValidationProcedure`)

A validação usa **9 alvos diferentes dos da calibração** (centro,
intermediários e bordas) e acontece em dois momentos, que medem coisas
diferentes:

- **Validação imediata**: roda sozinha logo depois da calibração
  (`ValidationConfig.after_calibration`). Mede o erro do **modelo**, sem
  tempo para a postura mudar.
- **Reteste** (tecla `T`, mais tarde): mede também a **deriva** desde a
  calibração. O primeiro alvo é o centro: o deslocamento medido nele é a
  deriva, e o erro nos outros 8 alvos é reportado bruto e **reancorado**
  (descontando a deriva do centro). No fim, a reancoragem é aplicada ao
  cursor, como na recentralização (se for menor que 25% da tela; senão o
  relatório pede para recalibrar).

Métricas, padrão em rastreamento ocular:

- **Acurácia**: distância média entre o alvo e a mediana das estimativas
  (erro sistemático);
- **Precisão (RMS)**: dispersão das estimativas em torno da própria média
  (ruído);
- **por região da tela**: centro, intermediária (até 25% do centro) e
  bordas;
- o **deslocamento médio** (erro constante) e a acurácia sem ele.

Todos os valores também saem em **graus de ângulo visual**,
θ = atan(erro_cm / distância_cm), com a largura física da tela e a
distância do rosto medida pelo próprio pipeline (ou `--distance-cm`).
Graus permitem comparar com a literatura e entre telas diferentes.

Cada validação gera três arquivos em `results/`:

- `validacao_<data>.csv`: uma linha por amostra (alvo, estimativa, ângulos,
  erro);
- `validacao_<data>.json`: resumo com o tipo (imediata ou reteste), os
  minutos desde a calibração, as métricas acima, a deriva no centro, a
  reancoragem aplicada e o resumo da calibração usada;
- `validacao_<data>.npz`: medidas brutas, instantes e landmarks, para
  reanálise.

## 7. Recentralização (`RecenterProcedure`)

O pipeline compensa rotação e translação da cabeça pela geometria. Ainda
assim, sobram deslocamentos sistemáticos: erro na distância estimada pelo
MediaPipe, a pupila que quase não contra-rotaciona na vertical (seção 5.6),
mudança grande de postura, óculos que escorregam. A recentralização mostra
só o alvo central, mede o deslocamento entre a estimativa e o centro e o
soma como um *offset* ao pixel do monitor real, sem refazer a calibração. A
postura em que se recentralizou passa a ser a referência do aviso de
postura.

Ao abrir o app com uma calibração salva (de outra sessão), a
recentralização roda **automaticamente** antes do uso
(`CalibrationConfig.recenter_on_start`): a postura e a posição da pessoa
mudam de uma sessão para outra, e esse desvio constante era a maior parte
do erro nas validações feitas depois de calibrar.

### 7.1 Aviso de postura

O modelo guarda a pose média da cabeça e a posição do rosto durante a
calibração, além da faixa de giro coberta pela etapa da cabeça. Durante o
uso, se a cabeça passar mais de 4° além dessa faixa ou o rosto se deslocar
mais de 4 cm, por mais de 1,5 s seguidos, o HUD mostra "Postura fora da
calibração: R para recentralizar". Movimentos rápidos não disparam o aviso.

### 7.2 Correção por alvos conhecidos

Quando a interface sabe para onde o usuário estava olhando (por exemplo, o
centro de uma tecla do teclado virtual que ele acabou de selecionar),
`EyeGazeApp.confirm_target(alvo)` usa as amostras dos últimos ~0,5 s para
corrigir o offset. O offset anda 30% do caminho até o erro medido, com
limite de 80 px por correção. Uma seleção errada desloca pouco o cursor, e
seleções corretas seguidas eliminam a deriva aos poucos (recalibração
implícita). Essa função é a base para o teclado virtual. Com o mouse do
sistema operacional, o app não sabe onde estão os botões e não pode
usá-la.

## 8. Controle do mouse e dwell-click (`io_devices.py`, `app.py`)

O cursor é movido com `pynput`, que funciona em Windows, macOS e Linux/X11
sem privilégios de administrador. A biblioteca `keyboard` do
Webcam3DTracker exige root no Linux.

O dwell-click dispara um clique quando o ponto estimado permanece dentro de
um raio (60 px) por um tempo (1 s), com intervalo mínimo entre cliques. É o
método de seleção mais comum em comunicação alternativa por olhar porque não
exige nenhum movimento além do olhar.

Regras de segurança:

- o mouse só liga com calibração;
- sem rosto no frame, o cursor fica parado;
- o dwell é reiniciado sempre que o controle é interrompido.

## 9. Visualização em tempo real (`visualization.py`)

Cada camada é ligada e desligada por uma tecla, durante a execução:

| Tecla | Camada |
|---|---|
| `L` | landmarks: malha facial, contorno dos olhos, contorno e centro das íris, cantos usados como âncora |
| `G` | olhar (etapas 2–3): para cada olho, o globo (círculo com raio de 0,4 × a largura do olho, o tamanho de um olho real), o centro, a pupila **medida** (ponto verde), a pupila **corrigida** pelo ganho da calibração (anel laranja, onde o feixe sai do globo) e o feixe; o vetor do olhar (seta), yaw/pitch e a posição da intersecção dos feixes. A distância entre o ponto verde e o anel laranja mostra quanto o MediaPipe encolhe o movimento da íris |
| `P` | eixos da cabeça (x vermelho, y verde, frente azul) e yaw/pitch/roll, com a fonte da pose |
| `V` | minimapa da tela: alvos de calibração, rastro recente, ponto filtrado (amarelo), ponto bruto (cinza) e progresso do dwell |
| `3` | vista 3D (`scene3d.py`): rosto, eixos da cabeça, globos (0,4 × largura do olho), pupilas medidas e corrigidas (laranja), feixes até a tela, intersecção (branco), vetor do olhar até o monitor virtual (magenta) e cursor no monitor real (amarelo). A diferença entre o magenta e o amarelo é o que a correção da etapa 5 faz |

As camadas geométricas são desenhadas no frame original e só depois
espelhadas (tecla `F`), para que textos e minimapa não fiquem invertidos.

## 10. Gravação e reprodução de sessões (`io_devices.py`)

`python main.py --gravar` grava o vídeo da câmera (MJPG, ~2–4 MB/s em 720p,
para preservar o detalhe da região dos olhos) e um JSON com o timestamp de
cada quadro, as teclas pressionadas (com o índice do quadro), a resolução
da tela e a semente da ordem aleatória dos alvos.
`python main.py --reproduzir results/sessao_X.json` roda o aplicativo
inteiro sobre o vídeo, usando os timestamps gravados como relógio e
reaplicando as teclas nos mesmos quadros. A sessão (calibração, validação e
recentralização) é refeita de forma determinística com o código atual, e os
resultados vão para `results/reproducoes/`, sem tocar na calibração salva.
Isso permite avaliar mudanças no estimador do olhar (por exemplo, o
refinamento da íris na imagem) com dados reais, coisa que as medidas
gravadas em `.npz` não permitem.

**Limitação.** A reprodução só é fiel enquanto o app se comporta no tempo
como na gravação. A coleta de cada alvo começa quando o olhar *estimado*
fixa, e o resultado de um procedimento (recentralização aplicada ou
rejeitada) muda quanto tempo a tela de resultado fica aberta. Com outro
modelo, o app pode mostrar um alvo enquanto a pessoa do vídeo ainda olha
o anterior: numa reprodução de 12/11 14:49, a validação ficou ~1 s
atrasada em relação ao vídeo e o erro medido (810 px) não tinha relação
com o modelo. A repetição de alvos ruins é desligada na reprodução, porque
o vídeo não atende a pedidos novos. Para comparar modelos, use
`tools/benchmark.py`, que reaproveita as janelas de coleta originais; a
reprodução serve para verificar o app de ponta a ponta.

## 11. Verificação (`tests/`)

Os testes usam um rosto 3D sintético, com pose da cabeça e direção do olhar
conhecidas, projetado em perspectiva por uma câmera pinhole no mesmo
formato de saída do MediaPipe. Eles verificam:

- sinais da pose da cabeça e a geometria (intersecção de retas, monitor
  virtual, orientação da tela);
- as medidas do olho em cm (âncoras a ~6 cm, pupila que ignora o z ruidoso
  da íris);
- que o ajuste do globo recupera o raio real num olho sintético "físico",
  que o feixe passa pela pupila e que os dois feixes se cruzam no alvo;
- que o raio efetivo por eixo absorve a compressão da íris, e que girar ou
  deslocar a cabeça mantém o erro baixo (seção 5.8);
- que a correção monitor virtual → real absorve uma largura de tela errada;
- que o estabilizador remove o tremor (< 3 px/frame), chega após sacadas em
  ≤ 5 frames, ignora picos isolados e acompanha deslocamentos lentos;
- recuperação exata da rotação quando a matriz de pose é fornecida;
- generalização da calibração para alvos fora da grade, simulando um monitor
  de 53×30 cm a 60 cm;
- o fluxo completo do aplicativo com um "usuário simulado" que olha para os
  alvos exibidos, segue as setas da etapa da cabeça e depois usa o cursor
  com a cabeça girada e deslocada (erro < 130 px, tremor < 4 px/frame);
- o protocolo de coleta (`tests/test_procedures.py`): ordem aleatória, coleta
  só depois da fixação (e no tempo limite se o olhar não estabiliza), etapa
  da cabeça para quem não move a cabeça e para quem gira demais, repetição
  de um alvo em que o usuário se distraiu, aviso de postura (sem disparar em
  giros rápidos) e correção por alvos conhecidos;
- gravação e reprodução de uma sessão com a mesma ordem de alvos e o mesmo
  resultado (`tests/test_recording.py`).

Esses testes validam a matemática e o software. A acurácia real depende da
câmera, da iluminação e do usuário, e deve ser medida com o procedimento de
validação (`T`).

## Referências

- Casiez, G.; Roussel, N.; Vogel, D. *1€ Filter: A Simple Speed-based
  Low-pass Filter for Noisy Input in Interactive Systems*. CHI 2012.
- Salvucci, D. D.; Goldberg, J. H. *Identifying Fixations and Saccades in
  Eye-Tracking Protocols*. ETRA 2000.
- Iglewicz, B.; Hoaglin, D. *How to Detect and Handle Outliers*. ASQC, 1993.
- Google. *MediaPipe Face Landmarker*. Documentação do MediaPipe Solutions.
- Holmqvist, K. et al. *Eye Tracking: A Comprehensive Guide to Methods and
  Measures*. Oxford University Press, 2011 (definições de acurácia e
  precisão).
- Projeto de referência: *Webcam3DTracker* (JEOresearch).
