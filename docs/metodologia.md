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
   pupila) (seção 3.2);
4. **projetar** cada feixe no **monitor virtual**, o plano da tela com a
   geometria suposta, e combinar os dois pontos numa média ponderada pela
   qualidade de cada olho (seção 3.3);
5. **calibrar** com vários pontos: ajustar o globo de cada olho e a correção
   entre o monitor virtual e o monitor real (seção 5).

Depois disso, o ponto passa pelo estabilizador de fixações (seção 4) e
move o mouse (seção 8).

| Etapa | Arquivo |
|---|---|
| 1. landmarks | `face_tracker.py`, `head_pose.py` |
| 2. globo e pupila | `gaze.py` (medidas), `eyeball.py` (modelo do globo) |
| 3. feixes | `eyeball.py` |
| 4. monitor virtual | `geometry.VirtualScreen`, `EyeballModel.screen_points` |
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
pupila vira ~2° no feixe: é daí que vinha a instabilidade dos feixes
desenhados. Filtrar aqui suaviza só o movimento do olho; a rotação e a
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

### 3.2 Etapa 3: feixe de cada olho

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

O **vetor do olhar** exibido sai do ponto médio entre os dois centros (o
"olho ciclópico") na direção da bissetriz dos dois feixes (com um olho
fechado, o feixe do outro).

**Por que não a intersecção dos feixes (versão até 15/11).** Até então, o
ponto fixado era a intersecção 3D dos dois feixes (ponto médio do segmento
perpendicular comum), e o vetor do olhar ia até ela. Com ~6 cm entre os
olhos e a tela a ~60 cm, a profundidade dessa intersecção é a coordenada
mais sensível ao ruído: ela oscilava 4–5 cm com o olhar parado, os feixes
convergiam à frente do rosto em só 50–100% das amostras, e usar sempre a
bissetriz dava exatamente o mesmo erro. A intersecção foi substituída pela
projeção de cada feixe (seção 3.3), como fazem os rastreadores comerciais.

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

O **feixe de cada olho** é prolongado até esse plano, e cada ponto
atingido vira um pixel pela geometria pura:

  x_px = L/2 − x_cm · px/cm        y_px = (y_cm − 1 cm) · px/cm

O pixel do olhar é a **média ponderada** dos pixels dos dois olhos. O peso
de cada olho é o inverso do seu erro angular de ajuste ao quadrado
(inverso da variância), medido na calibração; antes de calibrar, os dois
pesam igual. Com um olho fechado (ou com o feixe de um olho sem atingir a
tela), vale o outro. Nas 33 calibrações gravadas até 13/11, um olho errava
mais de 1,5× o outro em 7 sessões.

Esse é o "pixel virtual". Antes de calibrar, ele é o próprio cursor. A
etapa 5 aprende a correção entre ele e o pixel real.

### 3.4 Vetor bruto (entrada da calibração)

O estimador entrega, por quadro, um vetor com tudo o que as etapas 2–5
precisam, e é isso que é gravado nas sessões:

  raw = [R (9) | E (3) | A_d, P_d, A_e, P_e (12) | P_d e P_e refinadas (6) |
         olho direito aberto, olho esquerdo aberto | ponte do nariz (3) |
         largura do olho direito e esquerdo]

(A = meio dos cantos; pupilas sem filtro.) A abertura da boca saiu do vetor
em 16/11 (versão 10 das medidas), junto com a etapa da boca (seção 5.4).

(d = olho direito, e = esquerdo; tudo em cm no referencial da câmera.) O
globo, os feixes e o pixel virtual são recalculados a partir desse vetor, então as sessões gravadas podem ser reajustadas offline com
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
rápidos). Até 16/11, o `FixationStabilizer` (substituído pelo
`GazeFollower`, seção 4.0) foi inspirado nos detectores de fixação por
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

### 4.0 Versão de 16/11: perseguição suave no lugar da trava

O usuário relatou que o cursor "só se move em grandes blocos" e não segue
movimentos sutis do olhar. A causa era a trava: o raio crescia com o
passeio medido da estimativa (mediana de 188 px numa sessão de uso), e
movimentos menores que isso só apareciam como um salto quando o desvio
acumulado passava do raio. O estabilizador passou a ser o `GazeFollower`:

1. **sacada:** igual (limiar adaptativo, 2 amostras coerentes, salta para
   a mais recente), seguida de uma acomodação rápida até a mediana das
   amostras seguintes (8 quadros);
2. **fixação:** o destino é a mediana das últimas 7 amostras; a cada
   quadro o cursor anda uma fração α do caminho, com α = 0,01 abaixo de
   1,6 × o ruído de quadro a quadro (só ruído: o cursor quase não se mexe),
   α = 0,5 a partir de 2,5 × o ruído (movimento real) e transição linear
   entre os dois. O ruído é o de quadro a quadro (MAD das diferenças), não
   o passeio.

**Bancada.** As validações gravadas (20 sessões, 180 alvos, ~1,2 s cada)
com o ponto bruto de cada quadro; em cada alvo, a partir do 12º quadro,
soma-se um degrau sintético (um movimento sutil do olho) ao ruído real.
"Segue" = o cursor chega a menos de 40% do degrau do novo ponto e anda ao
menos metade do degrau, até o fim do alvo (~0,7 s depois):

| Filtro | segue 40 px | 70 px | 100 px | 150 px | cursor parado (olho parado) | tremor | erro no alvo |
|---|---|---|---|---|---|---|---|
| trava (até 16/11) | 0% | 5% | 11% | 75% | 97% | 1,3 px/quadro | 133 px |
| trava menor (lock_k 1,5, teto 0,5) | 2% | 42% | 69% | 83% | 91% | 3,1 | 135 |
| One-Euro na mediana (vários parâmetros) | 42-44% | 61-62% | 69-72% | 85-87% | 21-36% | 6,2-7,9 | 137-139 |
| perseguição suave (16/11, à tarde) | 38-41% | 61-62% (133 ms) | 66% | 83% | 72% | 3,9 | 135 |
| **histerese (adotada, 16/11 à noite)** | **28%** | **58%** (133 ms) | **64%** | **82%** | **86%** | **3,1** | **135** |

O passeio da estimativa (~30-50 px com o olho parado) limita os degraus
menores: qualquer filtro que siga um degrau de 40 px também segue um
passeio de 40 px. A perseguição suave foi escolhida numa varredura (mediana
de 5, 7 ou 9 quadros; limiares de 0,8 a 1,6 e de 2,5 a 3,5 × o ruído; α de
0 a 0,01 e de 0,3 a 0,5) como o menor tremor entre os que seguem ao menos
60% dos degraus de 70 px. O tremor com o olho parado é um deslize lento (o
cursor acompanha devagar o passeio), não um salto.

**Histerese (versão atual).** Com a perseguição suave o usuário relatou o
cursor "se movendo demais com a vista fixa": ele deslizava sempre um pouco
até o olhar e acompanhava o passeio. Na versão atual o cursor fica PARADO
enquanto o destino (mediana de 7 amostras) está a menos de 4 × o ruído de
quadro a quadro; ao passar disso, desliza (40% do caminho por quadro) até
ficar a menos de 1 × o ruído e para de novo. Numa varredura (liga a 2,5 a
4 × o ruído, desliga a 0,3 ou 1 ×, mediana de 7 ou 9), a combinação
escolhida deixa o cursor parado em 86% dos quadros com o olho fixo (72% na
perseguição suave), seguindo 58% dos movimentos de 70 px (62%).

As seções 4.1 e 4.2 descrevem a versão com trava, mantidas como histórico.

### 4.0.1 Ajuste de 17/11: movimentos sutis (`tools/bancada_cursor.py`)

Queixa no uso: o cursor "anda em blocos" e não acompanha ajustes pequenos.
A bancada (agora em `tools/bancada_cursor.py`) usa o ruído REAL de 428
fixações das validações gravadas (ponto bruto no monitor, quadro a
quadro), encadeia fixações com degraus de 40 a 400 px e mede a fração
seguida em até 0,5 s, a latência, o cursor parado com o olho fixo, o
tremor e o erro.

| Estabilizador | segue 40/70/100 px | latência | parado | tremor | erro |
|---|---|---|---|---|---|
| até 17/11 (liga a 4 × ruído, alpha 0,4, mediana 7) | 33% / 43% / 62% | 167-233 ms | 85% | 3,0 px | 27 px |
| **atual (liga a 2,5 ×, alpha 0,7, mediana 5)** | **53% / 58% / 76%** | **133-167 ms** | **87%** | 4,3 px | **25 px** |

Com o limiar de 4 × o ruído (≥ 60 px com o piso de 15 px), movimentos
abaixo de ~60 px nunca eram seguidos. Baixar só o limiar seguia mais, mas
deixava o cursor inquieto (parado em 72-76%); com o passo maior (alpha
0,7) o cursor chega e para em poucos quadros, e a mediana de 5 amostras
reage mais cedo. As sacadas (200-400 px) continuam em 67 ms.

Testado e descartado: manter o deslizamento até a mediana se renovar
(seguia 64% dos degraus de 40 px, mas o cursor ficava parado em só 65%
dos quadros com o olho fixo e o erro subia para 29 px).

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
4. **o olhar só "teleporta" o cursor**: cursor = âncora + ajuste da
   cabeça, e a âncora só muda numa nova fixação a mais de 20% da largura
   da tela do cursor (a pessoa olhou para outro lugar). Mudanças menores da
   estimativa do olhar são ignoradas; um giro maior que 10° zera o ajuste
   (mudança de postura);
5. depois de um clique, o cursor fica onde clicou (a âncora vai para ele e
   a cabeça vira a nova referência);
6. o clique por permanência só sai com o cursor parado (tolerância de
   12 px no modo A, contra 60 px fora dele).

**Por que a âncora (sessão de 16/11 12:49).** Com os olhos parados, mexer a
cabeça mexe a estimativa do olhar: ~28 px por grau na vertical (somando
com o ajuste da cabeça, o cursor andava o dobro) e ~25 px por grau na
HORIZONTAL ao inclinar a cabeça (o cursor "andava para o lado"). No modo A,
96% do movimento do cursor vinha do ponto do olhar e 7,5% do ajuste da
cabeça. Nas amostras do teste da cabeça (olhos fixos no centro, cabeça
girando até ±12°), o movimento do cursor que NÃO vinha da cabeça caiu de
545 para 220 px/s e de 1.146 para 437 px/s (saltos de 16 para 6 e de 31
para 11); o que sobra são giros grandes, em que a estimativa se afasta
mais que o limite de teleporte. Em ajustes de até ~5° (~125 px), o cursor
responde só à cabeça. Permitir o teleporte só com a cabeça parada foi
testado e não teve ganho consistente.

**Antes (sessão de 16/11 12:22, ~3,5 min de uso com o modo A).** Para ajustar o cursor com a cabeça, a pessoa olha para o
CURSOR, que está deslocado do alvo pelo erro do olhar (100-200 px). Esse
salto do olhar era tomado como uma nova fixação: o ajuste era zerado e o
cursor pulava para o novo ponto do olhar, num ciclo que impedia ajustes
delicados (23 ajustes de mais de 15 px perdidos assim, 43 px na mediana).
Além disso, o raio de trava do estabilizador chegou a 188 px (mediana):
movimentos do olhar menores que isso não movem o cursor, e o ajuste pela
cabeça é a única forma de chegar a um alvo pequeno. E, com a tolerância de
60 px do clique por permanência, os 24 cliques feitos com ajuste fino
saíram com o ajuste ainda em andamento (e a correção contínua, seção 7.3,
aprendia com esses pontos incompletos). O limite de giro também subiu de
6° para 10°: a 25 px/grau, 6° limitavam o ajuste a ~140 px, menos que o
erro do olhar. O ganho (25 px/grau) é menor que o físico (~38) para
favorecer a precisão. Serve só para quem tem algum controle da cabeça; o
modo só com o olhar continua sendo o padrão.

**Correção de 17/11: o cursor "ficava parado" no modo de precisão.** Na
sessão de 17/11 15:45, reprocessada quadro a quadro com o modelo da
própria sessão e o estado gravado (mouse, precisão), o olhar ficou 13
vezes por 0,5 s ou mais a 150-360 px do cursor sem que ele se movesse, e
TODAS no modo de precisão (nenhuma no modo normal). A âncora só mudava
numa nova fixação a mais de 20% da largura (384 px), e nunca dentro da
mesma fixação (olhar que desliza para outro lugar sem sacada nítida). Agora,
como no MAGIC pointing (Zhai et al., 1999), o olhar manda enquanto a
cabeça não está ajustando: a âncora acompanha o olhar estabilizado e o
cursor anda como no modo normal. Quando a cabeça passa da zona morta (o
ajuste começa), a âncora congela; ela volta a seguir o olhar numa nova
fixação a mais de 10% da largura (192 px, acima do erro do olhar) ou se o
olhar ficar longe por 0,3 s. Na mesma sessão: 13 -> 1 evento (162 px), e a
distância mediana entre o olhar e o cursor no modo de precisão caiu de
131 px (p90 313) para 54 px (p90 148), igual à do modo normal (48 / 158).

## 5. Etapa 5: calibração com vários pontos (`calibration.py`, `procedures.py`)

A pessoa olha para alvos conhecidos em duas etapas: a grade 3×4 e, depois,
4 alvos internos (refinamento; ~35 s no total). Com todas as amostras
juntas, a calibração ajusta, em ordem:

1. o **globo ocular** de cada olho (seção 5.2), que faz o feixe passar pelo
   alvo;
2. a **correção monitor virtual → monitor real** (seção 5.3), que leva o
   pixel virtual da etapa 4 ao pixel onde o alvo de fato está.

As duas etapas são problemas lineares (mínimos quadrados), sem otimização
iterativa e sem mínimos locais.

### 5.1 Alvos no referencial da câmera

Cada alvo da tela (pixel) vira um ponto 3D T no plano do monitor virtual
(seção 3.3).

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
globo. Cada alvo pesa 1, e uma regularização fraca (λ = 0,002 em unidades de tan²) puxa o ganho para 1,
com limites de 0,2 a 25. Como o centro depende de c e a direção até o alvo
depende do centro, o ajuste é repetido duas vezes (a mudança é de
milímetros a dezenas de cm).

Do projeto de referência, foi considerado e descartado o "travamento" do
globo num único quadro, olhando o centro da tela (centro = íris + raio na
direção da câmera). Ele herda todo o ruído daquele quadro, e o ajuste em
12 alvos estima o mesmo centro com centenas de amostras.

O mesmo ajuste dá o **peso de cada olho** no ponto da tela (seção 3.3): o
inverso do erro angular médio do olho ao quadrado, nas amostras de ajuste.

Até 15/11, a **etapa da cabeça** (olhar fixo, cabeça girando) também
entrava neste ajuste, para separar o movimento do olho do da cabeça. Nas 40
sessões gravadas, treinar com ela melhorava a validação em só ~3 px e o
erro com a cabeça em poses novas em ~12 px: a rotação da cabeça já entra
pela geometria (R). Ela passou a ser só um teste (seção 5.4).

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

**Grade 3×4 (3 colunas, 4 linhas):** margem de 5% e **ordem aleatória**. Alvos colados na borda
não ajudariam: ali a pálpebra cobre a íris e o MediaPipe é menos preciso.
Na ordem linha a linha, o tempo andava junto com a posição vertical: nos
dados reais, a abertura do olho caiu de 0,27 para 0,16 ao longo dos ~35 s
da grade 4×4 (cansaço, olho ressecando), e o modelo aprendia essa deriva
como "olhar para baixo". Com a ordem aleatória, a deriva vira ruído em vez
de viés.

**Quantos alvos.** `tools/ablacao.py` ajusta o pipeline com k alvos
sorteados de cada calibração gravada (grade 4×4) e mede o erro na validação
gravada logo depois (10 sessões, 5 sorteios por sessão; validação bruta e,
entre parênteses, sem o desvio constante):

| Alvos no ajuste | validação (px) |
|---|---|
| 6 | 158 (93) |
| 9 | 153 (87) |
| 12 | 151 (82) |
| 16 | 148 (80) |

Com 12 alvos sorteados a validação é a mesma de 16. Com grades REGULARES
de 12 alvos tiradas da 4×4 (mesmas 40 sessões; erro nos alvos fora do
ajuste e validação):

| Grade | alvos fora do ajuste | validação (px) |
|---|---|---|
| 4×4 (16) | 119 (LOO) | 148 (80) |
| 4×3, sem a 2ª linha | 125 | 157 (83) |
| 4×3, sem a 3ª linha | 112 | 155 (83) |
| **3×4, sem a 2ª coluna** | **121** | **148 (79)** |
| 3×3 (9) | 121 | 159 (89) |

Tirar uma coluna não muda nada; tirar uma linha piora ~5%: o eixo vertical
tem o sinal mais fraco e precisa das 4 linhas. A grade passou para 3×4
(de ~45–60 s para ~25–30 s, contando as checagens do centro que
existiam).

**Etapa 2: refinamento (16/11, à noite).** As plataformas de referência
calibram, validam com alvos novos e recalibram (EyeLink: calibração de 9
ou 13 pontos seguida de validação; Tobii: recalibração dos pontos ruins).
O teste com os dados gravados: em cada uma das 33 sessões com validação
gravada logo depois da calibração, os alvos da validação foram somados à
grade como pontos de calibração e o erro foi medido nos alvos que ficaram
de fora.

| Calibração | erro nos alvos fora do ajuste |
|---|---|
| só a grade | 105-111 px |
| grade + os outros 8 alvos internos (validação cruzada) | 86 px |
| **grade + 4 alvos nos quadrantes (30/70%)** | **85 px** (base 105) |
| grade + 4 quadrantes + centro | 86 px (base 106) |
| grade + cruz (centro, 20/80% na vertical, 15/85% na horizontal) | 95 px (base 119) |

Quatro alvos internos dão quase todo o ganho (−19%), por ~9 s a mais. O
ganho vem de cobrir melhor o interior e de os pontos serem vistos logo
antes do uso (parte do desvio constante, seção 5.11, já entra no ajuste).
A calibração passou a ter a etapa 2 (`CalibrationConfig.refine_points`),
e os 4 alvos diagonais da validação saíram de 30/70% para 20/80% da tela,
para a validação continuar independente da calibração. Nas duas primeiras
calibrações gravadas com a etapa 2 (16/11 17:17), a validação nos novos
alvos deu 77 e 56 px (2,0° e 1,4°; a segunda é o menor erro do projeto).

**Critério de aceitação (16/11, à noite).** A qualidade varia muito entre
calibrações da mesma sessão (LOO de 116 e 51 px nas duas acima). Testado
nas 35 sessões com validação:

| Estratégia | validação |
|---|---|
| repetir só os pontos ruins (regra atual: > 2,5× a mediana e > 154 px) | 107 px (sem repetir: 108) |
| repetir com regras mais rígidas (até 2,4 pontos por calibração) | 107 px |
| juntar duas calibrações seguidas num ajuste só | 108 px (só a mais recente: 90; só a anterior: 184) |
| calibrações com LOO <= 90 px (11 sessões) / > 90 px (24) | 76 / 123 px |

O problema é a calibração inteira (postura, luz, ruído naquele momento), não
pontos isolados, e uma calibração "envelhece" em minutos. A tela de
resultado passou a recomendar refazer quando o LOO passa de 4,7% da
largura da tela (`CalibrationConfig.redo_loo_frac`, ~90 px em 1920 px): o
ESPAÇO refaz na hora, sem salvar a calibração ruim; outra tecla aceita.

Para cada alvo, a coleta começa quando o olhar **fixa** no alvo: depois de
0,6 s de acomodação, espera-se que 6 quadros seguidos fiquem a menos de 0,8°
da mediana deles, com limite de 1,8 s. Assim, as primeiras amostras não são
mais da sacada ou de um olhar "procurando" o alvo (detecção de fixação por
dispersão, Salvucci e Goldberg, 2000). Seguem 1,2 s de coleta das amostras
válidas. Se o rosto sumir, o alvo espera até juntar pelo menos 8 amostras.
Em cada alvo, amostras com escore-Z modificado (MAD) > 3,5 são descartadas
(Iglewicz e Hoaglin, 1993).

**Teste da cabeça (`HeadStageProcedure`; no modo pesquisa, roda sozinho
depois da validação imediata; tecla `K` a qualquer momento):** o alvo fica
no centro da tela e uma seta pede que a pessoa gire a cabeça para
a direita, a esquerda, para cima e para baixo, sem tirar o olho do alvo, e
depois volte ao centro. A seta enche conforme a cabeça gira (medida em
relação à pose de referência do modelo, a mediana da grade ou a da última
recentralização) e fica verde ao chegar a 80% do
ângulo pedido (10° de yaw e 8° de pitch). Depois de 0,6 s ali, passa à
próxima direção. Quem não alcança o ângulo segue depois de 4 s. Se a
cabeça passar de 15° de yaw ou 12° de pitch, a seta fica vermelha e
aparece "girou demais": sem guia, as pessoas giravam até 30° (yaw) e 40°
(pitch), ângulos em que os landmarks do olho deixam de ser confiáveis.
Essas amostras **não entram no ajuste**: medem o erro com a cabeça em
poses novas (seção 5.5), gravado em `model.info` e em
`results/cabeca_<data>.npz`. Até 16/11 a etapa vinha logo depois da grade,
antes da validação, e o efeito passageiro dos giros contaminava a
validação (seção 5.11). No modo usuário (padrão) ela não roda: é
inadequada para quem não consegue mover a cabeça.

**Calibração manual (tecla C).** Cada alvo só avança quando o usuário,
olhando para ele, aperta ESPAÇO ou ENTER. As amostras usadas são as de 1 s
**antes** da tecla, sem os últimos 0,15 s (tempo de reação): ao apertar a
tecla é comum piscar ou olhar para o teclado. A confirmação é recusada (com
aviso) se o olhar ainda se movia nessa janela (80% das amostras a mais de
1,6° da mediana) ou se faltam amostras, e ignorada nos primeiros 0,5 s do
alvo. Ela não substitui a automática: o público do projeto (pessoas com deficiência
motora) muitas vezes não consegue apertar uma tecla a cada alvo.

**Etapas removidas em 16/11.** A calibração tinha mais duas etapas:

- *Boca* (~4–7 s): olhar fixo no centro, boca abrindo e fechando, para
  aprender quanto a abertura da boca desloca a estimativa (o MediaPipe
  deforma a malha inteira; com o olhar fixo, o ponto descia ~50 px de boca
  aberta) e descontar. Nas 40 sessões, a correção não mudou nenhuma métrica
  (LOO, cabeça e validação idênticos com e sem ela), porque só age de boca
  aberta, e o benefício de boca aberta nunca foi medido.
- *Checagens do centro* (5 pontos, ~10 s) para medir a deriva da pupila ao
  longo da grade e descontá-la por uma tendência linear (seção 5.4.1). Nas
  6 sessões com checagens, o LOO melhorou 8% e a validação piorou 6%:
  inconclusivo. Com a grade 3×4, a calibração ficou curta o bastante para
  derivar menos.

**Repetição de alvos ruins:** depois do ajuste, alvos com erro na validação
cruzada acima de 2,5× a mediana e acima de 8% da largura da tela (até 3
alvos) são coletados de novo, uma vez, e o modelo é reajustado. Isso cobre
piscadas longas, distração ou olhar fora do alvo.

As medidas brutas e os landmarks de cada calibração e validação são
gravados em `results/*.npz`. `tools/benchmark.py` recalcula as medidas com
o estimador atual e avalia variantes do pipeline em todas as sessões
gravadas. `tools/reavaliar.py` detalha uma sessão (globo ajustado,
convergência dos feixes, erro por alvo na validação).

#### 5.4.1 Correção da deriva (removida em 16/11)

Nas checagens do centro a pessoa olhava sempre para o mesmo ponto. A
diferença entre a pupila medida e a prevista em cada checagem (por olho e
eixo, no referencial da cabeça) era a deriva naquele instante; a calibração
ajustava uma tendência linear no tempo a essas 5 medidas, encolhida por
s²/(s² + 4·EP²) para não tomar ruído por deriva, e a subtraía de cada
amostra em relação ao fim da calibração. No usuário sintético com deriva
linear da pupila, o erro no estado do fim caía de 102 para 27 px (deriva de
0,04 cm). Com dados reais o ganho não se confirmou (seção 5.4). A deriva
DEPOIS da calibração continua tratada na validação (reteste) e na
recentralização (seções 6 e 7).

### 5.4.1 Calibração por perseguição (opcional, `--perseguicao`)

Na calibração por perseguição (*smooth pursuit*; Pfeuffer et al., 2013),
um ponto percorre a tela a velocidade constante e a pessoa o segue com os
olhos. Cada quadro vira uma amostra com o seu próprio alvo: centenas de
posições em ~13-21 s, contra 16 alvos em ~35 s na grade. Duas formas
(`CalibrationConfig.pursuit`):

- **"soma"**: grade 3x4 → perseguição pelo perímetro (margem de 5%) →
  etapa 2. O perímetro é onde o erro é maior (bordas, seção 5.11);
- **"alternativa"**: perseguição pelo perímetro e pelas linhas do meio
  (horizontal e vertical) + 5 alvos fixos (centro e 30/70%), no lugar da
  grade. Os alvos fixos mantêm o critério de aceitação (LOO) e a repetição
  de alvos ruins.

O ponto fica parado 1 s no início (o olho chega) e anda a 420 px/s (~7°/s
a 60 cm, abaixo do limite em que a perseguição suave passa a ter sacadas de
correção). Amostras descartadas: o primeiro 0,4 s de movimento (o olho
ainda acelera), 0,25 s depois de cada canto (o olho corta caminho e corrige
com uma sacada) e quadros com salto > 2° em relação ao anterior (sacadas,
olhar perdido). O alvo de cada amostra é a posição do ponto 0,1 s ANTES (o
atraso do olho).

No ajuste (`fit_from_samples`), as amostras da perseguição entram sempre
no treino com peso total `pursuit_weight` (6 "alvos fixos"; cada alvo fixo
pesa 1) e ficam fora das dobras da validação cruzada: o erro LOO continua
medido só nos alvos fixos, comparável ao da grade. A postura de referência
também usa só os alvos fixos. Os dados ficam no mesmo `calibracao_*.npz`
(`pursuit_*`), e `tools/reavaliar.py` os recalcula a partir dos landmarks.

Na simulação (usuário com o erro típico do MediaPipe, 3 sementes, erro no
reteste com 9 alvos novos): grade + etapa 2 22 px, "soma" 20 px,
"alternativa" 14 px. A simulação não tem o atraso variável nem as sacadas
de correção do olho real.

**Sessões reais (17/11, 4 calibrações com perseguição).** As amostras da
perseguição erram ~80-89 px no próprio ajuste (o olho real atrasa de forma
variável e corrige com sacadas), o dobro do erro de ajuste dos alvos
fixos. Na mesma sessão, sem a perseguição ("sem perseguição" no
benchmark): "soma" 164 -> 148 px e 92 -> 89 px na validação, ou seja, a
perseguição piorava um pouco; na "alternativa", ela é necessária (5 alvos
fixos sozinhos: 133 e 236 px, com ela 77 e 201 px), mas não supera a
grade. Pesos 3 e 12 não mudam o quadro. A perseguição continua opcional
e **desligada por padrão**.

### 5.5 Escolha e erro em pontos e poses novos

O tipo de correção e a força do ridge são escolhidos por **validação
cruzada deixando um alvo de fora**: para cada alvo, o globo E a
correção são reajustados sem ele, e o erro é a distância da mediana das
previsões ao alvo (pontos novos, cabeça parada). Fica a combinação de menor
erro médio.

Nas 40 sessões, fixar a correção (quadrática ou afim, α = 0,01) deu a
mesma validação (148 px) mas LOO pior (125 e 136 contra 119 px); a escolha
custa ~0,1 s e foi mantida.

**Cabeça em poses novas.** No modo pesquisa, as amostras da etapa da cabeça
(que não entram no ajuste) são divididas pela direção dominante do desvio
de pose (direita, esquerda, cima, baixo), descartadas as a menos de 2° da
pose da grade ou além de 15°/12°. O erro de cada direção é a mediana das
distâncias de suas amostras ao alvo central, e o reportado é a média das
direções (`erro_cabeca_px`).

A tela de resultado mostra o erro em pontos novos, o erro só com o monitor
virtual (sem correção, para ver quanto a etapa 5 corrige), o erro com a
cabeça em poses novas (modo pesquisa) e uma classificação da qualidade (boa até 4,5% da
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

**Versão de 16/11** (40 calibrações, 10 com validação; grade gravada 4×4;
o erro "cabeça" passou a ser medido com a etapa da cabeça FORA do ajuste,
então não é comparável à coluna acima):

| Variante | LOO | cabeça (teste) | validação (sem desvio) |
|---|---|---|---|
| versão de 15/11 (intersecção dos feixes, etapas da boca e da deriva) | 118 | — | 151 (82) |
| **16/11: ponto de cada olho na tela, média ponderada; sem boca e deriva** | **119** | **290** | **148 (80)** |
| olhos com peso igual | 118 | 289 | 151 (82) |
| correção quadrática fixa (α = 0,01) | 125 | 294 | 148 (82) |
| correção afim fixa (α = 0,01) | 136 | 282 | 148 (83) |
| só o monitor virtual | 153 | 307 | 160 (94) |
| pupila só do MediaPipe | 146 | 312 | 180 (105) |
| pupila só refinada | 133 | 334 | 151 (90) |

Remover as etapas da boca e da deriva e trocar a intersecção pela projeção
de cada feixe não piorou nada; a validação melhorou 3 px. Em graus, com a
distância medida pelo pipeline (mediana de 39 cm, que depende do campo de
visão suposto de 90°): LOO 3,0°, validação 3,9° (2,1° sem o desvio
constante). Com a distância real, informada em `tools/benchmark.py
--distance-cm`, os graus serão menores se a pessoa estava mais longe.

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

### 5.12 Análise de 17/11: o ganho vertical e o desvio que aparece depois

Com as sessões de 17/11 (validação por olho e perseguição) e as 44
validações gravadas:

- **O desvio constante continua dominando.** Benchmark: 110 px na
  validação, 74 px sem o desvio. Na sessão das 15:03, o alvo do centro
  errava +93 px (y) no fim da calibração e −299 px na validação, 6 s
  depois, sem mudança de postura (pitch da cabeça +0,5°, posição < 1 mm).
- **O sinal vertical é fraco e a calibração o amplifica.** Comparando o
  pitch das features entre o topo e a base da tela com a geometria real
  (largura da tela e distância medida): o ganho efetivo é ~2 × o
  geométrico na horizontal e **4-6 × na vertical** (o MediaPipe move a
  íris na vertical ~1/5 do que deveria). O ruído quadro a quadro das
  features é pequeno (6-26 px depois do ganho), mas um viés lento de 0,5°
  no sinal vertical vira 100-250 px na tela. Esse viés é o desvio
  constante.
- **Pálpebras não acrescentam informação ao modelo.** Num ajuste só no
  plano da imagem, somar a posição da pálpebra superior e a abertura do
  olho à íris reduzia o erro vertical (164 -> 104 px), mas o modelo
  geométrico atual já faz melhor (72 px no eixo y), e corrigir o resíduo
  dele com os sinais das pálpebras não mudou nada (96 -> 96-98 px).
- **Fonte da pupila.** A média das duas fontes continua a melhor (110 px;
  só MediaPipe 136, só refinada 115).
- **Correção do desvio logo depois de calibrar.** Os 4 primeiros alvos da
  validação (no tempo) estimam o desvio e os 5 seguintes medem o erro (48
  sessões): sem corrigir 105 px; corrigindo pela média de 1/2/3/4 alvos,
  122/99/94/90 px (com 1 piora, como já se sabia); pela média de 4
  encolhida pela incerteza (seção 7.1), **84 px** (mediana 88 -> 78),
  piorando só 19% das sessões (31% sem encolher). Daí a conferência de 4
  pontos depois de toda calibração.

### 5.11 De onde vem o desvio constante (análise de 16/11)

Nas validações, o erro bruto (137 px) é dominado por um desvio constante
(~106 px em média, de 7 a 332 px conforme a sessão; sem ele, 82 px). Quatro
hipóteses foram testadas com as 45 calibrações gravadas (15 com validação):

1. **Deriva no tempo durante a calibração.** Com a grade em ordem
   aleatória, o erro de cada alvo deixado fora do ajuste (com sinal) foi
   comparado ao instante em que o alvo foi visto: 0,5 px/s (correlação
   0,07). Não há deriva mensurável em ~25 s.
2. **Mudança de postura.** Na etapa da cabeça (olhar fixo no centro, fora
   do ajuste; 6.661 amostras com até ±15°/±12°), o erro vertical acompanha a
   inclinação da cabeça em ~12 px por grau (~47 px por cm de deslocamento
   vertical do rosto; rotação e translação andam juntas, correlação −0,96,
   e não dá para separá-las). Entre calibrar e validar, a postura mudou em
   geral menos de 2° e 1 cm: isso explica no máximo ~30 px.
3. **Viés de forma do modelo.** O erro com sinal dos alvos fora do ajuste
   depende da posição: +15 a +44 px em y nos alvos internos e −67 px na
   última linha (a correção extrapola para dentro). Na validação, só com
   alvos internos, o desvio médio das 15 sessões é (+34, +45) px: parte do
   desvio é esse viés, comum a todas as sessões.
4. **Efeito da etapa da cabeça (modo pesquisa).** Com a cabeça de volta à
   pose da grade (< 2°) e o olhar no centro, o erro no FIM da etapa da
   cabeça foi de ~207 px, quase sempre para cima (y de −115 a −358 px na
   maioria das sessões). O primeiro alvo da validação, logo depois e sempre
   o centro, tinha y negativo em 11 de 15 sessões, enquanto os demais alvos
   tinham y positivo na maioria. A etapa da cabeça deixa um efeito
   passageiro (postura que não volta exatamente, ou a malha do MediaPipe
   que se acomoda de outro jeito depois de giros grandes), que contamina o
   começo da validação. Isso também explica por que reancorar pelo centro
   (o primeiro alvo) piorava o erro (seção 7).

Como a ordem da validação era fixa (o centro sempre primeiro), o tempo
desde a etapa da cabeça e a posição do alvo ficavam confundidos. A partir
de 16/11 os alvos da validação são visitados em ordem aleatória (e o JSON
registra a ordem de cada alvo), e no modo pesquisa a validação imediata
vem ANTES do teste da cabeça, para que ela meça o modelo e não o efeito da
etapa (no modo usuário a etapa não existe).

**A precisão depois de mexer a cabeça (16/11, à tarde).** Nos testes da
cabeça das 5 calibrações de 16/11 com vídeo, com a cabeça de volta a menos
de 2° da pose da grade e o olhar no mesmo ponto, o erro mudou 90-330 px
(média 208 px) entre antes e depois dos giros, quase sempre para cima; a
postura em si explicaria ~30 px. Reprocessando os quadros dos vídeos:

| MediaPipe | mudança depois dos giros | precisão RMS | validação (sem desvio) |
|---|---|---|---|
| modo VIDEO (rastreamento entre quadros, o usado) | 208 px | 30 px | 99 (66) px |
| modo IMAGE (detecção do rosto em cada quadro) | 100 px | 85 px | 131 (87) px |
| VIDEO, reiniciando o rastreamento com a cabeça parada depois de um giro > 5° | 217 px | 36 px | 115 (76) px |

O modo IMAGE reduz o efeito à metade, mas quase triplica o tremor (o recorte
do rosto muda a cada quadro, e a posição dos landmarks depende do recorte:
o mesmo quadro dá um viés de −400 a −600 px em y entre os dois modos).
Reiniciar o rastreamento não muda nada, então o efeito não é o estado do
rastreamento. A causa continua aberta; as defesas são a recentralização de
4 pontos (seção 7.1) e a correção contínua pelos cliques (seção 7.3).

**Medida da íris (16/11, à tarde).** Com os quadros originais das 9
calibrações de 16/11 que têm vídeo e validação, o centro refinado da íris
foi recalculado com outras medidas (fonte da pupila: média com o MediaPipe):

| Medida do centro da íris | LOO | validação (sem desvio) | precisão RMS |
|---|---|---|---|
| **centróide dos pixels escuros (atual)** | **92 px** | **90 (68) px** | 34 px |
| centróide semeado pelo MediaPipe (peso gaussiano, σ = 1 raio) | 94 | 92 (69) | 33 |
| idem, σ = 0,7 raio | 95 | 93 (70) | 32 |
| gradientes (Timm e Barth, 2011) | 109 | 92 (69) | 35 |
| só o MediaPipe | 108 | 104 (79) | 21 |

Nenhuma supera a atual. Nas 24 sessões com validação, o peso da medida
refinada na média com o MediaPipe varia a acurácia e o tremor em sentidos
opostos (0,25: validação 130 px, RMS 28; 0,5: 119 px, RMS 37; 0,75: 116 px,
RMS 48); 0,5 continua. Coletar metade das amostras por alvo (0,6 s) piora
o LOO em 5% e a validação em 1,5%: alongar a coleta não compensa. O
próximo ganho provável é a resolução da imagem (1080p), ainda sem sessões
gravadas para medir.

**Bordas da tela (16/11, à noite).** Erro em pontos novos (LOO) por
posição do alvo, nas 60 calibrações gravadas: linha de cima (5%) 119 px,
linhas internas 88-104 px, linha de baixo (95%) 148 px; colunas laterais
123-129 px, coluna central 92 px. Duas tentativas de melhorar as bordas:

| Variante | LOO | borda de cima | borda de baixo | validação |
|---|---|---|---|---|
| correção até quadrática (atual) | 114 | — | 143 | 113 (75) |
| + termos cúbicos (u³, v³; escolhidos por validação cruzada) | 113 | — | 142 | 114 (75) |
| + abertura do olho (padronizada) na correção | 118 (base 120) | 114 (base 117) | 145 (base 149) | — |

Nenhuma muda o quadro: nos extremos, sobretudo olhando para baixo, a
pálpebra cobre a íris e o centro medido deixa de acompanhar o olho; a
correção não recupera um sinal que a medida não tem. O clique com zoom
(seção 8.1) é a forma de clicar com precisão também nas bordas.

## 6. Validação (`procedures.py` → `ValidationProcedure`)

A validação usa **9 alvos diferentes dos da calibração** (centro,
intermediários e bordas) e acontece em dois momentos:

- **Validação imediata** (modo pesquisa): roda sozinha logo depois da
  calibração (`ValidationConfig.after_calibration`) e só mede.
- **Reteste** (tecla `T`, mais tarde): mede e **corrige o desvio médio** dos
  alvos no cursor (se for menor que 25% da tela; senão o relatório pede
  para recalibrar).
- **Revalidação periódica** (`ValidationConfig.revalidate_every_min`,
  `--revalidar-min`; 10 min no modo pesquisa, desligada fora dele): o
  mesmo reteste, iniciado sozinho depois de N minutos sem calibração,
  recentralização ou validação, com o rosto visível. Como a checagem de
  deriva do EyeLink, mede quanto o erro cresce ao longo da sessão e corrige
  o desvio. Cancelar (ESC) só adia para o próximo intervalo.

**Por que a validação imediata continua (modo pesquisa).** Nas 15 sessões
com validação gravada até 16/11, o erro em pontos novos estimado na própria
calibração (LOO) quase não previa o erro da validação: correlação de 0,09
com o erro bruto e 0,43 com o erro sem o desvio constante. O erro da
validação é dominado por um **desvio constante** de ~106 px em média, que
aparece segundos depois de calibrar e varia de sessão para sessão (de 7 a
332 px). A postura mal muda nesse intervalo (em geral < 2° e < 1 cm), mas
o desvio acompanha o deslocamento vertical do rosto (correlação 0,68 com o
desvio em y): pequenas mudanças de postura viram desvios grandes. Só a
validação mede isso; por isso ela fica no modo pesquisa (no modo usuário,
não existe).

Métricas, padrão em rastreamento ocular:

- **Acurácia**: distância média entre o alvo e a mediana das estimativas
  (erro sistemático);
- **Precisão**: RMS em torno da própria média (desvio) e **RMS amostra a
  amostra** (RMS-S2S, o tremor quadro a quadro);
- **Perda de dados**: fração dos quadros da coleta sem amostra válida
  (rosto perdido, piscada);
- por região da tela: centro, intermediária (até 25% do centro) e bordas;
- **por olho** (`por_olho` no json): acurácia, precisão (RMS e RMS-S2S) e
  **disponibilidade** (fração dos quadros da coleta com o olho aberto e o
  feixe atingindo a tela) de cada olho sozinho, com a mesma correção e o
  mesmo offset do modelo, como no relatório do Tobii Pro. Mostra se um olho
  é o problema (pálpebra caída, reflexo nos óculos, luz de um lado só) e
  quanto a média ponderada dos dois ganha sobre o melhor olho;
- o **desvio médio** (constante), a acurácia sem ele (dentro da amostra) e
  a **acurácia recentralizada**: cada alvo corrigido pela média dos erros
  dos OUTROS alvos (validação cruzada), o erro que sobra depois de uma
  recentralização com vários pontos.

Todos os valores também saem em **graus de ângulo visual**,
θ = atan(erro_cm / distância_cm), com a largura física da tela e a
distância do rosto medida pelo próprio pipeline (ou `--distance-cm`).

Cada validação gera três arquivos em `results/`:

- `validacao_<data>.csv`: uma linha por amostra (alvo, estimativa, ângulos,
  erro);
- `validacao_<data>.json`: resumo com o tipo (imediata ou reteste), os
  minutos desde a calibração, as métricas acima, a correção aplicada e o
  resumo da calibração usada;
- `validacao_<data>.npz`: medidas brutas, instantes e landmarks, para
  reanálise.

## 7. Correção do desvio constante

Como o desvio constante domina o erro no uso, o app tem três formas de
corrigi-lo, todas somando um *offset* ao pixel do monitor real, sem refazer
a calibração. O que os dados dizem sobre cada uma vem das validações
gravadas (15 sessões; o erro em cada alvo depois de corrigir o desvio
estimado com outros alvos):

| Desvio estimado com | erro nos demais alvos |
|---|---|
| nada | 138 px |
| 1 ponto (o centro) | 147 px |
| 2 pontos | 103 px |
| 3 pontos | 96 px |
| 5 pontos | 91 px |
| 8 pontos | 88 px |

Com um ponto só a correção **piora** o erro (a estimativa de um ponto tem
o próprio erro do modelo naquele lugar, de 50-100 px); com 3 ou mais, ele
cai ~30%.

### 7.1 Recentralização (`RecenterProcedure`, tecla `R`)

Mostra **4 alvos** (30%/70% da largura e da altura, em ordem aleatória;
`CalibrationConfig.recenter_points`, ~9 s), mede o erro em cada um e soma a
**média** como offset. Até 16/11 era só o centro, o que piorava o erro em
média. A tela de resultado mostra também a dispersão entre os 4 erros (uma
dispersão alta indica que o desvio não é constante e que vale recalibrar).
Desvios maiores que 25% da tela não são aplicados. A postura em que se
recentralizou passa a ser a referência do aviso de postura.

Ao abrir o app com uma calibração salva, a recentralização roda
**automaticamente** antes do uso (`CalibrationConfig.recenter_on_start`).

**Conferência depois de calibrar (17/11).** A mesma recentralização roda
logo depois de toda calibração (`CalibrationConfig.check_after_calibration`,
~9 s), como a *drift correction* do EyeLink e a validação com correção do
Tobii: o desvio constante aparece nos primeiros segundos de uso (seção
5.12). No modo pesquisa, a validação imediata vem depois dela e mede o
erro já com a correção.

**Desvio encolhido pela incerteza** (`procedures.shrink_offset`; vale para
a recentralização, a conferência, o reteste e a revalidação periódica). A
média dos erros em poucos alvos é ruidosa: corrigir por ela inteira
piorava 31% das sessões. O desvio aplicado é a média multiplicada por
f = |m|² / (|m|² + 4·se²), em que se² é a variância dos erros entre os
alvos dividida pelo número de alvos (estimador de James-Stein). Um desvio
coerente (os 4 alvos erram para o mesmo lado) tem f ~ 1; um desvio do
tamanho do ruído, f ~ 0. Nas 48 validações: 90 px com a média inteira, 85
px com metade fixa, **84 px com f adaptativo**. Com um desvio artificial
de 180 px somado a todos os alvos (recentralização manual com o cursor
claramente deslocado), metade fixa deixava 123 px e o fator adaptativo
91 px.

### 7.2 Aviso de postura

O modelo guarda a pose média da cabeça e a posição do rosto durante a
calibração e a faixa de giro coberta na grade. Durante o uso, se a cabeça
passar mais de 4° além dessa faixa ou o rosto se deslocar mais de 4 cm, por
mais de 1,5 s seguidos, o HUD mostra "Postura fora da calibração: R para
recentralizar". Movimentos rápidos não disparam o aviso.

### 7.3 Correção contínua por alvos conhecidos

Quando se sabe para onde a pessoa olhava, `EyeGazeApp.confirm_target(alvo)`
usa as amostras dos últimos ~0,5 s para corrigir o offset: ele anda 30% do
caminho até o erro medido, com limite de 80 px por correção. Simulando nas
validações gravadas os alvos chegando um a um, essa regra levou o erro de
138 para 101 px (94 px depois de 7 alvos); a média de todos os alvos
anteriores dá o mesmo (99 px), e usar só o último alvo, 116 px.

A dificuldade é saber o alvo. Num clique pelo olhar, o clique cai onde o
cursor (a própria estimativa) está, e "a posição clicada" não ensina nada.
As fontes de alvos conhecidos são:

- **cliques depois do ajuste fino pela cabeça** (tecla `A`, seção 4.3): a
  pessoa corrigiu o cursor com a cabeça antes do clique, então o ponto
  final é o alvo e o ajuste feito é o erro do olhar ali. Cada clique por
  permanência com um ajuste de mais de 15 px corrige o offset
  (`AdaptConfig.learn_from_fine_clicks`) e zera o ajuste fino;
- **interfaces que sabem onde estão os alvos** (teclado virtual, ou, no
  Windows, a UI Automation, que informa o retângulo do botão sob o cursor;
  ainda não implementada), chamando `confirm_target` com o centro do
  elemento selecionado.

A correção acumulada é salva com a calibração ao sair do app.

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
- o dwell é reiniciado sempre que o controle é interrompido;
- olhar para a barra de controle nunca gera clique do sistema.

### 8.0 Uso sem teclado: barra de controle pelo olhar (`gaze_bar.py`, `control.py`)

O público do projeto muitas vezes não usa teclado. Como nas barras do Tobii
Dynavox e do Windows Eye Control, uma coluna de 5 botões grandes (170 × 100
px, maiores que o erro típico do olhar) fica na borda esquerda da tela,
centralizada na altura, durante o uso (longe dos menus e da barra de título,
no topo, e da barra de tarefas, embaixo):
Mouse (pausa/religa), Clique (por permanência), Precisão (ajuste pela
cabeça), Recentralizar e Recalibrar. Olhar 1,2 s para um botão o aciona,
com intervalo de 1,5 s entre acionamentos. A barra fica colada na borda de
esquerda: um olhar estimado à esquerda da tela ainda cai no botão (borda
"infinita", lei de Fitts). A barra funciona
com o mouse pausado, e olhar para ela não gera clique do sistema. Depois de
uma calibração (ou da recentralização ao abrir o app), o mouse liga
sozinho. Desenhar a barra custa ~4 ms quando algo muda e ~0,15 ms quando
nada muda (ela só é redesenhada quando um estado ou o progresso muda).

A janela da barra é do próprio OpenCV, sem moldura, sempre na frente e sem
roubar o foco (estilos do Windows ajustados pela API do sistema). A primeira
versão usava o tkinter, que a cada quadro processava todas as mensagens do
programa, inclusive as teclas destinadas à janela da câmera: os atalhos do
teclado paravam de funcionar.

O uso do cursor (mouse, clique por permanência, modo de precisão, correção
contínua e barra) fica em `control.CursorControl`; o `app.py` cuida da
câmera, do pipeline, dos procedimentos e do desenho.

### 8.0.1 Desempenho em segundo plano (`system.py`)

Nas sessões gravadas de 16/11, com o clique por permanência e o modo de
precisão ligados, o app caía de ~31 para 13-16 quadros por segundo depois
dos primeiros cliques e não voltava, mesmo com o clique desligado (sessão
12:22: 31,2 fps antes, 12,8 fps depois, 20,8 fps com o clique já
desligado). O processamento não explica isso: um quadro leva ~15 ms
(MediaPipe com a íris refinada 8,4 ms, desenho da câmera 3,8 ms, modelo
0,8 ms), contra 33 ms entre quadros da câmera. O clique cai em outra janela
e o app passa para segundo plano; o Windows 11 desacelera processos em
segundo plano (EcoQoS) e ignora o temporizador de 1 ms deles, e cada espera
curta do laço vira ~15 ms. Ao iniciar, o app pede prioridade acima do
normal e desliga essas duas desacelerações para o próprio processo
(`SetProcessInformation` com `ProcessPowerThrottling`, e `timeBeginPeriod(1)`).

### 8.1 Clique com zoom (removido)

Em 16/11 o app teve um clique com zoom (o primeiro clique por permanência
abria uma lupa 3× em volta do cursor e o segundo clicava no ponto
correspondente, dividindo o erro do olhar por 3). Foi removido a pedido do
usuário: a etapa extra dificultava o uso da ferramenta, e o foco passou a
ser a precisão da calibração e da detecção.

## 9. Visualização em tempo real (`visualization.py`)

Cada camada é ligada e desligada por uma tecla, durante a execução:

| Tecla | Camada |
|---|---|
| `L` | landmarks: malha facial, contorno dos olhos, contorno e centro das íris, cantos usados como âncora |
| `G` | olhar (etapas 2–3): para cada olho, o globo (círculo com raio de 0,4 × a largura do olho, o tamanho de um olho real), o centro, a pupila **medida** (ponto verde), a pupila **corrigida** pelo ganho da calibração (anel laranja, onde o feixe sai do globo) e o feixe; o vetor do olhar (seta), yaw/pitch e o peso de cada olho. A distância entre o ponto verde e o anel laranja mostra quanto o MediaPipe encolhe o movimento da íris |
| `P` | eixos da cabeça (x vermelho, y verde, frente azul) e yaw/pitch/roll, com a fonte da pose |
| `V` | minimapa da tela: alvos de calibração, rastro recente, ponto filtrado (amarelo), ponto bruto (cinza) e progresso do dwell |
| `3` | vista 3D (`scene3d.py`): rosto, eixos da cabeça, globos (0,4 × largura do olho), pupilas medidas e corrigidas (laranja), feixes até a tela com o ponto de cada olho (branco), ponto do olhar no monitor virtual, a média ponderada dos olhos (magenta) e cursor no monitor real (amarelo). A diferença entre o magenta e o amarelo é o que a correção da etapa 5 faz |

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

- sinais da pose da cabeça e a geometria (retas, monitor virtual,
  orientação da tela);
- as medidas do olho em cm (âncoras a ~6 cm, pupila que ignora o z ruidoso
  da íris);
- que o ajuste do globo recupera o raio real num olho sintético "físico",
  que o feixe passa pela pupila, que o feixe de cada olho atinge o alvo no
  monitor virtual e que o olho mais ruidoso pesa menos;
- que o raio efetivo por eixo absorve a compressão da íris, e que girar ou
  deslocar a cabeça mantém o erro baixo (seção 5.8);
- que a correção monitor virtual → real absorve uma largura de tela errada;
- que o estabilizador remove o tremor (< 3 px/frame), chega após sacadas em
  ≤ 5 frames, ignora picos isolados e acompanha deslocamentos lentos;
- recuperação exata da rotação quando a matriz de pose é fornecida;
- generalização da calibração para alvos fora da grade, simulando um monitor
  de 53×30 cm a 60 cm;
- o fluxo completo do aplicativo com um "usuário simulado" que olha para os
  alvos exibidos, segue as setas da etapa da cabeça (modo pesquisa) e
  depois usa o cursor
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
- Zhai, S.; Morimoto, C.; Ihde, S. *Manual and Gaze Input Cascaded (MAGIC)
  Pointing*. CHI 1999.
- Pfeuffer, K.; Vidal, M.; Turner, J.; Bulling, A.; Gellersen, H. *Pursuit
  Calibration: Making Gaze Calibration Less Tedious and More Flexible*.
  UIST 2013.
