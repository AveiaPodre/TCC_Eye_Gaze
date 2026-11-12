# Metodologia

Este documento descreve o método implementado, na ordem do pipeline. Cada
seção indica o arquivo correspondente. Use-o como base para o capítulo de
metodologia do TCC.

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

## 3. Modelo do globo ocular (`gaze.py`)

Cada olho é tratado como uma esfera de raio r que gira dentro da cabeça. A
íris fica na superfície da esfera e o olhar é a direção do centro da esfera
até a íris, como no Webcam3DTracker. Há três diferenças:

1. **Âncora a cada frame.** O centro do olho é derivado dos cantos do olho
   (pontos fixos no crânio) em todo frame. Ele não é "travado" uma vez na
   calibração, então não acumula erro quando a pessoa se mexe.
2. **Normalização.** O deslocamento da íris é medido no referencial da
   cabeça e dividido pela largura do olho. Fica independente da distância à
   câmera e da rotação da cabeça.
3. **Sem o z da íris.** O z é a coordenada mais ruidosa da íris. A
   profundidade sai da restrição de a íris estar na esfera.

Para cada olho, sendo m o ponto médio dos cantos e w a distância entre eles:

  (dx, dy, ·) = Rᵀ (íris − m) / w

  pitch_olho = asin(−dy / ρ)
  yaw_olho   = asin( dx / (ρ · cos pitch_olho))

ρ = r / w é o raio do globo em "larguras de olho". O padrão 0,40 vem da
anatomia: raio de ~12 mm sobre uma fissura palpebral de ~30 mm. Um erro em ρ
só altera a escala dos ângulos, e a calibração absorve isso.

A direção do olho no referencial da cabeça,
d_c = (sen yaw·cos pitch, −sen pitch, −cos yaw·cos pitch), vai para a câmera
com d = R · d_c. Os olhos abertos são combinados pela média das direções.
Saem daí dois ângulos no referencial da câmera:

  yaw   = atan2(−d_x, −d_z)        (> 0: olhando para a direita do usuário)
  pitch = atan2(−d_y, √(d_x² + d_z²)) (> 0: olhando para cima)

Como d combina a rotação da cabeça com a rotação do olho, uma pessoa que
olha para o mesmo ponto girando a cabeça (e compensando com os olhos) gera
os mesmos ângulos, **desde que ρ esteja certo**. O teste
`test_gaze_with_matrix_is_exact_and_head_invariant` verifica essa
propriedade num rosto sintético com ρ exato.

Com pessoas reais, ρ fixo não basta, por três motivos:

- o tamanho do olho e da fissura palpebral varia de pessoa para pessoa;
- na vertical, as pálpebras acompanham e cobrem a íris, e o deslocamento
  vertical medido é menor que o real;
- a inclinação da cabeça (pitch) altera a posição aparente da íris em
  relação aos cantos (perspectiva e pálpebras).

Com a cabeça parada, esses erros são absorvidos pela calibração. Quando a
cabeça gira, o olho compensa com o ganho errado e o cursor anda. Na versão
anterior, isso causava centenas de pixels de erro ao inclinar a cabeça 10°.
A seção 5 descreve como esses parâmetros são aprendidos por usuário.

### Saída para a calibração

Por isso o estimador não entrega só (yaw, pitch). Ele entrega um vetor bruto
por frame:

  raw = [R (9 valores), E (3 valores, cm), u, v]

- R: rotação da cabeça (seção 2).
- E: posição 3D do rosto em relação à câmera, em cm. Vem da translação da
  matriz de pose do MediaPipe. Sem a matriz, é estimada por câmera pinhole a
  partir da distância entre os cantos externos dos olhos (~9 cm em adultos).
- u, v: deslocamento médio da íris (dos olhos abertos) no referencial da
  cabeça, em larguras de olho.

(yaw, pitch) continuam sendo calculados com os parâmetros padrão. São usados
no HUD, antes da calibração e para rejeitar outliers.

### Piscadas

Para cada olho, EAR = distância entre as pálpebras / largura do olho. O olho
conta como fechado quando o EAR cai abaixo de 55% da sua linha de base. A
linha de base é uma média móvel do EAR com o olho aberto, então se adapta a
cada pessoa sem ajuste manual. Com um olho fechado, usa-se só o outro. Com
os dois fechados, o frame é marcado como piscada e o cursor mantém a última
posição.

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

## 5. Calibração (`calibration.py`, `procedures.py`)

### 5.1 Por que o modelo mudou (versão 3)

A versão anterior era um modelo geométrico do globo ocular com parâmetros
do usuário (ganhos, offsets, acoplamento) otimizados por Nelder-Mead. Nos
testes com webcam real ele falhou (sessão de 02/11/2026, validação com
erro médio de 182 px):

- **Sinal vertical fraco:** em alvos de cima a baixo da tela, o pitch
  medido variou só ~3°, contra os ~15–18° esperados a 60 cm. O centro da
  íris do MediaPipe quase não se move na vertical em relação aos cantos do
  olho. Todas as estimativas de y ficaram entre 421 e 577 px.
- **Viés de ~30° no pitch:** olhando o centro da tela (abaixo da câmera), o
  pitch medido era +13° ("para cima").
- **Ajuste degenerado:** com esses dados, o otimizador levou os parâmetros
  aos limites (offset horizontal de −31°) e nem os próprios alvos da
  calibração foram ajustados (RMSE de 302 px).

### 5.2 Medidas do olho

Além do deslocamento da íris em relação aos cantos (u, v), o estimador
mede as **pálpebras** no referencial da cabeça (média de 3 pontos em cada
pálpebra):

- lid_up: altura da pálpebra superior em relação ao meio dos cantos, com o
  z do próprio MediaPipe (ver 5.2.1);
- abertura: distância entre as pálpebras.

Todas são divididas pela largura do olho. A pálpebra superior acompanha o
olhar vertical (desce quando se olha para baixo), então lid_up e a abertura
trazem o sinal vertical que falta em v. A versão anterior usava, no lugar
de lid_up, lid_v (posição da íris entre as pálpebras), que nas sessões
reais não variava com o olhar (r² = 0,00 contra a posição vertical do alvo).

#### 5.2.1 Coerência com a rotação da cabeça: por que lid_up

Para compensar a cabeça, uma medida do olho precisa responder à rotação do
olho **dentro da cabeça** do mesmo jeito em duas situações: quando quem se
move é o olho (grade de alvos, cabeça parada) e quando quem se move é a
cabeça (etapa da cabeça: olhar fixo no centro, cabeça inclinando, olho
girando ao contrário). Para cada medida candidata, ajustou-se a inclinação
medida × ângulo do olho na cabeça separadamente na grade e na etapa da
cabeça (só inclinação vertical, |yaw| < 4°). O ângulo do olho na cabeça vem
da posição do alvo, da posição 3D do rosto e da rotação da cabeça. A razão
entre as duas inclinações deveria ser 1:

| Medida vertical | Razão (sessão 1 / sessão 2) | r² na grade | Ruído entre alvos |
|---|---|---|---|
| íris vertical v (profundidade pela esfera) | 0,15 / −0,04 | 0,68 / 0,80 | 5,5° / 4,0° |
| abertura (profundidade pela esfera) | 0,07 / 0,06 | 0,89 / 0,92 | 2,8° / 2,3° |
| pálpebra superior (profundidade pela esfera) | 0,01 / −0,07 | 0,87 / 0,92 | 3,0° / 2,4° |
| íris refinada na imagem | 0,47 / 0,29 | 0,86 / 0,93 | 3,1° / 2,2° |
| pálpebra inferior (z do MediaPipe) | −4,7 / −4,5 | 0,58 / 0,64 | 6,7° / 5,9° |
| **pálpebra superior (z do MediaPipe) = lid_up** | **1,03 / 1,11** | **0,92 / 0,94** | **2,3° / 1,9°** |

As medidas anteriores mudavam com o olhar na grade, mas praticamente não
mudavam quando a cabeça inclinava com o olhar fixo. A calibração então
aprendia relações contraditórias no eixo vertical: a etapa da cabeça piorava
o ajuste da grade sem trazer invariância. A diferença entre a pálpebra
superior "pela esfera" (razão ~0) e "pelo z do MediaPipe" (razão ~1) mostra
que o problema estava na correção de profundidade: a pálpebra não fica
exatamente sobre a superfície do globo, e trocar o z dela pelo da esfera
distorcia a medida conforme a inclinação. Para a íris, a correção pela
esfera continua necessária (ver abaixo): com o z do MediaPipe, a razão da
íris vertical fica em 5–6.

#### Profundidade da íris pela esfera do olho

Para levar a íris ao referencial da cabeça, o vetor íris → meio dos
cantos é girado por Rᵀ. A versão anterior usava o z da íris dado pelo
MediaPipe, que é ruidoso. Com a cabeça girada, esse z "vazava" para u e
v. Na sessão de 02/11 (13h30), u variava 4,5× mais por grau de cabeça do
que por grau de olho, e v ~14×. Agora só x e y da imagem são usados. A
profundidade é obtida pela interseção da linha de visada com a esfera do
olho, ancorada nos cantos (`gaze.on_eyeball`). Na abertura também; já
lid_up usa o z do MediaPipe (ver 5.2.1).

#### Limite de resolução e refinamento da íris na imagem

Com a câmera a 1280×720, o olho tem ~48 px de largura. Na sessão de
02/11 (13h52), a íris do MediaPipe andou só ~4 px da imagem para a tela
inteira, ou seja, 1 px da imagem ≈ 390 px da tela. A geometria do olho
prevê de 3 a 4 vezes mais deslocamento: o modelo do MediaPipe regulariza a
íris em direção ao centro do olho. Nenhum modelo de calibração ficou
abaixo de ~100 px de erro LOO com essas medidas.

`iris_refine.py` mede o centro da íris direto nos pixels: dentro do
polígono das pálpebras, calcula o centróide dos pixels mais escuros que o
percentil 35 da região, ponderado pelo quanto são mais escuros. A região
é ampliada 4× e suavizada. O resultado é sub-pixel e não é regularizado.
Ele entra como uma segunda fonte de medidas (colunas 16–19 do vetor
bruto). A calibração ajusta o modelo com as duas fontes e escolhe a de
menor erro LOO, então o refinamento nunca piora o resultado em relação ao
MediaPipe. Custo: ~3 ms por frame.

### 5.3 Modelo de regressão

A geometria fica só onde ela é confiável, na cabeça. O ponto para onde a
face aponta no plano da câmera (que coincide aproximadamente com o plano da
tela) é:

  P_cabeça = E_xy + E_z · tan(direção da face)   (cm)

O olho entra pelas medidas brutas φ = [u, v, lid_up, abertura],
centralizadas na média da calibração, mais os termos quadráticos u²,
u·lid_up, lid_up² e u·v no grau 2. Elas são multiplicadas pela distância E_z
(o mesmo ângulo de olho cobre mais centímetros de longe) e por
sec²(ângulo da cabeça), a derivada de tan(cabeça + olho):

  x = c + a·P_cabeça + Σ_k b_k · E_z · sec²(yaw_cabeça) · φ_k
  y = c + a·P_cabeça + Σ_k b_k · E_z · sec²(pitch_cabeça) · φ_k

O modelo é **linear nos coeficientes**, ajustado por mínimos quadrados com
ridge em forma fechada: sem otimização iterativa e sem mínimos locais.
As colunas são padronizadas, com desvio-padrão mínimo de 1 cm nas de
P_cabeça. Assim, se a cabeça ficou quase parada na calibração, o ruído
dessa coluna não é amplificado.

#### Modelo físico (candidato)

A regressão acima ("livre") tem ~20 coeficientes, e cada eixo tem os seus
para a cabeça. Quando há etapa da cabeça, entra na escolha (5.5) também um
modelo com só 3 coeficientes:

  x = c_x − a·P_x + b_x · E_z · sec²(yaw_cabeça) · u
  y = c_y + a·P_y + b_y · E_z · sec²(pitch_cabeça) · lid_up

O coeficiente a (px por cm na tela) é **o mesmo nos dois eixos**, porque os
pixels são quadrados. O sinal negativo em x vem da imagem não espelhada (a
direita do usuário é −x na câmera). Cada eixo usa só a medida do olho
coerente com a cabeça (u no horizontal, lid_up no vertical). O giro
horizontal da etapa da cabeça, que é o mais confiável, determina a, e esse
valor passa a valer também na vertical. As equações dos dois eixos são
resolvidas juntas (ridge com coeficientes compartilhados e um intercepto
por eixo). Nas sessões reais, o a ajustado ficou em 56–65 px/cm, compatível
com uma tela de 1920 px e ~30–34 cm de largura. Sem a etapa da cabeça, a
fica indeterminado (a cabeça quase não se move na grade) e o modelo físico
não é considerado. Quando a largura física da tela é conhecida, a não é
aprendido: ver a seguir.

#### Ganho da cabeça pela geometria (largura da tela conhecida)

**Problema relatado.** O eixo vertical "não alcançava" a parte de baixo
da tela, como se a câmera esperasse a cabeça sempre voltada para ela. O
ganho do modelo à inclinação da cabeça explica isso: girando a matriz de
rotação de uma amostra típica com as medidas do olho fixas (a cabeça
inclina e o olho acompanha, como ao abaixar a cabeça para olhar para
baixo), o cursor andava só 12, 27 e 23 px por grau de pitch nas três
sessões gravadas. A geometria pede ~38 px/grau (distância × π/180 ×
px/cm). A etapa da cabeça é curta (na sessão de 12/11, 10–19 amostras por
direção) e várias medidas do olho não respondem à cabeça, então o
coeficiente aprendido saía 1/3 a 2/3 do físico. Remover a penalidade do
ridge nesse coeficiente não mudou nada: o encolhimento vinha dos dados.

**Solução.** Se a cabeça gira θ com o olho parado dentro dela, o olhar
gira θ e o cursor tem de andar a·ΔP_cabeça, com a = px/cm da tela. Com a
largura física da tela, a é conhecido. No Windows ela é lida do monitor
(`GetDeviceCaps(HORZSIZE)`, valor do EDID; 34,4 cm no notebook dos testes,
aceito só se a proporção bater com a da resolução); nos outros sistemas,
`--screen-width-cm`. A coerência da pálpebra (5.2.1), calculada supondo
essa largura e a distância do MediaPipe, deu razão ≈ 1, o que indica que
essa escala está certa. O termo da cabeça vira então um deslocamento
fixo, −a·P_x e +a·P_y, e só o olho é ajustado.

**Só no eixo y.** No horizontal, o ganho fixo piorava o erro em pontos
novos (sessão de 12/11: 82 → 140 px de erro em x), porque a regressão livre
usa mais medidas do olho, inclusive a íris refinada, que nessa sessão teve
o dobro do sinal horizontal. O horizontal já funcionava bem no uso. Por
isso x continua vindo do melhor modelo aprendido, e y vem de um segundo
modelo (`y_model`) com a cabeça pela geometria. Na versão atual, esse modelo
é o olhar geométrico calibrado (ver adiante). Na primeira versão, ele era
escolhido pelo erro **só do eixo y** entre:

- o **compacto**: y = c + a·P_y + E_z·sec²(pitch)·(b₁·lid_up + b₂·lid_up²),
  só com a medida coerente com a cabeça (o termo quadrático cobre a
  saturação da pálpebra ao olhar para baixo);
- a **regressão livre** (todas as medidas do olho, grau 1 ou 2), com a
  cabeça fixa.

A validação cruzada é refeita para a combinação, e os erros da tela de
resultado são os da combinação. Mesmo sem a etapa da cabeça, a
compensação da cabeça continua valendo.

**Escolha do modelo de y só pelos alvos da grade.** Na etapa da cabeça o
alvo é sempre o **centro** da tela. Por isso o erro nessa etapa favorece
modelos que encolhem a previsão para o centro: um modelo que sempre
previsse o centro teria erro zero ali. Com o erro da cabeça pesando metade
na escolha (da força do ridge e do tipo de modelo), o eixo y era
comprimido. Na sessão de 12/11 09:07, a linha de baixo era prevista 185 px
acima do alvo na própria calibração (inclinação prevista/real de 0,70). Os
modelos de y passaram a ser escolhidos só pelo erro nos alvos da grade
(`score_head=False`); a etapa da cabeça continua no ajuste e o erro nela
continua sendo reportado.

**Viés da direção da face (`face_bias`).** O eixo "para a frente" do
modelo canônico do MediaPipe não coincide com a direção para onde a pessoa
vira o rosto. Olhando para a tela (abaixo da câmera), a face aparecia
inclinada ~13° para cima, quando a geometria pede ~20° para baixo: um viés
de 29–35° nas cinco sessões gravadas, muito estável. Com o ganho da cabeça
fixo, o viés distorce o termo E_z·tan(direção da face). Por exemplo, ao se
aproximar da tela o cursor subia em vez de descer (tan(+13°) = +0,23
contra tan(−20°) = −0,36). Supõe-se então que, na postura da grade, o rosto
aponta em média para o centro dos alvos. O viés é a diferença entre a
direção medida (mediana da grade) e a direção do rosto até esse ponto, e é
subtraído (no espaço dos ângulos) antes de calcular P_cabeça. A postura da
calibração vira a referência "de frente para a tela". O HUD e a vista 3D
mostram a cabeça em relação a essa referência; antes mostravam o ângulo
cru do MediaPipe (+13°), dando a impressão de que o sistema "achava" que a
cabeça estava sempre para cima.

**Trava na recentralização.** Desvios maiores que 25% da tela em qualquer
eixo (`CalibrationConfig.recenter_max_frac`) não são aplicados, e a tela
pede para recalibrar. Caso real: um R 23 s depois de calibrar, com o rosto
medido 8 cm mais baixo, aplicou −964 px no eixo y, e a parte de baixo da
tela ficou inalcançável.

#### Olhar geométrico calibrado (vista 3D e HUD)

O olhar geométrico de `gaze.py` (globo ocular + rotação da cabeça, sem
calibração) foi comparado com a direção real do rosto até cada alvo da
grade, nas nove sessões gravadas:

| | Olhar real | Olhar geométrico cru |
|---|---|---|
| inclinação média | −15° a −21° (a tela fica abaixo da câmera) | +18° a +31° |
| variação na tela toda | 100% | vertical: 18–29%; horizontal: 34–47% |

O raio cru aponta ~40° para **cima** do real. Cerca de 30° vêm do viés da
direção da face do MediaPipe (`face_bias`) e ~10° da íris, que em repouso
fica acima da linha que une os cantos do olho. Além disso, o movimento do
olho aparece comprimido em 2 a 5 vezes (íris regularizada pelo MediaPipe,
ver 5.2).

O modelo **geométrico** calibra esse raio, trabalhando com ângulos
(φ = atan das tangentes):

  φ = direção da face corrigida (face_bias) + ângulo do olho na cabeça
  ângulo do olho: φ_olho,x = c_x + b·u;  φ_olho,y = c_y + b₁·lid_up + b₂·lid_up² + b₃·v
  ponto na tela = E_xy + E_z · tan(φ)  (cm → px pela largura da tela)

O ajuste é linear: para cada amostra, o ângulo do olho **necessário** é o
ângulo do rosto até o alvo menos a direção corrigida da face, e as medidas
do olho são regredidas nele (ridge, força escolhida pelo erro no eixo y nos
alvos da grade). A cabeça e a distância entram pela geometria exata (sem o
ganho linearizado a·P_cabeça); só o olho é ajustado, com 4 coeficientes
por fonte da íris.

Ele chegou a ser o modelo do eixo y (`y_model`). Nas nove sessões (média,
erro só no eixo y, validação cruzada na grade), comparado com a escolha
anterior entre o compacto e a regressão livre com a cabeça fixa:

| | erro y em alvos novos (LOO) | erro y com a cabeça em poses novas |
|---|---|---|
| escolha entre compacto e livre | 94 px | 325 px |
| **olhar geométrico calibrado** | **91 px** | **261 px** |

Escolher por sessão entre todos os candidatos não ajudou (98 px / 332 px):
com 16 alvos, a escolha é ruidosa.

A íris vertical v não responde à inclinação da cabeça (razão ~0, ver
5.2.1), mas traz sinal na grade: sem ela, o erro y subia para 113 px (alvos
novos) e 289 px (cabeça). Por isso ela só entra com a etapa da cabeça,
cujas amostras (olhar fixo, cabeça inclinando) ensinam ao ajuste quanto
confiar nela. Sem a etapa, o modelo usaria v como na grade, e inclinar a
cabeça deslocaria o cursor (no usuário sintético, 206 px a 10° de
inclinação); aí entram só lid_up e lid_up². O usuário sintético dos testes
foi ajustado para reproduzir essa razão ~0 medida (`pitch_coupling` = 0,2).
O valor anterior, 0,4, imitava o estimador antigo, em que v respondia ao
contrário. Na vista 3D e no HUD, o "olhar calibrado" (magenta) é hoje o raio dos olhos
até o ponto estimado pelo **modelo do cursor** (antes do estabilizador).
Mostrar o raio de um modelo geométrico separado fazia o raio e o cursor
discordarem, o que confundia mais do que ajudava.

#### Validação real: modelo do eixo y atual

Com as sessões de 12/11 vieram três **validações** (9 alvos de outro
momento da sessão, que não entram na calibração). Elas são o teste mais
próximo do uso, porque a postura muda entre calibrar e usar. Erro médio e
escala do eixo y (inclinação estimado/real; 1 = sem compressão), média dos
três pares calibração → validação:

| Modelo do eixo y | Erro (px) | Escala y |
|---|---|---|
| ganho da cabeça aprendido (sem largura da tela) | 112 | 0,99 |
| compacto (só pálpebra, cabeça pela geometria) | 173 | 0,70 |
| olhar geométrico calibrado | 125 | 0,81 |
| **regressão livre, cabeça pela geometria** | **115** | **1,01** |

Os modelos que dependem só da pálpebra **comprimem** o eixo y na validação
(0,61–0,81). Na sessão de 12/11 13:25, a cabeça estava ~5° mais para cima
na validação do que na calibração. Para ver a tela, o olho fica mais
voltado para baixo, e a pálpebra entra na faixa em que satura: ela variou
13% menos entre os alvos, e a íris vertical, 15% menos. A pose medida pelo
MediaPipe também se mistura com o olhar (0,5–1,4° de "inclinação" por
1000 px de olhar para baixo, sem mexer a cabeça). A validação cruzada na
grade não mostrava isso, porque ali a postura é a mesma da calibração. Por
isso o modelo do eixo y passou a ser a regressão livre (todas as medidas do
olho) com o ganho da cabeça pela geometria (`Y_MODEL_KINDS`). Ela mantém
a escala e tem o ganho físico da cabeça.

#### Escolha do modelo do eixo x e número de pontos

**Viés de escolha no eixo x.** O erro na etapa da cabeça, em que o alvo é
sempre o **centro**, também entrava na escolha do modelo principal (que dá
o eixo x) e da força do ridge. Em 5 das 15 calibrações gravadas, isso
escolhia um modelo "encolhido" (o físico, ou ridge forte): escala
horizontal de 0,44 a 0,89, ou seja, o cursor não alcançava as laterais. Na
pior (12/11 09:45, ridge 1,0), o erro horizontal **nos próprios alvos da
calibração** era de 304 px. Com a escolha só pelos alvos da grade
(`X_SCORE_HEAD = False`), médias das 15 calibrações:

| | erro x nos alvos (ajuste) | erro em alvos novos (LOO) | menor escala x | validação (3 pares) |
|---|---|---|---|---|
| escolha com o erro da cabeça | 75 px | 130 px | 0,44 | 115 px |
| **escolha só pela grade** | **39 px** | **102 px** | **0,90** | **111 px** |

O custo é a compensação da cabeça no eixo x medida na etapa da cabeça:
o erro x com a cabeça em poses novas subiu de 150 para 184 px. Essa métrica
favorece exatamente os modelos encolhidos, mas parte da piora é real.

**Mais pontos de calibração?** Curva de aprendizado: o modelo foi ajustado
com subconjuntos sorteados de k alvos da grade 4×4 (6 sorteios por k, 13
calibrações) e medido nos alvos que ficaram de fora e na validação (3
pares):

| alvos usados | erro nos alvos de fora | erro na validação |
|---|---|---|
| 6 | 206 px | 176 px |
| 8 | 173 px | 141 px |
| 10 | 148 px | 133 px |
| 12 | 145 px | 125 px |
| 14 | 135 px | 115 px |
| 15–16 | 134 px | 118–121 px |

O erro cai até ~14 pontos e depois praticamente para. Ir de 16 para 25
pontos (grade 5×5) deve render pouco (≤ ~10 px, extrapolando a curva) e
alongaria a calibração em ~60%. O limite está nas medidas do olho e na
mudança de postura, não na quantidade de pontos. A grade continua 4×4 e
pode ser mudada em `CalibrationConfig.grid_cols/grid_rows`.

### 5.4 Coleta

**Etapa 1, grade 4×4:** margem de 5% (era 8%), percorrida em **ordem
aleatória**. Com 8%, a última linha ficava a 92% da altura, e abaixo dela,
na barra de tarefas, o modelo extrapolava medidas do olho que saturam ao
olhar para baixo. Alvos mais perto da borda dão ao ajuste exemplos dessa
região. Alvos colados na borda não ajudariam: ali a pálpebra cobre a íris e
o MediaPipe é menos preciso, e essas amostras ruidosas puxariam o ajuste.
Na ordem linha a linha, o tempo andava junto com a posição vertical: nos
dados reais, a abertura do olho caiu de 0,27 para 0,16 ao longo dos ~35 s
da grade (cansaço, olho ressecando), e o modelo aprendia essa deriva como
"olhar para baixo". Com a ordem aleatória, a deriva vira ruído em vez de
viés.

Para cada alvo, a coleta começa quando o olhar **fixa** no alvo: depois de
0,6 s de acomodação, espera-se que 6 quadros seguidos fiquem a menos de 0,8°
da mediana deles (o ruído típico com o olhar parado, medido nas sessões
reais, é ~0,2°), com limite de 1,8 s. Assim, as primeiras amostras não são
mais da sacada ou de um olhar "procurando" o alvo (detecção de fixação por
dispersão, Salvucci e Goldberg, 2000). Seguem 1,2 s de coleta das amostras
válidas. Se o rosto sumir, o alvo espera até juntar pelo menos 8 amostras.
Em cada alvo, amostras com escore-Z modificado (MAD) > 3,5 são descartadas
(Iglewicz e Hoaglin, 1993). A grade passou de 3×3 para 4×4 porque o eixo
vertical, de sinal mais fraco, precisa de mais linhas.

**Etapa 2, movimento da cabeça (guiada):** o alvo fica no centro da tela e
uma seta pede que a pessoa gire a cabeça para a direita, a esquerda, para
cima e para baixo, sem tirar o olho do alvo, e depois volte ao centro. A
seta enche conforme a cabeça gira (medida em relação à pose média durante a
grade) e fica verde ao chegar a 80% do ângulo pedido (10° de yaw e 8° de
pitch). Depois de 0,6 s ali, passa à próxima direção. Quem não alcança o
ângulo segue depois de 4 s, e quem não pode mover a cabeça só continua
olhando o ponto. Se a cabeça passar da faixa aproveitada (15° de yaw, 12° de
pitch), a seta fica vermelha e aparece "girou demais". A etapa determina o
coeficiente de P_cabeça e pesa o equivalente a 1 alvo. No ajuste entram só
as amostras dentro dessa faixa: na versão anterior, sem guia, as pessoas
giravam até 30° (yaw) e 40° (pitch). Nesses ângulos os landmarks do olho
deixam de ser confiáveis e o erro do modelo na própria etapa chegava a
~210 px (mediana).

**Calibração manual (tecla C).** Cada alvo só avança quando o usuário,
olhando para ele, aperta ESPAÇO ou ENTER. As amostras usadas são as de 1 s
**antes** da tecla, sem os últimos 0,15 s (tempo de reação): ao apertar a
tecla é comum piscar ou olhar para o teclado, então as amostras de depois
não servem. A confirmação é recusada (com aviso) se o olhar ainda se movia
nessa janela (80% das amostras a mais de 1,6° da mediana) ou se faltam
amostras, e ignorada nos primeiros 0,5 s do alvo. As etapas da cabeça e da
boca continuam automáticas. Ela não substitui a automática: o público do
projeto (pessoas com deficiência motora) muitas vezes não consegue
apertar uma tecla a cada alvo. Ela serve para quem pode apertar, ou com um
ajudante, e garante que cada alvo é coletado com a atenção da pessoa.

**Etapa 3, boca:** depois da etapa da cabeça, o ponto fica no centro e a
pessoa abre e fecha a boca devagar (≥ 4 s; segue em 7 s se a boca não
abrir). O MediaPipe estima a malha inteira de uma vez, e abrir a boca
contamina a pose da cabeça e os pontos do olho. Na sessão de 12/11 09:47,
nos quadros de boca aberta durante a coleta de alvos (olhar fixo e
conhecido), o ponto estimado descia ~50 px. A pose medida "inclinava"
−0,85° e a íris descia em relação aos cantos do olho. Uma pose só com
pontos rígidos (cantos dos olhos, ponte do nariz, testa, têmporas) foi
testada e não resolveu: esses pontos também se moviam. Na etapa da boca o
olhar está fixo, então todo deslocamento da previsão que acompanha a
abertura é artefato. Ajusta-se esse deslocamento por unidade de abertura
acima da abertura típica da grade (mínimos quadrados pela origem, com
rejeição de outliers e descontando o desvio com a boca fechada), e a
previsão passa a descontá-lo (`fit_mouth_correction`). Sem a etapa, ou se
a boca quase não abriu (variação < 0,04), a correção é zero. No usuário
sintético com o artefato, o cursor andava 451 px ao abrir a boca sem a
etapa, e 0 px com ela.

**Repetição de alvos ruins:** depois do ajuste, alvos com erro na validação
cruzada acima de 2,5× a mediana e acima de 8% da largura da tela (até 3
alvos) são coletados de novo, uma vez, e o modelo é reajustado. Isso cobre
piscadas longas, distração ou olhar fora do alvo.

As medidas brutas de cada calibração e validação são gravadas em
`results/*.npz`. `tools/benchmark.py` reajusta variantes do modelo em todas
as sessões gravadas e compara o erro em alvos novos, em poses novas da
cabeça e na validação. `tools/reavaliar.py` detalha uma sessão. Assim dá
para comparar alternativas com os mesmos dados, sem refazer as sessões.

### 5.5 Escolha do modelo e erro em pontos e poses novos

A força do ridge (α ∈ {10⁻⁴, …, 1}), o tipo de modelo (livre ou físico), o
grau dos termos do olho (1 ou 2, no livre) e a fonte da posição da íris
(MediaPipe ou refinada na imagem) são escolhidos por validação cruzada com
dois tipos de dobra:

- **alvos**: cada alvo da grade fica fora do ajuste por vez, e o erro é a
  distância da mediana das previsões a ele (pontos novos, cabeça parada);
- **direções da cabeça**: as amostras da etapa da cabeça são separadas pela
  direção dominante do desvio de pose (direita, esquerda, cima, baixo).
  Cada direção fica fora por vez, e o erro é a mediana das distâncias de suas
  amostras ao alvo central (poses novas da cabeça). Amostras a menos de 2°
  da pose da grade ficam sempre no treino.

O critério é a média simples dos dois erros. A escolha só pelo erro dos
alvos ignorava o problema principal com usuários reais, a invariância à
cabeça, porque durante a grade a cabeça fica quase parada. Os dois erros
aparecem na tela de resultado ("erro estimado em pontos novos" e "com a
cabeça em poses novas"), junto com uma classificação da qualidade (boa até
4,5% da largura da tela, razoável até 8%, ruim acima disso). Eles são mais
honestos que o erro de ajuste, que esconde sobreajuste.

### 5.6 Análise das sessões reais

Quatro sessões de calibração (duas com validação) foram reanalisadas
offline antes da versão atual. Resultados principais:

- **A acurácia era ruim, a precisão não.** Na validação, o erro médio ficava
  em 180–235 px, mas a dispersão das estimativas era de ~33 px. O problema
  era viés do modelo, não tremor.
- **A íris do MediaPipe se move muito pouco.** A tela inteira corresponde
  a uma variação de só ~0,09 na posição horizontal da íris (em larguras de
  olho). Uma mudança de 0,009, causada por 2–3° de giro de cabeça ou por
  deriva dos landmarks, já desloca o cursor 150–180 px.
- **Girar a cabeça mexe na medida do olho.** Com o olhar parado, 1° de giro
  de cabeça mudou a medida horizontal da íris 2× mais que 1° de movimento
  do olho (5× no estimador anterior, que usava o z da íris). Inclinar a
  cabeça para cima também altera a abertura medida das pálpebras. Por isso
  a compensação de cabeça precisa ser aprendida na faixa em que a cabeça de
  fato se move.
- **A postura muda entre calibrar e usar.** Entre a calibração e a validação,
  a pessoa girou ~3° e deslocou o rosto 2–3 cm. Numa sessão, 220 dos 236 px
  de erro eram um deslocamento constante que a recentralização corrige.

Comparação no benchmark (erro médio em px, mesmas sessões):

| Variante | Alvos novos (LOO) | Poses novas da cabeça | Validação |
|---|---|---|---|
| Faixa 15°/12° (padrão) | 111 | 376 | 183 |
| Sem filtro de faixa | 250 | 351 | 213 |
| Sem etapa da cabeça | 92 | 1010 | 281 |

O erro em poses novas continua alto nessas sessões porque as etapas da
cabeça foram gravadas sem guia, com giros extremos. Uma das duas validações
favoreceu o grau 1 e a outra o grau 2, por isso a escolha é feita por
sessão pela validação cruzada. Sessões novas, com o protocolo guiado, devem
ser comparadas no benchmark para confirmar essas escolhas.

#### Sessões de 11/11 (protocolo guiado) e efeito de lid_up

Duas sessões gravadas com a etapa da cabeça guiada, recalculadas a partir
dos landmarks com cada versão do estimador (`tools/benchmark.py`, erro
médio em px):

| Versão | Alvos novos (LOO) | Poses novas da cabeça |
|---|---|---|
| lid_v, só regressão livre (anterior) | 159 (166 / 153) | 412 (294 / 531) |
| lid_up, livre ou físico pela validação cruzada | 116 (92 / 140) | 321 (252 / 390) |

Na sessão 1 a validação cruzada escolheu a regressão livre; na sessão 2, o
modelo físico. Pela validação cruzada interna da sessão 2, o erro com a
cabeça em poses novas foi de 405 px (livre) para 290 px (físico). Também
foram testadas, sem ganho: a íris refinada como medida horizontal única, a
íris relativa ao centro do contorno do olho, a abertura com o z do
MediaPipe e termos quadráticos no modelo físico.

#### Sessões de 11/11 e 12/11: ganho da cabeça pela geometria

Três sessões, tela de 34,4 cm. Erro médio por eixo em px (LOO: alvos
novos; cabeça: direções da etapa da cabeça deixadas de fora) e ganho do
cursor por grau de inclinação da cabeça com o olho acompanhando:

| Sessão | Modelo | \|x\| LOO | \|y\| LOO | \|x\| cabeça | \|y\| cabeça | Ganho pitch (px/°) |
|---|---|---|---|---|---|---|
| 11/11 08:53 | aprendido | 57 | 65 | 102 | 203 | 12 |
| | **y pela geometria** | 57 | 72 | 102 | **102** | **35** |
| 11/11 08:57 | aprendido | 87 | 89 | 258 | 200 | 27 |
| | **y pela geometria** | 87 | 97 | 258 | **148** | **40** |
| 12/11 08:09 | aprendido | 82 | 106 | 186 | 371 | 23 |
| | **y pela geometria** | 82 | 128 | 186 | **349** | **37** |

O horizontal não muda. No vertical, o erro com a cabeça em movimento cai e
o ganho passa ao valor físico. Nessa primeira versão, o modelo de y ainda
era escolhido pela média do erro na grade e na etapa da cabeça, o que o
comprimia (ver "Escolha do modelo de y só pelos alvos da grade").

Versão final (escolha de y só pela grade e viés da face), com as cinco
sessões. Alcance vertical em pontos novos com a cabeça parada (LOO):
inclinação prevista/real no eixo y e erro médio na linha de baixo de
alvos:

| Sessão | Ganho aprendido: inclinação / linha de baixo | Versão final: inclinação / linha de baixo |
|---|---|---|
| 11/11 08:53 | 1,00 / −9 px | 0,99 / −12 px |
| 11/11 08:57 | 0,80 / −91 px | 0,94 / −46 px |
| 12/11 08:09 | 0,82 / −113 px | 0,83 / −156 px |
| 12/11 09:06 | 0,96 / −63 px | 0,90 / −74 px |
| 12/11 09:07 | 0,68 / −205 px | 0,88 / −96 px |
| média | 0,85 / −96 px | 0,91 / −77 px |

Ganho do cursor por grau de inclinação da cabeça: 10–27 px/° (aprendido)
contra 38–44 px/° (geometria). Em `tools/benchmark.py` (médias das cinco
sessões), a configuração padrão ficou com 117 px em alvos novos e 390 px
em poses novas da cabeça, contra 137 e 364 px com o ganho aprendido.

**Limite que continua.** Com a cabeça parada, as medidas verticais
saturam ao olhar para baixo (na sessão de 12/11 a pálpebra superior andou
0,030 entre a 2ª e a 3ª linha de alvos e 0,015 entre a 3ª e a 4ª), e a
linha de alvos mais baixa fica a 92% da altura da tela (margem de 8%).
Abaixo dela, por exemplo na barra de tarefas, o modelo extrapola uma medida
saturada. Na prática, para olhar bem para baixo as pessoas também abaixam
a cabeça, e aí o ganho geométrico cobre a diferença.

Para referência: a ~35 cm de um monitor de 34,5 cm de largura (1920 px),
1° de ângulo visual ≈ 34 px. 100 px ≈ 3°, próximo do que se obtém com
rastreadores por webcam baseados em landmarks.

### 5.7 Efeito (simulação)

O usuário sintético dos testes reproduz o que foi observado na webcam:

- íris com só 20% do movimento vertical;
- pálpebras acompanhando o olhar;
- ganho real do olho diferente do assumido;
- acoplamento com o pitch da cabeça.

Nele, com ruído nos landmarks, o erro fica em ~15–25 px com a cabeça
parada e abaixo de 50 px com yaw de ±15° e pitch de ±10°. Translação
lateral de 6 cm é compensada quase exatamente. Mudar a distância em 10 cm
deixa um resíduo de ~40 px, porque a aproximação linear de tan(olho) só é
exata perto da distância da calibração; nesse caso, a recentralização (R)
corrige. Para referência: num monitor de 53 cm a 60 cm, 1° de ângulo
visual ≈ 36 px. Os testes correspondentes estão em `tests/test_pipeline.py`
(`test_head_motion_stage_makes_gaze_head_invariant`,
`test_head_translation_is_compensated`,
`test_head_invariance_with_landmark_noise`). Os valores com webcam real
serão maiores e devem ser medidos com a validação (seção 6).

## 6. Validação (`procedures.py` → `ValidationProcedure`)

A validação usa **9 alvos diferentes dos da calibração**, em posições
intermediárias e de borda. São reportadas duas métricas padrão em
rastreamento ocular:

- **Acurácia**: distância média entre o alvo e a mediana das estimativas
  (erro sistemático);
- **Precisão (RMS)**: dispersão das estimativas em torno da própria média
  (ruído).

Com `--screen-width-cm` e `--distance-cm`, os valores também são convertidos
para graus de ângulo visual:

  θ = atan(erro_cm / distância_cm)

Graus permitem comparar com a literatura e entre telas diferentes.

Também são reportados o **deslocamento médio** (erro constante, igual em
todos os alvos) e a **acurácia sem ele**. A diferença entre as duas mostra
quanto do erro a recentralização (seção 7) corrigiria. Nas sessões reais,
foi a maior parte.

Cada validação gera três arquivos em `results/`:

- `validacao_<data>.csv`: uma linha por amostra (alvo, estimativa, ângulos,
  erro);
- `validacao_<data>.json`: resumo com as métricas por alvo e o resumo do
  modelo usado (fonte da íris, grau, erros da validação cruzada);
- `validacao_<data>.npz`: medidas brutas e landmarks, para reanálise.

## 7. Recentralização (`RecenterProcedure`)

O modelo já compensa rotação e translação da cabeça. Ainda assim, sobram
pequenos deslocamentos sistemáticos: erro na distância estimada pelo
MediaPipe, mudança grande de postura, óculos que escorregam. A
recentralização mostra só o alvo central, mede o deslocamento entre a
estimativa e o centro e o soma como um *offset* à saída do modelo, sem
refazer a calibração. A postura em que se recentralizou passa a ser a
referência do aviso de postura.

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
| `G` | olhar: globo ocular estimado (círculo), raio de cada olho, raio combinado (seta) e os ângulos yaw/pitch |
| `P` | eixos da cabeça (x vermelho, y verde, frente azul) e yaw/pitch/roll, com a fonte da pose |
| `V` | minimapa da tela: alvos de calibração, rastro recente, ponto filtrado (amarelo), ponto bruto (cinza) e progresso do dwell |

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

## 11. Verificação (`tests/`)

Os testes usam um rosto 3D sintético, com pose da cabeça e direção do olhar
conhecidas, projetado no mesmo formato de saída do MediaPipe. Eles verificam:

- sinais e invariâncias da pose e do olhar;
- que a etapa da cabeça reduz o erro com a cabeça girada a menos da metade
  do erro sem ela, e a menos de 50 px, num usuário com erros de ganho e
  acoplamento;
- compensação da translação (±6 cm lateral, 50–70 cm de distância);
- que o estabilizador remove o tremor (< 3 px/frame), chega após sacadas em
  ≤ 5 frames, ignora picos isolados e acompanha deslocamentos lentos;
- recuperação exata dos ângulos quando a matriz de pose é fornecida;
- generalização da calibração para alvos fora da grade, simulando um monitor
  de 53×30 cm a 60 cm;
- o fluxo completo do aplicativo com um "usuário simulado" que olha para os
  alvos exibidos, segue as setas da etapa da cabeça e depois usa o cursor
  com a cabeça girada e deslocada (erro < 80 px, tremor < 4 px/frame);
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
