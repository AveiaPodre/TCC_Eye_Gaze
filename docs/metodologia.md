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

1. **Durante uma fixação**, o cursor segue a média das últimas N amostras
   (N = 20, ~0,7 s). Ruído com desvio σ vira σ/√N. Uma zona morta de 6 px
   impede micro-movimentos, e o cursor percorre 30% do caminho até a média
   a cada frame.
2. **Sacada:** uma amostra a mais de L px da média da fixação é candidata.
   O cursor só salta quando 3 amostras seguidas estão fora e são coerentes
   entre si (dispersão < L). Picos isolados (quase-piscadas, erros de
   detecção) são ignorados.
3. **Limiar adaptativo:** L = max(7% da largura da tela, 3 × ruído medido na
   fixação atual). Em condições ruins (pouca luz, rosto longe), o ruído
   aumenta e fica mais difícil confundir tremor com sacada.

Resultado num sinal simulado com ruído de 45 px por eixo:

| | movimento do cursor por frame durante a fixação | frames até chegar após uma sacada |
|---|---|---|
| sem filtro | ~82 px | 0 |
| One-Euro (versão anterior) | ~46 px | 0 |
| estabilizador de fixação | ~2 px | 2–3 (~100 ms) |

O One-Euro continua sendo usado, com parâmetros mais suaves, só nos ângulos
mostrados no HUD e no mapeamento antes da calibração. O HUD mostra o ruído
estimado (`ruido ~N px`). O minimapa mostra o ponto bruto (cinza) e o cursor
estabilizado (amarelo).

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

- lid_v: posição vertical da íris em relação à linha média das pálpebras;
- abertura: distância entre as pálpebras.

Todas são divididas pela largura do olho. A pálpebra superior acompanha o
olhar vertical (desce quando se olha para baixo), então lid_v e a abertura
trazem o sinal vertical que falta em v.

#### Profundidade da íris pela esfera do olho

Para levar a íris ao referencial da cabeça, o vetor íris → meio dos
cantos é girado por Rᵀ. A versão anterior usava o z da íris dado pelo
MediaPipe, que é ruidoso. Com a cabeça girada, esse z "vazava" para u e
v. Na sessão de 02/11 (13h30), u variava 4,5× mais por grau de cabeça do
que por grau de olho, e v ~14×. Agora só x e y da imagem são usados. A
profundidade é obtida pela interseção da linha de visada com a esfera do
olho, ancorada nos cantos (`gaze.on_eyeball`). O mesmo vale para as
pálpebras.

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

O olho entra pelas medidas brutas φ = [u, v, lid_v, abertura],
centralizadas na média da calibração, mais os termos quadráticos u²,
u·lid_v, lid_v² e u·v no grau 2. Elas são multiplicadas pela distância E_z
(o mesmo ângulo de olho cobre mais centímetros de longe) e por
sec²(ângulo da cabeça), a derivada de tan(cabeça + olho):

  x = c + a·P_cabeça + Σ_k b_k · E_z · sec²(yaw_cabeça) · φ_k
  y = c + a·P_cabeça + Σ_k b_k · E_z · sec²(pitch_cabeça) · φ_k

O modelo é **linear nos coeficientes**, ajustado por mínimos quadrados com
ridge em forma fechada: sem otimização iterativa e sem mínimos locais.
As colunas são padronizadas, com desvio-padrão mínimo de 1 cm nas de
P_cabeça. Assim, se a cabeça ficou quase parada na calibração, o ruído
dessa coluna não é amplificado.

### 5.4 Coleta

**Etapa 1, grade 4×4:** margem de 8%, percorrida em **ordem aleatória**.
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

A força do ridge (α ∈ {10⁻⁴, …, 1}), o grau dos termos do olho (1 ou 2) e a
fonte da posição da íris (MediaPipe ou refinada na imagem) são escolhidos por
validação cruzada com dois tipos de dobra:

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
