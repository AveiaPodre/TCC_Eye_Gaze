# Roteiro de implementação

Premissas:

- O **MediaPipe continua sendo a base**: ele encontra os landmarks do rosto
  e a íris, e a pose da cabeça sai dele.
- O pipeline continua **geométrico**: feixe de cada olho → projeção no
  monitor virtual → correção monitor virtual → real.
- O foco é **enxugar** (tirar o que os dados mostraram ser inefetivo) e
  **refinar a correção**, que é a contribuição do projeto.

Regra: nada é removido ou adotado sem número. Toda mudança passa por
`tools/benchmark.py` (e `tools/ablacao.py` quando mexe no protocolo) nas
sessões gravadas e não pode piorar a validação.

Linha de base (15/11, 40 calibrações, 10 com validação): LOO 118 px,
validação 151 px (82 px sem o desvio constante). Em 16/11, com as sessões
novas (45 calibrações, 15 com validação): LOO 119 px, validação 137 px
(82 px sem o desvio constante).

## O que os dados mostraram (16/11)

- O erro no uso é dominado por um **desvio constante** (~106 px em média),
  que aparece segundos depois de calibrar e varia de sessão para sessão. O
  LOO da calibração não o prevê (correlação 0,09 com a validação).
- Corrigir esse desvio com **1 ponto piora** (138 → 147 px); com 3 pontos ou
  mais, melhora ~30% (96 px com 3, 91 com 5).
- O passo de 0,3 por alvo conhecido (`confirm_target`) leva o erro de 138
  a 101 px com os alvos chegando um a um (94 px depois de 7 alvos).

## Fase 0: base de medição (feita)

- [x] `tools/ablacao.py`: erro com k alvos sorteados (alvos fora do ajuste e
      validação).
- [x] Benchmark em px e em graus; `--distance-cm` para a distância real.
- [x] Cache das medidas recalculadas (`results/.cache`, refeito quando o
      código do estimador muda).
- [x] Testes lentos marcados (`slow`, automático em `tests/conftest.py`):
      `pytest -m "not slow"` roda 59 testes em ~6 s.
- [x] Validação com RMS amostra a amostra, perda de dados e acurácia
      recentralizada (validação cruzada).

## Fase 1: captura da câmera (feita; falta 1080p)

- [x] MJPG pedido antes da resolução; `--backend`, `--exposicao`,
      `--sem-autofoco`; formato, fps do driver e exposição no início.
- [x] Fps real da câmera no HUD (amarelo abaixo de 25).
- [x] Sessões de 16/11: 720p a 30,3 fps reais.
- [ ] Gravar sessões em 1080p (`--width 1920 --height 1080`) com exposição
      fixa e comparar no benchmark com as de 720p.
- [ ] Converter para cinza só o recorte dos olhos em `iris_refine.py`.

## Fase 2: limpeza do pipeline (feita)

- [x] Intersecção 3D dos feixes → **ponto de cada olho no monitor virtual,
      média ponderada** pelo inverso do erro angular² de cada olho.
- [x] Removidas: âncoras alternativas (fica a mista), profundidade pela
      matriz do MediaPipe, escolha da fonte da pupila (fica a média).
- [x] Removida a etapa da boca e a correção da boca.
- [~] Filtro da pupila só para exibição: mantido (a detecção de fixação na
      coleta depende dele).

## Fase 3: calibração enxuta (feita)

- [x] Grade 3×4 (12 alvos: 3 colunas, 4 linhas): mesma validação que 16;
      a 4×3 piorou ~5% (o eixo vertical precisa das 4 linhas). Sessões de
      16/11: ~23 s por calibração.
- [x] Checagens do centro e correção da deriva removidas.
- [x] Etapa da cabeça só no **modo pesquisa** e só como **teste**.
- [x] Validação imediata só no modo pesquisa. **Mantida** (não removida):
      é a única medida do desvio constante, que o LOO não prevê.
- [x] Protocolo da calibração gravado com a sessão.

## Fase 4: refinar a correção monitor virtual → real (em andamento)

- [x] Correção fixa testada: mesma validação, LOO pior; a escolha por
      validação cruzada foi mantida.
- [x] **Recentralização com 4 pontos** (era 1, que piorava o erro).
- [x] **Reteste corrige pelo desvio médio** dos 9 alvos (era pelo centro).
- [x] **Correção contínua pelo ajuste fino da cabeça**: cada clique por
      permanência depois de um ajuste fino > 15 px corrige o offset
      (passo 0,3) e zera o ajuste; salva ao sair.
- [ ] (Windows) UI Automation: centro do botão clicado como alvo conhecido
      e *snap* do cursor. Precisa de `comtypes` (não instalado); fazer
      quando a dependência for aceita.
- [ ] Medir a correção contínua em uso real: gravar uma sessão usando o
      modo A + dwell por alguns minutos e um reteste (T) no fim.

## Fase 5: medida da pupila sobre o MediaPipe

- [x] Testado e **descartado**: invalidar ~120 ms depois de cada piscada
      ou interrupção. Nas sessões gravadas, perdia metade das amostras e
      piorava o LOO (119 → 134 px) e a validação (137 → 142 px).
- [x] Testado, **sem efeito mensurável**: piscada pela média dos 3 pares de
      pontos das pálpebras (as sessões gravadas só guardam quadros de olho
      aberto). Não adotado.
- [x] Testado nos quadros originais (9 calibrações com vídeo de 16/11):
      centróide semeado pelo MediaPipe (peso gaussiano, σ = 1 e 0,7 raio)
      e método de gradientes (Timm e Barth, 2011). Nenhum supera o
      centróide atual (LOO 92 px, validação 90/68 px); os gradientes são
      piores (LOO 109 px): a 720p a íris tem ~5 px de raio.
- [x] Mistura MediaPipe/refinada na pupila (24 sessões): 50/50 (atual) é o
      equilíbrio; 60% da refinada ganha ~2 px de acurácia e perde ~5 px de
      tremor.
- [x] Tempo de coleta por alvo: com metade (0,6 s), LOO +5% e validação
      +1,5%; alongar a calibração não compensa.
- [x] Resolução: a webcam do notebook de testes só entrega 1280×720 (em
      todos os backends, com e sem MJPG). O ganho de resolução depende de
      uma webcam externa (de preferência 1080p, embaixo do monitor).

## Fase 6: o desvio constante e a postura

- [x] Investigação (metodologia, seção 5.11): não é deriva no tempo durante
      a calibração (0,5 px/s); a postura explica no máximo ~30 px (~12 px
      por grau de inclinação); há um viés de forma comum (+40 px em y nos
      alvos internos); e a etapa da cabeça deixa um efeito passageiro de
      ~200 px que contamina o começo da validação.
- [x] Validação em ordem aleatória (separa tempo de posição do alvo).
- [x] Modo pesquisa: validação imediata ANTES do teste da cabeça
      (`HeadStageProcedure`, gravado em `cabeca_*.npz`; tecla K). Confirmar
      com sessões novas que o desvio da validação imediata diminui.
- [ ] Viés de forma: testar margem maior na grade (alvos menos colados na
      borda, onde a pálpebra cobre a íris e a correção extrapola). Precisa
      de sessões novas.
- [ ] Posição da câmera configurável no monitor virtual (`gap_cm` com
      sinal); sessões com a webcam embaixo do monitor.

## Fase 6b: precisão para alvos pequenos (sessão de 16/11 12:22)

- [x] Modo A = âncora + cabeça: o olhar só teleporta o cursor (nova
      fixação a > 20% da tela do cursor). Mexer a cabeça com os olhos
      parados movia a estimativa ~25-28 px/grau (inclusive de lado); nos
      dados do teste da cabeça, o movimento que não vinha da cabeça caiu
      ~60%.
- [x] Investigado nos 47 testes da cabeça gravados: NÃO há acoplamento
      sistemático "inclinar a cabeça -> cursor para o lado" (−0,8 ± 9
      px/grau; o −25 px/grau medido no uso vinha de mexer os olhos junto). O
      que varia muito é a sensibilidade VERTICAL à inclinação: −11 ± 28 px
      por grau, de sessão para sessão.
- [ ] Compensação da cabeça fora do modo A: aprender, por sessão, a
      sensibilidade vertical à inclinação com o teste da cabeça (hoje só
      teste). Antes, ela entrava no ajuste e melhorava pouco a validação;
      reavaliar com o teste da cabeça vindo depois da validação.
- [x] Dwell-click no modo A só com o cursor parado (12 px): os 24 cliques
      com ajuste fino saíam com o ajuste em andamento.
- [x] Limite de giro do ajuste fino 6° → 10° (6° limitava o ajuste a
      ~140 px, menos que o erro do olhar).
- [ ] Gravar uma sessão com o modo A para medir o efeito (a reprodução de
      uma sessão gravada não mostra o efeito interativo: a pessoa reagia ao
      cursor antigo).
- [ ] Clique por permanência acidental: 58 cliques em ~90 s de uso (um a
      cada ~2 s). Avaliar dwell mais longo, ou, no modo A, clicar só depois
      de um ajuste.

## Fase 6c: cursor que segue o olhar (16/11, à tarde)

- [x] Estabilizador novo (`GazeFollower`, "perseguição suave") no lugar da
      trava: o cursor andava em blocos (seguia 0% dos movimentos sutis de
      40 px e 5% dos de 70 px; agora 41% e 61%, em ~130 ms), com o cursor
      parado em 72% dos quadros de fixação (97% antes) e o mesmo erro.
- [x] Investigada a piora depois de mexer a cabeça (~208 px de mudança com a
      cabeça de volta): não é o estado do rastreamento do MediaPipe
      (reiniciá-lo não muda nada); o modo IMAGE reduz à metade mas triplica
      o tremor. Causa em aberto; defesas: R (4 pontos) e correção contínua.
- [x] Cursor "se movendo demais com a vista fixa": histerese no lugar da
      perseguição suave (parado em 86% dos quadros com o olho fixo, contra
      72%; segue 58% dos movimentos de 70 px, contra 62%).
- [x] Bordas (sobretudo a de baixo, 148 px contra ~100 px no meio):
      correção cúbica e abertura do olho na correção testadas, ganho de
      0-4 px; não adotadas. Limite da medida da íris com a pálpebra.

## Fase 6d: calibração em duas etapas (16/11, à noite)

- [x] Removidos o clique com zoom e a firmeza ajustável por teclas (fora
      do escopo: dificultavam o uso).
- [x] Etapa 2 da calibração: 4 alvos internos depois da grade, ajuste com
      os 16 pontos juntos. Nas 33 sessões com validação: erro nos alvos fora
      do ajuste 105 → 85 px (−19%). Alvos diagonais da validação movidos
      para 20/80% (independentes da calibração).
- [x] Borda de baixo: a estimativa fica ~62 px ACIMA do alvo na última
      linha (saturação: a pálpebra cobre a íris; o olho fecha ~25% olhando
      para baixo); não é piscada falsa (nenhuma amostra perdida).
- [x] Sessão de 16/11 17:17: validação 77 e 56 px (2,0° e 1,4°).
- [x] Critério de aceitação: LOO > 4,7% da largura -> "recomendado refazer"
      (ESPAÇO refaz sem salvar). LOO <= 90 px: validação 76 px; > 90: 123 px.
      Repetir só pontos ruins ou juntar calibrações não ajudou.

## Fase 6f: uso pleno sem teclado e desempenho (17/11)

- [x] Lentidão com o clique e o modo de precisão: o app caía de ~31 para
      13-16 fps depois dos primeiros cliques (segundo plano no Windows).
      `system.py` desliga a desaceleração em segundo plano e mantém o
      temporizador de 1 ms.
- [x] Barra de controle pelo olhar na borda esquerda, centralizada na
      altura (Mouse, Clique, Precisão, Recentralizar, Recalibrar); janela do
      OpenCV sem moldura e sem roubar o foco (com o tkinter, os atalhos do
      teclado paravam); o mouse liga sozinho depois da calibração.
- [x] Uso do cursor reorganizado em `control.py` (`CursorControl`).
- [ ] Sessão de uso real só pelo olhar (barra + clique + precisão) para
      confirmar os ~30 fps e medir a taxa de acerto.

## Fase 6g: métricas por olho, revalidação e perseguição (17/11)

- [x] Métricas por olho na validação (acurácia, precisão, disponibilidade
      de cada olho), no json e na tela de resultado.
- [x] Revalidação periódica (`--revalidar-min`; 10 min no modo pesquisa):
      mede a deriva ao longo da sessão e corrige o desvio.
- [x] Calibração por perseguição (`--perseguicao soma|alternativa`),
      desligada por padrão; simulação: 22 -> 20 px (soma), 14 px
      (alternativa).
- [ ] Gravar sessões com `--gravar --perseguicao soma` e
      `--gravar --perseguicao alternativa` e comparar a validação com a
      calibração atual; adotar como padrão só se o erro cair.
- [ ] Analisar `por_olho` nas sessões: se um olho erra sistematicamente
      mais, rever os pesos dos olhos.
- [ ] Usar as revalidações periódicas para medir a deriva por minuto.

## Fase 6h: análise das sessões de 17/11 (15:00)

- [x] Perseguição nas sessões reais: sem ganho ("soma" piorava um pouco;
      "alternativa" não supera a grade). Fica opcional e desligada.
- [x] Causa do desvio constante: o ganho vertical da calibração é 4-6 × o
      geométrico (íris do MediaPipe), e um viés lento de 0,5° vira 100-250
      px. Pálpebras não acrescentam informação ao modelo atual.
- [x] Conferência de 4 pontos depois de toda calibração, com o desvio
      encolhido pela incerteza (105 -> 84 px nas 48 validações).
- [x] Estabilizador: movimentos sutis seguidos 33/43/62% -> 53/58/76%
      (40/70/100 px), parado 87% (`tools/bancada_cursor.py`).
- [x] Desempenho: gravação numa thread, painéis do HUD só na região deles,
      prévia da câmera a 10 fps com o mouse em uso; tempo por quadro e
      estado gravados com a sessão.
- [x] Teste da cabeça fora do protocolo (tecla K ou `--cabeca`).
- [ ] Gravar uma sessão de USO (barra, clique, precisão) e conferir
      `proc_ms` e o fps com tudo ligado.
- [ ] Viés vertical lento: testar o modo IMAGE do MediaPipe só para a
      íris ou um filtro de deriva com os cliques confirmados.

## Fase 6i: cursor parado no modo de precisão e limpeza (17/11, noite)

- [x] Cursor "parado" ao olhar para outro lugar: só acontecia no modo de
      precisão (13 eventos na sessão das 15:45, nenhum no modo normal).
      O olhar agora manda enquanto a cabeça não ajusta: 13 -> 1 evento.
- [x] Revisão do projeto: código morto removido (atalhos antigos do app,
      funções só usadas em testes, campo de configuração sem uso),
      comentários desatualizados corrigidos.

## Fase 7: uso real e documentação

- [ ] Indicador do dwell sobre a área de trabalho.
- [ ] (Opcional) zoom em dois passos.
- [ ] Alinhar os padrões de `FixationStabilizer.__init__` com o config.
- [ ] Protocolo com mais participantes, em graus.
