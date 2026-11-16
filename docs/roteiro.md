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
- [ ] Refinamento semeado pelo MediaPipe: buscar o centro só a ~1 íris do
      ponto do MediaPipe, com peso gaussiano. Precisa dos quadros do
      vídeo: sessões com vídeo têm os timestamps de cada quadro, que
      batem com os dos landmarks gravados.
- [ ] Comparar o centróide com o método de gradientes (Timm e Barth, 2011)
      nesses quadros.

## Fase 6: o desvio constante e a postura

- [ ] O desvio da validação acompanha o deslocamento vertical do rosto
      (correlação 0,68): investigar se a compensação da translação vertical
      da cabeça está subestimada (ex.: escala da distância, âncora vertical).
- [ ] Posição da câmera configurável no monitor virtual (`gap_cm` com
      sinal); sessões com a webcam embaixo do monitor.

## Fase 7: uso real e documentação

- [ ] Indicador do dwell sobre a área de trabalho.
- [ ] (Opcional) zoom em dois passos.
- [ ] Alinhar os padrões de `FixationStabilizer.__init__` com o config.
- [ ] Protocolo com mais participantes, em graus.
