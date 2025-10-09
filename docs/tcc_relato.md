# Análise Acadêmica do Pipeline de Previsão Hierárquica

## Introdução
O presente documento descreve, em linguagem acadêmica, as principais decisões
arquiteturais e parametrizações disponíveis no pipeline de previsão
hierárquica implementado no script `hierarchical_forecaster.py`. O objetivo é
subsidiar trabalhos de conclusão de curso com uma visão estruturada das etapas
do processo, destacando possibilidades de customização e interpretação dos
hiperparâmetros.

## Estrutura Geral do Sistema
O pipeline encontra-se organizado na classe `Forecaster`, responsável por
orquestrar sete macroetapas: (i) carregamento e pré-processamento dos dados,
(ii) classificação ABC multidimensional, (iii) análise estatística das séries,
(iv) segmentação hierárquica com transformações do alvo, (v) modelagem por
perfil com mecanismos de fallback, (vi) recalibração, capping e validação, e
(vii) projeção futura hierarquizada seguida de exportação dos resultados. Essa
sequência garante que cada decisão metodológica seja contextualizada e possa
ser documentada adequadamente no relatório técnico do TCC.

## Hiperparâmetros e Possibilidades de Ajuste
A seguir apresenta-se um quadro descritivo dos principais hiperparâmetros e
configurações passíveis de ajuste para experimentação acadêmica:

### 1. Fonte de Dados e Horizonte de Previsão
- **`FILE_PATH` / `OUTPUT_PATH`**: determinam, respectivamente, os arquivos de
  entrada e saída. A alteração desses caminhos permite testar diferentes bases
  ou gerar relatórios em locais controlados.
- **`HORIZONTE`**: define o número de meses projetados para o futuro. Pode ser
  ajustado conforme o problema de planejamento, impactando a abrangência da
  projeção YOY + Rolling.
- **`CUTOFF`**: estabelece a data de separação entre treino e teste, permitindo
  análises de sensibilidade sobre o desempenho preditivo ao variar o período de
  validação.

### 2. Qualidade Histórica e Segmentação Inicial
- **`MIN_HISTORICO_MESES`**: requisito mínimo de observações mensais por série
  para habilitar modelagem individual. Aumentar esse valor torna o pipeline
  mais rigoroso, enquanto reduzi-lo amplia a cobertura com risco de menor
  robustez estatística.
- **`SEASONAL_CV_THRESHOLD`** e **`SEASONAL_ACF12_MIN`**: parâmetros para
  detecção de sazonalidade via coeficiente de variação e correlação no lag 12.
  Alterações nesses limiares permitem calibrar a sensibilidade da classificação
  sazonal, o que se reflete na escolha de modelos sazonais versus não sazonais.
- **`VOLATILE_CV_THRESHOLD`**: controla o ponto de corte para definir séries
  voláteis, direcionando-as a modelos de Holt-Winters mais responsivos.
- **`INTERMITTENT_ZERO_RATIO`**: taxa mínima de zeros para classificar uma série
  como intermitente, influenciando o direcionamento para Croston.

### 3. Modelagem e Transformações
- **`ALPHA_CROSTON` / `BETA_CROSTON`** e grades associadas: parâmetros do
  método de Croston com busca opcional (`CROSTON_TUNE`) para séries
  intermitentes. Ajustá-los modifica a suavização da demanda e a responsividade
  a novas ocorrências.
- **`FAST_MODE`**: ao ser ativado, prioriza Holt-Winters em configurações
  enxutas para séries sazonais, reduzindo custo computacional em detrimento de
  uma busca SARIMAX mais aprofundada.
- **Transformações do alvo**: o pipeline seleciona automaticamente entre
  Box-Cox, log1p ou identidade por segmento. Essa decisão pode ser documentada
  como parte da metodologia, ressaltando que a escolha decorre do teste de
  normalidade (Shapiro) comparativo.

### 4. Recalibração e Controle de Valores Extremos
- **Parâmetros de capping (`CAP_P95_MULT`, `CAP_MAX_MULT`,
  `CAP_RECENT_WINDOW`, `CAP_RECENT_MULT`, `CAP_FUT_MULT`)**: definem limites
  superiores para as previsões tanto no histórico quanto no futuro, reduzindo o
  risco de extrapolações inviáveis. Tais valores podem ser ajustados para
  refletir políticas corporativas de crescimento máximo.
- **`PESO_YOY`, `WINDOW_BOUND`, `FATOR_LIMITE`**: regulam a composição do
  forecast hierárquico futuro ao ponderar tendências recentes e limites por
  linha/grupo. São importantes para estudos que comparam cenários conservadores
  e agressivos de expansão.

### 5. Validação Estatística
- O pipeline reporta **R², MAPE, WAPE, RMSE, MAE**, além de testes de Shapiro e
  Breusch-Pagan. Esses indicadores podem ser utilizados como critérios de
  avaliação em experimentos, sendo possível registrar em relatório como cada
  hiperparâmetro impacta os diagnósticos de aderência.

## Conclusões
O script oferece um arcabouço robusto para trabalhos acadêmicos que necessitam
justificar tecnicamente cada etapa da previsão hierárquica. A organização em
blocos comentados favorece a elaboração de capítulos metodológicos, enquanto a
lista de hiperparâmetros fornece um menu de experimentos possíveis. Recomenda-se
que cada ajuste seja acompanhado de análise estatística e discussão crítica,
consolidando a consistência científica do TCC.
