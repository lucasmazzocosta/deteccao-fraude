# Detecção de fraude em cartão de crédito

Projeto de machine learning que treina e compara modelos para detectar **transações fraudulentas de cartão de crédito**, numa base em que a fraude é extremamente rara. O foco é escolher as métricas certas para esse tipo de problema e explicar por que o modelo marca uma transação como fraude.

| Arquivo | Conteúdo |
|---|---|
| [`deteccao_fraude.ipynb`](deteccao_fraude.ipynb) | notebook completo, com as saídas, tabelas e gráficos salvos |
| [`deteccao_fraude.py`](deteccao_fraude.py) | o mesmo código em formato de script Python |
| [`EXPLICACAO_PASSO_A_PASSO.md`](EXPLICACAO_PASSO_A_PASSO.md) | explicação do código, linha por linha, para quem está começando |
| [`requirements.txt`](requirements.txt) | bibliotecas necessárias |

---

## 1. O problema

A base tem **284.807 transações** reais feitas por cartões europeus em dois dias de setembro de 2013. Só **492 delas são fraude (0,17%)**.

| Coluna | Significado |
|---|---|
| `Time` | segundos desde a primeira transação da base |
| `Amount` | valor da transação |
| `V1` a `V28` | variáveis transformadas por PCA para proteger a privacidade dos clientes |
| `Class` | a resposta: **1 = fraude**, 0 = normal |

### Por que o desbalanceamento muda a forma de avaliar

Com 99,83% de transações normais, um modelo que responde "não é fraude" para tudo atinge **99,83% de acurácia** sem detectar uma única fraude. A acurácia, portanto, não diz nada aqui.

Por isso todos os modelos são avaliados pelas métricas da **classe de fraude (1)**:

- **Recall**: das fraudes reais, quantas o modelo encontrou. É a métrica principal: uma fraude que passa vira prejuízo direto.
- **Precisão**: das transações marcadas como fraude, quantas eram fraude de verdade. Precisão baixa significa muitos clientes legítimos com o cartão bloqueado.
- **F1**: média harmônica entre precisão e recall. Só fica alta quando as duas estão altas.
- **Matriz de confusão**: fraudes pegas, fraudes perdidas e alarmes falsos em números absolutos.
- **Curva de precisão e recall (Average Precision)**: mais informativa que a curva ROC num problema tão desbalanceado. Como a quantidade de transações normais é enorme, a taxa de falsos positivos da curva ROC fica sempre pequena, e a ROC acaba parecendo boa até para modelos medianos.

---

## 2. Preparação dos dados

1. **Leitura direta do link** com `pd.read_csv`, sem baixar o arquivo. O CSV passa do limite de 100 MB do GitHub e por isso não está no repositório.
2. **Exploração**: primeiras linhas, proporção de classes com `value_counts(normalize=True)` e histograma do valor antes e depois do logaritmo.
3. **Log do valor**: `Amount_log = np.log1p(Amount)`. O valor das transações é muito assimétrico (muitas compras pequenas, poucas enormes), e o logaritmo comprime os valores extremos. A coluna original `Amount` sai do modelo para não repetir a mesma informação.
4. **Separação treino/teste** (70% / 30%) com `stratify=y`, mantendo a proporção de fraudes nos dois conjuntos.
5. **Padronização com `StandardScaler` dentro de um `Pipeline`**: a média e o desvio são calculados só com o treino e depois aplicados ao teste, sem vazamento de informação.
6. **Balanceamento só no treino**: o undersampling e o SMOTE são aplicados apenas aos dados de treino. O teste mantém sempre a proporção real, para as métricas refletirem o cenário verdadeiro.

---

## 3. Modelos e comparação

| Modelo | Como lida com o desbalanceamento |
|---|---|
| **Regressão logística** (baseline) | nenhum ajuste; serve de referência |
| **Random Forest** | `class_weight='balanced'`: erros em fraude pesam proporcionalmente mais |
| **XGBoost** | `scale_pos_weight` = nº de normais ÷ nº de fraudes no treino (≈ 578) |
| **Regressão logística + undersampling** | treino reduzido a 50% fraude / 50% normal, sorteando transações normais |
| **Regressão logística + SMOTE** | cria fraudes sintéticas no treino até igualar as classes |
| **XGBoost + GridSearchCV** | busca de hiperparâmetros escolhida pelo recall |

Cada modelo passa pela mesma função de avaliação, que guarda **precisão, recall, F1, fraudes pegas, fraudes perdidas e alarmes falsos** da classe de fraude. No fim, o notebook monta uma **tabela única ordenada pelo recall** (seção 5.4) e desenha as **curvas ROC e de precisão e recall** dos três modelos principais no mesmo gráfico (seção 4), com AUC e Average Precision na legenda.

### O que cada abordagem tende a fazer

- **Undersampling** costuma levar o recall para perto do máximo, mas derruba a precisão: o modelo vê pouquíssimas transações normais no treino e passa a desconfiar de muita coisa.
- **SMOTE** mantém todas as transações normais e equilibra as classes com exemplos sintéticos. Em geral aumenta o recall com uma perda de precisão menor que o undersampling.
- **Pesos de classe** (`class_weight`, `scale_pos_weight`) produzem um efeito parecido sem alterar os dados, e por isso são a forma mais simples de tratar o desbalanceamento.

---

## 4. Limiar de decisão e explicabilidade

### Limiar de decisão: 0,5 → 0,3

Os modelos devolvem uma **probabilidade** de fraude, e o padrão é marcar como fraude quando ela passa de 0,5. O limiar foi baixado para **0,3** nos três modelos principais, e há uma tabela com limiares de 0,1 a 0,7 para o Random Forest:

- **limiar menor** → mais transações marcadas → **recall sobe** e precisão cai;
- **limiar maior** → menos transações marcadas → precisão sobe e recall cai.

O limiar de **0,3** foi escolhido porque, em detecção de fraude, deixar uma fraude passar costuma custar mais caro do que verificar uma transação legítima. Aceitar alguns alarmes falsos a mais em troca de encontrar mais fraudes é a troca que faz sentido para o negócio.

### Importância das variáveis e SHAP

- **Importância das variáveis do XGBoost**: as 15 colunas mais usadas pelo modelo, num gráfico de barras com os nomes das variáveis.
- **SHAP**: calculado sobre uma amostra com **todas as fraudes do teste e 300 transações normais**, para garantir que existam fraudes a explicar. São três gráficos:
  - **barras**: impacto médio de cada variável nas decisões;
  - **beeswarm**: para cada variável, se valores altos (vermelho) ou baixos (azul) empurram a previsão para fraude;
  - **waterfall**: a decomposição completa de **uma transação de fraude real**, mostrando quanto cada variável somou ou subtraiu da probabilidade final.

Como `V1` a `V28` são componentes de PCA, o SHAP mostra **quais componentes** pesam na decisão, mas não é possível traduzi-los de volta para características originais da compra (como loja ou país). Essa é uma limitação da própria base, que foi anonimizada.

---

## 5. O que mudei em relação à Expert

- **Padronização dentro do Pipeline** em todos os modelos, ajustada só com os dados de treino, em vez de aplicar o `StandardScaler` na base inteira antes da separação.
- **Undersampling e SMOTE aplicados só no treino** e efetivamente treinados e comparados pelo recall. O SMOTE roda dentro do `Pipeline` do `imbalanced-learn`, depois da padronização, porque ele calcula distâncias entre exemplos.
- **`scale_pos_weight` do XGBoost calculado** a partir da proporção real das classes, em vez de um valor fixo.
- **Ajuste de limiar estendido ao Random Forest e ao XGBoost**, com uma tabela de vários limiares para visualizar a troca entre recall e precisão.
- **GridSearchCV ampliado** com `learning_rate`, usando `StratifiedKFold` para manter a proporção de fraudes em cada parte da validação cruzada, e escolhendo pelo recall.
- **Amostra do SHAP com fraudes garantidas** e os gráficos beeswarm e waterfall além do gráfico de barras.
- **Tabela única de comparação** reunindo todos os modelos e variações testados.

---

## Como rodar

```bash
pip install -r requirements.txt
```

- **Notebook:** abra `deteccao_fraude.ipynb` no Jupyter, no VS Code (extensão Jupyter) ou no Google Colab e execute todas as células.
- **Script:** `python deteccao_fraude.py`. Cada gráfico abre numa janela, e o script continua quando ela é fechada.

O dataset é carregado automaticamente pelo link. A primeira execução leva alguns minutos por causa do download (~150 MB) e do GridSearchCV.

## Próximos passos

- Criar variáveis de comportamento ao longo do tempo, como a quantidade de transações em sequência e o intervalo desde a transação anterior.
- Ampliar a busca de hiperparâmetros, por exemplo com `RandomizedSearchCV`.
- Testar outros classificadores (LightGBM, Isolation Forest) e comparar todos pelo recall.
- Escolher o limiar a partir de um custo estimado para cada fraude perdida e cada alarme falso.

## Fonte dos dados

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), Machine Learning Group, Université Libre de Bruxelles.
Arquivo usado: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
