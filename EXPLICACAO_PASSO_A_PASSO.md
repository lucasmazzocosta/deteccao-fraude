# Explicando o código passo a passo (para quem nunca viu isso)

Imagine que você trabalha num banco e recebe uma planilha gigante com **284 mil compras** feitas com cartão. Algumas poucas (492) foram fraudes. A ideia é ensinar o computador a olhar uma compra e dizer **"essa parece fraude"** ou **"essa parece normal"**.

Isso se chama **machine learning** (aprendizado de máquina): em vez de escrever regras à mão ("se o valor for maior que X, é fraude"), mostramos ao computador muitos exemplos já com a resposta, e ele descobre sozinho os padrões.

O código segue o caminho de uma receita: **pegar os ingredientes → preparar → cozinhar → provar → ajustar o tempero → explicar o prato**.

---

## Parte 0: Bibliotecas (as "caixas de ferramentas")

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

Python sozinho sabe fazer pouca coisa. **Bibliotecas** são pacotes de ferramentas prontas que outras pessoas escreveram. `import` significa "traga essa caixa de ferramentas", e `as np` dá um apelido curto para não precisar escrever o nome inteiro toda vez.

- **numpy (`np`)**: faz contas com muitos números de uma vez.
- **pandas (`pd`)**: trabalha com tabelas, como um Excel dentro do Python.
- **matplotlib (`plt`)**: desenha gráficos.

```python
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
...
```

`from X import Y` significa "da caixa X, pegue só a ferramenta Y". O **scikit-learn** (`sklearn`) é a principal biblioteca de machine learning em Python. Dela vêm:

- `train_test_split`: divide os dados em parte de estudo e parte de prova.
- `StandardScaler`: coloca os números na mesma escala.
- `Pipeline`: uma "linha de montagem" que junta várias etapas.
- `LogisticRegression` e `RandomForestClassifier`: dois tipos de modelo.
- as funções de métricas (`classification_report`, `roc_curve`, `precision_score` etc.): medem a qualidade das respostas.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import shap
```

- **imbalanced-learn (`imblearn`)**: ferramentas para quando uma resposta é muito mais rara que a outra, que é o nosso caso.
- **XGBoost**: um modelo muito usado em competições.
- **SHAP**: explica por que o modelo tomou cada decisão.

```python
SEED = 42
```

Vários passos sorteiam coisas, como quais linhas vão para teste. A **semente** fixa esse sorteio para que, toda vez que você rodar, o resultado seja o mesmo. O número 42 é só uma tradição.

---

## Parte 1: Explorar (conhecer os dados)

```python
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
```

- `url = "..."` guarda o endereço do arquivo numa **variável**, que é uma etiqueta com um nome para guardar um valor.
- `pd.read_csv(url)` lê o arquivo CSV (uma planilha em texto, com valores separados por vírgula) direto da internet e transforma em tabela.
- `df` é a tabela. O nome vem de **DataFrame**, que é como o pandas chama suas tabelas.

```python
print("Linhas e colunas:", df.shape)
df.head()
```

- `df.shape` diz o tamanho da tabela: (284807, 31), ou seja, 284.807 linhas (compras) e 31 colunas (informações de cada compra).
- `print(...)` mostra algo na tela.
- `df.head()` mostra as 5 primeiras linhas, para dar uma olhada.

As colunas são:
- **Time**: em que segundo a compra aconteceu.
- **Amount**: o valor da compra.
- **V1 a V28**: informações da compra "embaralhadas" matematicamente (por uma técnica chamada PCA) para que ninguém descubra dados pessoais. Não sabemos o que cada uma significa, mas o modelo consegue usá-las.
- **Class**: a resposta. 1 = fraude, 0 = normal.

```python
print(df["Class"].value_counts())
df["Class"].value_counts(normalize=True)
```

- `df["Class"]` pega só a coluna Class.
- `.value_counts()` conta quantas vezes cada valor aparece: 284.315 zeros e 492 uns.
- `normalize=True` mostra isso em proporção: 0,9983 e 0,0017. **Só 0,17% são fraude.** É uma agulha num palheiro.

```python
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(df["Amount"], bins=60)
ax[1].hist(np.log1p(df["Amount"]), bins=60)
plt.show()
```

- `plt.subplots(1, 2, ...)` cria uma área de desenho com 1 linha e 2 gráficos lado a lado. `figsize` é o tamanho.
- `.hist(...)` desenha um **histograma**, um gráfico de barras que mostra quantas compras caem em cada faixa de valor. `bins=60` divide em 60 faixas.
- O primeiro gráfico mostra os valores originais: quase tudo amontoado perto de zero, com poucas compras enormes.
- O segundo mostra os valores depois do logaritmo (explicado abaixo): bem mais espalhados.
- `plt.show()` exibe os gráficos.

---

## Parte 2: Preparar (arrumar os ingredientes)

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

Cria uma **coluna nova** chamada `Amount_log`. O **logaritmo** "encolhe" números grandes muito mais que os pequenos: R$ 10 vira cerca de 2,4 e R$ 10.000 vira cerca de 9,2. Assim uma compra gigante não "grita" mais alto que todas as outras. O `log1p` calcula log(1 + valor), o que evita erro quando o valor é 0.

```python
X = df.drop(columns=["Class", "Amount"])
y = df["Class"]
```

Separamos a tabela em duas partes:
- **X** (as pistas): tudo que o modelo pode olhar para decidir. `.drop` remove colunas. Tiramos `Class` porque é a resposta (seria "colar na prova") e `Amount` porque já temos a versão com log.
- **y** (o gabarito): a resposta certa de cada compra.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED,
)
```

Pense num aluno estudando para uma prova. Se a prova tiver exatamente as mesmas questões dos exercícios, ele pode só decorar. Para saber se ele **aprendeu de verdade**, a prova precisa ter questões que ele nunca viu.

- **Treino (70%)**: o modelo estuda aqui, vendo as pistas e o gabarito.
- **Teste (30%)**: guardado para a prova final. O modelo nunca vê esses dados durante o estudo.
- `stratify=y`: garante que treino e teste tenham a mesma proporção de fraudes (0,17%). Sem isso, por azar, o teste poderia ficar quase sem fraudes.
- O resultado são quatro pedaços: pistas e gabarito de treino, pistas e gabarito de teste.

```python
print("Treino:", X_train.shape, "| fraudes:", int(y_train.sum()))
```

Só confere os tamanhos. `y_train.sum()` soma os 1s, ou seja, conta as fraudes.

### A função `avaliar`

```python
resultados = []

def avaliar(nome, y_true, y_pred):
    ...
```

- `resultados = []` cria uma **lista vazia**, onde vamos guardar as notas de cada modelo.
- `def` cria uma **função**: um bloco de código com nome que podemos reaproveitar várias vezes. Em vez de repetir as mesmas 15 linhas para cada modelo, chamamos `avaliar(...)`.
- Ela recebe três coisas: o nome do modelo, as respostas certas (`y_true`) e as respostas do modelo (`y_pred`).

Dentro dela:

```python
print(classification_report(y_true, y_pred, digits=3))
```

Mostra o **boletim** do modelo, com precisão, recall e F1 para cada classe. Olhamos a linha **1** (fraude).

Para entender as métricas, imagine um detector de metais no aeroporto:
- **Recall**: de todas as pessoas que realmente carregavam metal, quantas o detector apitou? Recall baixo = armas passando.
- **Precisão**: de todas as vezes que o detector apitou, quantas eram metal de verdade? Precisão baixa = muita gente sendo revistada à toa.
- **F1**: uma nota única que equilibra as duas.

```python
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
```

A **matriz de confusão** conta quatro situações:
- **tp** (verdadeiro positivo): era fraude e o modelo disse fraude. ✅
- **fn** (falso negativo): era fraude e o modelo disse normal. ❌ fraude perdida
- **fp** (falso positivo): era normal e o modelo disse fraude. ⚠️ alarme falso
- **tn** (verdadeiro negativo): era normal e o modelo disse normal. ✅

`.ravel()` "achata" a tabelinha 2×2 numa fila de 4 números para guardar cada um numa variável.

```python
resultados.append({ "modelo": nome, "precisao": ..., "recall": ..., ... })
```

`.append` adiciona uma linha nova à lista. As chaves `{}` criam um **dicionário**, um conjunto de pares "nome: valor". No fim, essa lista vira uma tabela comparando todos os modelos.

---

## Parte 3: Treinar (ensinar os modelos)

### 3.1 Regressão logística

```python
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
```

Um **Pipeline** é uma linha de montagem: os dados entram e passam por cada etapa em ordem.

1. **`StandardScaler`**: as colunas têm escalas muito diferentes (Time vai até 172 mil, as V's ficam perto de 0). O scaler deixa todas com média 0 e mesma "espalhada", como converter tudo para a mesma unidade de medida. Assim nenhuma coluna domina só por ter números maiores.
2. **`LogisticRegression`**: o modelo mais simples para respostas sim/não. Ele dá um "peso" para cada pista e soma tudo para calcular a chance de fraude. `max_iter=1000` dá a ele até 1000 tentativas para achar os melhores pesos.

Usamos esse modelo como **baseline**: a régua mínima. Se um modelo complicado não for melhor que ele, não vale a pena.

```python
pipe_lr.fit(X_train, y_train)
```

**`.fit` = estudar.** O modelo olha as pistas e o gabarito do treino e aprende os padrões. Como o scaler está dentro do Pipeline, ele também aprende a escala **só com o treino**, sem espiar a prova.

```python
y_pred_lr = pipe_lr.predict(X_test)
y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
```

- **`.predict` = fazer a prova.** Para cada compra do teste, o modelo responde 0 ou 1.
- **`.predict_proba`** dá a **probabilidade** em vez da resposta seca: "acho que tem 73% de chance de ser fraude". Ele devolve duas colunas (chance de 0 e chance de 1), e `[:, 1]` pega só a segunda, a chance de fraude.

```python
avaliar("Regressão logística", y_test, y_pred_lr)
```

Corrige a prova comparando as respostas do modelo com o gabarito.

### 3.2 Random Forest

```python
RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=SEED)
```

Uma **árvore de decisão** é uma sequência de perguntas: "V14 é menor que -3? Se sim, V12 é menor que -2? ..." até chegar numa resposta. Uma **floresta aleatória** cria várias árvores, cada uma vendo um pedaço diferente dos dados, e elas **votam**.

- `n_estimators=50`: 50 árvores.
- `max_depth=10`: cada árvore faz no máximo 10 perguntas em sequência. Árvores muito fundas decoram o treino.
- `class_weight="balanced"`: como há 578 compras normais para cada fraude, o modelo poderia ignorar as fraudes. Isso faz **errar uma fraude "doer" muito mais** que errar uma compra normal.
- `n_jobs=-1`: usa todos os núcleos do processador, para ir mais rápido.

O resto (`fit`, `predict`, `predict_proba`, `avaliar`) é igual ao anterior.

### 3.3 XGBoost

```python
peso_fraude = (y_train == 0).sum() / (y_train == 1).sum()
```

- `(y_train == 0)` cria uma lista de Verdadeiro/Falso dizendo se cada linha é normal. `.sum()` conta os Verdadeiros.
- Dividimos normais por fraudes: cerca de **578**. Esse vai ser o peso da fraude.

```python
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=peso_fraude, ...)
```

O XGBoost também usa árvores, mas em **sequência**: a primeira tenta acertar, a segunda foca nos erros da primeira, a terceira nos erros que sobraram, e assim por diante. É como uma equipe em que cada pessoa revisa o trabalho da anterior.

- `learning_rate=0.1`: o quanto cada árvore nova corrige. Valores menores aprendem mais devagar, porém com mais cuidado.
- `scale_pos_weight=peso_fraude`: a mesma ideia do `class_weight` do Random Forest.
- `eval_metric="logloss"`: a medida que o XGBoost usa internamente para saber se está melhorando.

---

## Parte 4: Avaliar (comparar com gráficos)

```python
probs = {"Regressão logística": y_prob_lr, "Random Forest": y_prob_rf, "XGBoost": y_prob_xgb}
```

Um dicionário juntando as probabilidades dos três modelos, para desenhar todos no mesmo gráfico.

### Curva ROC

```python
for nome, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{nome} (AUC = {roc_auc_score(y_test, p):.3f})")
```

- `for ... in ...` é um **laço**: repete o bloco para cada modelo.
- `roc_curve` testa vários pontos de corte e, para cada um, calcula quantas fraudes o modelo pega (`tpr`) e quantos alarmes falsos dá (`fpr`). O `_` ignora uma terceira informação que não usamos.
- `plt.plot` desenha a linha.
- `f"..."` é um texto com valores encaixados entre `{}`. `:.3f` mostra 3 casas decimais.
- **AUC** é a área debaixo da curva: 1 é perfeito e 0,5 é chute.

```python
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chute aleatório")
```

Desenha uma linha diagonal tracejada preta (`k--`) para referência. Um modelo que chuta fica em cima dela.

O problema da curva ROC aqui: como existem muitas compras normais, ela costuma parecer ótima mesmo para modelos medianos.

### Curva de precisão e recall

```python
precision, recall, _ = precision_recall_curve(y_test, p)
plt.plot(recall, precision, label=f"... (AP = {average_precision_score(y_test, p):.3f})")
plt.axhline(y_test.mean(), ...)
```

Mesma ideia, mas mostrando **recall × precisão**, que só olham para a fraude. É o gráfico mais honesto para esse problema. O **AP** (Average Precision) resume a curva num número: quanto mais perto de 1, melhor. `axhline` desenha uma linha horizontal na altura do chute, que aqui é só 0,0017.

---

## Parte 5: Ajustar (afinar o tempero)

### 5.1 Limiar de decisão

```python
threshold = 0.3
y_pred_lr_03 = (y_prob_lr >= threshold).astype(int)
```

Por padrão, o modelo diz "fraude" quando a chance passa de 50%. Mas para um banco, **deixar passar uma fraude costuma ser pior** que ligar para um cliente à toa. Então baixamos a régua para 30%: qualquer compra com 30% ou mais de chance já é marcada.

- `y_prob_lr >= threshold` compara cada probabilidade com 0,3 e dá Verdadeiro ou Falso.
- `.astype(int)` transforma Verdadeiro em 1 e Falso em 0.

Fazemos isso para os três modelos e avaliamos de novo.

```python
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (y_prob_rf >= t).astype(int)
    linhas.append({...})
pd.DataFrame(linhas).round(3)
```

Testa vários limiares no Random Forest e monta uma tabela. Assim dá para **ver a troca**: limiar baixo pega mais fraudes com mais alarmes falsos, e limiar alto o contrário. `.round(3)` arredonda para 3 casas.

### 5.2 Undersampling × oversampling

Outra forma de lidar com a raridade da fraude é **mudar a proporção dos dados de estudo**.

**Undersampling** (diminuir os normais):

```python
treino = X_train.copy()
treino["Class"] = y_train
```

Faz uma cópia do treino e coloca a resposta de volta como coluna, para separar por classe.

```python
fraudes = treino[treino["Class"] == 1]
normais = treino[treino["Class"] == 0].sample(len(fraudes), random_state=SEED)
```

- Pega todas as fraudes do treino.
- Das compras normais, **sorteia** (`.sample`) a mesma quantidade que o número de fraudes (`len(fraudes)`).

```python
treino_under = pd.concat([fraudes, normais]).sample(frac=1, random_state=SEED)
```

`pd.concat` junta as duas tabelas, uma embaixo da outra. `.sample(frac=1)` sorteia 100% das linhas, o que na prática **embaralha** a tabela.

O resultado é um treino pequeno e 50/50. O problema é que jogamos fora quase todas as compras normais, e o modelo tende a ficar "paranoico": recall alto, precisão baixa.

**Oversampling com SMOTE** (criar fraudes de mentira):

```python
pipe_smote = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=SEED)),
    ("model", LogisticRegression(max_iter=1000)),
])
```

O **SMOTE** pega uma fraude real, encontra outras fraudes parecidas com ela (as "vizinhas") e cria uma fraude nova **no meio do caminho** entre elas. Faz isso até as fraudes ficarem na mesma quantidade das normais.

Usamos o Pipeline do `imblearn` porque ele sabe que o SMOTE só deve rodar no **estudo**, nunca na **prova**. O teste continua com fraudes só reais e na proporção real. O scaler vem antes porque o SMOTE mede distâncias, e as escalas precisam estar iguais.

### 5.3 GridSearchCV

```python
param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
}
```

Os **hiperparâmetros** são os botões de configuração do modelo. Aqui listamos opções para três botões. São 2 × 2 × 2 = **8 combinações**.

```python
grid = GridSearchCV(XGBClassifier(...), param_grid, scoring="recall",
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED))
grid.fit(X_train, y_train)
```

O GridSearchCV **testa todas as 8 combinações** e escolhe a melhor:
- `scoring="recall"`: o critério de "melhor" é pegar mais fraudes.
- `cv=StratifiedKFold(n_splits=3, ...)`: **validação cruzada**. Divide o treino em 3 partes; estuda com 2 e testa na 3ª, repetindo 3 vezes, sempre trocando qual parte é teste. A nota final é a média. `Stratified` mantém a proporção de fraudes em cada parte. Tudo isso acontece **dentro do treino**: a prova final continua intocada.

```python
print("Melhores hiperparâmetros:", grid.best_params_)
xgb_best = grid.best_estimator_
```

Mostra a combinação vencedora e guarda o modelo já treinado com ela, que depois é avaliado no teste.

### 5.4 Tabela de comparação

```python
comparacao = (
    pd.DataFrame(resultados)
    .sort_values(["recall", "precisao"], ascending=False)
    .reset_index(drop=True)
)
```

- `pd.DataFrame(resultados)` transforma a lista de notas em tabela.
- `.sort_values(..., ascending=False)` ordena do maior recall para o menor. Em caso de empate, desempata pela precisão.
- `.reset_index(drop=True)` renumera as linhas 0, 1, 2...

**Essa é a tabela que vai para o README.**

---

## Parte 6: Explicar (por que o modelo decidiu assim?)

### Importância das variáveis

```python
importancias = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values()
importancias.tail(15).plot(kind="barh", ...)
```

- `xgb.feature_importances_` diz o quanto o XGBoost usou cada coluna ao montar suas árvores.
- `pd.Series(..., index=X_train.columns)` cola o nome de cada coluna no seu número.
- `.sort_values()` ordena do menor para o maior, e `.tail(15)` pega as 15 últimas, que são as mais importantes.
- `kind="barh"` desenha barras horizontais.

Isso mostra o que importa **no geral**, mas não explica uma compra específica.

### SHAP

```python
amostra = pd.concat([
    X_test[y_test == 1],
    X_test[y_test == 0].sample(300, random_state=SEED),
])
```

O SHAP é lento, então usamos uma amostra: **todas as fraudes do teste + 300 compras normais sorteadas**. `X_test[y_test == 1]` significa "as linhas do teste cuja resposta é fraude".

```python
explainer = shap.Explainer(xgb)
shap_values = explainer(amostra)
```

- `shap.Explainer(xgb)` cria um "explicador" para o modelo XGBoost.
- `explainer(amostra)` calcula, **para cada compra e cada coluna**, quanto aquela coluna empurrou a decisão para "fraude" (valor positivo) ou para "normal" (valor negativo).

A ideia vem da teoria dos jogos: é como dividir o mérito de um gol entre os jogadores que participaram da jogada.

```python
shap.plots.bar(shap_values, max_display=12)
```

Barras com o **impacto médio** de cada variável. Parecido com a importância, só que calculado de outro jeito.

```python
shap.plots.beeswarm(shap_values, max_display=12)
```

O gráfico de "enxame de abelhas": cada ponto é uma compra.
- **Posição horizontal**: à direita empurrou para fraude, à esquerda para normal.
- **Cor**: vermelho significa que o valor daquela variável era alto, azul que era baixo.
- Exemplo de leitura: "pontos azuis de V14 ficam à direita" quer dizer "**valores baixos de V14 aumentam a chance de fraude**".

```python
indice_fraude = 0
shap.plots.waterfall(shap_values[indice_fraude], max_display=12)
```

O gráfico de "cascata" explica **uma única compra** (a primeira da amostra, que é uma fraude real). Ele começa na chance média do modelo e cada barra soma (vermelho) ou subtrai (azul) até chegar na resposta final. É como uma nota fiscal da decisão: "começou em X, V14 somou tanto, V4 somou tanto, Time tirou tanto... resultado: fraude".

---

## Resumo em uma frase por etapa

1. **Explorar**: descobrimos que só 0,17% das compras são fraude.
2. **Preparar**: arrumamos o valor com logaritmo e separamos estudo de prova.
3. **Treinar**: ensinamos três modelos, dando mais peso para a fraude.
4. **Avaliar**: medimos com recall, precisão e F1, porque acurácia engana aqui.
5. **Ajustar**: baixamos a régua de decisão, testamos balancear os dados e procuramos a melhor configuração.
6. **Explicar**: o SHAP mostra quais informações fizeram cada compra parecer fraude.
