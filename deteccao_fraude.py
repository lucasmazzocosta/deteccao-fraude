# Detecção de fraude em cartão de crédito
#
# Neste código eu treino modelos que tentam descobrir quais transações de cartão de crédito são fraude.
#
# O detalhe que muda tudo: só 0,17% das transações são fraude. Um modelo que responde "não é fraude" para tudo
# acerta 99,8% das vezes e mesmo assim não serve para nada. Por isso aqui eu não olho para a acurácia,
# e sim para três métricas da classe de fraude (Class = 1):
#
# - Recall: de todas as fraudes reais, quantas o modelo encontrou?
# - Precisão: de tudo que o modelo marcou como fraude, quanto era fraude mesmo?
# - F1: um equilíbrio entre as duas.
#
# ======================================================================


# 0. Bibliotecas

# Instalação (só precisa rodar uma vez, se ainda não tiver as bibliotecas):
# pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib

import numpy as np                     # contas numéricas (ex.: logaritmo)
import pandas as pd                    # tabelas de dados (DataFrames)
import matplotlib.pyplot as plt        # gráficos

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # Pipeline que aceita SMOTE como etapa
from xgboost import XGBClassifier
import shap

SEED = 42

# Mostra todas as colunas das tabelas no terminal (sem cortar com "...")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)   # "semente" fixa: faz os sorteios darem sempre o mesmo resultado, para o código ser reproduzível


# ======================================================================
# 1. Explorar
#
# O arquivo é lido direto do link, sem baixar. Ele tem mais de 100 MB, então não vai para o GitHub.


url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print("Linhas e colunas:", df.shape)
print(df.head())

# Proporção de cada classe: 0 = normal, 1 = fraude.
# normalize=True devolve porcentagem em vez de contagem.
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))

# O valor (Amount) das transações é muito "torto": a maioria é pequena e poucas são enormes.
# Os dois histogramas mostram isso antes e depois do logaritmo.
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(df["Amount"], bins=60)
ax[0].set_title("Amount (original)")
ax[1].hist(np.log1p(df["Amount"]), bins=60)
ax[1].set_title("log(1 + Amount)")
plt.tight_layout()
plt.show()


# ======================================================================
# 2. Preparar
#
# - Log do valor: np.log1p "achata" os valores muito altos, deixando a escala mais comportada.
#   (log1p(x) = log(1 + x), que funciona até para valor 0.)
# - Separar treino e teste com stratify=y, para os dois lados terem a mesma proporção de fraudes.
# - Padronização (StandardScaler): fica dentro do Pipeline, para ser ajustada só com os dados de treino.
#   Se eu padronizasse a base inteira antes de separar, o modelo "espiaria" o teste (vazamento de dados).

df["Amount_log"] = np.log1p(df["Amount"])

# X = as "pistas" que o modelo usa; y = a resposta certa (fraude ou não).
# Tiro Class (é a resposta) e Amount (já está representado por Amount_log; deixar os dois seria repetido).
X = df.drop(columns=["Class", "Amount"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,      # 30% para teste, 70% para treino
    stratify=y,         # mantém 0,17% de fraude nos dois lados
    random_state=SEED,
)

print("Treino:", X_train.shape, "| fraudes:", int(y_train.sum()))
print("Teste: ", X_test.shape,  "| fraudes:", int(y_test.sum()))

# Função de avaliação
#
# Para comparar os modelos sempre do mesmo jeito, uso uma função que imprime o relatório e guarda
# precisão, recall e F1 da classe de fraude numa lista.

resultados = []   # cada modelo avaliado vira uma linha aqui

def avaliar(nome, y_true, y_pred):
    """Mostra o relatório e a matriz de confusão e guarda as métricas da classe 1 (fraude)."""
    print(f"=== {nome} ===")
    print(classification_report(y_true, y_pred, digits=3))

    # Matriz de confusão: [[normais certos, alarmes falsos], [fraudes perdidas, fraudes pegas]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Fraudes pegas: {tp} | fraudes perdidas: {fn} | alarmes falsos: {fp}\n")

    resultados.append({
        "modelo": nome,
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "fraudes_pegas": tp,
        "fraudes_perdidas": fn,
        "alarmes_falsos": fp,
    })


# ======================================================================
# 3. Treinar
#
# 3.1 Regressão logística (baseline)
# É o modelo mais simples: serve de régua para saber se os mais complexos valem a pena.
# O Pipeline junta padronização + modelo num objeto só.

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe_lr.fit(X_train, y_train)

y_pred_lr = pipe_lr.predict(X_test)             # resposta 0/1 (limiar padrão de 0,5)
y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1] # probabilidade de ser fraude, de 0 a 1

avaliar("Regressão logística", y_test, y_pred_lr)

# 3.2 Random Forest com class_weight='balanced'
# Muitas árvores de decisão votando juntas. class_weight='balanced' faz cada erro em fraude "custar"
# muito mais que um erro em transação normal, compensando o desbalanceamento.

pipe_rf = Pipeline([
    ("scaler", StandardScaler()),   # árvores não precisam, mas mantenho o mesmo formato de Pipeline
    ("model", RandomForestClassifier(
        n_estimators=50,            # 50 árvores
        max_depth=10,               # cada árvore com no máximo 10 níveis (evita decorar o treino)
        class_weight="balanced",
        n_jobs=-1,                  # usa todos os núcleos do processador
        random_state=SEED,
    )),
])
pipe_rf.fit(X_train, y_train)

y_pred_rf = pipe_rf.predict(X_test)
y_prob_rf = pipe_rf.predict_proba(X_test)[:, 1]

avaliar("Random Forest (balanced)", y_test, y_pred_rf)

# 3.3 XGBoost
# Árvores construídas em sequência, cada uma corrigindo os erros da anterior.
# scale_pos_weight é o "peso da fraude". O valor recomendado é nº de normais ÷ nº de fraudes do treino.

peso_fraude = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {peso_fraude:.1f}")

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=peso_fraude,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=SEED,
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

avaliar("XGBoost", y_test, y_pred_xgb)


# ======================================================================
# 4. Avaliar
#
# 4.1 Curva ROC
# Mostra o equilíbrio entre pegar fraudes (eixo Y) e dar alarme falso (eixo X).
# Num problema tão desbalanceado ela costuma parecer ótima para todos, por isso também olho a próxima curva.


probs = {
    "Regressão logística": y_prob_lr,
    "Random Forest": y_prob_rf,
    "XGBoost": y_prob_xgb,
}

plt.figure(figsize=(6, 5))
for nome, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{nome} (AUC = {roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chute aleatório")
plt.xlabel("Taxa de falsos positivos")
plt.ylabel("Taxa de verdadeiros positivos (recall)")
plt.title("Curva ROC")
plt.legend()
plt.show()

# 4.2 Curva de precisão e recall
# É a curva mais útil aqui, porque olha só para a classe de fraude.
# Quanto mais a curva fica no canto superior direito, melhor.
# O número na legenda (Average Precision, AP) resume a curva: 1 é perfeito.

plt.figure(figsize=(6, 5))
for nome, p in probs.items():
    precision, recall, _ = precision_recall_curve(y_test, p)
    plt.plot(recall, precision, label=f"{nome} (AP = {average_precision_score(y_test, p):.3f})")
plt.axhline(y_test.mean(), color="k", ls="--", lw=1, label="Chute aleatório")
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.title("Curva de precisão e recall")
plt.legend()
plt.show()


# ======================================================================
# 5. Ajustar
#
# 5.1 Limiar de decisão: 0,5 → 0,3
# O modelo devolve uma probabilidade. Por padrão, acima de 0,5 vira "fraude".
# Baixando para 0,3, ele marca mais transações como suspeitas: pega mais fraudes (recall sobe),
# mas dá mais alarmes falsos (precisão cai).

threshold = 0.3

y_pred_lr_03 = (y_prob_lr >= threshold).astype(int)
y_pred_rf_03 = (y_prob_rf >= threshold).astype(int)
y_pred_xgb_03 = (y_prob_xgb >= threshold).astype(int)

avaliar("Regressão logística (limiar 0,3)", y_test, y_pred_lr_03)
avaliar("Random Forest (limiar 0,3)", y_test, y_pred_rf_03)
avaliar("XGBoost (limiar 0,3)", y_test, y_pred_xgb_03)

# Como as métricas mudam conforme o limiar, para o Random Forest.
# Ajuda a escolher o limiar olhando a troca entre recall e precisão.
linhas = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (y_prob_rf >= t).astype(int)
    linhas.append({
        "limiar": t,
        "precisao": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    })
print(pd.DataFrame(linhas).round(3))

# 5.2 Undersampling × oversampling (SMOTE)
# Duas formas de "equilibrar" o treino:
# - Undersampling: joga fora transações normais até sobrar a mesma quantidade das fraudes.
# - Oversampling (SMOTE): cria fraudes "sintéticas" parecidas com as reais.
#
# Importante: os dois são feitos só no treino. O teste continua com a proporção real,
# senão as métricas ficam falsamente boas.

# --- Undersampling (feito à mão com pandas) ---
treino = X_train.copy()
treino["Class"] = y_train

fraudes = treino[treino["Class"] == 1]
normais = treino[treino["Class"] == 0].sample(len(fraudes), random_state=SEED)  # sorteia o mesmo nº de normais
treino_under = pd.concat([fraudes, normais]).sample(frac=1, random_state=SEED) # junta e embaralha

X_under = treino_under.drop(columns="Class")
y_under = treino_under["Class"]
print("Treino com undersampling:", y_under.value_counts().to_dict())

pipe_under = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe_under.fit(X_under, y_under)
avaliar("Regressão logística + undersampling", y_test, pipe_under.predict(X_test))

# --- Oversampling com SMOTE ---
# O SMOTE cria exemplos novos olhando os "vizinhos" de cada fraude, então ele precisa dos dados
# já padronizados. O Pipeline do imbalanced-learn faz: padroniza -> SMOTE -> treina,
# e só aplica o SMOTE durante o treino (nunca no teste).
pipe_smote = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=SEED)),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe_smote.fit(X_train, y_train)
avaliar("Regressão logística + SMOTE", y_test, pipe_smote.predict(X_test))

# 5.3 GridSearchCV no XGBoost, escolhendo pelo recall
# Testa combinações de hiperparâmetros (as "configurações" do modelo) e fica com a que teve o maior recall
# numa validação cruzada de 3 partes, feita só dentro do treino.

param_grid = {
    "max_depth": [3, 5],            # profundidade das árvores
    "n_estimators": [50, 100],      # quantidade de árvores
    "learning_rate": [0.05, 0.1],   # tamanho do "passo" de cada árvore nova
}

grid = GridSearchCV(
    XGBClassifier(scale_pos_weight=peso_fraude, eval_metric="logloss", n_jobs=-1, random_state=SEED),
    param_grid,
    scoring="recall",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),  # cada parte com a mesma % de fraude
    n_jobs=1,
)
grid.fit(X_train, y_train)

print("Melhores hiperparâmetros:", grid.best_params_)
print(f"Recall médio na validação cruzada: {grid.best_score_:.3f}\n")

xgb_best = grid.best_estimator_
avaliar("XGBoost (GridSearch)", y_test, xgb_best.predict(X_test))

# 5.4 Comparação final
# Todos os modelos testados, ordenados pelo recall da classe de fraude.

comparacao = (
    pd.DataFrame(resultados)
    .sort_values(["recall", "precisao"], ascending=False)
    .reset_index(drop=True)
)
print(comparacao.round(3))


# ======================================================================
# 6. Explicar
#
# 6.1 Importância das variáveis (XGBoost)
# Quais colunas o modelo mais usou para decidir, no geral.

importancias = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values()

importancias.tail(15).plot(kind="barh", figsize=(7, 5))
plt.title("15 variáveis mais importantes (XGBoost)")
plt.xlabel("Importância")
plt.tight_layout()
plt.show()

# 6.2 SHAP
# A importância acima é geral. O SHAP mostra, para cada transação, quanto cada variável empurrou
# a decisão para "fraude" ou para "normal".
#
# Uso uma amostra do teste que tem todas as fraudes do teste + 300 normais, para o SHAP ter fraudes para explicar.
# (Pegar só X_test[:100] quase sempre resulta em zero fraudes.)

amostra = pd.concat([
    X_test[y_test == 1],
    X_test[y_test == 0].sample(300, random_state=SEED),
])

explainer = shap.Explainer(xgb)
shap_values = explainer(amostra)

# Gráfico de barras: impacto médio de cada variável (em valor absoluto)
shap.plots.bar(shap_values, max_display=12)

# Beeswarm: cada ponto é uma transação.
# Cor = valor da variável (vermelho alto, azul baixo); posição à direita = empurra para "fraude".
shap.plots.beeswarm(shap_values, max_display=12)

# Waterfall: explica UMA transação de fraude, variável por variável.
# Começa no valor médio do modelo (E[f(x)]) e cada barra soma ou subtrai até chegar na previsão final.
indice_fraude = 0   # a primeira linha da amostra é uma fraude real
print("Probabilidade de fraude dada pelo modelo:",
      round(float(xgb.predict_proba(amostra.iloc[[indice_fraude]])[:, 1][0]), 3))
shap.plots.waterfall(shap_values[indice_fraude], max_display=12)
