# 🔍 Detecção de Fraude em Cartão de Crédito

Projeto de pipeline completo de detecção de fraude em transações de cartão de crédito utilizando machine learning.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Desafio](#desafio)
- [Stack Utilizada](#stack-utilizada)
- [Estrutura do Notebook](#estrutura-do-notebook)
- [Resultados](#resultados)
- [Como Executar](#como-executar)
- [Instalação](#instalação)
- [Licença](#licença)

## 📋 Sobre o Projeto

Este projeto implementa um pipeline completo de detecção de fraude em transações de cartão de crédito, abordando um dos problemas mais clássicos e desafiadores do machine learning: **a classificação de dados altamente desbalanceados**.

O dataset utilizado contém transações de cartão de crédito feitas por titulares europeus em setembro de 2013. Das 284.807 transações, apenas **492 (0,17%) são fraudes**.

### O Problema da Acurácia Enganosa

Com 99,83% das transações sendo normais, um modelo que sempre prevê "não fraude" alcançaria **99,83% de acurácia** — mas seria completamente inútil na prática. Por isso, nosso foco são as métricas que realmente importam:

- **Recall (Detecção de Fraude):** Quantas fraudes o modelo consegue identificar?
- **Precisão:** Das transações marcadas como fraude, quantas realmente são?
- **F1-Score:** Média harmônica entre precisão e recall — nossa métrica principal.
- **AUC-ROC:** Capacidade discriminatória do modelo.

## 🎯 Desafio

- [x] Explorar o dataset e entender a distribuição das classes
- [x] Criar features de engenharia (log do Amount, log do Time)
- [x] Padronizar variáveis com StandardScaler
- [x] Separar treino/teste com stratify para manter a proporção de fraude
- [x] Comparar abordagens de balanceamento: baseline, undersampling (RandomUnderSampler), oversampling (SMOTE)
- [x] Treinar e comparar modelos: Logistic Regression, Random Forest, XGBoost
- [x] Avaliar com curvas ROC e Precision-Recall
- [x] Otimizar o limiar de decisão para maximizar o F1
- [x] Explicar as decisões do modelo com SHAP
- [x] Identificar as features mais importantes para detecção de fraude

## 🛠️ Stack Utilizada

| Tecnologia | Uso |
|------------|-----|
| **Python 3.14** | Linguagem principal |
| **pandas** | Manipulação e análise de dados |
| **numpy** | Transformações numéricas |
| **scikit-learn 1.9** | Modelos de ML, métricas, pré-processamento |
| **XGBoost 3.4** | Gradient Boosting |
| **imbalanced-learn 0.14** | Técnicas de balanceamento (SMOTE, RandomUnderSampler) |
| **SHAP 0.52** | Explicação de predições do modelo |
| **matplotlib / seaborn** | Visualizações |

## 📂 Estrutura do Notebook

O arquivo `fraud_detection.ipynb` contém **32 células** organizadas em 11 seções:

1. **Importar Bibliotecas** — Todos os imports necessários
2. **Explorar os Dados** — `shape`, distribuição de classes, estatísticas descritivas, valores nulos
3. **Preparação dos Dados** — Feature engineering (log transforms), padronização com `StandardScaler`, `train_test_split` com `stratify=y`
4. **Tratamento do Desbalanceamento** — `RandomUnderSampler` e `SMOTE`
5. **Treinamento e Comparação de Modelos** — Baseline + abordagens com undersampling/oversampling para cada algoritmo
6. **Comparação dos Modelos** — Ranking por F1-Score com gráficos de acurácia, F1 e AUC-ROC
7. **Curvas de Avaliação** — Curvas ROC e Precision-Recall para todos os modelos
8. **Ajuste do Limiar de Decisão** — Otimização do threshold para máximo F1
9. **Importância das Variáveis** — Top features do Random Forest
10. **Explicação com SHAP** — `summary_plot`, `waterfall_plot` e contribuições individuais
11. **Conclusão** — Resumo consolidado dos resultados

## 📊 Resultados

### Comparação de Modelos (F1-Score da classe fraude)

| Modelo | F1-Score | AUC-ROC | Observação |
|--------|----------|---------|------------|
| Random Forest (Baseline) | **~0,86** | ~0,96 | Melhor desempenho geral |
| XGBoost (Baseline) | **~0,85** | ~0,97 | Competitivo |
| Logistic Regression (Baseline) | ~0,10 | ~0,97 | Fraco, mas AUC alto |
| Undersampling / SMOTE | ~0,07-0,08 | ~0,97 | Pioraram com balanceamento |

### Principais Conclusões

- **`class_weight='balanced'`** no Random Forest funciona melhor que técnicas de reamostragem para este dataset
- O **AUC-ROC** é alto para todos os modelos (~0,95+), mas o F1-Score é a métrica decisiva
- A **otimização do limiar** de decisão pode melhorar ainda mais o F1
- O SHAP revela quais features (V14, V12, V10, V4, V17) mais influenciam a detecção de fraude

## ▶️ Como Executar

### Pré-requisitos

- Python 3.14+
- pip

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/fraud-detection.git
cd fraud-detection

# Crie e ative um ambiente virtual (opcional, recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Baixe o dataset do Kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Coloque o arquivo creditcard.csv na pasta raiz do projeto

# Execute o notebook
jupyter notebook fraud_detection.ipynb
```

### Executando diretamente

```bash
# Se o jupyter não estiver instalado como comando, use:
python -m jupyter notebook fraud_detection.ipynb
```

## 📦 Dependências

Ver `requirements.txt` para a lista completa de pacotes e versões.

## 📝 Dataset

- **Fonte:** Kaggle - Credit Card Fraud Detection ([mlg-ulb](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Tamanho:** 284.807 transações, 31 features
- **Features:** `Time`, `V1-V28` (PCA), `Amount`, `Class`
- **Fraudes:** 492 (0,17%)
- **Licença:** CC BY-NC-SA 4.0

## 🔮 Evoluções Possíveis

- [ ] Testar outros modelos (LightGBM, CatBoost)
- [ ] Expandir busca de hiperparâmetros com `GridSearchCV`
- [ ] Criar features de comportamento temporal (transações em sequência)
- [ ] Implementar detecção de anomalias não supervisionadas
- [ ] Comparar undersampling vs oversampling com métricas detalhadas

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

**Autor:** Seu Nome

**Data:** Setembro 2026
