---
title: Qualidade Ambiental ML
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
sdk_version: "1.0.0"
app_file: app/app.py
pinned: false
---

# Projeto de Machine Learning - Qualidade Ambiental

## Visão Geral

Este projeto apresenta um pipeline completo de Machine Learning para previsão da **qualidade ambiental**, desenvolvido como atividade final da unidade curricular de Aprendizado de Máquina.

O sistema foi construído com base nas etapas do **CRISP-DM**, incluindo análise exploratória dos dados, preparação e transformação, treinamento e comparação de modelos, rastreamento de experimentos com MLflow, salvamento do melhor modelo e disponibilização de uma aplicação web com Streamlit.

## Objetivo

O objetivo do projeto é prever a variável **`Qualidade_Ambiental`** com base em medições ambientais, como:

- Temperatura
- Umidade relativa
- CO2
- CO
- Pressão atmosférica
- NO2
- SO2
- O3

Trata-se de um problema de **classificação multiclasse**.

## Dataset

O dataset utilizado contém **10.000 registros** e **9 variáveis**, sendo:

### Variáveis preditoras

- `Temperatura`
- `Umidade`
- `CO2`
- `CO`
- `Pressao_Atm`
- `NO2`
- `SO2`
- `O3`

### Variável alvo

- `Qualidade_Ambiental`

### Classes da variável alvo

- `Boa`
- `Excelente`
- `Moderada`
- `Ruim`
- `Muito Ruim`

## Etapas do Projeto

### 1. Análise Exploratória dos Dados

Na etapa de EDA, foram realizadas análises para compreender a estrutura dos dados e identificar problemas de qualidade.

Principais pontos observados:

- valores ausentes em `Temperatura` e `Umidade`
- presença de valores inválidos em `Pressao_Atm`, como `erro_sensor`
- desbalanceamento entre as classes da variável alvo
- indícios de outliers, especialmente em `CO2`

Também foram gerados gráficos de distribuição, boxplots e mapa de correlação.

### 2. Preparação dos Dados

A preparação incluiu:

- conversão da coluna `Pressao_Atm` para formato numérico
- transformação de valores inválidos em `NaN`
- imputação de valores ausentes com a mediana
- padronização das variáveis numéricas
- separação entre treino e teste com estratificação

Essa etapa foi implementada com `Pipeline` e `ColumnTransformer`, tornando o processo reproduzível.

### 3. Treinamento e Comparação de Modelos

Foram treinados e comparados os seguintes modelos:

- Logistic Regression
- Logistic Regression com balanceamento de classes
- Random Forest
- Random Forest com balanceamento de classes
- Gradient Boosting
- HistGradientBoosting

As métricas principais utilizadas foram:

- `Accuracy`
- `F1 Macro`

A escolha do melhor modelo foi baseada principalmente no **F1 Macro**, por ser mais adequado para um cenário com classes desbalanceadas.

Na versão final do projeto, o melhor desempenho foi obtido com o modelo **HistGradientBoostingClassifier**, alcançando:

- `Accuracy`: **0.9275**
- `F1 Macro`: **0.6724**

### Comparação final dos modelos

| Modelo | Accuracy | F1 Macro |
|---|---:|---:|
| Logistic Regression | 0.7630 | 0.4676 |
| Logistic Regression com balanceamento | 0.6730 | 0.5424 |
| Random Forest | 0.9065 | 0.5670 |
| Random Forest com balanceamento | 0.9190 | 0.6291 |
| Gradient Boosting | 0.9205 | 0.5968 |
| HistGradientBoosting | **0.9275** | **0.6724** |

### Interpretação dos resultados

O projeto apresentou desempenho elevado em termos de acurácia global, indicando boa capacidade de classificação geral. No entanto, como o dataset possui classes desbalanceadas, o `F1 Macro` foi adotado como métrica principal de decisão, pois avalia de forma mais justa o desempenho em todas as classes, inclusive nas menos representadas.

Esse critério foi importante porque permitiu escolher um modelo não apenas com bom desempenho médio, mas também com melhor equilíbrio entre as categorias da variável alvo.

### 4. Registro de Experimentos com MLflow

Os experimentos foram rastreados com `MLflow`, registrando:

- nome do modelo
- parâmetros
- métricas
- artefatos do modelo treinado

Isso permite comparar execuções e reforça a parte de MLOps do projeto.

### 5. Aplicação Web

Foi desenvolvida uma aplicação com `Streamlit` para permitir que o usuário insira valores ambientais manualmente e obtenha uma previsão da qualidade ambiental.

A interface também exibe:

- o modelo selecionado
- a accuracy do melhor modelo
- o F1 Macro do melhor modelo
- probabilidades por classe
- cenários prontos para teste

### Prints da aplicação


- Tela inicial da aplicação
  <img src="https://i.imgur.com/MYoOAEh.png" alt="Tela inicial" />
- Print de uma previsão realizada
  <img src="https://i.imgur.com/9VfaH2c.png" alt="Previsao" />
- Print do gráfico de probabilidades por classe
  <img src="https://i.imgur.com/GoIdgab.png" alt="Grafico" />
  
Além disso, a página contém o aviso obrigatório da atividade:

> Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.

## Estrutura do Projeto

```text
ambiental/
├─ app/
│  └─ app.py
├─ data/
│  ├─ ambiental.txt
│  └─ dataset_ambiental.csv
├─ models/
│  ├─ best_model.pkl
│  └─ model_info.json
├─ notebooks/
│  └─ 01_eda.ipynb
├─ src/
│  ├─ data_prep.py
│  └─ train.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- MLflow
- Joblib
- Streamlit
- Jupyter Notebook

## Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd ambiental
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o treinamento

```bash
python src/train.py
```

Esse comando irá:

- treinar os modelos
- comparar os resultados
- registrar os experimentos no MLflow
- salvar o melhor modelo em `models/best_model.pkl`
- salvar as informações do modelo em `models/model_info.json`

### 4. Abrir a interface do MLflow

```bash
mlflow ui
```

Depois, acesse no navegador:

```text
http://127.0.0.1:5000
```

### 5. Executar a aplicação web ou acessar localmente

A aplicação está disponível publicamente em:

https://huggingface.co/spaces/matheusdc1/qualidade-ambiental-ml

Para acessar localmente execute o comando no terminal:

```bash
streamlit run app/app.py
```

## Como Usar a Aplicação

1. Abra a aplicação Streamlit no navegador.
2. Escolha um cenário pronto ou preencha os valores manualmente.
3. Clique no botão de previsão.
4. Visualize a classe prevista, a interpretação textual e as probabilidades por classe.

## Resultados do Projeto

O projeto permitiu construir uma solução completa de Machine Learning, passando por todas as etapas principais esperadas em um fluxo real de trabalho.

Entre os principais resultados obtidos:

- compreensão e análise do dataset
- tratamento de inconsistências e valores ausentes
- comparação entre múltiplos modelos
- registro dos experimentos com MLflow
- salvamento do melhor modelo
- criação de uma interface web interativa para uso do sistema
- melhoria do desempenho do pipeline após uma segunda rodada de experimentos

### Melhor modelo final

- Modelo: `hist_gradient_boosting`
- Accuracy: `0.9275`
- F1 Macro: `0.6724`

## Limitações e Trabalhos Futuros

Apesar dos bons resultados obtidos, o projeto apresenta algumas limitações importantes:

- o conjunto de dados utilizado é ilustrativo e educacional
- existe desbalanceamento entre as classes da variável alvo
- algumas previsões em cenários muito incomuns podem não seguir uma interpretação totalmente intuitiva
- o desempenho em classes raras ainda pode ser melhorado

Como possibilidades de evolução futura, poderiam ser exploradas:

- técnicas de reamostragem, como `SMOTE`
- ajuste fino de hiperparâmetros com `GridSearchCV` ou `RandomizedSearchCV`
- inclusão de novas variáveis ambientais
- uso de dados reais para aumentar a robustez do sistema

## Conclusão

O projeto atendeu à proposta de construir um pipeline completo de Machine Learning, passando por todas as etapas centrais de um fluxo real: análise exploratória, preparação dos dados, modelagem, rastreamento de experimentos, escolha do melhor modelo, criação de interface e deploy público.

Além disso, a segunda rodada de experimentos contribuiu para melhorar o desempenho do sistema, resultando na escolha do modelo `HistGradientBoosting`, que apresentou o melhor equilíbrio entre desempenho global e qualidade de classificação entre as diferentes classes.

## Checklist do Projeto

### 1. Estrutura inicial

- [x] Criar estrutura de pastas
- [x] Adicionar `requirements.txt`
- [x] Adicionar `.gitignore`
- [x] Organizar arquivos do dataset

### 2. Entendimento dos dados

- [x] Criar notebook de EDA
- [x] Analisar tipos de dados
- [x] Analisar valores ausentes
- [x] Analisar distribuição das classes
- [x] Analisar outliers
- [x] Gerar gráficos e insights

### 3. Preparação dos dados

- [x] Tratar valores ausentes
- [x] Corrigir `Pressao_Atm`
- [x] Preparar variáveis preditoras e alvo
- [x] Separar treino e teste
- [x] Criar pipeline de pré-processamento

### 4. Modelagem

- [x] Treinar modelo baseline
- [x] Treinar modelos adicionais
- [x] Comparar métricas
- [x] Escolher melhor modelo

### 5. MLOps

- [x] Configurar MLflow
- [x] Registrar parâmetros e métricas
- [x] Salvar melhor modelo

### 6. Aplicação web

- [x] Criar interface com Streamlit
- [x] Inserir aviso educacional obrigatório
- [x] Testar previsões

### 7. Documentação

- [x] Escrever README completo
- [x] Explicar como rodar o projeto
- [x] Explicar como usar a aplicação

### 8. Publicação

- [x] Subir para GitHub
- [x] Fazer deploy público
