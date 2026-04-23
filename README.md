# Diabetes Predictor App

Aplicativo Streamlit para prever risco de diabetes a partir de variáveis clínicas do Pima Indians Diabetes Dataset.

## App ao vivo

Link do Streamlit Cloud: adicione aqui o link gerado após o deploy.

Sugestão para o currículo:

`Diabetes Predictor App - modelo Random Forest com Scikit-Learn e interface Streamlit: COLE_AQUI_O_LINK_DO_STREAMLIT`

## Funcionalidades

- Modelo de classificação com Scikit-Learn usando Random Forest.
- Formulário de entrada para informar dados do paciente.
- Predição ao vivo com probabilidade estimada.
- Métricas de avaliação do modelo.
- Gráfico de importância das features.

## Tecnologias

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Matplotlib

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

O projeto usa o Pima Indians Diabetes Dataset, salvo em `data/diabetes.csv`.

Colunas usadas pelo modelo:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Variável alvo:

- Outcome

## Deploy gratuito no Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io/).
2. Conecte sua conta GitHub.
3. Escolha o repositório `nicolasagra-dev/projeto-3`.
4. Configure:
   - Branch: `main`
   - Main file path: `app.py`
5. Clique em Deploy.
6. Copie o link gerado e substitua o campo "Link do Streamlit Cloud" neste README.
7. Use o mesmo link no currículo.

## Aviso

Este projeto tem finalidade educacional e não substitui avaliação médica.

