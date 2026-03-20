import json

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/best_model.pkl"
MODEL_INFO_PATH = "models/model_info.json"


SCENARIOS = {
    "Personalizado": {
        "Temperatura": 25.0,
        "Umidade": 50.0,
        "CO2": 1000.0,
        "CO": 10.0,
        "Pressao_Atm": 1013.0,
        "NO2": 30.0,
        "SO2": 20.0,
        "O3": 40.0,
    },
    "Cenário Saudável": {
        "Temperatura": 22.0,
        "Umidade": 55.0,
        "CO2": 450.0,
        "CO": 2.0,
        "Pressao_Atm": 1015.0,
        "NO2": 12.0,
        "SO2": 5.0,
        "O3": 18.0,
    },
    "Cenário Moderado": {
        "Temperatura": 28.0,
        "Umidade": 60.0,
        "CO2": 1200.0,
        "CO": 12.0,
        "Pressao_Atm": 1008.0,
        "NO2": 35.0,
        "SO2": 18.0,
        "O3": 55.0,
    },
    "Cenário Crítico": {
        "Temperatura": 35.0,
        "Umidade": 78.0,
        "CO2": 3500.0,
        "CO": 32.0,
        "Pressao_Atm": 990.0,
        "NO2": 85.0,
        "SO2": 38.0,
        "O3": 105.0,
    },
}


@st.cache_resource
def load_model():
    """
    Carrega o modelo treinado salvo em disco.
    """
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_info():
    """
    Carrega as informações do melhor modelo.
    """
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_result_style(prediction: str):
    """
    Define cor e mensagem explicativa para a classe prevista.
    """
    styles = {
        "Excelente": {
            "color": "#1b8a5a",
            "message": "O ambiente apresenta condições bastante favoráveis.",
        },
        "Boa": {
            "color": "#2eaf7d",
            "message": "O ambiente apresenta boas condições gerais.",
        },
        "Moderada": {
            "color": "#d4a017",
            "message": "Há sinais moderados de atenção nas condições ambientais.",
        },
        "Ruim": {
            "color": "#d96c06",
            "message": "O ambiente apresenta sinais relevantes de deterioração.",
        },
        "Muito Ruim": {
            "color": "#c0392b",
            "message": "O ambiente está em condição crítica e exige maior atenção.",
        },
    }
    return styles.get(
        prediction,
        {
            "color": "#444444",
            "message": "Resultado gerado pelo modelo.",
        },
    )


def build_input_dataframe(values: dict) -> pd.DataFrame:
    """
    Constrói o DataFrame no mesmo formato esperado pelo modelo.
    """
    return pd.DataFrame(
        {
            "Temperatura": [values["Temperatura"]],
            "Umidade": [values["Umidade"]],
            "CO2": [values["CO2"]],
            "CO": [values["CO"]],
            "Pressao_Atm": [values["Pressao_Atm"]],
            "NO2": [values["NO2"]],
            "SO2": [values["SO2"]],
            "O3": [values["O3"]],
        }
    )


def main():
    st.set_page_config(
        page_title="Predição de Qualidade Ambiental",
        page_icon="🌿",
        layout="wide",
    )

    st.title("Predição de Qualidade Ambiental")
    st.markdown(
        """
        Este sistema utiliza um modelo de Machine Learning para prever a
        **qualidade ambiental** com base em medições como temperatura,
        umidade e concentração de gases.
        """
    )

    st.warning(
        "Este conteúdo é destinado apenas para fins educacionais. "
        "Os dados exibidos são ilustrativos e podem não corresponder a situações reais."
    )

    model = load_model()
    model_info = load_model_info()

    st.subheader("Informações do modelo")
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Modelo", model_info["model_name"])

    with metric_col2:
        st.metric("Accuracy", f"{model_info['accuracy']:.4f}")

    with metric_col3:
        st.metric("F1 Macro", f"{model_info['f1_macro']:.4f}")

    st.divider()

    st.sidebar.header("Entradas do Sistema")
    scenario_name = st.sidebar.selectbox(
        "Escolha um cenário",
        list(SCENARIOS.keys()),
    )
    selected = SCENARIOS[scenario_name]

    temperatura = st.sidebar.number_input(
        "Temperatura (°C)", min_value=0.0, max_value=60.0, value=float(selected["Temperatura"])
    )
    umidade = st.sidebar.number_input(
        "Umidade Relativa (%)", min_value=0.0, max_value=100.0, value=float(selected["Umidade"])
    )
    co2 = st.sidebar.number_input(
        "CO2 (ppm)", min_value=0.0, max_value=30000.0, value=float(selected["CO2"])
    )
    co = st.sidebar.number_input(
        "CO (ppm)", min_value=0.0, max_value=100.0, value=float(selected["CO"])
    )
    pressao_atm = st.sidebar.number_input(
        "Pressão Atmosférica (hPa)", min_value=800.0, max_value=1200.0, value=float(selected["Pressao_Atm"])
    )
    no2 = st.sidebar.number_input(
        "NO2 (ppb)", min_value=0.0, max_value=200.0, value=float(selected["NO2"])
    )
    so2 = st.sidebar.number_input(
        "SO2 (ppb)", min_value=0.0, max_value=200.0, value=float(selected["SO2"])
    )
    o3 = st.sidebar.number_input(
        "O3 (ppb)", min_value=0.0, max_value=200.0, value=float(selected["O3"])
    )

    predict_button = st.sidebar.button("Prever qualidade ambiental", use_container_width=True)

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.subheader("Dados informados")
        input_values = {
            "Temperatura": temperatura,
            "Umidade": umidade,
            "CO2": co2,
            "CO": co,
            "Pressao_Atm": pressao_atm,
            "NO2": no2,
            "SO2": so2,
            "O3": o3,
        }
        input_data = build_input_dataframe(input_values)
        st.dataframe(input_data, use_container_width=True)

        st.subheader("Sobre o sistema")
        st.markdown(
            """
            - O modelo foi escolhido com base na comparação entre diferentes algoritmos.
            - As métricas principais usadas na avaliação foram `Accuracy` e `F1 Macro`.
            - O `F1 Macro` recebeu atenção especial por causa do desbalanceamento entre classes.
            """
        )

    with right_col:
        st.subheader("Resultado da previsão")

        if predict_button:
            prediction = model.predict(input_data)[0]
            style = get_result_style(prediction)

            st.markdown(
                f"""
                <div style="
                    background-color: {style['color']};
                    color: white;
                    padding: 18px;
                    border-radius: 12px;
                    text-align: center;
                    font-size: 26px;
                    font-weight: bold;
                    margin-bottom: 12px;
                ">
                    {prediction}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(style["message"])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                class_names = model.classes_

                proba_df = pd.DataFrame(
                    {
                        "Classe": class_names,
                        "Probabilidade": probabilities,
                    }
                ).sort_values("Probabilidade", ascending=False)

                proba_df["Probabilidade"] = proba_df["Probabilidade"].round(4)

                st.subheader("Probabilidades por classe")
                st.bar_chart(
                    proba_df.set_index("Classe")["Probabilidade"],
                    use_container_width=True,
                )
                st.dataframe(proba_df, use_container_width=True)

        else:
            st.write("Preencha os dados na barra lateral e clique em prever para gerar o resultado.")

    st.divider()

    st.subheader("Limitações")
    st.markdown(
        """
        - O modelo foi treinado com um conjunto de dados ilustrativo.
        - Os resultados não devem ser interpretados como diagnóstico real.
        - O sistema foi desenvolvido exclusivamente para fins educacionais.
        """
    )


if __name__ == "__main__":
    main()
