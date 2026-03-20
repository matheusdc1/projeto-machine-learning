import json
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/best_model.pkl"
MODEL_INFO_PATH = "models/model_info.json"


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


def main():
    st.set_page_config(
        page_title="Predição de Qualidade Ambiental",
        page_icon="🌿",
        layout="centered"
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

    st.subheader("Modelo selecionado")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Modelo", model_info["model_name"])

    with col_b:
        st.metric("Accuracy", f"{model_info['accuracy']:.4f}")

    with col_c:
        st.metric("F1 Macro", f"{model_info['f1_macro']:.4f}")

    st.divider()

    st.subheader("Informe os dados ambientais")

    col1, col2 = st.columns(2)

    with col1:
        temperatura = st.number_input("Temperatura (°C)", min_value=0.0, max_value=60.0, value=25.0)
        umidade = st.number_input("Umidade Relativa (%)", min_value=0.0, max_value=100.0, value=50.0)
        co2 = st.number_input("CO2 (ppm)", min_value=0.0, max_value=30000.0, value=1000.0)
        co = st.number_input("CO (ppm)", min_value=0.0, max_value=100.0, value=10.0)

    with col2:
        pressao_atm = st.number_input("Pressão Atmosférica (hPa)", min_value=800.0, max_value=1200.0, value=1013.0)
        no2 = st.number_input("NO2 (ppb)", min_value=0.0, max_value=200.0, value=30.0)
        so2 = st.number_input("SO2 (ppb)", min_value=0.0, max_value=200.0, value=20.0)
        o3 = st.number_input("O3 (ppb)", min_value=0.0, max_value=200.0, value=40.0)

    if st.button("Prever qualidade ambiental", use_container_width=True):
        input_data = pd.DataFrame(
            {
                "Temperatura": [temperatura],
                "Umidade": [umidade],
                "CO2": [co2],
                "CO": [co],
                "Pressao_Atm": [pressao_atm],
                "NO2": [no2],
                "SO2": [so2],
                "O3": [o3],
            }
        )

        prediction = model.predict(input_data)[0]

        st.subheader("Resultado da predição")
        st.success(f"A qualidade ambiental prevista é: {prediction}")

        st.dataframe(input_data, use_container_width=True)


if __name__ == "__main__":
    main()