import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/best_model.pkl"


@st.cache_resource
def load_model():
    """
    Carrega o modelo treinado salvo em disco.
    """
    model = joblib.load(MODEL_PATH)
    return model


def main():
    st.set_page_config(
        page_title="Qualidade Ambiental",
        page_icon="🌿",
        layout="centered"
    )

    st.title("Sistema de Predição de Qualidade Ambiental")

    st.write(
        "Preencha os valores das variáveis ambientais para obter uma previsão "
        "da qualidade ambiental."
    )

    st.warning(
        "Este conteúdo é destinado apenas para fins educacionais. "
        "Os dados exibidos são ilustrativos e podem não corresponder a situações reais."
    )

    model = load_model()

    temperatura = st.number_input("Temperatura (°C)", min_value=0.0, max_value=60.0, value=25.0)
    umidade = st.number_input("Umidade Relativa (%)", min_value=0.0, max_value=100.0, value=50.0)
    co2 = st.number_input("CO2 (ppm)", min_value=0.0, max_value=30000.0, value=1000.0)
    co = st.number_input("CO (ppm)", min_value=0.0, max_value=100.0, value=10.0)
    pressao_atm = st.number_input("Pressão Atmosférica (hPa)", min_value=800.0, max_value=1200.0, value=1013.0)
    no2 = st.number_input("NO2 (ppb)", min_value=0.0, max_value=200.0, value=30.0)
    so2 = st.number_input("SO2 (ppb)", min_value=0.0, max_value=200.0, value=20.0)
    o3 = st.number_input("O3 (ppb)", min_value=0.0, max_value=200.0, value=40.0)

    if st.button("Prever Qualidade Ambiental"):
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

        st.success(f"Qualidade Ambiental prevista: {prediction}")


if __name__ == "__main__":
    main()