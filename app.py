import streamlit as st
import pandas as pd
from pycaret.regression import setup, compare_models, predict_model, pull
import tempfile
import os

st.set_page_config(page_title="Previsão com IA", layout="wide")

st.title("🔮 Plataforma de Previsão com IA")
st.markdown("Suba seus dados, escolha a variável a ser prevista, e obtenha predições automáticas com Machine Learning.")

uploaded_file = st.file_uploader("📁 Faça upload do seu arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Pré-visualização dos dados:")
        st.dataframe(df.head())

        target = st.selectbox("🎯 Escolha a variável que deseja prever", df.columns)

        if st.button("🚀 Gerar Previsões"):
            with st.spinner("Treinando modelo..."):
                # Criar um diretório temporário para evitar conflitos de sessão
                with tempfile.TemporaryDirectory() as temp_dir:
                    os.chdir(temp_dir)
                    setup(data=df, target=target, silent=True, session_id=42)
                    best_model = compare_models()
                    resultados = predict_model(best_model)

                st.success("Modelo treinado com sucesso!")

                st.subheader("🔍 Resultados com predições:")
                st.dataframe(resultados[[target, "Label"]].rename(columns={"Label": "Previsão"}))

                st.subheader("📊 Importância das variáveis:")
                interpret_df = pull()
                st.dataframe(interpret_df)

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
