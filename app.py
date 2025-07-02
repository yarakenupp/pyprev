import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

st.title("🔮 Previsor com Treino + Previsão Futura")

st.header("1️⃣ Upload da base de treino (com variável alvo)")
uploaded_file = st.file_uploader("📂 Envie o CSV de treino", type="csv")

modelo = None  # vai guardar o modelo treinado
colunas_entrada = []

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("👀 Prévia dos dados de treino")
    st.dataframe(df.head())

    colunas_numericas = df.select_dtypes(include=["number"]).columns

    if len(colunas_numericas) < 2:
        st.warning("⚠️ A base precisa ter pelo menos duas colunas numéricas.")
    else:
        target = st.selectbox("🎯 Qual variável você quer prever?", colunas_numericas)

        if st.button("🚀 Treinar modelo"):
            try:
                X = df.drop(columns=[target])
                y = df[target]
                X = X.select_dtypes(include=["number"])
                colunas_entrada = X.columns.tolist()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                modelo = RandomForestRegressor()
                modelo.fit(X_train, y_train)

                y_pred = modelo.predict(X_test)
                rmse = sqrt(mean_squared_error(y_test, y_pred))

                st.success(f"✅ Modelo treinado com sucesso! RMSE: {rmse:.2f}")

                st.subheader("🔍 Amostra de previsões no treino")
                preview = X_test.copy()
                preview["Real"] = y_test.values
                preview["Previsto"] = y_pred
                st.dataframe(preview.head())

            except Exception as e:
                st.error(f"❌ Erro ao treinar: {e}")

# --------------------------------------------------------

st.header("2️⃣ Prever nova base (sem a variável alvo)")

uploaded_novos = st.file_uploader("📂 Envie nova base para previsão", type="csv", key="previsao")

if uploaded_novos and modelo:
    try:
        novos_dados = pd.read_csv(uploaded_novos)
        st.subheader("🧾 Dados recebidos para previsão")
        st.dataframe(novos_dados.head())

        # Garante as colunas certas
        novos_X = novos_dados[colunas_entrada].select_dtypes(include=["number"])

        previsoes = modelo.predict(novos_X)

        resultado = novos_dados.copy()
        resultado["Previsao"] = previsoes

        st.success("✅ Previsões geradas com sucesso!")
        st.dataframe(resultado.head())

        # Opção para baixar
        csv = resultado.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar CSV com previsões", data=csv, file_name="previsoes.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Erro ao prever novos dados: {e}")

elif uploaded_novos and not modelo:
    st.warning("⚠️ Você precisa treinar o modelo primeiro.")
