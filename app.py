import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Mini Previsor IA", layout="centered")
st.title("🔮 Previsão Automática com IA")

uploaded_file = st.file_uploader("📁 Envie um arquivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Prévia dos dados")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Qual variável você quer prever?", df.columns)

    if st.button("🚀 Rodar previsão"):
        try:
            X = df.drop(columns=[target])
            y = df[target]

            # Remove colunas não numéricas automaticamente
            X = X.select_dtypes(include=["number"])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            st.success(f"Modelo treinado com sucesso! RMSE: {rmse:.2f}")

            st.subheader("🔎 Amostra de Previsões")
            preview = X_test.copy()
            preview["Real"] = y_test.values
            preview["Previsto"] = y_pred
            st.dataframe(preview.head(10))

        except Exception as e:
            st.error(f"Erro: {e}")
