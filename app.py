import streamlit as st

# Título do app
st.title("👋 Olá, Streamlit!")

# Caixa de texto para o usuário digitar o nome
nome = st.text_input("Digite seu nome:")

# Botão para confirmar
if st.button("Dizer Olá"):
    if nome:
        st.success(f"Olá, {nome}! Seja bem-vindo(a) ao seu primeiro app com Streamlit.")
    else:
        st.warning("Por favor, digite seu nome antes de continuar.")
