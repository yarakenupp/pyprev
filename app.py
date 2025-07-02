import streamlit as st

st.title("Olá, Streamlit!")
nome = st.text_input("Digite seu nome:")
if st.button("Dizer Olá"):
    st.write(f"Olá, {nome or 'visitante'}! Bem-vindo(a)!")
