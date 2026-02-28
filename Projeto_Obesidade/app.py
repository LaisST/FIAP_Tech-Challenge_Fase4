# Importar bibliotecas
import streamlit as st
import pandas as pd
import joblib
import os

# Dicionários de Tradução

map_genero = {
    "Masculino": "Male",
    "Feminino": "Female"
}

map_sim_nao = {
    "Sim": "yes",
    "Não": "no"
}

map_caec = {
    "Não": "no",
    "As vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always"
}

map_alcool = {
    "Não bebe": "no",
    "As vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always"
}

map_transporte = {
    "Transporte Público": "Public_Transportation",
    "Carro": "Automóvel",
    "A Pé": "Walking",
    "Bicicleta": "Bike",
    "Moto": "Motorbike"
}

map_frequencia = {
    "0 - Nunca": 0,
    "1 - Raramente": 1,
    "2 - Às vezes": 2,
    "3 - Frequente": 3
}

map_resultado = {
    "Insufficient_Weight": "Abaixo do peso",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidade I",
    "Obesity_Type_II": "Obesidade II",
    "Obesity_Type_III": "Obesidade III"
}

# Carregar modelo e scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
COLUNAS_PATH = os.path.join(BASE_DIR, "models", "colunas.pkl")

modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
colunas = joblib.load(COLUNAS_PATH)

# Interface
st.set_page_config(
    page_title="Predição de Obesidade",
    layout="centered"
)

st.title("Sistema de Predição de Obesidade")
st.write("Preencha os dados do paciente:")

# Dados do usuário
# Genero
gender_pt = st.selectbox("Gênero", ["Masculino", "Feminino"])
gender = map_genero[gender_pt]

# Idade
age = st.number_input("Idade", min_value=1, max_value=120, value=25)

# Altura
height = st.number_input("Altura (em metros)", min_value=1.0, max_value=2.5, value=1.70)

# Peso
weight = st.number_input("Peso (kg)", min_value=20.0, max_value=300.0, value=70.0)

# Historico Familiar
family_pt = st.selectbox(
    "Histórico familiar de sobrepeso?",
    ["Sim", "Não"]
)
family_history = map_sim_nao[family_pt]

# Consumo de Calorias
favc_pt = st.selectbox(
    "Consome alimentos calóricos frequentemente?",
    ["Sim", "Não"]
)
favc = map_sim_nao[favc_pt]

# Consumo de Vegetais
fcvc_pt = st.select_slider(
"Consumo de vegetais", 
options=[
        "0 - Nunca",
        "1 - Raramente",
        "2 - Às vezes",
        "3 - Frequente"
    ]
)
fcvc = map_frequencia[fcvc_pt]

# Número de Refeicoes
ncp_pt = st.select_slider(
    "Número de refeições",
    options=[
    "0 - Nunca",
    "1 - Raramente",
    "2 - Às vezes",
    "3 - Frequente"
]
)
ncp = map_frequencia[ncp_pt]

# Alimentacao entre as Refeicoes
caec_pt = st.selectbox(
    "Come entre as refeições?",
    ["Não", "As vezes", "Frequentemente", "Sempre"]
)
caec = map_caec[caec_pt]

# Habito de Fumar
smoke_pt = st.selectbox("Fuma?", ["Sim", "Não"])
smoke = map_sim_nao[smoke_pt]

# Consumo de Agua
ch2o_pt = st.select_slider(
    "Consumo de água",
    options=[
        "0 - Nunca",
        "1 - Raramente",
        "2 - Às vezes",
        "3 - Frequente"
    ]
    )
ch2o = map_frequencia[ch2o_pt]

# Controle de Calorias
scc_pt = st.selectbox(
    "Monitora calorias?",
    ["Sim", "Não"]
)
scc = map_sim_nao[scc_pt]

# Habito de Atividade Fisica
faf_pt = st.select_slider(
    "Atividade física", 
    options=[
        "0 - Nunca",
        "1 - Raramente",
        "2 - Às vezes",
        "3 - Frequente"
    ]
)
faf = map_frequencia[faf_pt]

# Habito de Telas
tue_pt = st.select_slider(
    "Tempo em telas",
    options=[
        "0 - Nunca",
        "1 - Raramente",
        "2 - Às vezes",
        "3 - Frequente"
    ]
)
tue = map_frequencia[tue_pt]

# Consumo de Alcool
calc_pt = st.selectbox(
    "Consumo de álcool",
    ["Não bebe", "As vezes", "Frequentemente", "Sempre"]
)
calc = map_alcool[calc_pt]

# Meios de Transporte
mtrans_pt = st.selectbox(
    "Meio de transporte",
    [
        "Transporte Público",
        "Carro",
        "A Pé",
        "Bicicleta",
        "Moto"
    ]
)
mtrans = map_transporte[mtrans_pt]

# Botão de previsão
if st.button("Prever"):

    dados = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }

    df = pd.DataFrame([dados])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Ajustar colunas ao modelo
    colunas_modelo = modelo.feature_names_in_

    df = df.reindex(columns=colunas_modelo, fill_value=0)

    # Escalonar
    df_scaled = scaler.transform(df)

    # Previsão
    pred = modelo.predict(df_scaled)[0]

    # Traduzir resultado
    resultado_pt = map_resultado.get(pred, pred)
    st.success(f"Resultado: {resultado_pt}")

