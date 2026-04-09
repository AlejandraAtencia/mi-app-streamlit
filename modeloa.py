import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── 1. CARGAR MODELO Y OBJETOS
# Se asume que el archivo 'modelo-class.pkl' contiene: [modelNN, labelencoder, variables, scaler]
with open('modelo-class.pkl', 'rb') as f:
    modelNN, labelencoder, variables, scaler = pickle.load(f)

# ── 2. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Viabilidad de Tokenización Inmobiliaria",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Viabilidad de Tokenización Inmobiliaria")
st.markdown("**Antioquia, Colombia** — Modelo de clasificación basado en datos CEED-DANE")
st.markdown("---")

# ── 3. FORMULARIO DE ENTRADA
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Parámetros Económicos")
    PRECIOVTAX = st.number_input("Precio por m² (miles COP)", min_value=0, value=2500)
    
    TIPOVRDEST = st.selectbox(
        "Tipo de valor del destino",
        options=[1, 2],
        format_func=lambda x: "Real" if x == 1 else "Estimado"
    )

    ESTRATO = st.selectbox("Estrato socioeconómico", options=[1, 2, 3, 4, 5, 6], index=2)
    
    RANVIVI = st.selectbox(
        "Rango de precio de vivienda",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: {1:"VIP", 2:"VIS", 3:"No VIS Bajo", 4:"No VIS Medio", 5:"No VIS Alto", 6:"Premium"}[x],
        index=1
    )

with col2:
    st.subheader("🔨 Avance de Obra")
    CAPITULO = st.selectbox(
        "Capítulo actual",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: f"Capítulo {x}",
        index=1
    )

    GRADOAVANC = st.slider("Grado de avance actual (%)", 0, 100, 50)

    st.subheader("📋 Legalidad y Tipo")
    OB_FORMAL = st.selectbox(
        "Formalidad de la obra",
        options=[1, 2],
        format_func=lambda x: "Formal" if x == 1 else "Informal"
    )

    AMPLIACION = st.selectbox(
        "¿Es una ampliación?",
        options=[1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No",
        index=1
    )

st.markdown("---")

# ── 4. LÓGICA DE PREDICCIÓN
if st.button("🔍 Evaluar viabilidad del proyecto", use_container_width=True):
    try:
        # A. Crear un diccionario con TODAS las variables que el modelo espera (64 columnas)
        # Inicializamos todas en 0.0 para que el Scaler no encuentre faltantes ni extras
        fila = {col: 0.0 for col in variables}

        # B. Asignar variables numéricas directas
        if 'PRECIOVTAX' in fila: fila['PRECIOVTAX'] = float(PRECIOVTAX)
        if 'GRADOAVANC' in fila: fila['GRADOAVANC'] = float(GRADOAVANC)
        if 'ESTRATO' in fila:    fila['ESTRATO'] = float(ESTRATO)
        if 'RANVIVI' in fila:    fila['RANVIVI'] = float(RANVIVI)
        if 'CAPITULO' in fila:   fila['CAPITULO'] = float(CAPITULO)

        # C. Asignar Dummies con el formato exacto del Colab (float .0)
        # TIPOVRDEST
        col_tipo = f'TIPOVRDEST_{float(TIPOVRDEST)}'
        if col_tipo in fila: fila[col_tipo] = 1.0

        # OB_FORMAL
        col_formal = f'OB_FORMAL_{float(OB_FORMAL)}'
        if col_formal in fila: fila[col_formal] = 1.0

        # AMPLIACION
        col_ampli = f'AMPLIACION_{float(AMPLIACION)}'
        if col_ampli in fila: fila[col_ampli] = 1.0

        # D. Crear DataFrame y REORDENAR columnas exactamente como en el entrenamiento
        # Esto elimina columnas que no deberían estar y ordena las 64 variables correctamente
        entrada_df = pd.DataFrame([fila])[variables]

        # E. Escalar y Predecir
        entrada_scaled = scaler.transform(entrada_df)
        prediccion = modelNN.predict(entrada_scaled)[0]
        probabilidades = modelNN.predict_proba(entrada_scaled)[0]

        # ── 5. DESPLIEGUE DE RESULTADOS
        st.subheader("Resultado del Análisis Predictivo")
        
        if prediccion == 1:
            st.success("✅ **PROYECTO VIABLE** para inversión/tokenización")
            st.metric("Nivel de Confianza", f"{probabilidades[1]*100:.2f}%")
        else:
            st.error("❌ **PROYECTO NO VIABLE** bajo las condiciones actuales")
            st.metric("Probabilidad de Riesgo", f"{probabilidades[0]*100:.2f}%")

    except Exception as e:
        st.error("Error en el procesamiento técnico.")
        st.info(f"Detalle: {e}")

st.caption("Desarrollado para la materia de Minería de Datos - Maestría en Ciencia de Datos")
