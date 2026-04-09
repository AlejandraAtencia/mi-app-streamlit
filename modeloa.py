import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── 1. CARGAR MODELO Y OBJETOS
# Asegúrate de que el archivo 'modelo-class.pkl' esté en la misma carpeta
with open('modelo-class.pkl', 'rb') as f:
    modelNN, labelencoder, variables, scaler = pickle.load(f)

# ── 2. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Viabilidad de Tokenización Inmobiliaria",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Viabilidad de Tokenización Inmobiliaria")
st.markdown("**Antioquia, Colombia** — Modelo basado en datos CEED-DANE")
st.markdown("---")

# ── 3. FORMULARIO DE ENTRADA
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Parámetros Económicos")
    PRECIOVTAX = st.number_input("Precio por m² (COP)", min_value=0, value=2500)
    
    TIPOVRDEST = st.selectbox(
        "Tipo de valor del destino",
        options=[1, 2],
        format_func=lambda x: "Real" if x == 1 else "Estimado"
    )

    ESTRATO = st.selectbox("Estrato", options=[1, 2, 3, 4, 5, 6], index=2)
    
    RANVIVI = st.selectbox(
        "Rango de precio",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: {1:"VIP", 2:"VIS", 3:"No VIS Bajo", 4:"No VIS Medio", 5:"No VIS Alto", 6:"Premium"}[x],
        index=1
    )

with col2:
    st.subheader("🔨 Estado de Obra")
    CAPITULO = st.selectbox(
        "Capítulo actual",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: f"Capítulo {x}",
        index=1
    )

    GRADOAVANC = st.slider("Grado de avance (%)", 0, 100, 50)

    st.subheader("📋 Legalidad")
    OB_FORMAL = st.selectbox(
        "Formalidad",
        options=[1, 2],
        format_func=lambda x: "Formal" if x == 1 else "Informal"
    )

    AMPLIACION = st.selectbox(
        "¿Es ampliación?",
        options=[1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No",
        index=1
    )

st.markdown("---")

# ── 4. LÓGICA DE PREDICCIÓN
if st.button("🔍 Evaluar viabilidad del proyecto", use_container_width=True):
    try:
        # Crear un diccionario con TODAS las columnas que el modelo vio en el fit
        # Inicializadas en 0.0 para cubrir municipios y otras categorías
        fila = {col: 0.0 for col in variables}

        # Asignar variables numéricas directas (si existen en el modelo)
        if 'PRECIOVTAX' in fila: fila['PRECIOVTAX'] = float(PRECIOVTAX)
        if 'GRADOAVANC' in fila: fila['GRADOAVANC'] = float(GRADOAVANC)
        if 'ESTRATO' in fila:    fila['ESTRATO'] = float(ESTRATO)
        if 'RANVIVI' in fila:    fila['RANVIVI'] = float(RANVIVI)
        if 'CAPITULO' in fila:   fila['CAPITULO'] = float(CAPITULO)

        # Asignar Dummies (usando el formato float .0 que generó el Colab)
        # TIPOVRDEST
        cat_tipovr = f'TIPOVRDEST_{float(TIPOVRDEST)}'
        if cat_tipovr in fila: fila[cat_tipovr] = 1.0

        # OB_FORMAL
        cat_formal = f'OB_FORMAL_{float(OB_FORMAL)}'
        if cat_formal in fila: fila[cat_formal] = 1.0

        # AMPLIACION
        cat_ampli = f'AMPLIACION_{float(AMPLIACION)}'
        if cat_ampli in fila: fila[cat_ampli] = 1.0

        # IMPORTANTE: Crear DataFrame asegurando el orden EXACTO de las columnas
        # Esto elimina automáticamente cualquier columna extra como USO_DOS_3
        entrada = pd.DataFrame([fila])[variables]

        # Escalar
        entrada_scaled = scaler.transform(entrada)

        # Predicción
        pred = modelNN.predict(entrada_scaled)[0]
        prob = modelNN.predict_proba(entrada_scaled)[0]

        # ── 5. MOSTRAR RESULTADOS
        st.subheader("Resultado del Análisis")
        if pred == 1:
            st.success("✅ **PROYECTO VIABLE**")
            st.metric("Confianza de Viabilidad", f"{prob[1]*100:.2f}%")
        else:
            st.error("❌ **PROYECTO NO VIABLE**")
            st.metric("Riesgo detectado", f"{prob[0]*100:.2f}%")

        # Detalle técnico opcional
        with st.expander("Ver detalle de probabilidades"):
            st.write(f"Clase 0 (No Viable): {prob[0]:.4f}")
            st.write(f"Clase 1 (Viable): {prob[1]:.4f}")

    except Exception as e:
        st.error("Ocurrió un error en el procesamiento de datos.")
        st.info(f"Detalle: {e}")

st.caption("Modelo Predictivo - Ingeniería y Ciencia de Datos")
