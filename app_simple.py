import streamlit as st
import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

# Configuración de página
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide"
)

# Título
st.title("📱 SMS Spam Detector")
st.markdown("Detecta si un mensaje SMS es spam o no usando machine learning")

# Inicializar session state para el modelo
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.baseline_model = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Botón para cargar modelo
    if st.button("🔄 Cargar Modelo Baseline", type="primary"):
        try:
            from baseline_model import BaselineModel
            
            with st.spinner("Cargando modelo..."):
                baseline = BaselineModel()
                baseline.load_model()
                
            st.session_state.baseline_model = baseline
            st.session_state.model_loaded = True
            st.success("✅ Modelo cargado correctamente!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.model_loaded = False
    
    # Mostrar estado del modelo
    if st.session_state.model_loaded:
        st.success("🟢 Modelo listo para usar")
    else:
        st.warning("🟡 Modelo no cargado")
    
    # Información
    with st.expander("ℹ️ Información"):
        st.markdown("""
        **Modelo**: TF-IDF + Regresión Logística
        **F1-Score**: 0.91 (91% precisión)
        **Dataset**: 107 mensajes (ES + EN)
        **Balance**: 54 HAM / 53 SPAM
        **Características**: N-gramas 1-3
        **Objetivo**: F1 ≥ 0.95
        """)

# Interfaz principal
if not st.session_state.model_loaded:
    st.warning("⚠️ Primero carga el modelo usando el botón en la barra lateral")
    st.stop()

# Área principal
st.subheader("📝 Analizar Mensaje SMS")

# Ejemplos predefinidos
examples = {
    "Escribir mensaje personalizado": "",
    "Ejemplo HAM": "Hola, ¿cómo estás? ¿Quieres almorzar juntos hoy?",
    "Ejemplo SPAM": "¡FELICIDADES! Has ganado $5000 USD. Envía GANADOR al 4567 para reclamar tu premio AHORA",
    "Ejemplo HAM 2": "La reunión es mañana a las 2 PM en la sala de conferencias",
    "Ejemplo SPAM 2": "ALERTA: Tu tarjeta de crédito ha sido bloqueada. Llama YA al 800-123-456",
    "Ejemplo SPAM 3": "🎉 ¡Ganaste un viaje a Europa GRATIS! Confirma enviando VIAJE al 7777",
    "Ejemplo HAM 3": "Disculpa, llegaré 10 minutos tarde a la reunión",
    "Ejemplo SPAM 4": "💰 INVIERTE $100 y gana $5000 en 24 horas. Oportunidad única: inversion-fake.com",
    "Ejemplo HAM 4": "¿Puedes recogerme del hospital? Estoy al lado del edificio principal"
}

# Selector de ejemplo
selected_example = st.selectbox("🎯 Usar ejemplo predefinido:", list(examples.keys()))

# Área de texto
user_input = st.text_area(
    "Introduce el mensaje SMS:",
    value=examples[selected_example],
    height=100,
    max_chars=500,
    placeholder="Escribe aquí tu mensaje SMS..."
)

# Botón de análisis
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_button = st.button(
        "🔍 Analizar Mensaje", 
        type="primary", 
        use_container_width=True,
        disabled=not user_input.strip()
    )

# Procesar análisis
if analyze_button and user_input.strip():
    
    with st.spinner("Analizando mensaje..."):
        try:
            # Hacer predicción
            model = st.session_state.baseline_model
            predictions = model.model.predict(model.vectorizer.transform([user_input]))
            probabilities = model.model.predict_proba(model.vectorizer.transform([user_input]))
            
            pred = predictions[0]
            prob = probabilities[0]
            
            result = "SPAM" if pred == 1 else "HAM"
            confidence = float(prob[pred])
            spam_prob = float(prob[1])
            ham_prob = float(prob[0])
            
            # Mostrar resultados
            st.subheader("📋 Resultado del Análisis")
            
            if result == "SPAM":
                st.error(f"🚨 **SPAM DETECTADO** (Confianza: {confidence:.1%})")
            else:
                st.success(f"✅ **MENSAJE LEGÍTIMO** (Confianza: {confidence:.1%})")
            
            # Métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🎯 Probabilidad SPAM", f"{spam_prob:.1%}")
            
            with col2:
                st.metric("✅ Probabilidad HAM", f"{ham_prob:.1%}")
            
            # Gráfico simple
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['HAM (Legítimo)', 'SPAM'],
                    y=[ham_prob, spam_prob],
                    marker_color=['#2E8B57', '#DC143C'],
                    text=[f'{ham_prob:.1%}', f'{spam_prob:.1%}'],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Probabilidades de Clasificación",
                yaxis_title="Probabilidad",
                yaxis=dict(range=[0, 1]),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("📊 **Modelo utilizado**: TF-IDF + Regresión Logística")
            
        except Exception as e:
            st.error(f"❌ Error procesando mensaje: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>SMS Spam Detector - Proyecto de Inteligencia Artificial UAO</small>
</div>
""", unsafe_allow_html=True)
