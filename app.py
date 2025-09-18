import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import sys
import os

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from baseline_model import BaselineModel
from distilbert_model import DistilBERTModel
from data_preprocessing import DataPreprocessor


class SpamDetectorApp:
    """
    Aplicación Streamlit para detección de spam en SMS
    """
    
    def __init__(self):
        """Inicializa la aplicación"""
        self.config = self.load_config()
        self.baseline_model = None
        self.distilbert_model = None
        self.best_model_type = None
        
        # Configurar página
        st.set_page_config(
            page_title=self.config['app']['title'],
            page_icon="📱",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def load_config(self) -> Dict:
        """Carga la configuración desde archivo YAML"""
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        else:
            # Configuración por defecto si no existe el archivo
            return {
                'app': {
                    'title': 'SMS Spam Detector',
                    'description': 'Detecta si un mensaje SMS es spam o no usando modelos de machine learning',
                    'max_input_length': 500
                },
                'evaluation': {'target_f1_score': 0.95},
                'paths': {'models_dir': 'models', 'results_dir': 'results'}
            }
    
    def load_models(self) -> Tuple[bool, bool]:
        """Carga los modelos entrenados"""
        baseline_loaded = False
        distilbert_loaded = False
        
        # Intentar cargar modelo baseline
        try:
            self.baseline_model = BaselineModel()
            self.baseline_model.load_model()
            baseline_loaded = True
            st.success("✅ Modelo Baseline cargado correctamente")
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar el modelo Baseline: {e}")
        
        # Intentar cargar modelo DistilBERT
        try:
            self.distilbert_model = DistilBERTModel()
            self.distilbert_model.load_model()
            distilbert_loaded = True
            st.success("✅ Modelo DistilBERT cargado correctamente")
        except Exception as e:
            st.info(f"ℹ️ DistilBERT no disponible (se requiere entrenamiento): {str(e)[:50]}...")
            st.info("💡 Por ahora puedes usar el modelo Baseline que funciona perfectamente")
        
        return baseline_loaded, distilbert_loaded
    
    def get_best_model_type(self) -> Optional[str]:
        """Determina cuál es el mejor modelo basado en los resultados de evaluación"""
        results_path = Path(self.config['paths']['results_dir']) / 'evaluation_results.json'
        
        if not results_path.exists():
            return None
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return results.get('comparison', {}).get('best_model', None)
        except:
            return None
    
    def predict_with_model(self, text: str, model_type: str) -> Tuple[str, float, Dict]:
        """Realiza predicción con el modelo especificado"""
        if model_type == 'baseline' and self.baseline_model:
            try:
                # Limpiar el texto de entrada
                clean_text = self.baseline_model.vectorizer.transform([text])
                predictions = self.baseline_model.model.predict(clean_text)
                probabilities = self.baseline_model.model.predict_proba(clean_text)
                
                pred = predictions[0]
                prob = probabilities[0]
                
                result = "SPAM" if pred == 1 else "HAM"
                confidence = float(prob[pred])
                
                return result, confidence, {
                    'spam_probability': float(prob[1]) if len(prob) > 1 else 0.0,
                    'ham_probability': float(prob[0]) if len(prob) > 0 else 0.0,
                    'model_used': 'TF-IDF + Logistic Regression'
                }
            except Exception as e:
                st.error(f"Error en predicción baseline: {e}")
                return "ERROR", 0.0, {}
        
        elif model_type == 'distilbert' and self.distilbert_model:
            try:
                predictions, probabilities = self.distilbert_model.predict([text])
                pred = predictions[0]
                prob = probabilities[0]
                
                result = "SPAM" if pred == 1 else "HAM"
                confidence = prob[pred] if len(prob) > pred else 0.5
                
                return result, confidence, {
                    'spam_probability': float(prob[1]) if len(prob) > 1 else 0,
                    'ham_probability': float(prob[0]) if len(prob) > 0 else 0,
                    'model_used': 'DistilBERT'
                }
            except Exception as e:
                st.error(f"Error en predicción DistilBERT: {e}")
                return "ERROR", 0.0, {}
        
        return "ERROR", 0.0, {}
    
    def create_probability_chart(self, details: Dict) -> go.Figure:
        """Crea gráfico de probabilidades"""
        if not details:
            return go.Figure()
        
        labels = ['HAM (No Spam)', 'SPAM']
        values = [details.get('ham_probability', 0), details.get('spam_probability', 0)]
        colors = ['#2E8B57', '#DC143C']  # Verde para HAM, Rojo para SPAM
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'{v:.2%}' for v in values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Probabilidades de Clasificación",
            xaxis_title="Clase",
            yaxis_title="Probabilidad",
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def show_model_comparison(self):
        """Muestra comparación de modelos si está disponible"""
        results_path = Path(self.config['paths']['results_dir']) / 'evaluation_results.json'
        
        if not results_path.exists():
            st.info("📊 Ejecuta la evaluación de modelos para ver la comparación")
            return
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            comparison = results.get('comparison', {})
            if not comparison:
                return
            
            st.subheader("📈 Comparación de Modelos")
            
            # Métricas en columnas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 Objetivo F1-Score",
                    f"{comparison['target_f1']:.3f}",
                    delta=None
                )
            
            with col2:
                baseline_f1 = comparison['baseline']['f1_score']
                baseline_status = "✅" if comparison['baseline_achieves_target'] else "❌"
                st.metric(
                    f"{baseline_status} Baseline F1",
                    f"{baseline_f1:.3f}",
                    delta=f"{baseline_f1 - comparison['target_f1']:.3f}"
                )
            
            with col3:
                distilbert_f1 = comparison['distilbert']['f1_score']
                distilbert_status = "✅" if comparison['distilbert_achieves_target'] else "❌"
                st.metric(
                    f"{distilbert_status} DistilBERT F1",
                    f"{distilbert_f1:.3f}",
                    delta=f"{distilbert_f1 - comparison['target_f1']:.3f}"
                )
            
            # Gráfico comparativo
            metrics_df = pd.DataFrame({
                'Modelo': ['Baseline', 'DistilBERT'],
                'F1-Score': [baseline_f1, distilbert_f1],
                'Accuracy': [comparison['baseline']['accuracy'], comparison['distilbert']['accuracy']],
                'Precision': [comparison['baseline']['precision'], comparison['distilbert']['precision']],
                'Recall': [comparison['baseline']['recall'], comparison['distilbert']['recall']]
            })
            
            fig = px.bar(
                metrics_df.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'),
                x='Métrica',
                y='Valor',
                color='Modelo',
                barmode='group',
                title="Comparación Detallada de Métricas",
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mejor modelo
            best_model = "DistilBERT" if comparison['best_model'] == 'distilbert' else "Baseline"
            improvement = comparison['improvement']
            st.success(f"🏆 **Mejor modelo**: {best_model} (mejora de +{improvement:.4f} en F1-Score)")
            
        except Exception as e:
            st.error(f"Error cargando resultados de comparación: {e}")
    
    def main_interface(self):
        """Interfaz principal de la aplicación"""
        st.title(self.config['app']['title'])
        st.markdown(self.config['app']['description'])
        
        # Sidebar para configuración
        with st.sidebar:
            st.header("⚙️ Configuración")
            
            # Cargar modelos
            if st.button("🔄 Cargar Modelos", type="primary"):
                with st.spinner("Cargando modelos..."):
                    baseline_loaded, distilbert_loaded = self.load_models()
                
                if baseline_loaded or distilbert_loaded:
                    self.best_model_type = self.get_best_model_type()
                    if self.best_model_type:
                        st.info(f"🏆 Mejor modelo detectado: {self.best_model_type}")
            
            # Selección de modelo
            model_options = []
            if self.baseline_model:
                model_options.append("baseline")
            if self.distilbert_model:
                model_options.append("distilbert")
            
            if model_options:
                if len(model_options) > 1:
                    # Si hay ambos modelos, usar el mejor por defecto
                    default_idx = 0
                    if self.best_model_type and self.best_model_type in model_options:
                        default_idx = model_options.index(self.best_model_type)
                    
                    selected_model = st.selectbox(
                        "📊 Seleccionar Modelo",
                        options=model_options,
                        format_func=lambda x: "🔄 Baseline (TF-IDF + LR)" if x == "baseline" else "🤖 DistilBERT",
                        index=default_idx
                    )
                else:
                    selected_model = model_options[0]
                    model_name = "🔄 Baseline (TF-IDF + LR)" if selected_model == "baseline" else "🤖 DistilBERT"
                    st.info(f"Modelo disponible: {model_name}")
            else:
                st.error("❌ No hay modelos cargados")
                selected_model = None
            
            st.markdown("---")
            
            # Mostrar información del proyecto
            with st.expander("ℹ️ Información del Proyecto"):
                st.markdown("""
                **SMS Spam Detector**
                
                Este proyecto implementa dos enfoques para la detección de spam:
                
                1. **Modelo Baseline**: TF-IDF + Regresión Logística
                2. **Modelo Avanzado**: DistilBERT fine-tuned
                
                **Objetivo**: Alcanzar F1-Score ≥ 0.95
                """)
        
        # Interfaz principal
        if not selected_model:
            st.warning("⚠️ Primero debes cargar los modelos usando el botón en la barra lateral")
            return
        
        # Input del usuario
        st.subheader("📝 Analizar Mensaje SMS")
        
        # Ejemplos predefinidos
        examples = {
            "Ejemplo HAM": "Hi! How are you doing today? Want to grab coffee later?",
            "Ejemplo SPAM": "CONGRATULATIONS! You've won a $1000 gift card! Click here to claim now: http://spam-link.com",
            "Ejemplo Neutro": "Thanks for your message. I'll get back to you soon."
        }
        
        example_choice = st.selectbox("🎯 Usar ejemplo predefinido:", ["Escribir mensaje personalizado"] + list(examples.keys()))
        
        if example_choice != "Escribir mensaje personalizado":
            default_text = examples[example_choice]
        else:
            default_text = ""
        
        user_input = st.text_area(
            "Introduce el mensaje SMS a analizar:",
            value=default_text,
            height=100,
            max_chars=self.config['app']['max_input_length'],
            placeholder="Ejemplo: Free msg: Txt STOP to 85543 to stop receiving messages..."
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            analyze_button = st.button("🔍 Analizar Mensaje", type="primary", use_container_width=True)
        
        if analyze_button and user_input.strip():
            with st.spinner("Analizando mensaje..."):
                # Realizar predicción
                result, confidence, details = self.predict_with_model(user_input.strip(), selected_model)
                
                if result != "ERROR":
                    # Mostrar resultado principal
                    st.subheader("📋 Resultado del Análisis")
                    
                    # Resultado con color
                    if result == "SPAM":
                        st.error(f"🚨 **SPAM DETECTADO** (Confianza: {confidence:.1%})")
                    else:
                        st.success(f"✅ **MENSAJE LEGÍTIMO** (Confianza: {confidence:.1%})")
                    
                    # Detalles en columnas
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "🎯 Probabilidad SPAM",
                            f"{details.get('spam_probability', 0):.1%}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "✅ Probabilidad HAM",
                            f"{details.get('ham_probability', 0):.1%}",
                            delta=None
                        )
                    
                    # Gráfico de probabilidades
                    if details:
                        fig = self.create_probability_chart(details)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Información del modelo usado
                    st.info(f"📊 Modelo utilizado: **{details.get('model_used', 'Desconocido')}**")
                    
                else:
                    st.error("❌ Error al procesar el mensaje")
        
        elif analyze_button:
            st.warning("⚠️ Por favor, introduce un mensaje para analizar")
        
        # Mostrar comparación de modelos
        st.markdown("---")
        self.show_model_comparison()
    
    def run(self):
        """Ejecuta la aplicación"""
        self.main_interface()


def main():
    """Función principal"""
    app = SpamDetectorApp()
    app.run()


if __name__ == "__main__":
    main()
