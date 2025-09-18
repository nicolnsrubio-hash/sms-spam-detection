# 📱 SMS Spam Detector

Un sistema completo de detección de SPAM en mensajes SMS que compara un modelo baseline clásico (TF‑IDF + Regresión Logística) con un modelo Transformer ligero (DistilBERT) afinado para clasificación binaria.

## 🎯 Objetivos del Proyecto

- **Objetivo General**: Diseñar, entrenar/evaluar y desplegar un detector de SPAM en SMS con interfaz web y empaquetado en contenedor.

- **Objetivos Específicos**:
  - Implementar un baseline (TF‑IDF + Regresión Logística) y persistirlo como .pkl
  - Afinar DistilBERT para la tarea y comparar vs baseline
  - Desarrollar una app en Streamlit para inferencia en tiempo real
  - Dockerizar la solución y documentar ejecución
  - **Lograr F1 ≥ 0.95 en el conjunto de prueba**

## 🏗️ Arquitectura del Proyecto

```
proyecto_de_curso/
├── src/                          # Código fuente
│   ├── data_preprocessing.py     # Preprocesamiento de datos
│   ├── baseline_model.py         # Modelo TF-IDF + Regresión Logística
│   ├── distilbert_model.py       # Modelo DistilBERT
│   └── model_evaluation.py       # Evaluación y comparación
├── app.py                        # Aplicación Streamlit
├── config.yaml                   # Configuración del proyecto
├── requirements.txt              # Dependencias Python
├── Dockerfile                    # Configuración Docker
├── docker-compose.yml           # Orquestación de contenedores
├── data/                         # Datos del proyecto
├── models/                       # Modelos entrenados
├── results/                      # Resultados y gráficos
└── logs/                         # Logs de entrenamiento
```

## 🚀 Inicio Rápido

### Opción 1: Ejecución con Docker (Recomendado)

1. **Clonar el repositorio y construir la imagen:**
   ```bash
   git clone <repository-url>
   cd proyecto_de_curso
   docker-compose build
   ```

2. **Entrenar los modelos (opcional):**
   ```bash
   docker-compose --profile training up model-trainer
   ```

3. **Ejecutar la aplicación web:**
   ```bash
   docker-compose up sms-spam-detector
   ```

4. **Acceder a la aplicación:**
   - Abrir navegador en: http://localhost:8501

### Opción 2: Ejecución Local

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Entrenar modelos:**
   ```bash
   # Preprocesar datos
   python src/data_preprocessing.py
   
   # Entrenar modelo baseline
   python src/baseline_model.py
   
   # Entrenar modelo DistilBERT
   python src/distilbert_model.py
   
   # Evaluar y comparar modelos
   python src/model_evaluation.py
   ```

3. **Ejecutar aplicación web:**
   ```bash
   streamlit run app.py
   ```

## 📊 Modelos Implementados

### 1. Modelo Baseline: TF-IDF + Regresión Logística

- **Vectorización**: TF-IDF con n-gramas (1,2)
- **Clasificador**: Regresión Logística
- **Características**: 5000 features máximo
- **Ventajas**: Rápido, interpretable, bajo consumo de recursos

### 2. Modelo Avanzado: DistilBERT

- **Modelo base**: `distilbert-base-uncased` de Hugging Face
- **Arquitectura**: Transformer con 6 capas
- **Fine-tuning**: Supervisado sobre dataset SMS Spam
- **Ventajas**: Mayor capacidad de comprensión contextual

## 🔧 Configuración

El archivo `config.yaml` contiene todas las configuraciones del proyecto:

```yaml
# Configuración de datos
data:
  dataset_url: "https://raw.githubusercontent.com/justmarkham/pydata-berlin-2016/master/sms.tsv"
  test_size: 0.2
  random_state: 42

# Configuración modelo baseline
baseline:
  max_features: 5000
  ngram_range: [1, 2]
  C: 1.0

# Configuración DistilBERT
distilbert:
  model_name: "distilbert-base-uncased"
  max_length: 128
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3

# Objetivo de rendimiento
evaluation:
  target_f1_score: 0.95
```

## 📈 Evaluación y Métricas

El sistema evalúa automáticamente ambos modelos usando:

- **F1-Score**: Métrica principal (objetivo ≥ 0.95)
- **Accuracy**: Precisión general
- **Precision**: Precisión por clase
- **Recall**: Cobertura por clase
- **Matriz de Confusión**: Análisis detallado de errores

Los resultados se guardan en `results/evaluation_results.json` y se generan gráficos comparativos.

## 🌐 Aplicación Web

La interfaz Streamlit proporciona:

- **Análisis en tiempo real** de mensajes SMS
- **Selección de modelo** (Baseline o DistilBERT)
- **Visualización de probabilidades** con gráficos interactivos
- **Ejemplos predefinidos** para pruebas rápidas
- **Comparación de modelos** con métricas detalladas

### Características de la App:

- 📱 Interfaz intuitiva y responsiva
- 🎯 Predicciones con nivel de confianza
- 📊 Gráficos interactivos con Plotly
- ⚙️ Configuración flexible de modelos
- 🏆 Identificación automática del mejor modelo

## 🐳 Docker y Despliegue

### Servicios Docker:

1. **sms-spam-detector**: Aplicación web principal
2. **model-trainer**: Entrenamiento automático de modelos

### Comandos útiles:

```bash
# Construir imágenes
docker-compose build

# Ejecutar solo la app
docker-compose up sms-spam-detector

# Entrenar modelos
docker-compose --profile training up model-trainer

# Ejecutar en background
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar servicios
docker-compose down
```

## 📁 Dataset

- **Fuente**: SMS Spam Collection Dataset
- **Tamaño**: ~5574 mensajes SMS
- **Clases**: 
  - `ham`: Mensajes legítimos (~4827)
  - `spam`: Mensajes spam (~747)
- **División**: 80% entrenamiento, 20% prueba
- **Preprocesamiento**: Limpieza de texto, tokenización, normalización

## 🛠️ Desarrollo y Testing

### Estructura del Código:

- **Modular**: Cada modelo en su propio archivo
- **Configurable**: Parámetros centralizados en YAML
- **Reproducible**: Seeds fijos para consistencia
- **Documentado**: Docstrings y comentarios extensivos

### Flujo de Trabajo:

1. **Preprocesamiento**: Limpieza y preparación de datos
2. **Entrenamiento**: Modelos baseline y DistilBERT
3. **Evaluación**: Comparación automática de rendimiento
4. **Despliegue**: Interfaz web con el mejor modelo

## 📋 Requisitos del Sistema

### Mínimos:
- Python 3.9+
- 4GB RAM
- 2GB espacio disco

### Recomendados:
- Python 3.9+
- 8GB RAM
- GPU (para DistilBERT)
- 5GB espacio disco

## 🚨 Solución de Problemas

### Problemas Comunes:

1. **Error de memoria con DistilBERT**:
   - Reducir `batch_size` en config.yaml
   - Usar CPU en lugar de GPU

2. **Modelos no se cargan en Streamlit**:
   - Verificar que los modelos estén entrenados
   - Revisar rutas en config.yaml

3. **Docker build falla**:
   - Verificar conexión a internet
   - Limpiar cache: `docker system prune`

### Logs y Debugging:

- Logs de entrenamiento: `logs/`
- Logs de Docker: `docker-compose logs`
- Logs de Streamlit: Consola del navegador

## 📊 Resultados Esperados

Basado en experimentos preliminares, se espera:

- **Baseline**: F1 ~0.92-0.96
- **DistilBERT**: F1 ~0.95-0.98
- **Tiempo de entrenamiento**:
  - Baseline: ~30 segundos
  - DistilBERT: ~10-20 minutos (CPU)

## 🤝 Contribuciones

1. Fork el repositorio
2. Crear branch: `git checkout -b feature/nueva-caracteristica`
3. Commit: `git commit -am 'Agregar nueva característica'`
4. Push: `git push origin feature/nueva-caracteristica`
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para detalles.

## 👥 Autores

- **Desarrollador**: [Tu Nombre]
- **Universidad**: UAO
- **Curso**: Desarrollo de Proyectos de Inteligencia Artificial
- **Año**: 2024

## 🔗 Enlaces Útiles

- [DistilBERT en Hugging Face](https://huggingface.co/distilbert-base-uncased)
- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

---

## 📞 Soporte

Para preguntas o problemas:

1. Revisar la sección de troubleshooting
2. Buscar en issues existentes
3. Crear un nuevo issue con detalles
4. Contactar al desarrollador

**¡Feliz detección de spam! 🎉📱**
