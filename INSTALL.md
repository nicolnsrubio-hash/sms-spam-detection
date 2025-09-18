# 🛠️ Guía de Instalación Detallada

Esta guía te ayudará a instalar y ejecutar el proyecto SMS Spam Detection desde cero en tu computadora.

## 📋 Requisitos del Sistema

### Requisitos Mínimos
- **Sistema Operativo**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 o superior
- **RAM**: 4GB mínimo (8GB recomendado para DistilBERT)
- **Espacio en disco**: 2GB libres
- **Conexión a internet**: Para descargar dependencias y dataset

### Software Necesario
- **Git**: Para clonar el repositorio
- **Python**: Con pip incluido
- **Navegador web**: Para usar la interfaz Streamlit

## 🚀 Instalación Paso a Paso

### Paso 1: Verificar Python

```bash
# Verificar versión de Python (debe ser 3.8 o superior)
python --version

# En algunos sistemas Linux/Mac puede ser:
python3 --version
```

Si no tienes Python instalado:
- **Windows**: Descargar desde [python.org](https://python.org/downloads/)
- **macOS**: `brew install python3` (con Homebrew)
- **Linux**: `sudo apt install python3 python3-pip`

### Paso 2: Verificar Git

```bash
git --version
```

Si no tienes Git instalado:
- **Windows**: Descargar desde [git-scm.com](https://git-scm.com/downloads)
- **macOS**: `brew install git`
- **Linux**: `sudo apt install git`

### Paso 3: Clonar el Repositorio

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/sms-spam-detection.git

# Navegar al directorio
cd sms-spam-detection
```

### Paso 4: Crear Entorno Virtual

**En Windows:**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate

# Verificar que está activado (debe aparecer (venv) en el prompt)
```

**En Linux/Mac:**
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Verificar que está activado (debe aparecer (venv) en el prompt)
```

### Paso 5: Actualizar pip

```bash
# Actualizar pip a la última versión
python -m pip install --upgrade pip
```

### Paso 6: Instalar Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

**Nota**: Esta instalación puede tomar varios minutos, especialmente PyTorch y Transformers.

### Paso 7: Verificar Instalación

```bash
# Verificar que las librerías principales se instalaron correctamente
python -c "import pandas, numpy, sklearn, torch, transformers, streamlit; print('Todas las dependencias instaladas correctamente!')"
```

## 🏃‍♂️ Primer Uso

### 1. Preparar Datos y Entrenar Modelos

```bash
# Preprocesar datos (descarga el dataset automáticamente)
python src/data_preprocessing.py

# Entrenar modelo baseline (rápido: ~1-2 minutos)
python src/baseline_model.py

# Entrenar modelo DistilBERT (más lento: ~10-15 minutos)
python src/distilbert_model.py

# Evaluar y comparar ambos modelos
python src/model_evaluation.py
```

### 2. Ejecutar la Aplicación Web

```bash
# Iniciar Streamlit
streamlit run streamlit_app/app.py
```

La aplicación se abrirá automáticamente en tu navegador en: `http://localhost:8501`

## 🐛 Solución de Problemas Comunes

### Error: "Python no es reconocido como comando"

**Windows**: Asegúrate de que Python esté en el PATH del sistema.
```bash
# Usar el launcher de Python
py --version
py -m pip install -r requirements.txt
```

### Error: "pip no es reconocido"

```bash
# Usar el módulo pip directamente
python -m pip install -r requirements.txt
```

### Error de memoria con DistilBERT

Editar `config.yaml` y reducir el batch_size:
```yaml
distilbert:
  batch_size: 8  # En lugar de 16
```

### Error de descarga del dataset

Si el dataset no se descarga automáticamente:

1. Descargar manualmente desde: https://raw.githubusercontent.com/justmarkham/pydata-berlin-2016/master/sms.tsv
2. Guardar como `data/sms.tsv`
3. Crear la carpeta `data/` si no existe

### Error: "Modelos no encontrados"

Asegúrate de entrenar los modelos primero:
```bash
python src/baseline_model.py
python src/distilbert_model.py
```

### Problemas con PyTorch en Windows

Si PyTorch no se instala correctamente:
```bash
# Instalar versión específica para Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Error de permisos en Linux/Mac

```bash
# Usar pip con --user si hay problemas de permisos
pip install --user -r requirements.txt
```

## 🔧 Instalación Avanzada

### Usando Conda

Si prefieres usar Conda:
```bash
# Crear entorno con Conda
conda create -n sms-spam python=3.9
conda activate sms-spam

# Instalar PyTorch desde conda-forge
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Instalar el resto con pip
pip install -r requirements.txt
```

### Desarrollo

Para contribuir al proyecto:
```bash
# Instalar dependencias adicionales de desarrollo
pip install -r requirements.txt
pip install pre-commit

# Configurar pre-commit hooks
pre-commit install
```

## 📊 Verificación de Instalación

Para verificar que todo funciona correctamente:

```bash
# Ejecutar tests
pytest tests/ -v

# Verificar formateo de código
black --check src/ tests/

# Verificar linting
flake8 src/ tests/
```

## 💡 Consejos

1. **Usa un entorno virtual**: Siempre activa el entorno virtual antes de trabajar
2. **Actualiza regularmente**: `pip install --upgrade -r requirements.txt`
3. **Mantén los modelos**: Una vez entrenados, los modelos se guardan automáticamente
4. **Revisa los logs**: Los archivos de log te ayudarán a depurar problemas

## 📞 Soporte

Si encuentras problemas durante la instalación:

1. Revisa esta guía nuevamente
2. Verifica los requisitos del sistema
3. Busca el error específico en [Issues](https://github.com/tu-usuario/sms-spam-detection/issues)
4. Crea un nuevo issue si el problema persiste

## 🎉 ¡Listo!

Una vez completada la instalación, deberías poder:
- ✅ Ejecutar la aplicación web
- ✅ Entrenar modelos
- ✅ Ver resultados de evaluación
- ✅ Hacer predicciones de spam

¡Disfruta usando el SMS Spam Detector!
