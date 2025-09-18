# 🚀 Guía para Subir el Proyecto a GitHub

Esta guía te ayudará a crear un repositorio en GitHub y subir el proyecto completo.

## 📋 Pasos para Crear el Repositorio

### 1. Crear Repositorio en GitHub (Interfaz Web)

1. **Ir a GitHub**: Abre tu navegador y ve a [github.com](https://github.com)

2. **Iniciar sesión**: Ingresa con tu cuenta de GitHub

3. **Crear nuevo repositorio**:
   - Haz clic en el botón verde "New" o "+"
   - Selecciona "New repository"

4. **Configurar el repositorio**:
   ```
   Repository name: sms-spam-detection
   Description: 🚀 Sistema completo de detección de spam en SMS con ML y Deep Learning. Incluye modelo baseline (TF-IDF) y DistilBERT, interfaz Streamlit, gRPC API, MLflow tracking y pruebas completas. F1-Score ≥0.95
   
   ✅ Public (recomendado para que cualquiera pueda instalarlo)
   ❌ NO marcar "Add a README file" (ya tenemos uno)
   ❌ NO agregar .gitignore (ya tenemos uno)
   ❌ NO agregar license (ya tenemos uno)
   ```

5. **Crear repositorio**: Haz clic en "Create repository"

### 2. Conectar tu Proyecto Local con GitHub

GitHub te mostrará comandos similares a estos. Usa la segunda opción "push an existing repository from the command line":

```bash
git remote add origin https://github.com/TU-USUARIO/sms-spam-detection.git
git branch -M main
git push -u origin main
```

**Reemplaza `TU-USUARIO` con tu nombre de usuario de GitHub**

### 3. Ejecutar los Comandos

En tu terminal (desde la carpeta del proyecto):

```bash
# Agregar el repositorio remoto (cambiar TU-USUARIO por tu username)
git remote add origin https://github.com/TU-USUARIO/sms-spam-detection.git

# Cambiar a la rama main
git branch -M main

# Subir todo el código a GitHub
git push -u origin main
```

Si es la primera vez que usas Git con GitHub, puede pedirte credenciales:
- **Usuario**: Tu nombre de usuario de GitHub
- **Contraseña**: Tu Personal Access Token (no tu contraseña normal)

### 4. Crear Personal Access Token (si es necesario)

Si GitHub te pide contraseña:

1. Ve a GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Clic en "Generate new token (classic)"
3. Selecciona scopes: `repo` (repositorios)
4. Genera el token y úsalo como contraseña

## ✅ Verificar que Todo Esté Subido

Después de hacer `git push`, deberías ver:

1. **Todos los archivos** en tu repositorio GitHub
2. **README.md** con la documentación completa
3. **Archivos principales**:
   - `src/` con todos los modelos
   - `tests/` con las pruebas
   - `requirements.txt`
   - `config.yaml`
   - `LICENSE`
   - `INSTALL.md`

## 🎯 Configurar el Repositorio para Instalación Fácil

### 1. Agregar Topics (Etiquetas)

En GitHub, ve a tu repositorio y agrega topics:
```
machine-learning, deep-learning, nlp, spam-detection, sms, python, 
streamlit, distilbert, transformers, mlflow, grpc, tensorflow, 
scikit-learn, artificial-intelligence
```

### 2. Crear Release (Opcional)

1. Ve a "Releases" en tu repositorio
2. Clic en "Create a new release"
3. Tag: `v1.0.0`
4. Title: `SMS Spam Detection v1.0.0`
5. Descripción:
```markdown
🚀 **Primera versión completa del SMS Spam Detection**

## ✨ Características
- Modelo Baseline (TF-IDF + Regresión Logística)
- Modelo Avanzado (DistilBERT Fine-tuned)
- Interfaz web con Streamlit
- API gRPC
- Seguimiento con MLflow
- Suite completa de pruebas
- F1-Score ≥0.95 alcanzado

## 📦 Instalación
```bash
git clone https://github.com/TU-USUARIO/sms-spam-detection.git
cd sms-spam-detection
pip install -r requirements.txt
python src/baseline_model.py
streamlit run app.py
```

## 📊 Resultados
- Baseline F1-Score: ~0.91
- DistilBERT F1-Score: ~0.95+
- Tiempo de inferencia: <100ms
```

### 3. Actualizar README con URL Real

Una vez que tengas el repositorio, actualiza las URLs en README.md:

```bash
# Editar README.md y cambiar las URLs de ejemplo por las reales:
git clone https://github.com/TU-USUARIO/sms-spam-detection.git
```

Luego commit y push:
```bash
git add README.md
git commit -m "docs: update GitHub URLs in documentation"
git push
```

## 🌟 Hacer el Repositorio Atractivo

### 1. Descripción Clara
En la página principal del repo, asegúrate de tener:
- Descripción clara y concisa
- Website URL (si tienes demo online)
- Topics relevantes

### 2. README Perfecto
Tu README.md ya incluye:
- ✅ Badges de tecnologías
- ✅ Descripción clara
- ✅ Instrucciones de instalación
- ✅ Ejemplos de uso
- ✅ Capturas (puedes agregar después)
- ✅ Documentación completa

### 3. Issues Template (Opcional)

Crear `.github/ISSUE_TEMPLATE.md`:
```markdown
## Descripción del Problema
Descripción clara del problema encontrado.

## Pasos para Reproducir
1. Ejecutar '...'
2. Ver error en '...'
3. Error aparece

## Comportamiento Esperado
Lo que esperabas que pasara.

## Entorno
- OS: [Windows/Mac/Linux]
- Python: [3.8/3.9/3.10]
- Versión del proyecto: [v1.0.0]
```

## 📚 Documentación Adicional

El proyecto ya incluye:
- ✅ `README.md` - Documentación principal
- ✅ `INSTALL.md` - Guía de instalación detallada
- ✅ `LICENSE` - Licencia MIT
- ✅ `requirements.txt` - Dependencias
- ✅ `.gitignore` - Archivos a ignorar
- ✅ Código bien documentado con docstrings

## 🎉 ¡Listo!

Una vez completados estos pasos, cualquier persona podrá:

1. **Clonar tu repositorio**:
   ```bash
   git clone https://github.com/TU-USUARIO/sms-spam-detection.git
   ```

2. **Instalar dependencias**:
   ```bash
   cd sms-spam-detection
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

3. **Ejecutar el proyecto**:
   ```bash
   python src/baseline_model.py
   streamlit run app.py
   ```

## 🔄 Comandos Git Útiles para el Futuro

```bash
# Ver estado del repositorio
git status

# Agregar cambios
git add .

# Hacer commit descriptivo
git commit -m "feat: nueva característica"

# Subir cambios
git push

# Ver historial
git log --oneline

# Crear nueva rama
git checkout -b feature/nueva-funcionalidad
```

¡Tu proyecto está listo para ser compartido con el mundo! 🌍
