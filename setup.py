#!/usr/bin/env python3
"""
Setup script para SMS Spam Detection Project
"""

import os
import sys
from pathlib import Path
import subprocess

def create_directories():
    """Crear directorios necesarios para el proyecto"""
    directories = [
        "data",
        "models", 
        "logs",
        "results",
        "mlruns"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Directorio '{directory}' creado/verificado")

def check_python_version():
    """Verificar versión de Python"""
    if sys.version_info < (3, 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        print(f"Versión actual: {sys.version}")
        sys.exit(1)
    else:
        print(f"✓ Python {sys.version.split()[0]} - OK")

def install_requirements():
    """Instalar dependencias"""
    try:
        print("📦 Instalando dependencias...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✓ Dependencias instaladas correctamente")
    except subprocess.CalledProcessError as e:
        print("❌ Error instalando dependencias:")
        print(e.stderr)
        sys.exit(1)

def verify_installation():
    """Verificar que las librerías principales estén instaladas"""
    try:
        import pandas
        import numpy
        import sklearn
        import torch
        import transformers
        import streamlit
        print("✓ Todas las dependencias principales verificadas")
    except ImportError as e:
        print(f"❌ Error importando librería: {e}")
        sys.exit(1)

def main():
    """Función principal de setup"""
    print("🚀 Configurando SMS Spam Detection Project...")
    print("=" * 50)
    
    # Verificar versión de Python
    check_python_version()
    
    # Crear directorios
    create_directories()
    
    # Instalar dependencias si se pide
    if "--install-deps" in sys.argv:
        install_requirements()
        verify_installation()
    
    print("\n✅ Setup completado!")
    print("\nPróximos pasos:")
    print("1. Activar entorno virtual: source venv/bin/activate (Linux/Mac) o venv\\Scripts\\activate (Windows)")
    print("2. Instalar dependencias: pip install -r requirements.txt")
    print("3. Entrenar modelos: python src/baseline_model.py")
    print("4. Ejecutar app: streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    main()
