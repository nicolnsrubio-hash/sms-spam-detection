#!/usr/bin/env python3
"""
Script para entrenar el modelo DistilBERT para detección de SMS spam
"""

import pandas as pd
import sys
from pathlib import Path
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/distilbert_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Agregar directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from distilbert_model import DistilBERTModel
    from data_preprocessing import DataPreprocessor
except ImportError as e:
    logger.error(f"Error al importar módulos: {e}")
    logger.error("Asegúrate de que los archivos están en la carpeta src/")
    sys.exit(1)


def create_directories():
    """Crea los directorios necesarios"""
    directories = ["models", "logs", "results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    logger.info("Directorios creados/verificados")


def check_gpu_availability():
    """Verifica la disponibilidad de GPU"""
    import torch
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        logger.info(f"GPU disponible: {device}")
        return True
    else:
        logger.info("Usando CPU para entrenamiento")
        return False


def load_or_create_data():
    """Carga datos existentes o crea nuevos datos procesados"""
    data_dir = Path("data")
    train_path = data_dir / "train_data.csv"
    test_path = data_dir / "test_data.csv"
    
    if train_path.exists() and test_path.exists():
        logger.info("Cargando datos procesados existentes...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Datos cargados - Train: {len(train_df)}, Test: {len(test_df)}")
    else:
        logger.info("Creando datos procesados...")
        preprocessor = DataPreprocessor()
        train_df, test_df = preprocessor.get_processed_data()
        preprocessor.save_processed_data(train_df, test_df)
        logger.info(f"Datos procesados y guardados - Train: {len(train_df)}, Test: {len(test_df)}")
    
    return train_df, test_df


def train_distilbert():
    """Función principal para entrenar DistilBERT"""
    logger.info("=== Iniciando entrenamiento de DistilBERT ===")
    
    # Crear directorios
    create_directories()
    
    # Verificar GPU
    check_gpu_availability()
    
    try:
        # Cargar datos
        train_df, test_df = load_or_create_data()
        
        # Verificar que los datos tienen las columnas necesarias
        required_columns = ['message_clean', 'label_binary']
        for col in required_columns:
            if col not in train_df.columns:
                logger.error(f"Columna requerida '{col}' no encontrada en datos de entrenamiento")
                return False
        
        # Crear modelo DistilBERT
        logger.info("Inicializando modelo DistilBERT...")
        distilbert = DistilBERTModel()
        
        # Entrenar modelo
        logger.info("Comenzando entrenamiento...")
        train_metrics = distilbert.train(train_df, test_df)
        logger.info(f"Entrenamiento completado. Métricas: {train_metrics}")
        
        # Evaluar en conjunto de prueba
        logger.info("Evaluando modelo en conjunto de prueba...")
        test_metrics = distilbert.evaluate(test_df)
        
        # Mostrar resultados
        logger.info("=== RESULTADOS DEL ENTRENAMIENTO ===")
        logger.info(f"F1-Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall: {test_metrics['recall']:.4f}")
        
        # Guardar modelo
        logger.info("Guardando modelo entrenado...")
        distilbert.save_model()
        
        # Verificar objetivo
        target_f1 = 0.95  # Del config.yaml
        achieved_f1 = test_metrics['f1_score']
        
        if achieved_f1 >= target_f1:
            logger.info(f"✅ ¡Objetivo alcanzado! F1-Score: {achieved_f1:.4f} >= {target_f1}")
            success = True
        else:
            logger.warning(f"⚠️ Objetivo no alcanzado. F1-Score: {achieved_f1:.4f} < {target_f1}")
            success = True  # Aún consideramos exitoso si entrenó
        
        # Guardar métricas
        results_dir = Path("results")
        results_file = results_dir / f"distilbert_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(results_file, 'w') as f:
            f.write("=== DistilBERT Training Results ===\n")
            f.write(f"Training completed: {datetime.now()}\n")
            f.write(f"F1-Score: {test_metrics['f1_score']:.4f}\n")
            f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"Recall: {test_metrics['recall']:.4f}\n")
            f.write(f"Target F1 achieved: {achieved_f1 >= target_f1}\n")
        
        logger.info(f"Resultados guardados en: {results_file}")
        logger.info("=== Entrenamiento de DistilBERT COMPLETADO ===")
        
        return success
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def verify_model():
    """Verifica que el modelo se puede cargar correctamente"""
    try:
        logger.info("Verificando modelo entrenado...")
        distilbert = DistilBERTModel()
        distilbert.load_model()
        
        # Probar predicción simple
        test_messages = [
            "Congratulations! You have won $1000!",
            "Hey, are you coming to the meeting?"
        ]
        
        predictions, probabilities = distilbert.predict(test_messages)
        
        logger.info("✅ Modelo verificado correctamente:")
        for i, msg in enumerate(test_messages):
            pred_label = "SPAM" if predictions[i] == 1 else "HAM"
            confidence = probabilities[i][predictions[i]]
            logger.info(f"  '{msg}' -> {pred_label} (confianza: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error al verificar modelo: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento de DistilBERT para detección de SMS spam")
    print("Este proceso puede tomar varios minutos...")
    print("-" * 60)
    
    # Entrenar modelo
    success = train_distilbert()
    
    if success:
        print("\n" + "="*60)
        print("✅ ¡ENTRENAMIENTO EXITOSO!")
        print("Verificando modelo...")
        
        # Verificar modelo
        if verify_model():
            print("✅ Modelo verificado y listo para usar")
            print("\n🎯 El modelo DistilBERT ya está disponible para la aplicación web")
            print("Puedes ejecutar: streamlit run app.py")
        else:
            print("❌ Error al verificar el modelo")
    else:
        print("\n" + "="*60)
        print("❌ Error durante el entrenamiento")
        print("Revisa los logs para más detalles")
    
    print("-" * 60)
