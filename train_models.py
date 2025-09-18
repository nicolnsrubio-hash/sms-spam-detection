#!/usr/bin/env python3
"""
Script principal para entrenar todos los modelos del proyecto SMS Spam Detection
Ejecuta secuencialmente: preprocesamiento, baseline, DistilBERT y evaluación
"""

import sys
import time
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Función principal que ejecuta todo el pipeline de entrenamiento"""
    start_time = time.time()
    
    print("🚀 Iniciando pipeline completo de entrenamiento de modelos SMS Spam Detection")
    print("=" * 80)
    
    try:
        # 1. Preprocesamiento de datos
        print("\n📊 PASO 1: Preprocesamiento de datos")
        print("-" * 50)
        from src.data_preprocessing import main as preprocess_main
        preprocess_main()
        
        # 2. Entrenamiento del modelo baseline
        print("\n🔄 PASO 2: Entrenamiento del modelo Baseline (TF-IDF + Regresión Logística)")
        print("-" * 50)
        from src.baseline_model import main as baseline_main
        baseline_main()
        
        # 3. Entrenamiento del modelo DistilBERT
        print("\n🤖 PASO 3: Entrenamiento del modelo DistilBERT")
        print("-" * 50)
        from src.distilbert_model import main as distilbert_main
        distilbert_main()
        
        # 4. Evaluación y comparación
        print("\n📈 PASO 4: Evaluación y comparación de modelos")
        print("-" * 50)
        from src.model_evaluation import main as evaluation_main
        evaluation_main()
        
        # Resumen final
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE!")
        print(f"⏱️  Tiempo total de ejecución: {total_time:.2f} segundos ({total_time/60:.1f} minutos)")
        print("\n📁 Archivos generados:")
        print("   • Datos procesados: data/train_data.csv, data/test_data.csv")
        print("   • Modelo Baseline: models/baseline_model.pkl, models/tfidf_vectorizer.pkl")
        print("   • Modelo DistilBERT: models/distilbert_spam_classifier/, models/distilbert_tokenizer/")
        print("   • Resultados: results/evaluation_results.json")
        print("   • Gráficos: results/model_comparison.png, results/confusion_matrices.png")
        print("\n🌐 Para usar la aplicación web:")
        print("   streamlit run app.py")
        print("\n🐳 O con Docker:")
        print("   docker-compose up sms-spam-detector")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR EN EL PIPELINE: {e}")
        print("Revisa los logs para más detalles.")
        sys.exit(1)


if __name__ == "__main__":
    main()
