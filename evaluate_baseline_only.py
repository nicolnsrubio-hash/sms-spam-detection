import pandas as pd
import json
import yaml
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from baseline_model import BaselineModel
from data_preprocessing import DataPreprocessor


def load_config():
    """Carga la configuración"""
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate_baseline_only():
    """Evalúa solo el modelo baseline y genera el archivo de resultados"""
    print("🔄 Evaluando modelo baseline...")
    
    config = load_config()
    
    # Cargar datos de prueba
    data_dir = Path(config["paths"]["data_dir"])
    test_path = data_dir / "test_data.csv"
    
    if not test_path.exists():
        print("❌ Datos de prueba no encontrados")
        return
    
    test_df = pd.read_csv(test_path)
    print(f"📊 Datos de prueba: {len(test_df)} muestras")
    
    # Evaluar modelo baseline
    try:
        baseline = BaselineModel()
        baseline.load_model()
        metrics = baseline.evaluate(test_df)
        
        # Calcular métricas agregadas para HAM y SPAM
        report = metrics["classification_report"]
        
        # Extraer métricas principales
        baseline_metrics = {
            "model_type": "baseline",
            "model_name": "TF-IDF + Logistic Regression",
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
            "precision": (report["Ham"]["precision"] + report["Spam"]["precision"]) / 2,
            "recall": (report["Ham"]["recall"] + report["Spam"]["recall"]) / 2,
            "precision_ham": report["Ham"]["precision"],
            "recall_ham": report["Ham"]["recall"],
            "precision_spam": report["Spam"]["precision"],
            "recall_spam": report["Spam"]["recall"],
            "confusion_matrix": metrics["confusion_matrix"]
        }
        
        # Crear resultados simulando que solo tenemos baseline
        target_f1 = config["evaluation"]["target_f1_score"]
        
        comparison = {
            "baseline": {
                "f1_score": baseline_metrics["f1_score"],
                "accuracy": baseline_metrics["accuracy"], 
                "precision": baseline_metrics["precision"],
                "recall": baseline_metrics["recall"]
            },
            "distilbert": {
                "f1_score": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            },
            "best_model": "baseline",
            "improvement": 0.0,
            "target_f1": target_f1,
            "baseline_achieves_target": baseline_metrics["f1_score"] >= target_f1,
            "distilbert_achieves_target": False
        }
        
        # Preparar resultados completos
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "test_samples": len(test_df),
            "baseline_metrics": baseline_metrics,
            "distilbert_metrics": {},
            "comparison": comparison,
            "summary": {
                "models_evaluated": ["baseline"],
                "best_model": "baseline", 
                "best_f1_score": baseline_metrics["f1_score"],
                "target_achieved": baseline_metrics["f1_score"] >= target_f1
            }
        }
        
        # Guardar resultados
        results_dir = Path(config["paths"]["results_dir"])
        results_dir.mkdir(exist_ok=True)
        
        results_path = results_dir / "evaluation_results.json"
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados guardados en: {results_path}")
        
        # Mostrar resumen
        print("\n" + "="*50)
        print("📊 RESUMEN DE EVALUACIÓN")
        print("="*50)
        print(f"Modelo: TF-IDF + Regresión Logística")
        print(f"F1-Score: {baseline_metrics['f1_score']:.4f}")
        print(f"Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"Precision: {baseline_metrics['precision']:.4f}")
        print(f"Recall: {baseline_metrics['recall']:.4f}")
        
        status = "✅ ALCANZADO" if comparison["baseline_achieves_target"] else "❌ NO ALCANZADO"
        print(f"Objetivo F1 >= {target_f1}: {status}")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error evaluando modelo: {e}")


if __name__ == "__main__":
    evaluate_baseline_only()
