import pandas as pd
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")

# Imports de nuestros modelos
from baseline_model import BaselineModel
from distilbert_model import DistilBERTModel
from data_preprocessing import DataPreprocessor


class ModelEvaluator:
    """
    Clase para evaluar y comparar el modelo baseline vs DistilBERT
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa el evaluador de modelos"""
        self.config = self.load_config(config_path)
        self.results = {}

    def load_config(self, config_path: str) -> Dict:
        """Carga la configuración desde archivo YAML"""
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def load_test_data(self) -> pd.DataFrame:
        """Carga los datos de prueba"""
        data_dir = Path(self.config["paths"]["data_dir"])
        test_path = data_dir / "test_data.csv"

        if not test_path.exists():
            print("Datos de prueba no encontrados. Procesando datos...")
            preprocessor = DataPreprocessor()
            _, test_df = preprocessor.get_processed_data()
            preprocessor.save_processed_data(_, test_df)
            return test_df
        else:
            return pd.read_csv(test_path)

    def evaluate_baseline_model(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa el modelo baseline"""
        print("Evaluando modelo baseline...")

        try:
            baseline = BaselineModel()
            baseline.load_model()
            metrics = baseline.evaluate(test_df)

            # Agregar información adicional
            metrics["model_type"] = "baseline"
            metrics["model_name"] = "TF-IDF + Logistic Regression"

            return metrics

        except Exception as e:
            print(f"Error evaluando modelo baseline: {e}")
            return {}

    def evaluate_distilbert_model(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa el modelo DistilBERT"""
        print("Evaluando modelo DistilBERT...")

        try:
            distilbert = DistilBERTModel()
            distilbert.load_model()
            metrics = distilbert.evaluate(test_df)

            # Agregar información adicional
            metrics["model_type"] = "distilbert"
            metrics["model_name"] = "DistilBERT"

            return metrics

        except Exception as e:
            print(f"Error evaluando modelo DistilBERT: {e}")
            return {}

    def compare_models(
        self, baseline_metrics: Dict, distilbert_metrics: Dict
    ) -> Dict[str, Any]:
        """Compara los dos modelos y determina el mejor"""

        if not baseline_metrics or not distilbert_metrics:
            print("No se pueden comparar los modelos - falta información de evaluación")
            return {}

        comparison = {
            "baseline": {
                "f1_score": baseline_metrics.get("f1_score", 0),
                "accuracy": baseline_metrics.get("accuracy", 0),
                "precision": baseline_metrics.get("precision", 0),
                "recall": baseline_metrics.get("recall", 0),
            },
            "distilbert": {
                "f1_score": distilbert_metrics.get("f1_score", 0),
                "accuracy": distilbert_metrics.get("accuracy", 0),
                "precision": distilbert_metrics.get("precision", 0),
                "recall": distilbert_metrics.get("recall", 0),
            },
        }

        # Determinar el mejor modelo basado en F1-score
        baseline_f1 = comparison["baseline"]["f1_score"]
        distilbert_f1 = comparison["distilbert"]["f1_score"]

        if distilbert_f1 > baseline_f1:
            best_model = "distilbert"
            improvement = distilbert_f1 - baseline_f1
        else:
            best_model = "baseline"
            improvement = baseline_f1 - distilbert_f1

        comparison["best_model"] = best_model
        comparison["improvement"] = improvement
        comparison["target_f1"] = self.config["evaluation"]["target_f1_score"]

        # Verificar si algún modelo alcanza el objetivo
        comparison["baseline_achieves_target"] = baseline_f1 >= comparison["target_f1"]
        comparison["distilbert_achieves_target"] = (
            distilbert_f1 >= comparison["target_f1"]
        )

        return comparison

    def create_comparison_plots(self, baseline_metrics: Dict, distilbert_metrics: Dict):
        """Crea gráficos de comparación entre modelos"""
        results_dir = Path(self.config["paths"]["results_dir"])
        results_dir.mkdir(exist_ok=True)

        # Configurar estilo
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Gráfico de barras comparativo de métricas principales
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Comparación de Modelos: Baseline vs DistilBERT",
            fontsize=16,
            fontweight="bold",
        )

        models = ["Baseline\\n(TF-IDF + LR)", "DistilBERT"]

        # F1-Score
        f1_scores = [
            baseline_metrics.get("f1_score", 0),
            distilbert_metrics.get("f1_score", 0),
        ]
        axes[0, 0].bar(models, f1_scores, color=["skyblue", "lightcoral"])
        axes[0, 0].set_title("F1-Score")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        # Accuracy
        accuracies = [
            baseline_metrics.get("accuracy", 0),
            distilbert_metrics.get("accuracy", 0),
        ]
        axes[0, 1].bar(models, accuracies, color=["lightgreen", "orange"])
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 1].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        # Precision
        precisions = [
            baseline_metrics.get("precision", 0),
            distilbert_metrics.get("precision", 0),
        ]
        axes[1, 0].bar(models, precisions, color=["gold", "mediumpurple"])
        axes[1, 0].set_title("Precision")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(precisions):
            axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        # Recall
        recalls = [
            baseline_metrics.get("recall", 0),
            distilbert_metrics.get("recall", 0),
        ]
        axes[1, 1].bar(models, recalls, color=["lightsteelblue", "salmon"])
        axes[1, 1].set_title("Recall")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_ylim(0, 1)
        for i, v in enumerate(recalls):
            axes[1, 1].text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(results_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Matrices de confusión
        if (
            "confusion_matrix" in baseline_metrics
            and "confusion_matrix" in distilbert_metrics
        ):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Matrices de Confusión", fontsize=16, fontweight="bold")

            # Baseline confusion matrix
            cm_baseline = np.array(baseline_metrics["confusion_matrix"])
            sns.heatmap(
                cm_baseline,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"],
                ax=axes[0],
            )
            axes[0].set_title("Baseline (TF-IDF + LR)")
            axes[0].set_xlabel("Predicción")
            axes[0].set_ylabel("Verdadero")

            # DistilBERT confusion matrix
            cm_distilbert = np.array(distilbert_metrics["confusion_matrix"])
            sns.heatmap(
                cm_distilbert,
                annot=True,
                fmt="d",
                cmap="Reds",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"],
                ax=axes[1],
            )
            axes[1].set_title("DistilBERT")
            axes[1].set_xlabel("Predicción")
            axes[1].set_ylabel("Verdadero")

            plt.tight_layout()
            plt.savefig(
                results_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        print(f"Gráficos guardados en: {results_dir}")

    def save_results(self, results: Dict[str, Any]):
        """Guarda los resultados en archivo JSON"""
        results_dir = Path(self.config["paths"]["results_dir"])
        results_dir.mkdir(exist_ok=True)

        results_path = results_dir / "evaluation_results.json"

        # Convertir numpy arrays a listas para serialización JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_clean = convert_numpy(results)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)

        print(f"Resultados guardados en: {results_path}")

    def print_summary(self, comparison: Dict[str, Any]):
        """Imprime un resumen de la comparación"""
        print("\\n" + "=" * 60)
        print("RESUMEN DE EVALUACIÓN DE MODELOS")
        print("=" * 60)

        target_f1 = comparison["target_f1"]

        print("\\nMODELO BASELINE (TF-IDF + Regresión Logística):")
        print(f"  F1-Score: {comparison['baseline']['f1_score']:.4f}")
        print(f"  Accuracy: {comparison['baseline']['accuracy']:.4f}")
        print(f"  Precision: {comparison['baseline']['precision']:.4f}")
        print(f"  Recall: {comparison['baseline']['recall']:.4f}")
        status = (
            "✅ ALCANZADO"
            if comparison["baseline_achieves_target"]
            else "❌ NO ALCANZADO"
        )
        print(f"  Objetivo F1 >= {target_f1}: {status}")

        print("\\nMODELO DISTILBERT:")
        print(f"  F1-Score: {comparison['distilbert']['f1_score']:.4f}")
        print(f"  Accuracy: {comparison['distilbert']['accuracy']:.4f}")
        print(f"  Precision: {comparison['distilbert']['precision']:.4f}")
        print(f"  Recall: {comparison['distilbert']['recall']:.4f}")
        status = (
            "✅ ALCANZADO"
            if comparison["distilbert_achieves_target"]
            else "❌ NO ALCANZADO"
        )
        print(f"  Objetivo F1 >= {target_f1}: {status}")

        print("\\nCOMPARACIÓN:")
        best_model_name = (
            "DistilBERT" if comparison["best_model"] == "distilbert" else "Baseline"
        )
        print(f"  Mejor modelo: {best_model_name}")
        print(f"  Mejora en F1-Score: +{comparison['improvement']:.4f}")

        print("\\n" + "=" * 60)

    def run_full_evaluation(self):
        """Ejecuta la evaluación completa de ambos modelos"""
        print("Iniciando evaluación completa de modelos...")

        # Cargar datos de prueba
        test_df = self.load_test_data()
        print(f"Datos de prueba cargados: {len(test_df)} muestras")

        # Evaluar modelo baseline
        baseline_metrics = self.evaluate_baseline_model(test_df)

        # Evaluar modelo DistilBERT
        distilbert_metrics = self.evaluate_distilbert_model(test_df)

        if not baseline_metrics or not distilbert_metrics:
            print(
                "No se pudo completar la evaluación. "
                "Verifique que los modelos estén entrenados."
            )
            return

        # Comparar modelos
        comparison = self.compare_models(baseline_metrics, distilbert_metrics)

        # Crear gráficos
        self.create_comparison_plots(baseline_metrics, distilbert_metrics)

        # Preparar resultados completos
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "test_samples": len(test_df),
            "baseline_metrics": baseline_metrics,
            "distilbert_metrics": distilbert_metrics,
            "comparison": comparison,
        }

        # Guardar resultados
        self.save_results(results)

        # Mostrar resumen
        self.print_summary(comparison)

        return results


def main():
    """Función principal para ejecutar la evaluación"""
    evaluator = ModelEvaluator()
    results = evaluator.run_full_evaluation()

    if results:
        print("\\nEvaluación completada exitosamente!")
        print(
            "Revisa la carpeta 'results' para ver los gráficos y resultados detallados."
        )
    else:
        print("\\nNo se pudo completar la evaluación.")


if __name__ == "__main__":
    main()
