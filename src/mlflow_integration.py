"""
Integración con MLflow para tracking de experimentos y gestión de modelos
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import Dict
import joblib
import os
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Gestor de experimentos y modelos con MLflow
    """

    def __init__(self, experiment_name: str = "SMS_Spam_Detection"):
        """
        Inicializa el gestor MLflow
        """
        self.experiment_name = experiment_name
        self.setup_mlflow()

    def setup_mlflow(self):
        """
        Configura MLflow tracking
        """
        # Configurar directorio de tracking
        tracking_dir = Path("mlruns")
        tracking_dir.mkdir(exist_ok=True)

        mlflow.set_tracking_uri(f"file://{tracking_dir.absolute()}")

        # Crear o establecer experimento
        try:
            mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            # El experimento ya existe
            pass

        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow configurado. Experimento: {self.experiment_name}")

    def log_baseline_experiment(
        self, model, vectorizer, train_metrics: Dict, test_metrics: Dict, config: Dict
    ):
        """
        Registra experimento del modelo baseline
        """
        with mlflow.start_run(run_name="Baseline_TF-IDF_LogisticRegression") as run:
            # Log de parámetros
            baseline_config = config.get("baseline", {})
            mlflow.log_params(
                {
                    "model_type": "baseline",
                    "algorithm": "TF-IDF + Logistic Regression",
                    "max_features": baseline_config.get("max_features", 5000),
                    "ngram_range": str(baseline_config.get("ngram_range", [1, 2])),
                    "C": baseline_config.get("C", 1.0),
                    "solver": baseline_config.get("solver", "liblinear"),
                    "max_iter": baseline_config.get("max_iter", 1000),
                }
            )

            # Log de métricas de entrenamiento
            mlflow.log_metrics(
                {
                    "train_f1": train_metrics.get("train_f1", 0),
                    "train_samples": train_metrics.get("n_samples", 0),
                    "train_features": train_metrics.get("n_features", 0),
                }
            )

            # Log de métricas de prueba
            mlflow.log_metrics(
                {
                    "test_f1": test_metrics.get("f1_score", 0),
                    "test_accuracy": test_metrics.get("accuracy", 0),
                    "test_precision": test_metrics.get("precision", 0),
                    "test_recall": test_metrics.get("recall", 0),
                    "test_precision_ham": test_metrics.get("precision_ham", 0),
                    "test_recall_ham": test_metrics.get("recall_ham", 0),
                    "test_precision_spam": test_metrics.get("precision_spam", 0),
                    "test_recall_spam": test_metrics.get("recall_spam", 0),
                }
            )

            # Log de artefactos (modelo y vectorizador)
            temp_model_path = "temp_baseline_model.pkl"
            temp_vectorizer_path = "temp_vectorizer.pkl"

            joblib.dump(model, temp_model_path)
            joblib.dump(vectorizer, temp_vectorizer_path)

            mlflow.log_artifact(temp_model_path, "model")
            mlflow.log_artifact(temp_vectorizer_path, "model")

            # Registrar modelo
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn_model",
                registered_model_name="SMS_Spam_Baseline",
            )

            # Log de matriz de confusión si está disponible
            if "confusion_matrix" in test_metrics:
                cm_data = pd.DataFrame(
                    test_metrics["confusion_matrix"],
                    index=["Ham", "Spam"],
                    columns=["Predicted_Ham", "Predicted_Spam"],
                )
                mlflow.log_text(cm_data.to_string(), "confusion_matrix.txt")

            # Limpiar archivos temporales
            os.remove(temp_model_path)
            os.remove(temp_vectorizer_path)

            # Log de tags
            mlflow.set_tags(
                {
                    "model_family": "sklearn",
                    "dataset_version": "enhanced_spanish",
                    "training_date": datetime.now().isoformat(),
                    "status": "trained",
                }
            )

            logger.info(f"Experimento baseline registrado. Run ID: {run.info.run_id}")
            return run.info.run_id

    def log_distilbert_experiment(
        self, model, tokenizer, train_metrics: Dict, test_metrics: Dict, config: Dict
    ):
        """
        Registra experimento del modelo DistilBERT
        """
        with mlflow.start_run(run_name="DistilBERT_Fine_Tuned") as run:
            # Log de parámetros
            distilbert_config = config.get("distilbert", {})
            mlflow.log_params(
                {
                    "model_type": "distilbert",
                    "algorithm": "DistilBERT Fine-tuned",
                    "base_model": distilbert_config.get(
                        "model_name", "distilbert-base-uncased"
                    ),
                    "max_length": distilbert_config.get("max_length", 128),
                    "batch_size": distilbert_config.get("batch_size", 16),
                    "learning_rate": distilbert_config.get("learning_rate", 2e-5),
                    "num_epochs": distilbert_config.get("num_epochs", 3),
                    "warmup_steps": distilbert_config.get("warmup_steps", 500),
                    "weight_decay": distilbert_config.get("weight_decay", 0.01),
                }
            )

            # Log de métricas de entrenamiento
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics.get("train_loss", 0),
                    "train_f1": train_metrics.get("train_f1", 0),
                    "train_accuracy": train_metrics.get("train_accuracy", 0),
                    "global_step": train_metrics.get("global_step", 0),
                }
            )

            # Log de métricas de prueba
            mlflow.log_metrics(
                {
                    "test_f1": test_metrics.get("f1_score", 0),
                    "test_accuracy": test_metrics.get("accuracy", 0),
                    "test_precision": test_metrics.get("precision", 0),
                    "test_recall": test_metrics.get("recall", 0),
                    "test_precision_ham": test_metrics.get("precision_ham", 0),
                    "test_recall_ham": test_metrics.get("recall_ham", 0),
                    "test_precision_spam": test_metrics.get("precision_spam", 0),
                    "test_recall_spam": test_metrics.get("recall_spam", 0),
                }
            )

            # Log de artefactos del modelo (si existe)
            model_dir = Path(config["distilbert"]["model_path"])
            if model_dir.exists():
                mlflow.log_artifacts(str(model_dir), "distilbert_model")

            tokenizer_dir = Path(config["distilbert"]["tokenizer_path"])
            if tokenizer_dir.exists():
                mlflow.log_artifacts(str(tokenizer_dir), "tokenizer")

            # Log de matriz de confusión si está disponible
            if "confusion_matrix" in test_metrics:
                cm_data = pd.DataFrame(
                    test_metrics["confusion_matrix"],
                    index=["Ham", "Spam"],
                    columns=["Predicted_Ham", "Predicted_Spam"],
                )
                mlflow.log_text(cm_data.to_string(), "confusion_matrix.txt")

            # Log de tags
            mlflow.set_tags(
                {
                    "model_family": "transformers",
                    "dataset_version": "enhanced_spanish",
                    "training_date": datetime.now().isoformat(),
                    "status": "trained",
                }
            )

            logger.info(f"Experimento DistilBERT registrado. Run ID: {run.info.run_id}")
            return run.info.run_id

    def log_model_comparison(self, comparison_results: Dict):
        """
        Registra comparación entre modelos
        """
        with mlflow.start_run(run_name="Model_Comparison") as run:
            # Log de métricas de comparación
            baseline_metrics = comparison_results.get("baseline_metrics", {})
            distilbert_metrics = comparison_results.get("distilbert_metrics", {})
            comparison = comparison_results.get("comparison", {})

            mlflow.log_metrics(
                {
                    "baseline_f1": baseline_metrics.get("f1_score", 0),
                    "baseline_accuracy": baseline_metrics.get("accuracy", 0),
                    "distilbert_f1": distilbert_metrics.get("f1_score", 0),
                    "distilbert_accuracy": distilbert_metrics.get("accuracy", 0),
                    "improvement": comparison.get("improvement", 0),
                    "target_f1": comparison.get("target_f1", 0.95),
                    "baseline_achieves_target": int(
                        comparison.get("baseline_achieves_target", False)
                    ),
                    "distilbert_achieves_target": int(
                        comparison.get("distilbert_achieves_target", False)
                    ),
                }
            )

            # Log del mejor modelo
            mlflow.log_param("best_model", comparison.get("best_model", "baseline"))

            # Log de tags
            mlflow.set_tags(
                {
                    "experiment_type": "comparison",
                    "comparison_date": datetime.now().isoformat(),
                }
            )

            logger.info(f"Comparación de modelos registrada. Run ID: {run.info.run_id}")
            return run.info.run_id

    def get_best_model_info(self):
        """
        Obtiene información del mejor modelo registrado
        """
        try:
            # Buscar el mejor modelo por F1-score
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.test_f1 DESC"],
                max_results=1,
            )

            if len(runs) > 0:
                best_run = runs.iloc[0]
                return {
                    "run_id": best_run["run_id"],
                    "model_type": best_run.get("params.model_type", "unknown"),
                    "test_f1": best_run.get("metrics.test_f1", 0),
                    "test_accuracy": best_run.get("metrics.test_accuracy", 0),
                }

            return None

        except Exception as e:
            logger.error(f"Error obteniendo mejor modelo: {e}")
            return None

    def start_mlflow_ui(self, port: int = 5000):
        """
        Inicia la interfaz web de MLflow
        """
        tracking_dir = Path("mlruns").absolute()
        command = f"mlflow ui --backend-store-uri file://{tracking_dir} --port {port}"
        logger.info(f"Para ver la interfaz MLflow, ejecuta: {command}")
        return command


def load_config(config_path: str = "config.yaml") -> Dict:
    """Carga configuración desde YAML"""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar MLflow
    mlflow_manager = MLflowManager()

    # Ejemplo de logging de experimento
    config = load_config()

    # Simular métricas de ejemplo
    train_metrics = {"train_f1": 1.0, "n_samples": 85, "n_features": 1502}
    test_metrics = {
        "f1_score": 0.91,
        "accuracy": 0.91,
        "precision": 0.91,
        "recall": 0.91,
        "confusion_matrix": [[10, 1], [1, 10]],
    }

    print("MLflow configurado correctamente!")
    print("Para ver la interfaz web, ejecuta:")
    print(mlflow_manager.start_mlflow_ui())
