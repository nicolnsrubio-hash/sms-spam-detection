import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMSDataset(Dataset):
    """Dataset personalizado para SMS spam detection"""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenizar
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DistilBERTModel:
    """
    Modelo DistilBERT afinado para detección de spam en SMS
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa el modelo DistilBERT"""
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.is_trained = False

    def load_config(self, config_path: str) -> Dict:
        """Carga la configuración desde archivo YAML"""
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def setup_model_and_tokenizer(self):
        """Configura el modelo y tokenizador DistilBERT"""
        model_name = self.config["distilbert"]["model_name"]
        num_labels = self.config["distilbert"]["num_labels"]

        logger.info(f"Cargando modelo y tokenizador: {model_name}")

        # Cargar tokenizador
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Cargar modelo
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # Mover modelo al dispositivo
        self.model.to(self.device)

        logger.info("Modelo y tokenizador configurados correctamente")

    def create_datasets(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[SMSDataset, SMSDataset]:
        """Crea datasets para entrenamiento y evaluación"""
        max_length = self.config["distilbert"]["max_length"]

        # Preparar datos de entrenamiento
        train_texts = train_df["message_clean"].tolist()
        train_labels = train_df["label_binary"].tolist()

        # Preparar datos de prueba
        test_texts = test_df["message_clean"].tolist()
        test_labels = test_df["label_binary"].tolist()

        # Crear datasets
        train_dataset = SMSDataset(
            train_texts, train_labels, self.tokenizer, max_length
        )
        test_dataset = SMSDataset(test_texts, test_labels, self.tokenizer, max_length)

        logger.info(f"Dataset de entrenamiento: {len(train_dataset)} muestras")
        logger.info(f"Dataset de evaluación: {len(test_dataset)} muestras")

        return train_dataset, test_dataset

    def compute_metrics(self, eval_pred):
        """Función para computar métricas durante el entrenamiento"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calcular métricas
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def setup_training_arguments(self) -> TrainingArguments:
        """Configura los argumentos de entrenamiento"""
        distilbert_config = self.config["distilbert"]
        training_config = self.config["training"]

        output_dir = Path(self.config["paths"]["logs_dir"]) / "distilbert_training"

        # Desactivar integración MLflow/WandB
        import os
        os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"
        os.environ["WANDB_DISABLED"] = "true"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=int(distilbert_config["num_epochs"]),
            per_device_train_batch_size=int(distilbert_config["batch_size"]),
            per_device_eval_batch_size=int(distilbert_config["batch_size"]),
            warmup_steps=int(distilbert_config["warmup_steps"]),
            weight_decay=float(distilbert_config["weight_decay"]),
            learning_rate=float(distilbert_config["learning_rate"]),
            logging_steps=int(training_config["logging_steps"]),
            eval_strategy=training_config["evaluation_strategy"],
            eval_steps=int(training_config["eval_steps"]),
            save_strategy=training_config["save_strategy"],
            save_steps=int(training_config["save_steps"]),
            load_best_model_at_end=bool(training_config["load_best_model_at_end"]),
            metric_for_best_model=training_config["metric_for_best_model"],
            greater_is_better=True,
            logging_dir=str(output_dir / "logs"),
            report_to=[],  # Lista vacía para desactivar todos los reportes
            dataloader_pin_memory=False,
            disable_tqdm=False,
        )

        return training_args

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena el modelo DistilBERT"""
        logger.info("Iniciando entrenamiento de DistilBERT...")

        # Configurar modelo y tokenizador
        if self.tokenizer is None or self.model is None:
            self.setup_model_and_tokenizer()

        # Crear datasets
        train_dataset, eval_dataset = self.create_datasets(train_df, test_df)

        # Configurar argumentos de entrenamiento
        training_args = self.setup_training_arguments()

        # Crear trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Entrenar modelo
        logger.info("Comenzando entrenamiento...")
        train_result = self.trainer.train()

        self.is_trained = True

        # Evaluar en conjunto de entrenamiento
        logger.info("Evaluando en conjunto de entrenamiento...")
        train_metrics = self.trainer.evaluate(train_dataset)

        f1_score_train = train_metrics.get("eval_f1", "N/A")
        logger.info(
            f"Entrenamiento completado. F1-Score en entrenamiento: {f1_score_train:.4f}"
        )

        return {
            "train_loss": train_result.training_loss,
            "train_f1": train_metrics.get("eval_f1", 0),
            "train_accuracy": train_metrics.get("eval_accuracy", 0),
            "global_step": train_result.global_step,
        }

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones en nuevos textos"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for text in texts:
                # Tokenizar
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.config["distilbert"]["max_length"],
                    return_tensors="pt",
                )

                # Mover a dispositivo
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Predicción
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Aplicar softmax para obtener probabilidades
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = np.argmax(probs)

                predictions.append(pred)
                probabilities.append(probs)

        return np.array(predictions), np.array(probabilities)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa el modelo en el conjunto de prueba"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")

        logger.info("Evaluando modelo DistilBERT...")

        # Extraer datos de prueba
        test_texts = test_df["message_clean"].tolist()
        y_test = test_df["label_binary"].values

        # Realizar predicciones
        predictions, probabilities = self.predict(test_texts)

        # Calcular métricas
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted"
        )
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        # Métricas por clase
        precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(
            y_test, predictions, average=None
        )

        logger.info(f"F1-Score en prueba: {f1:.4f}")
        logger.info(f"Accuracy en prueba: {accuracy:.4f}")

        return {
            "f1_score": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "precision_ham": precision_class[0] if len(precision_class) > 0 else 0,
            "recall_ham": recall_class[0] if len(recall_class) > 0 else 0,
            "precision_spam": precision_class[1] if len(precision_class) > 1 else 0,
            "recall_spam": recall_class[1] if len(recall_class) > 1 else 0,
            "confusion_matrix": cm.tolist(),
        }

    def save_model(self):
        """Guarda el modelo y tokenizador entrenados"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardar")

        models_dir = Path(self.config["paths"]["models_dir"])
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / "distilbert_spam_classifier"
        tokenizer_path = models_dir / "distilbert_tokenizer"

        # Guardar modelo y tokenizador
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)

        logger.info(f"Modelo DistilBERT guardado en: {model_path}")
        logger.info(f"Tokenizador guardado en: {tokenizer_path}")

    def load_model(self):
        """Carga el modelo y tokenizador desde archivos"""
        models_dir = Path(self.config["paths"]["models_dir"])

        model_path = models_dir / "distilbert_spam_classifier"
        tokenizer_path = models_dir / "distilbert_tokenizer"

        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError("Archivos del modelo DistilBERT no encontrados")

        # Cargar tokenizador y modelo
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)

        self.is_trained = True

        logger.info(f"Modelo DistilBERT cargado desde: {model_path}")
        logger.info(f"Tokenizador cargado desde: {tokenizer_path}")


def main():
    """Función principal para entrenar el modelo DistilBERT"""
    from data_preprocessing import DataPreprocessor

    # Cargar y preprocesar datos
    preprocessor = DataPreprocessor()

    # Verificar si ya existen datos procesados
    data_dir = Path(preprocessor.config["paths"]["data_dir"])
    train_path = data_dir / "train_data.csv"
    test_path = data_dir / "test_data.csv"

    if train_path.exists() and test_path.exists():
        print("Cargando datos procesados existentes...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    else:
        print("Procesando datos por primera vez...")
        train_df, test_df = preprocessor.get_processed_data()
        preprocessor.save_processed_data(train_df, test_df)

    # Crear y entrenar modelo DistilBERT
    distilbert = DistilBERTModel()

    # Entrenar
    train_metrics = distilbert.train(train_df, test_df)
    print(f"\nMétricas de entrenamiento: {train_metrics}")

    # Evaluar
    test_metrics = distilbert.evaluate(test_df)

    # Guardar modelo
    distilbert.save_model()

    # Verificar objetivo de F1 >= 0.95
    target_f1 = distilbert.config["evaluation"]["target_f1_score"]
    achieved_f1 = test_metrics["f1_score"]

    if achieved_f1 >= target_f1:
        print(f"\n✅ Objetivo alcanzado! F1-Score: {achieved_f1:.4f} >= {target_f1}")
    else:
        print(f"\n❌ Objetivo no alcanzado. F1-Score: {achieved_f1:.4f} < {target_f1}")

    print("\nEntrenamiento del modelo DistilBERT completado!")


if __name__ == "__main__":
    main()
