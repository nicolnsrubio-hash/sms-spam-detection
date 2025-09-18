import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any


class BaselineModel:
    """
    Modelo baseline usando TF-IDF + Regresión Logística para detección de spam
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa el modelo baseline con configuración"""
        self.config = self.load_config(config_path)
        self.vectorizer = None
        self.model = None
        self.is_trained = False

    def load_config(self, config_path: str) -> Dict:
        """Carga la configuración desde archivo YAML"""
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def create_vectorizer(self) -> TfidfVectorizer:
        """Crea el vectorizador TF-IDF con la configuración especificada"""
        baseline_config = self.config["baseline"]

        vectorizer = TfidfVectorizer(
            max_features=baseline_config["max_features"],
            ngram_range=tuple(baseline_config["ngram_range"]),
            min_df=baseline_config["min_df"],
            max_df=baseline_config["max_df"],
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
        )

        return vectorizer

    def create_model(self) -> LogisticRegression:
        """Crea el modelo de regresión logística"""
        baseline_config = self.config["baseline"]

        model = LogisticRegression(
            C=baseline_config["C"],
            max_iter=baseline_config["max_iter"],
            solver=baseline_config["solver"],
            random_state=self.config["data"]["random_state"],
        )

        return model

    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """Entrena el modelo baseline"""
        print("Entrenando modelo baseline (TF-IDF + Regresión Logística)...")

        # Extraer textos y etiquetas
        X_train_text = train_df["message_clean"].values
        y_train = train_df["label_binary"].values

        # Crear y entrenar vectorizador
        self.vectorizer = self.create_vectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train_text)

        print(f"Características TF-IDF: {X_train_tfidf.shape[1]}")

        # Crear y entrenar modelo
        self.model = self.create_model()
        self.model.fit(X_train_tfidf, y_train)

        # Evaluar en datos de entrenamiento
        train_pred = self.model.predict(X_train_tfidf)
        train_f1 = f1_score(y_train, train_pred)

        print(f"F1-Score en entrenamiento: {train_f1:.4f}")

        self.is_trained = True

        return {
            "train_f1": train_f1,
            "n_features": X_train_tfidf.shape[1],
            "n_samples": X_train_tfidf.shape[0],
        }

    def predict(self, texts: list) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones en nuevos textos"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")

        # Vectorizar textos
        X_tfidf = self.vectorizer.transform(texts)

        # Predicciones
        predictions = self.model.predict(X_tfidf)
        probabilities = self.model.predict_proba(X_tfidf)

        return predictions, probabilities

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa el modelo en el conjunto de prueba"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")

        print("Evaluando modelo baseline...")

        # Extraer datos de prueba
        X_test_text = test_df["message_clean"].values
        y_test = test_df["label_binary"].values

        # Realizar predicciones
        predictions, probabilities = self.predict(X_test_text)

        # Calcular métricas
        f1 = f1_score(y_test, predictions)
        report = classification_report(
            y_test, predictions, target_names=["Ham", "Spam"], output_dict=True
        )
        cm = confusion_matrix(y_test, predictions)

        print(f"F1-Score en prueba: {f1:.4f}")
        print("\\nReporte de clasificación:")
        print(classification_report(y_test, predictions, target_names=["Ham", "Spam"]))

        return {
            "f1_score": f1,
            "accuracy": report["accuracy"],
            "precision_ham": report["Ham"]["precision"],
            "recall_ham": report["Ham"]["recall"],
            "precision_spam": report["Spam"]["precision"],
            "recall_spam": report["Spam"]["recall"],
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    def save_model(self) -> None:
        """Guarda el modelo y vectorizador entrenados"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de guardar")

        models_dir = Path(self.config["paths"]["models_dir"])
        models_dir.mkdir(exist_ok=True)

        # Rutas de guardado
        model_path = models_dir / self.config["baseline"]["model_path"].split("/")[-1]
        vectorizer_path = (
            models_dir / self.config["baseline"]["vectorizer_path"].split("/")[-1]
        )

        # Guardar modelo y vectorizador
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

        print(f"Modelo guardado en: {model_path}")
        print(f"Vectorizador guardado en: {vectorizer_path}")

    def load_model(self) -> None:
        """Carga el modelo y vectorizador desde archivos"""
        models_dir = Path(self.config["paths"]["models_dir"])

        model_path = models_dir / self.config["baseline"]["model_path"].split("/")[-1]
        vectorizer_path = (
            models_dir / self.config["baseline"]["vectorizer_path"].split("/")[-1]
        )

        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError("Archivos del modelo no encontrados")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.is_trained = True

        print(f"Modelo cargado desde: {model_path}")
        print(f"Vectorizador cargado desde: {vectorizer_path}")


def main():
    """Función principal para entrenar el modelo baseline"""
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

    # Crear y entrenar modelo baseline
    baseline = BaselineModel()

    # Entrenar
    train_metrics = baseline.train(train_df)
    print(f"\\nMétricas de entrenamiento: {train_metrics}")

    # Evaluar
    test_metrics = baseline.evaluate(test_df)

    # Guardar modelo
    baseline.save_model()

    # Verificar objetivo de F1 >= 0.95
    target_f1 = baseline.config["evaluation"]["target_f1_score"]
    achieved_f1 = test_metrics["f1_score"]

    if achieved_f1 >= target_f1:
        print(f"\\n✅ Objetivo alcanzado! F1-Score: {achieved_f1:.4f} >= {target_f1}")
    else:
        print(f"\\n❌ Objetivo no alcanzado. F1-Score: {achieved_f1:.4f} < {target_f1}")

    print("\\nEntrenamiento del modelo baseline completado!")


if __name__ == "__main__":
    main()
