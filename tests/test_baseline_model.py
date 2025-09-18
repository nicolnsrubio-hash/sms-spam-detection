"""
Pruebas unitarias para el modelo baseline
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
import tempfile
import os

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from baseline_model import BaselineModel
from data_preprocessing import DataPreprocessor


class TestBaselineModel:
    """
    Test suite para el modelo baseline
    """

    @pytest.fixture
    def sample_data(self):
        """Fixture con datos de prueba"""
        data = {
            "label": ["ham", "spam", "ham", "spam"],
            "message": [
                "Hola como estas",
                "FELICIDADES ganaste premio",
                "Nos vemos mañana",
                "GRATIS dinero URGENTE",
            ],
            "message_clean": [
                "hola como estas",
                "felicidades ganaste premio",
                "nos vemos manana",
                "gratis dinero urgente",
            ],
            "label_binary": [0, 1, 0, 1],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def baseline_model(self):
        """Fixture para modelo baseline"""
        return BaselineModel()

    def test_model_initialization(self, baseline_model):
        """Prueba inicialización del modelo"""
        assert baseline_model.config is not None
        assert baseline_model.vectorizer is None
        assert baseline_model.model is None
        assert baseline_model.is_trained is False

    def test_create_vectorizer(self, baseline_model):
        """Prueba creación del vectorizador TF-IDF"""
        vectorizer = baseline_model.create_vectorizer()

        assert vectorizer is not None
        assert hasattr(vectorizer, "fit_transform")
        assert (
            vectorizer.max_features == baseline_model.config["baseline"]["max_features"]
        )

    def test_create_model(self, baseline_model):
        """Prueba creación del modelo de regresión logística"""
        model = baseline_model.create_model()

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert model.C == baseline_model.config["baseline"]["C"]

    def test_train(self, baseline_model, sample_data):
        """Prueba entrenamiento del modelo"""
        metrics = baseline_model.train(sample_data)

        assert baseline_model.is_trained is True
        assert baseline_model.vectorizer is not None
        assert baseline_model.model is not None

        assert "train_f1" in metrics
        assert "n_features" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == len(sample_data)

    def test_predict(self, baseline_model, sample_data):
        """Prueba predicción del modelo"""
        # Primero entrenar
        baseline_model.train(sample_data)

        # Probar predicción
        test_texts = ["hola que tal", "GRATIS premio URGENTE"]
        predictions, probabilities = baseline_model.predict(test_texts)

        assert len(predictions) == len(test_texts)
        assert len(probabilities) == len(test_texts)
        assert all(pred in [0, 1] for pred in predictions)
        assert all(len(prob) == 2 for prob in probabilities)

    def test_evaluate(self, baseline_model, sample_data):
        """Prueba evaluación del modelo"""
        # Entrenar con datos de muestra
        baseline_model.train(sample_data)

        # Evaluar con los mismos datos
        metrics = baseline_model.evaluate(sample_data)

        assert "f1_score" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics

        # Verificar rangos de métricas
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["accuracy"] <= 1

    def test_save_load_model(self, baseline_model, sample_data):
        """Prueba guardado y carga del modelo"""
        # Entrenar modelo
        baseline_model.train(sample_data)

        # Guardar modelo
        with tempfile.TemporaryDirectory() as temp_dir:
            # Cambiar temporalmente las rutas
            original_model_path = baseline_model.config["baseline"]["model_path"]
            original_vectorizer_path = baseline_model.config["baseline"][
                "vectorizer_path"
            ]

            temp_model_path = os.path.join(temp_dir, "test_model.pkl")
            temp_vectorizer_path = os.path.join(temp_dir, "test_vectorizer.pkl")

            baseline_model.config["baseline"]["model_path"] = temp_model_path
            baseline_model.config["baseline"]["vectorizer_path"] = temp_vectorizer_path

            # Guardar
            baseline_model.save_model()

            # Crear nuevo modelo y cargar
            new_model = BaselineModel()
            new_model.config["baseline"]["model_path"] = temp_model_path
            new_model.config["baseline"]["vectorizer_path"] = temp_vectorizer_path

            new_model.load_model()

            # Verificar que se cargó correctamente
            assert new_model.is_trained is True
            assert new_model.model is not None
            assert new_model.vectorizer is not None

            # Restaurar rutas originales
            baseline_model.config["baseline"]["model_path"] = original_model_path
            baseline_model.config["baseline"][
                "vectorizer_path"
            ] = original_vectorizer_path

    def test_predict_without_training(self, baseline_model):
        """Prueba que falla la predicción sin entrenamiento"""
        with pytest.raises(ValueError):
            baseline_model.predict(["test message"])

    def test_evaluate_without_training(self, baseline_model, sample_data):
        """Prueba que falla la evaluación sin entrenamiento"""
        with pytest.raises(ValueError):
            baseline_model.evaluate(sample_data)


class TestDataPreprocessor:
    """
    Test suite para el preprocesador de datos
    """

    @pytest.fixture
    def preprocessor(self):
        """Fixture para el preprocesador"""
        return DataPreprocessor()

    def test_clean_text(self, preprocessor):
        """Prueba limpieza de texto"""
        # Casos de prueba
        test_cases = [
            ("HOLA MUNDO!", "hola mundo"),
            ("Números 123 y símbolos @#$", "numeros 123 y simbolos"),
            ("  espacios   múltiples  ", "espacios multiples"),
            ("", ""),
            (None, ""),
        ]

        for input_text, expected in test_cases:
            result = preprocessor.clean_text(input_text)
            assert result == expected

    def test_preprocess_data(self, preprocessor):
        """Prueba preprocesamiento completo"""
        # Crear dataframe de prueba
        test_data = pd.DataFrame(
            {
                "label": ["ham", "spam", "ham"],
                "message": [
                    "Hola como estas",
                    "FELICIDADES!!! Ganaste $1000",
                    "Nos vemos mañana",
                ],
            }
        )

        result = preprocessor.preprocess_data(test_data)

        # Verificar columnas
        assert "message_clean" in result.columns
        assert "label_binary" in result.columns

        # Verificar limpieza
        assert result["message_clean"].iloc[0] == "hola como estas"

        # Verificar codificación binaria
        assert result["label_binary"].iloc[0] == 0  # ham
        assert result["label_binary"].iloc[1] == 1  # spam

    def test_split_data(self, preprocessor):
        """Prueba división de datos"""
        # Crear dataframe balanceado
        test_data = pd.DataFrame(
            {
                "label": ["ham"] * 10 + ["spam"] * 10,
                "message": [f"mensaje ham {i}" for i in range(10)]
                + [f"mensaje spam {i}" for i in range(10)],
                "label_binary": [0] * 10 + [1] * 10,
            }
        )

        train_df, test_df = preprocessor.split_data(test_data)

        # Verificar tamaños
        assert len(train_df) + len(test_df) == len(test_data)
        assert len(test_df) == int(
            len(test_data) * preprocessor.config["data"]["test_size"]
        )

        # Verificar que ambos conjuntos tienen ambas clases
        assert 0 in train_df["label_binary"].values
        assert 1 in train_df["label_binary"].values
        assert 0 in test_df["label_binary"].values
        assert 1 in test_df["label_binary"].values


# Ejecutar pruebas si se ejecuta directamente
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
