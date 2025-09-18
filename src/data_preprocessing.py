import pandas as pd
import re
import requests
import os
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import yaml
from pathlib import Path


class DataPreprocessor:
    """
    Clase para preprocesar datos de SMS Spam Detection
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa el preprocesador con configuración"""
        self.config = self.load_config(config_path)
        self.data_path = self.config["data"]["data_path"]
        self.dataset_url = self.config["data"]["dataset_url"]

    def load_config(self, config_path: str) -> Dict:
        """Carga la configuración desde archivo YAML"""
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def download_dataset(self) -> None:
        """Descarga el dataset SMS Spam Collection si no existe"""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        if not os.path.exists(self.data_path):
            print(f"Descargando dataset desde: {self.dataset_url}")
            response = requests.get(self.dataset_url)
            response.raise_for_status()

            with open(self.data_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Dataset guardado en: {self.data_path}")
        else:
            print(f"Dataset ya existe en: {self.data_path}")

    def load_dataset(self) -> pd.DataFrame:
        """Carga el dataset desde archivo local"""
        try:
            df = pd.read_csv(
                self.data_path, sep="\t", header=None, names=["label", "message"]
            )
            print(f"Dataset cargado: {len(df)} muestras")
            return df
        except Exception as e:
            print(f"Error cargando dataset: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto"""
        if pd.isna(text):
            return ""

        # Convertir a minúsculas
        text = text.lower()

        # Remover caracteres especiales pero mantener espacios
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # Remover espacios múltiples
        text = re.sub(r"\s+", " ", text)

        # Remover espacios al inicio y final
        text = text.strip()

        return text

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesa el dataframe completo"""
        df_processed = df.copy()

        # Limpiar textos
        df_processed["message_clean"] = df_processed["message"].apply(self.clean_text)

        # Convertir labels a binario (0: ham, 1: spam)
        df_processed["label_binary"] = df_processed["label"].map({"ham": 0, "spam": 1})

        # Remover filas con mensajes vacíos
        df_processed = df_processed[df_processed["message_clean"].str.len() > 0]

        # Información del dataset
        print(f"Datos procesados: {len(df_processed)} muestras")
        print("Distribución de clases:")
        print(df_processed["label"].value_counts())

        return df_processed

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide los datos en entrenamiento y prueba"""
        test_size = self.config["data"]["test_size"]
        random_state = self.config["data"]["random_state"]

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label_binary"],
        )

        print(f"Conjunto de entrenamiento: {len(train_df)} muestras")
        print(f"Conjunto de prueba: {len(test_df)} muestras")

        return train_df, test_df

    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pipeline completo de procesamiento de datos"""
        # Descargar dataset si no existe
        self.download_dataset()

        # Cargar datos
        df = self.load_dataset()

        # Preprocesar
        df_processed = self.preprocess_data(df)

        # Dividir en train/test
        train_df, test_df = self.split_data(df_processed)

        return train_df, test_df

    def save_processed_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """Guarda los datos procesados"""
        data_dir = Path(self.config["paths"]["data_dir"])

        train_path = data_dir / "train_data.csv"
        test_path = data_dir / "test_data.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Datos de entrenamiento guardados en: {train_path}")
        print(f"Datos de prueba guardados en: {test_path}")


def main():
    """Función principal para ejecutar el preprocesamiento"""
    preprocessor = DataPreprocessor()
    train_df, test_df = preprocessor.get_processed_data()
    preprocessor.save_processed_data(train_df, test_df)
    print("\nPreprocesamiento completado exitosamente!")


if __name__ == "__main__":
    main()
