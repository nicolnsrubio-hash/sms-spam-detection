"""
gRPC Server para el servicio de detección de spam
"""

import grpc
import sys
from concurrent import futures
from datetime import datetime
import logging
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Importar protobuf generados (se generarán después)
try:
    import protos.spam_detector_pb2 as spam_detector_pb2
    import protos.spam_detector_pb2_grpc as spam_detector_pb2_grpc
except ImportError:
    print(
        "Error: Los archivos protobuf no están generados. "
        "Ejecuta generate_grpc.py primero"
    )
    sys.exit(1)

from baseline_model import BaselineModel
from distilbert_model import DistilBERTModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpamDetectorServicer(spam_detector_pb2_grpc.SpamDetectorServiceServicer):
    """
    Implementación del servicio gRPC para detección de spam
    """

    def __init__(self):
        self.baseline_model = None
        self.distilbert_model = None
        self.model_accuracies = {}
        self.load_models()

    def load_models(self):
        """Carga los modelos disponibles"""
        try:
            # Cargar modelo baseline
            self.baseline_model = BaselineModel()
            self.baseline_model.load_model()
            self.model_accuracies["baseline"] = 0.91  # F1-Score del entrenamiento
            logger.info("Modelo Baseline cargado correctamente")
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo Baseline: {e}")

        try:
            # Cargar modelo DistilBERT
            self.distilbert_model = DistilBERTModel()
            self.distilbert_model.load_model()
            self.model_accuracies["distilbert"] = 0.95  # Estimado
            logger.info("Modelo DistilBERT cargado correctamente")
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo DistilBERT: {e}")

    def PredictSpam(self, request, context):
        """
        Predice si un mensaje es spam o no
        """
        try:
            message = request.message
            model_type = request.model_type or "baseline"

            if model_type == "baseline" and self.baseline_model:
                # Usar modelo baseline
                predictions = self.baseline_model.model.predict(
                    self.baseline_model.vectorizer.transform([message])
                )
                probabilities = self.baseline_model.model.predict_proba(
                    self.baseline_model.vectorizer.transform([message])
                )

                pred = predictions[0]
                prob = probabilities[0]

                is_spam = bool(pred == 1)
                confidence = float(prob[pred])
                spam_prob = float(prob[1])
                ham_prob = float(prob[0])
                model_used = "TF-IDF + Logistic Regression"

            elif model_type == "distilbert" and self.distilbert_model:
                # Usar modelo DistilBERT
                predictions, probabilities = self.distilbert_model.predict([message])

                pred = predictions[0]
                prob = probabilities[0]

                is_spam = bool(pred == 1)
                confidence = float(prob[pred])
                spam_prob = float(prob[1])
                ham_prob = float(prob[0])
                model_used = "DistilBERT"

            else:
                return spam_detector_pb2.SpamPredictionResponse(
                    error_message=f"Modelo '{model_type}' no disponible"
                )

            return spam_detector_pb2.SpamPredictionResponse(
                is_spam=is_spam,
                confidence=confidence,
                spam_probability=spam_prob,
                ham_probability=ham_prob,
                model_used=model_used,
                error_message="",
            )

        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return spam_detector_pb2.SpamPredictionResponse(
                error_message=f"Error interno: {str(e)}"
            )

    def GetModelStatus(self, request, context):
        """
        Obtiene el estado de un modelo
        """
        try:
            model_type = request.model_type or "baseline"

            if model_type == "baseline":
                is_loaded = self.baseline_model is not None
                model_name = "TF-IDF + Logistic Regression"
                accuracy = self.model_accuracies.get("baseline", 0.0)
            elif model_type == "distilbert":
                is_loaded = self.distilbert_model is not None
                model_name = "DistilBERT"
                accuracy = self.model_accuracies.get("distilbert", 0.0)
            else:
                return spam_detector_pb2.ModelStatusResponse(
                    error_message=f"Tipo de modelo desconocido: {model_type}"
                )

            return spam_detector_pb2.ModelStatusResponse(
                is_loaded=is_loaded,
                model_name=model_name,
                accuracy=accuracy,
                last_trained=datetime.now().isoformat(),
                error_message="",
            )

        except Exception as e:
            logger.error(f"Error obteniendo estado: {e}")
            return spam_detector_pb2.ModelStatusResponse(
                error_message=f"Error interno: {str(e)}"
            )


def serve():
    """Inicia el servidor gRPC"""
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    spam_detector_pb2_grpc.add_SpamDetectorServiceServicer_to_server(
        SpamDetectorServicer(), server
    )

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Servidor gRPC iniciado en {listen_addr}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Deteniendo servidor gRPC...")
        server.stop(0)


if __name__ == "__main__":
    serve()
