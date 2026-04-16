import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import HousingPriceModel
from app.s3_utils import S3Utils
from app.config import Config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLOps Proyecto Final", description="API con S3, logging y DVC")

# Variables globales
model = HousingPriceModel()
s3 = S3Utils()

class PredictionRequest(BaseModel):
    features: list[float]

@app.on_event("startup")
async def startup_event():
    """Carga el modelo desde S3 al iniciar"""
    try:
        logger.info("Iniciando aplicación...")
        if s3.download_file(Config.MODEL_PATH, 'models/model_tmp.pkl'):
            model.load('models/model_tmp.pkl')
            logger.info("✅ Modelo cargado exitosamente desde S3")
        else:
            logger.warning("⚠️ No se pudo cargar modelo de S3")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")

@app.get("/")
def root():
    logger.info("Root endpoint llamado")
    return {
        "message": "MLOps Proyecto Final - API con S3",
        "status": "running",
        "endpoints": ["/health", "/predict", "/train"]
    }

@app.get("/health")
def health():
    logger.debug("Health check llamado")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        logger.info(f"Predicción solicitada con features: {request.features}")
        
        # Intentar cargar modelo si no está disponible
        if model.model is None:
            logger.info("Modelo no cargado, intentando cargar desde S3")
            if s3.download_file(Config.MODEL_PATH, 'models/model_tmp.pkl'):
                model.load('models/model_tmp.pkl')
        
        prediction = model.predict(request.features)
        logger.info(f"Predicción generada: {prediction}")
        
        return {
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train():
    """Endpoint para entrenar y subir a S3"""
    import subprocess
    logger.info("Iniciando reentrenamiento manual...")
    result = subprocess.run(['python', 'scripts/train.py'], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Reentrenamiento exitoso")
    else:
        logger.error(f"Reentrenamiento falló: {result.stderr}")
    
    return {
        "message": "Entrenamiento completado",
        "output": result.stdout,
        "timestamp": datetime.now().isoformat()
    }
