# TP7 - MLOps: CI/CD, S3 y Reentrenamiento

## Descripción
API de Machine Learning para predicción de precios de viviendas con:
- CI/CD con GitHub Actions
- Almacenamiento en AWS S3
- Reentrenamiento automático diario

## Tecnologías
- FastAPI
- AWS S3
- GitHub Actions
- Scikit-learn

## Endpoints
- `GET /health` - Health check
- `POST /predict` - Realizar predicción
- `POST /train` - Reentrenar modelo

## Configuración
1. Clonar repositorio
2. Crear archivo `.env` con credenciales AWS
3. `pip install -r requirements.txt`
4. `python scripts/train.py`
5. `uvicorn app.main:app --reload`

## CI/CD
El pipeline se ejecuta en cada push y entrena/subel modelo a S3 automáticamente.


Ivan Cespedes

## URL de la API desplegada
Próximamente: https://tp7-mlops-api.onrender.com

## 📊 Diagrama de Arquitectura

```mermaid
graph TB
    subgraph "Desarrollador"
        A[Git Push] --> B[GitHub]
    end
    
    subgraph "CI/CD Pipeline"
        B --> C[GitHub Actions]
        C --> D[Tests]
        C --> E[Entrenamiento]
        C --> F[Deploy]
    end
    
    subgraph "Almacenamiento"
        E --> G[S3 Bucket]
        G --> H[Modelos]
        G --> I[Métricas]
        G --> J[Datos]
    end
    
    subgraph "Infraestructura"
        F --> K[EC2]
        K --> L[API FastAPI]
        L --> M[/health]
        L --> N[/predict]
        L --> O[/train]
    end
    
    subgraph "Reentrenamiento"
        P[Cron diario] --> Q[GitHub Actions]
        Q --> E
    end

