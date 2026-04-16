import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import HousingPriceModel
from app.s3_utils import S3Utils
from app.config import Config

def generate_data():
    np.random.seed(42)
    size = np.random.normal(150, 50, 500)
    bedrooms = np.random.randint(1, 5, 500)
    age = np.random.uniform(0, 50, 500)
    
    price = 3000*size + 20000*bedrooms - 1000*age + np.random.normal(0, 50000, 500)
    
    X = pd.DataFrame({'size': size, 'bedrooms': bedrooms, 'age': age})
    y = price
    return X, y

def save_metrics(metrics, filename='metrics.json'):
    """Guarda métricas en archivo JSON con timestamp"""
    metrics['timestamp'] = datetime.now().isoformat()
    os.makedirs('metrics', exist_ok=True)
    with open(f'metrics/{filename}', 'w') as f:
        json.dump(metrics, f, indent=2)
    return f'metrics/{filename}'

def main():
    print("🚀 Iniciando entrenamiento...")
    
    # Inicializar S3
    s3 = S3Utils()
    
    # Generar o cargar datos
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar
    model = HousingPriceModel()
    metrics = model.train(X_train, y_train)
    
    # Evaluar en test
    y_pred = [model.predict(X_test.iloc[i].values.tolist()) for i in range(len(X_test))]
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Registrar todas las métricas
    full_metrics = {
        'train_score': metrics['score'],
        'test_mse': float(mse),
        'test_r2': float(r2),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    print(f"📊 Métricas: {full_metrics}")
    
    # Guardar localmente
    os.makedirs('models', exist_ok=True)
    model.save('models/model.pkl')
    
    # Guardar métricas en JSON
    metrics_file = save_metrics(full_metrics, 'metrics.json')
    print(f"✅ Métricas guardadas en {metrics_file}")
    
    # Subir a S3
    s3.upload_file('models/model.pkl', Config.MODEL_PATH)
    s3.upload_file(metrics_file, f'metrics/{datetime.now().strftime("%Y%m%d")}_metrics.json')
    
    print("✅ Entrenamiento completado!")

if __name__ == "__main__":
    main()
