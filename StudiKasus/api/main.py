from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
import os

app = FastAPI(
    title="Sepsis Prediction API",
    description="AI-powered API untuk prediksi risiko sepsis pada pasien UGD",
    version="1.0.0"
)

# Load models dan preprocessors
try:
    # Load main models (2 models sesuai requirements)
    scaler = joblib.load('models/scaler.pkl')
    imputer = joblib.load('models/imputer.pkl')
    ensemble_weights = joblib.load('models/ensemble_weights.pkl')
    
    models = {
        'model_a_tree_based': joblib.load('models/model_a_tree_based_model.pkl'),
        'model_b_neural_network': joblib.load('models/model_b_neural_network_model.pkl')
    }
    
    print("Main models (2 models) loaded successfully!")
    print(f"   - Model A (Tree-based): Gradient Boosting")
    print(f"   - Model B (Neural Network): MLP Classifier") 
    print(f"   - Ensemble weights: {ensemble_weights}")
    
except Exception as e:
    print(f"Error loading models: {e}")
    models = None

class PatientData(BaseModel):
    """Model untuk input data pasien"""
    heart_rate: float
    respiratory_rate: float
    temperature: float
    wbc_count: float
    lactate_level: float = None  # Optional, could be missing
    age: float
    num_comorbidities: int

class PredictionResponse(BaseModel):
    """Model untuk response prediksi"""
    sepsis_risk: int
    risk_probability: float
    risk_level: str
    confidence: str

def preprocess_input(data: PatientData) -> np.ndarray:
    """Preprocess input data"""
    # Convert to dataframe
    df = pd.DataFrame([data.dict()])
    
    # Handle missing lactate_level
    if df['lactate_level'].isna().any():
        df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    
    # Scale features
    scaled_data = scaler.transform(df)
    
    return scaled_data

def get_ensemble_prediction(scaled_data: np.ndarray) -> tuple:
    """Get ensemble prediction"""
    # Get predictions from all models
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        pred_proba = model.predict_proba(scaled_data)[0, 1]
        predictions[name] = model.predict(scaled_data)[0]
        probabilities[name] = pred_proba
    
    # Calculate ensemble probability
    if len(models) > 1:
        # Multiple models - use weighted ensemble
        ensemble_prob = sum(
            probabilities[name] * ensemble_weights[name] 
            for name in models.keys()
        )
    else:
        # Single model - use direct probability
        ensemble_prob = list(probabilities.values())[0]
    
    ensemble_pred = 1 if ensemble_prob >= 0.5 else 0
    
    return ensemble_pred, ensemble_prob, probabilities

def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

def get_confidence(probability: float) -> str:
    """Determine confidence level"""
    confidence_score = max(probability, 1 - probability)
    if confidence_score > 0.8:
        return "High Confidence"
    elif confidence_score > 0.6:
        return "Moderate Confidence"
    else:
        return "Low Confidence"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sepsis Prediction API",
        "version": "1.0.0",
        "status": "active" if models else "models not loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if models else "unhealthy",
        "models_loaded": models is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(patient_data: PatientData):
    """
    Prediksi risiko sepsis untuk pasien
    
    Parameters:
    - heart_rate: Detak jantung (bpm)
    - respiratory_rate: Laju pernapasan (per menit)
    - temperature: Suhu tubuh (Celsius)
    - wbc_count: Jumlah sel darah putih
    - lactate_level: Kadar laktat (optional)
    - age: Usia pasien
    - num_comorbidities: Jumlah komorbiditas
    
    Returns:
    - sepsis_risk: 0 (tidak berisiko) atau 1 (berisiko)
    - risk_probability: Probabilitas risiko (0-1)
    - risk_level: Level risiko (Low/Moderate/High)
    - confidence: Tingkat kepercayaan prediksi
    """
    
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Preprocess input
        scaled_data = preprocess_input(patient_data)
        
        # Get ensemble prediction
        prediction, probability, individual_probs = get_ensemble_prediction(scaled_data)
        
        # Determine risk level and confidence
        risk_level = get_risk_level(probability)
        confidence = get_confidence(probability)
        
        return PredictionResponse(
            sepsis_risk=prediction,
            risk_probability=round(probability, 3),
            risk_level=risk_level,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(patients_data: List[PatientData]):
    """
    Prediksi batch untuk multiple pasien
    """
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if len(patients_data) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 patients per batch")
    
    try:
        results = []
        
        for patient_data in patients_data:
            # Preprocess input
            scaled_data = preprocess_input(patient_data)
            
            # Get ensemble prediction
            prediction, probability, _ = get_ensemble_prediction(scaled_data)
            
            # Determine risk level and confidence
            risk_level = get_risk_level(probability)
            confidence = get_confidence(probability)
            
            results.append({
                "sepsis_risk": prediction,
                "risk_probability": round(probability, 3),
                "risk_level": risk_level,
                "confidence": confidence
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """
    Informasi tentang model yang digunakan
    """
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "models": list(models.keys()),
        "ensemble_weights": ensemble_weights,
        "features": [
            "heart_rate", "respiratory_rate", "temperature", 
            "wbc_count", "lactate_level", "age", "num_comorbidities"
        ],
        "preprocessing": {
            "scaler": "StandardScaler",
            "imputer": "SimpleImputer (median strategy)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)