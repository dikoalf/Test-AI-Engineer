# Sepsis Prediction System
## Deskripsi Proyek

Sistem ini mengimplementasikan layanan AI yang dapat menerima data vital dan hasil lab awal dari seorang pasien di UGD, lalu memberikan prediksi risiko sepsis secara real-time. Layanan ini akan membantu dokter memprioritaskan pasien dan memulai perawatan lebih awal, yang secara signifikan dapat meningkatkan hasil klinis.

## Ringkasan Performa Model Ensemble

| Model | Cross-validation AUC | Test AUC | Precision | Recall | F1-Score |
|-------|---------------------|-----------|-----------|--------|----------|
| Model A (Gradient Boosting) | 1.0000 | 1.0000 | 1.00 | 1.00 | 1.00 |
| Model B (Neural Network) | 1.0000 | 1.0000 | 1.00 | 1.00 | 1.00 |
| Ensemble | 1.0000 | 1.0000 | 1.00 | 1.00 | 1.00 |

Sistem ensemble menggunakan soft voting untuk menggabungkan prediksi dari kedua model dengan optimasi weight menggunakan GridSearchCV.

## Instruksi Langkah-demi-Langkah

### Cara Menjalankan Program dari Awal ke Akhir

#### Opsi 1: Menjalankan Secara Lokal (Tanpa Docker)

1. Setup Environment Python
   # Pastikan Python 3.8+ terinstall
   python --version
   
   # Clone repository (atau extract dari zip)
   git clone <repository-url>
   cd Test AI Engineer\StudiKasus
   
   # Buat virtual environment (opsional tapi direkomendasikan)
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate

2. Install Dependencies
   # Install semua package yang diperlukan
   pip install -r requirements.txt

3. Train Model (Jika Model Belum Ada)
   # Jalankan script untuk melatih kedua model
   python final_two_models.py
   
   # Script ini akan:
      - Load dan preprocess data dari data/sepsis_emr_data.csv
      - Train Model A (Gradient Boosting) dan Model B (Neural Network)
      - Optimasi hyperparameter menggunakan GridSearchCV
      - Simpan model ke folder models/
      - Output performance metrics

4. Jalankan API Server
   # Masuk ke folder api
   cd api
   
   # Start FastAPI server
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   
   # Server akan berjalan di http://localhost:8000
   # API documentation tersedia di http://localhost:8000/docs

5. Test API (Buka Terminal Baru)
   # Test health check
   curl -X GET "http://localhost:8000/health"
   
   # Test single prediction
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "heart_rate": 95,
          "respiratory_rate": 22,
          "temperature": 38.5,
          "wbc_count": 12.8,
          "lactate_level": 2.1,
          "age": 65,
          "num_comorbidities": 2
        }'

#### Opsi 2: Menggunakan Docker

1. Setup Docker
   # Pastikan Docker terinstall dan berjalan
   docker --version
   
   # Clone repository
   git clone <repository-url>
   cd sepsis_prediction_project

2. Build Docker Image
   # Build image dari Dockerfile
   docker build -t sepsis-prediction .
   
   # Proses ini akan:
      - Install Python dan dependencies
      - Copy semua file project
      - Setup environment untuk production

3. Run Container
   # Jalankan container
   docker run -p 8000:8000 sepsis-prediction
   
   # Container akan:
      - Start FastAPI server otomatis
      - Expose API di port 8000
      - Load pre-trained models

4. Test API
   # Test dari host machine
   curl -X GET "http://localhost:8000/health"
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "heart_rate": 120,
          "respiratory_rate": 25,
          "temperature": 39.2,
          "wbc_count": 15.5,
          "lactate_level": 3.2,
          "age": 70,
          "num_comorbidities": 3
        }'

### Cara Membangun Image Docker dan Menjalankan Container

# Build Docker image
docker build -t sepsis-prediction .

# Run container
docker run -p 8000:8000 sepsis-prediction

### Troubleshooting Common Issues

1. Model Files Not Found
   # Jika error "model not found", jalankan training terlebih dahulu:
   python final_two_models.py

2. Port Already in Use
   # Jika port 8000 sudah digunakan, ganti port:
   uvicorn main:app --host 0.0.0.0 --port 8001
   # atau untuk Docker:
   docker run -p 8001:8000 sepsis-prediction

3. Permission Denied (Linux/Mac)
   # Tambahkan sudo jika diperlukan:
   sudo docker build -t sepsis-prediction .

### Expected Output

Saat Training Model:
Training Model A (Gradient Boosting)...
Best parameters: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}
Model A CV AUC: 1.0000
Model A Test AUC: 1.0000

Training Model B (Neural Network)...
Best parameters: {'alpha': 0.0001, 'hidden_layer_sizes': (50,)}
Model B CV AUC: 1.0000
Model B Test AUC: 1.0000

Ensemble Model Test AUC: 1.0000
Models saved successfully!

Saat API Response:
{
  "sepsis_risk_prediction": 0,
  "risk_probability": 0.05,
  "model_confidence": "high",
  "model_a_prediction": 0,
  "model_b_prediction": 0,
  "ensemble_prediction": 0
}


### Contoh Perintah Curl untuk Mengirim Permintaan ke API

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "heart_rate": 95,
       "respiratory_rate": 22,
       "temperature": 38.5,
       "wbc_count": 12.8,
       "lactate_level": 2.1,
       "age": 65,
       "num_comorbidities": 2
     }'

# Health check
curl -X GET "http://localhost:8000/health"

# Model information
curl -X GET "http://localhost:8000/model_info"

### Contoh Skrip Python untuk Mengirim Permintaan ke API

import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample patient data
patient_data = {
    "heart_rate": 95,
    "respiratory_rate": 22,
    "temperature": 38.5,
    "wbc_count": 12.8,
    "lactate_level": 2.1,
    "age": 65,
    "num_comorbidities": 2
}

# Send request
response = requests.post(url, json=patient_data)
result = response.json()

print("Prediction Result:")
print(json.dumps(result, indent=2))

# Test multiple patients
batch_data = {
    "patients": [
        {
            "heart_rate": 95,
            "respiratory_rate": 22,
            "temperature": 38.5,
            "wbc_count": 12.8,
            "lactate_level": 2.1,
            "age": 65,
            "num_comorbidities": 2
        },
        {
            "heart_rate": 110,
            "respiratory_rate": 28,
            "temperature": 39.8,
            "wbc_count": 18.5,
            "lactate_level": 4.2,
            "age": 75,
            "num_comorbidities": 4
        }
    ]
}

# Test batch prediction (if endpoint exists)
batch_response = requests.post("http://localhost:8000/predict_batch", json=batch_data)
print("\nBatch Prediction Results:")
print(json.dumps(batch_response.json(), indent=2))


## Penjelasan Singkat tentang Pilihan Desain atau Asumsi

### Model Selection Approach

#### Model A: Gradient Boosting (Tree-based)
- Justifikasi: Tree-based algorithm sesuai requirements, excellent untuk medical data dengan mixed feature types
- Kelebihan: Handle missing values secara natural, feature importance yang mudah diinterpretasi untuk clinical decision
- Architecture: GradientBoostingClassifier dengan hyperparameter tuning menggunakan GridSearchCV

#### Model B: Neural Network (ANN)  
- Justifikasi: Neural Network sesuai requirements, excellent untuk non-linear pattern recognition
- Kelebihan: Mampu capture complex interactions antar vital signs, universal function approximator untuk medical patterns
- Architecture: MLPClassifier dengan single hidden layer (50 neurons)

#### Ensemble Strategy
- Approach: Soft voting untuk menggabungkan prediksi dari kedua model
- Optimization: Weight optimization menggunakan GridSearchCV untuk optimal ensemble performance
- Benefit: Menggabungkan kekuatan dari kedua paradigma pembelajaran yang berbeda

### Data Preprocessing Assumptions
- Missing Values: Menggunakan median imputation untuk lactate_level (asumsi missing at random)
- Feature Scaling: StandardScaler untuk normalisasi fitur numerik
- Train-Test Split: 80:20 dengan stratifikasi untuk menjaga distribusi target

### API Design Decisions
- Framework: FastAPI dipilih untuk performance dan automatic documentation
- Response Format: Include individual model predictions untuk transparency
- Error Handling: Comprehensive validation untuk input data medical
- Health Check: Built-in monitoring untuk production deployment