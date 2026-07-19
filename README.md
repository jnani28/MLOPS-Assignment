# 🏦 Bank Term Deposit Prediction – End-to-End MLOps Pipeline

## Author
**Anish Jain**

**Course:** Indian School of Business (ISB)

**Assignment:** End-to-End MLOps Pipeline with Docker & AWS Deployment

---

# Project Overview

Banks spend significant resources contacting customers during marketing campaigns. Many customers contacted never subscribe to a term deposit, resulting in unnecessary operational costs.

This project develops an end-to-end MLOps pipeline that predicts whether a customer will subscribe to a term deposit while demonstrating industry-standard machine learning deployment practices.

The project intentionally separates **Model Development (Training Environment)** from **Model Serving (Production Environment)** to simulate a real-world MLOps workflow.

---

# Training Environment (Local Development)

The local environment is responsible for building, validating, tracking, and packaging the machine learning model.

```
                 Raw Bank Dataset
                        │
                        ▼
           Great Expectations
            Data Validation
                        │
                        ▼
            Feature Engineering
                        │
                        ▼
      Random Forest Model Training
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
     MLflow Tracking            Model Artifacts
                              (model.pkl, metadata)
          │                           │
          └─────────────┬─────────────┘
                        ▼
                 OpenLineage Events
```

### Components

| Component | Technology |
|------------|------------|
| Programming Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Validation | Great Expectations |
| Experiment Tracking | MLflow |
| Pipeline Versioning | DVC |
| Data Lineage | OpenLineage |
| Drift Detection | Evidently AI |
| Version Control | Git + GitHub |

---

# Production Environment (AWS)

Only the inference service is deployed to AWS.

The training pipeline remains local, while AWS hosts the prediction API.

```
                 GitHub Repository
                        │
                        ▼
                   AWS EC2 Instance
                        │
                        ▼
                 Docker Container
                        │
                        ▼
                  FastAPI Application
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
   Swagger UI      Health Check      Predictions
     (/docs)         (/healthz)       (/predict)
```

### Production Stack

| Component | Technology |
|------------|------------|
| Cloud Platform | AWS EC2 |
| Containerization | Docker |
| REST API | FastAPI |
| API Documentation | Swagger UI |
| Health Monitoring | Health Endpoint |

---

# Complete Solution Architecture

```
                      LOCAL DEVELOPMENT

                Raw Bank Dataset
                        │
                        ▼
            Great Expectations Validation
                        │
                        ▼
              Feature Engineering
                        │
                        ▼
         Random Forest Model Training
                        │
                        ▼
            MLflow Experiment Tracking
                        │
                        ▼
        model.pkl + metadata + schema
                        │
                        ▼
              Push to GitHub Repository

────────────────────────────────────────────────────────────

                      PRODUCTION (AWS)

                 GitHub Repository
                        │
                        ▼
                  Amazon EC2 Instance
                        │
                        ▼
                 Docker Container
                        │
                        ▼
                  FastAPI Application
                        │
      ┌─────────────────┼─────────────────┐
      ▼                 ▼                 ▼
  /healthz          /predict          Swagger UI
                                       /docs
```

---

# Repository Structure

```
.
├── app.py
├── model_training.py
├── feature_engineering.py
├── data_validation.py
├── predict.py
├── openlineage_tracking.py
├── evidently_monitoring.py
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
├── requirements-api.txt
├── model.pkl
├── model_metadata.json
├── columns_info.json
├── bank.csv
├── artifacts/
├── monitoring/
├── reports/
├── tests/
├── docs/
├── mlruns/
└── README.md
```

---

# MLOps Components

## Data Validation

Implemented using **Great Expectations**.

Validation checks include:

- Missing values
- Duplicate records
- Schema validation
- Column names
- Data types
- Dataset integrity

Run

```bash
python data_validation.py
```

---

## Feature Engineering

Pipeline performs

- Missing value handling
- Feature encoding
- Scaling
- Pipeline creation

---

## Model Training

Algorithm used

- Random Forest Classifier

Generated artifacts

```
model.pkl
model_metadata.json
columns_info.json
```

Run

```bash
python model_training.py
```

---

## MLflow

Tracks

- Parameters
- Accuracy
- Precision
- Recall
- F1 Score
- Model artifacts

Launch

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

---

## FastAPI

Launch

```bash
uvicorn app:app --reload
```

Available Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | / | Home |
| GET | /healthz | Health Check |
| POST | /predict | Prediction |
| GET | /metrics | Prometheus Metrics |
| GET | /docs | Swagger Documentation |

Swagger

```
http://localhost:8000/docs
```

---

## Docker

Build

```bash
docker build -t bank-prediction-api:v1 .
```

Run

```bash
docker run -d -p 8000:8000 bank-prediction-api:v1
```

---

## AWS Deployment

Deployment steps

1. Launch Amazon Linux EC2
2. Install Docker
3. Clone GitHub repository
4. Build Docker image
5. Run Docker container
6. Access FastAPI

Example

```
http://<EC2-PUBLIC-IP>:8000/docs
```

---

## Docker Compose

Launch Monitoring Stack

```bash
docker compose up
```

Available Services

| Service | URL |
|----------|-----|
| FastAPI | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

---

## Prometheus

Collects

- API Requests
- Response Latency
- Error Counts

Metrics Endpoint

```
http://localhost:8000/metrics
```

---

## Grafana

Dashboard

```
http://localhost:3001
```

Visualizes

- Request Rate
- API Latency
- Error Rate
- Prometheus Metrics

---

## Evidently AI

Generate Drift Report

```bash
python evidently_monitoring.py
```

Output

```
reports/data_drift_report.html
```

---

## OpenLineage

Tracks

- Pipeline execution
- Dataset lineage
- Model lineage

Run

```bash
python openlineage_tracking.py
```

---

## GitHub Actions

CI Pipeline automatically performs

- Dependency installation
- Data validation
- Model training
- API smoke testing
- Docker build

Workflow

```
.github/workflows/ci.yml
```

---

# Running the Project

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Validate Data

```bash
python data_validation.py
```

## 3. Train Model

```bash
python model_training.py
```

## 4. Launch MLflow

```bash
mlflow ui
```

## 5. Start FastAPI

```bash
uvicorn app:app --reload
```

## 6. Build Docker

```bash
docker build -t bank-prediction-api:v1 .
```

## 7. Run Docker

```bash
docker run -p 8000:8000 bank-prediction-api:v1
```

## 8. Open Swagger

```
http://localhost:8000/docs
```

---

# Sample Prediction Request

```json
{
  "age": 35,
  "job": "management",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "balance": 2500,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "may",
  "duration": 180,
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

---

# Sample Prediction Response

```json
{
  "prediction": "yes",
  "probability_yes": 0.91,
  "probability_no": 0.09,
  "risk_level": "HIGH - Very likely to subscribe"
}
```

---

# Assignment Deliverables

✔ Machine Learning Model

✔ Data Validation

✔ Feature Engineering

✔ MLflow Experiment Tracking

✔ Docker Containerization

✔ FastAPI REST API

✔ Swagger Documentation

✔ AWS EC2 Deployment

✔ Health Check Endpoint

✔ Real-Time Inference

✔ Prometheus Monitoring

✔ Grafana Dashboard

✔ Evidently Drift Detection

✔ OpenLineage Integration

✔ GitHub Actions CI

---

# Future Enhancements

- Kubernetes Deployment
- AWS ECS/EKS Deployment
- MLflow Model Registry
- Automated Model Retraining
- Feature Store Integration
- CI/CD Deployment to AWS
- Canary Model Deployment

---

# Author

**Anish Jain**

Indian School of Business

End-to-End MLOps Assignment