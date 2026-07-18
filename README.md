# 🏦 Bank Term Deposit Prediction - End-to-End MLOps Pipeline

## Project Overview

This project predicts whether a bank customer will subscribe to a term deposit using Machine Learning while demonstrating a complete production-grade MLOps workflow.

Instead of only training a model, this repository covers the complete lifecycle:

- Data Validation
- Feature Engineering
- Model Training
- Experiment Tracking (MLflow)
- REST API Deployment (FastAPI)
- Containerization (Docker)
- Monitoring (Prometheus + Grafana)
- Data Drift Detection (Evidently)
- Data Lineage (OpenLineage)
- CI/CD (GitHub Actions)
- Reproducible Pipelines (DVC)

---

# Project Architecture

```
                    Bank Dataset
                         │
                         ▼
              Great Expectations
               Data Validation
                         │
                         ▼
               Feature Engineering
                         │
                         ▼
               Model Training (RF)
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
     MLflow                       Model Artifacts
 Experiment Tracking          model.pkl + metadata
        │                                 │
        └────────────────┬────────────────┘
                         ▼
                    FastAPI Service
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Swagger UI    Prometheus      Predictions
                         │
                         ▼
                     Grafana
                         │
                         ▼
                 Evidently Reports
                         │
                         ▼
                  OpenLineage Events
```

---

# Technologies Used

| Category | Tool |
|----------|------|
| Language | Python 3.12 |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Validation | Great Expectations |
| Experiment Tracking | MLflow |
| API | FastAPI |
| API Documentation | Swagger UI |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Monitoring | Prometheus |
| Dashboard | Grafana |
| Drift Detection | Evidently AI |
| Data Lineage | OpenLineage |
| Version Control | Git |
| CI/CD | GitHub Actions |
| Pipeline | DVC |

---

# Repository Structure

```
.
├── app.py
├── model_training.py
├── predict.py
├── data_validation.py
├── feature_engineering.py
├── openlineage_tracking.py
├── evidently_monitoring.py
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
├── requirements-api.txt
├── model.pkl
├── bank.csv
├── docs/
├── monitoring/
├── tests/
├── mlruns/
├── artifacts/
└── README.md
```

---

# Features Implemented

## Data Validation

✔ Great Expectations

Validates:

- Missing values
- Column names
- Data types
- Duplicate records
- Null percentages
- Dataset integrity

Run

```bash
python data_validation.py
```

---

## Feature Engineering

Includes

- Missing value handling
- Encoding
- Scaling
- Pipeline creation

---

## Model Training

Algorithm

- Random Forest Classifier

Artifacts generated

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

## MLflow Experiment Tracking

Tracks

- Parameters
- Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Model artifacts

Start UI

```bash
mlflow ui
```

Open

```
http://localhost:5000
```

---

## FastAPI Deployment

Run

```bash
uvicorn app:app --reload
```

Available Endpoints

| Endpoint | Purpose |
|-----------|----------|
| / | Home |
| /healthz | Health Check |
| /predict | Prediction |
| /metrics | Prometheus Metrics |
| /docs | Swagger Documentation |

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

Health Check

```
http://localhost:8000/healthz
```

---

## Docker Compose

Launch complete monitoring stack

```bash
docker compose up
```

Services

| Service | URL |
|----------|-----|
| FastAPI | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

---

## Prometheus

Metrics endpoint

```
http://localhost:8000/metrics
```

Collects

- API requests
- Latency
- Error counts

---

## Grafana

Dashboard

```
http://localhost:3001
```

Can visualize

- Request rate
- API latency
- Error rate
- Custom Prometheus metrics

---

## Evidently AI

Generate Data Drift Report

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
- Metadata

Run

```bash
python openlineage_tracking.py
```

---

## GitHub Actions

Automatically performs

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

# Running the Complete Pipeline

### Step 1

Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2

Validate data

```bash
python data_validation.py
```

### Step 3

Train model

```bash
python model_training.py
```

### Step 4

View experiments

```bash
mlflow ui
```

### Step 5

Run API

```bash
uvicorn app:app --reload
```

### Step 6

Test API

```
http://localhost:8000/docs
```

### Step 7

Run Docker

```bash
docker build -t bank-prediction-api:v1 .
docker run -p 8000:8000 bank-prediction-api:v1
```

### Step 8

Launch Monitoring

```bash
docker compose up
```

---

# Future Improvements

- Kubernetes Deployment
- AWS ECS/EKS Deployment
- Model Registry
- Automated Retraining
- Feature Store
- CI/CD to AWS
- Canary Deployments

---

# Author

Anish Jain

MLOps Assignment

Indian School of Business