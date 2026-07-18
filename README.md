# Bank Term Deposit Prediction - MLOps Assignment

## Problem Summary
A bank wants to target customers who are likely to subscribe to a term deposit.
This project provides a full local MLOps workflow with data validation, model training,
MLflow tracking, API serving, Docker packaging, and monitoring hooks.

## Key Production Fixes Applied
- Removed `duration` from training and inference to prevent feature leakage.
- Added robust logging and startup checks in training, validation, and API.
- Split API runtime dependencies from full project dependencies.
- Added Prometheus-compatible metrics endpoint (`/metrics`).
- Added Docker Compose stack for API + Prometheus + Grafana.
- Added OpenLineage tracking wrapper script.
- Added CI workflow for validation, training, API smoke test, and Docker build.

## Project Structure
- `model_training.py`: Training pipeline + MLflow logging + artifact generation
- `data_validation.py`: Great Expectations validation stage
- `app.py`: FastAPI inference service (`/`, `/healthz`, `/predict`, `/metrics`)
- `openlineage_tracking.py`: Optional training wrapper with OpenLineage events
- `evidently_monitoring.py`: Data drift report generation
- `dvc.yaml`: Reproducible validation and training stages
- `Dockerfile`: API container image
- `docker-compose.yml`: API + Prometheus + Grafana stack
- `.github/workflows/ci.yml`: CI checks
- `requirements.txt`: Full project dependencies
- `requirements-api.txt`: Lean API runtime dependencies

## Local Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run Pipeline
### 1) Data Validation
```bash
python data_validation.py
```

### 2) Model Training
```bash
python model_training.py
```

Artifacts produced:
- `model.pkl`
- `model_metadata.json`
- `columns_info.json`

### 3) Start API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints
- `GET /` basic service status
- `GET /healthz` readiness with model load state
- `POST /predict` model inference
- `GET /metrics` Prometheus metrics
- `GET /docs` Swagger UI

### Sample Prediction Request
```json
{
  "age": 35,
  "job": "admin.",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "campaign": 3,
  "pdays": 999,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp.var.rate": 1.1,
  "cons.price.idx": 93.994,
  "cons.conf.idx": -36.4,
  "euribor3m": 4.855,
  "nr.employed": 5191.0
}
```

## Docker
### Build and Run API Container
```bash
docker build -t bank-prediction-api:v1 .
docker run -d -p 8000:8000 --name bank-api bank-prediction-api:v1
```

### Verify
```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/docs
```

## Monitoring Stack
```bash
docker compose up -d
```

Services:
- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001`

## Data Drift Report (Evidently)
```bash
python evidently_monitoring.py
```
Output: `reports/data_drift_report.html`

## OpenLineage (Optional)
Set lineage backend URL and run:
```bash
# Example
set OPENLINEAGE_URL=http://localhost:5000
python openlineage_tracking.py
```

## AWS EC2 Deployment Notes
- Build image on EC2 and expose port `8000`.
- Keep security group limited to trusted IPs where possible.
- Use `/healthz` for load balancer health checks.
- Persist logs and model artifacts outside ephemeral containers in production.

## CI
GitHub Actions workflow runs:
1. Dependency install
2. Data validation
3. Model training
4. API smoke tests
5. Docker build
