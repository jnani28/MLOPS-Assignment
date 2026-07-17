# Bank Term Deposit Prediction - MLOps Assignment

## Problem Statement

A leading bank runs marketing campaigns to offer term deposits to clients. The goal is to **predict whether a client will subscribe to a term deposit (yes/no)** based on demographic, financial, and campaign-related features — enabling the bank to target the right customers and reduce marketing costs.

## Dataset

- **Source**: Bank Marketing Dataset (UCI Machine Learning Repository variant)
- **Size**: 4,119 rows × 21 columns
- **Target Variable**: `y` (yes/no — whether the client subscribed to a term deposit)
- **Class Imbalance**: 89% No, 11% Yes

## Model

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categorical features (full sklearn Pipeline)
- **Class Balancing**: `class_weight='balanced'` to handle the 8:1 imbalance ratio
- **Performance**:
  | Metric    | Score  |
  |-----------|--------|
  | Accuracy  | 0.8896 |
  | Precision | 0.4962 |
  | Recall    | 0.7222 |
  | F1-Score  | 0.5882 |
  | ROC-AUC   | 0.9425 |

## Project Structure

```
├── model_training.py       # Model training script (EDA + preprocessing + training + evaluation)
├── model_training.ipynb    # Jupyter notebook version of the training pipeline
├── app.py                  # FastAPI application for serving predictions
├── model.pkl               # Trained model pipeline (preprocessor + classifier)
├── Dockerfile              # Docker container configuration
├── requirements.txt        # Python dependencies for the API
├── bank-additional.csv     # Dataset
├── model_metadata.json     # Model performance metadata
└── README.md               # This file
```

---

## Docker Setup

### Prerequisites
- Docker Desktop installed and running

### Build the Docker Image

```bash
docker build -t bank-prediction-api:v1 .
```

### Run the Container Locally

```bash
docker run -d -p 8000:8000 --name bank-api bank-prediction-api:v1
```

### Verify it's Running

```bash
curl http://localhost:8000
# Response: {"status":"API is running","model":"Bank Term Deposit Prediction - RandomForest","version":"1.0.0"}
```

### Stop the Container

```bash
docker stop bank-api
docker rm bank-api
```

---

## AWS Deployment (EC2)

### Step 1: Launch EC2 Instance

1. Login to **AWS Console** → EC2 → **Launch Instance**
2. Settings:
   - **AMI**: Amazon Linux 2023 or Ubuntu 22.04
   - **Instance Type**: `t2.micro` (free tier eligible)
   - **Key Pair**: Create or select an existing key pair (`.pem` file)
   - **Security Group**: Add inbound rules:
     - SSH: Port 22 (your IP)
     - Custom TCP: Port 8000 (Anywhere 0.0.0.0/0) — for API access

### Step 2: Connect to EC2

```bash
# Set permissions for key file (Linux/macOS)
chmod 400 your-key.pem

# SSH into the instance
ssh -i your-key.pem ec2-user@<EC2-PUBLIC-IP>
# For Ubuntu AMI, use: ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 3: Install Docker on EC2

```bash
# For Amazon Linux 2023
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Log out and back in for group changes
exit
ssh -i your-key.pem ec2-user@<EC2-PUBLIC-IP>
```

### Step 4: Transfer Files to EC2

```bash
# From your local machine, copy the required files
scp -i your-key.pem app.py model.pkl requirements.txt Dockerfile ec2-user@<EC2-PUBLIC-IP>:~/
```

### Step 5: Build and Run on EC2

```bash
# On EC2 instance
docker build -t bank-prediction-api:v1 .
docker run -d -p 8000:8000 --name bank-api bank-prediction-api:v1

# Verify
curl http://localhost:8000
```

### Step 6: Access via Public IP

The API is now accessible at:
```
http://<EC2-PUBLIC-IP>:8000
```

---

## Testing the Endpoint

### Health Check

```bash
curl http://<EC2-PUBLIC-IP>:8000
```

### Sample Prediction Request (Likely NO)

```bash
curl -X POST http://<EC2-PUBLIC-IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "duration": 120,
    "campaign": 3,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.855,
    "nr.employed": 5191.0
  }'
```

**Expected Response:**
```json
{
  "prediction": "no",
  "probability_yes": 0.12,
  "probability_no": 0.88,
  "risk_level": "LOW - Unlikely to subscribe"
}
```

### Sample Prediction Request (Likely YES)

```bash
curl -X POST http://<EC2-PUBLIC-IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 60,
    "job": "retired",
    "marital": "married",
    "education": "basic.4y",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "mar",
    "day_of_week": "wed",
    "duration": 800,
    "campaign": 1,
    "pdays": 6,
    "previous": 2,
    "poutcome": "success",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.313,
    "nr.employed": 5099.1
  }'
```

### Using Postman

1. Open Postman → New Request
2. Method: **POST**
3. URL: `http://<EC2-PUBLIC-IP>:8000/predict`
4. Body → raw → JSON → Paste the JSON above
5. Click **Send**

### Swagger UI (Interactive API Docs)

Visit: `http://<EC2-PUBLIC-IP>:8000/docs`

---

## Notes

- The `duration` feature is kept in the model as this assignment focuses on the MLOps pipeline. In a real production scenario, it would be removed due to data leakage concerns.
- The model uses `class_weight='balanced'` to handle the class imbalance (8:1 ratio of No vs Yes).
- The full preprocessing pipeline (scaling + encoding) is bundled inside `model.pkl`, so the API accepts raw feature values directly.
