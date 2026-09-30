# Production-Grade MLOps Pipeline for Elderly Health Risk Monitoring

[![CI/CD Pipeline](https://github.com/user/elderly-health-risk-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/user/elderly-health-risk-mlops/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-oriented Machine Learning Operations (MLOps) system designed to assess, classify, monitor, and govern health risks among elderly populations using real-world clinical, functional, and sleep disruption indicators from the **UCI National Poll on Healthy Aging (NPHA)**.

---

## 1. Project Overview & Problem Statement

### 1.1 Problem Statement
Geriatric populations face compound, multi-system health vulnerabilities involving chronic pain, physical mobility limitations, psychological stress, and multi-factor sleep disruptions. In clinical practice, these underlying risk factors often translate into erratic and intensive emergency healthcare utilization (e.g. repeated physician and specialist visits). 

Traditional health risk scoring mechanisms are typically static, siloed, and lack automated lifecycle management. This project bridges this critical gap by engineering a **production-grade, reproducible MLOps pipeline** that automates the complete lifecycle: from standardized survey data ingestion, strict schema validation, and leakage-free feature engineering, to model training, experiment tracking with MLflow, automated model versioning, FastAPI containerized microservices, live inference telemetry logging, and statistical data drift detection.

### 1.2 Core Objectives
1. **End-to-End MLOps Lifecycle:** Implement every phase from raw data ingestion to continuous monitoring and CI/CD automation without manual intervention.
2. **Zero Data Leakage:** Ensure training split isolation for imputation, scaling, and feature engineering, serialized as reproducible pipeline artifacts.
3. **Rigorous Experiment Tracking & Versioning:** Track model parameters, diagnostic metrics (Accuracy, F1-Score, ROC-AUC), and visual artifacts across multiple experiment runs with MLflow, registering the champion model in the local model registry.
4. **Resilient Production API:** Deploy a high-throughput, low-latency REST API using FastAPI with strict Pydantic payload validation and async lifespan management.
5. **Statistical Observability:** Monitor live production requests, measure response latency, and detect population data drift using the **Population Stability Index (PSI)**.
6. **Academic Reproducibility & Low Footprint:** Run 100% locally on standard student/developer hardware using open-source tools with zero paid cloud dependencies.

---

## 2. End-to-End MLOps Architecture

```
                                  +-------------------------------------+
                                  |   UCI NPHA Raw Survey Dataset       |
                                  |   (714 Senior Survey Records, 14 Feat)|
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  | Phase 1: Data Ingestion & Lineage   |
                                  | SHA256 Checksum | Metadata JSON     |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  | Phase 2: Schema & Data Validation   |
                                  | Range Checks | Refusal Code Det.    |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  | Stratified Split (80 Train / 20 Test)|
                                  +-------------------------------------+
                                                     |
                                                     v
               +---------------------------------------------------------------------------+
               | Phase 2: Feature Engineering & Preprocessor (Fit on Train ONLY)           |
               | - Refusal Code Median Imputation                                          |
               | - Sleep Disturbance Composite Score (Stress, Meds, Pain, Nocturia)        |
               | - Health Deficit Composite Score (Physical + Mental + Dental)             |
               | - Standard Scaler                                                         |
               | Artifact: models/preprocessor.joblib                                      |
               +---------------------------------------------------------------------------+
                                                     |
                                                     v
               +---------------------------------------------------------------------------+
               | Phase 3 & 4: Model Training & MLflow Experiment Tracking                  |
               | - Baseline: Logistic Regression (L2 Balanced)                             |
               | - Baseline: Decision Tree Classifier (Depth 4)                            |
               | - Candidate: Random Forest Shallow (Depth 4, 50 Trees)                    |
               | - Champion: Random Forest Tuned (Depth 6, 120 Trees, Balanced Weights)    |
               | Artifacts: metrics.json, confusion_matrix.png, feature_importance.png    |
               +---------------------------------------------------------------------------+
                                                     |
                                                     v
               +---------------------------------------------------------------------------+
               | Phase 5: Model Registry & Governance                                      |
               | Registered Model: 'ElderlyHealthRiskClassifier' (Active Version: v2)      |
               | Status: Production | Artifact: models/health_risk_model.joblib            |
               +---------------------------------------------------------------------------+
                                                     |
                                                     v
               +---------------------------------------------------------------------------+
               | Phase 6: REST API Microservice (FastAPI + Pydantic + Uvicorn)             |
               | Endpoints: GET /, GET /health, GET /model-info, POST /predict             |
               | Containerized via Dockerfile & docker-compose.yml                         |
               +---------------------------------------------------------------------------+
                                                     |
                                                     v
               +---------------------------------------------------------------------------+
               | Phase 7: Observability, Telemetry & Data Drift Monitoring                 |
               | - Structured Request Logging (data/monitoring/api_requests.jsonl)         |
               | - Population Stability Index (PSI) Drift Engine (Threshold: 0.10 / 0.25)  |
               | - Automatic Retraining Alerts                                             |
               +---------------------------------------------------------------------------+
                                                     |
                                                     v
               +---------------------------------------------------------------------------+
               | Phase 8 & 9: CI/CD Automation & Quality Assurance                         |
               | - GitHub Actions Automated Workflow (Lint, Test, Docker Build)           |
               | - Pytest 29-Test Unit & Integration Suite (100% Pass)                     |
               +---------------------------------------------------------------------------+
```

---

## 3. Technology Stack

| Layer | Technologies Used | Rationale |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11 / 3.14 | Ubiquitous ML ecosystem and native type hinting |
| **Data Engineering** | Pandas, NumPy, ucimlrepo | Fast tabular processing, numerical operations, official UCI data fetch |
| **Machine Learning** | Scikit-learn, Joblib | Industry-standard Random Forest, custom pipelines, artifact serialization |
| **Experiment Tracking** | MLflow | Local SQLite-backed run comparison, parameter/metric logging, artifact store |
| **Model Registry** | MLflow Model Registry | Production versioning, lifecycle transitions (Staging -> Production) |
| **API Framework** | FastAPI, Pydantic, Uvicorn | High-performance async ASGI server with schema validation & OpenAPI docs |
| **Monitoring & Drift** | Custom PSI Engine, JSONL | Lightweight statistical monitoring (Population Stability Index), no heavy DB |
| **Containerization** | Docker, Docker Compose | Reproducible environments, microservice isolation |
| **CI/CD** | GitHub Actions | Automated build, schema verification, pytest execution, container build |
| **Testing** | Pytest, TestClient (HTTPX) | Comprehensive unit, data integrity, integration, and API endpoint testing |

---

## 4. Dataset Description & Lineage

The system utilizes the **National Poll on Healthy Aging (NPHA)** senior health risk and healthcare utilization dataset, curated by the **University of Michigan Institute for Healthcare Policy and Innovation (IHPI)** and sponsored by **AARP** and **Michigan Medicine**.

- **Source:** UCI Machine Learning Repository (Dataset ID: `936`)
- **DOI:** `10.3886/ICPSR37305.v1`
- **Total Records:** 714 survey respondents (seniors aged 50–80+)
- **Total Features:** 14 clinical and lifestyle predictors
- **Target Variable:** `Number_of_Doctors_Visited` mapped to `Health_Risk_Level`
  - **Class 0 (Low Risk):** 0–1 doctor visits (131 respondents)
  - **Class 1 (Moderate Risk):** 2–3 doctor visits (372 respondents)
  - **Class 2 (High Risk):** 4+ doctor visits (211 respondents)
- **Data Versioning:** SHA256 Checksum: `dec78ee6357fec20ef0035bd197e9a8b7724eb20e4497b1e7dd60d95c67577a8` stored in `data/raw/dataset_metadata.json`.

### Features Breakdown
1. `Age`: Age group bracket (1: 50–64, 2: 65–80)
2. `Physical_Health`: Self-reported physical status (1: Excellent to 5: Poor, -1: Refused)
3. `Mental_Health`: Self-reported psychological status (1: Excellent to 5: Poor, -1: Refused)
4. `Dental_Health`: Oral health status (1: Excellent to 5: Poor, 6: Dentures, -1: Refused)
5. `Employment`: Employment status (1: Full-time, 2: Part-time, 3: Retired, 4: Not working)
6. `Stress_Keeps_Patient_from_Sleeping`: Sleep disrupted by anxiety/stress (0: No, 1: Yes)
7. `Medication_Keeps_Patient_from_Sleeping`: Sleep disrupted by meds (0: No, 1: Yes)
8. `Pain_Keeps_Patient_from_Sleeping`: Sleep disrupted by chronic pain (0: No, 1: Yes)
9. `Bathroom_Needs_Keeps_Patient_from_Sleeping`: Sleep disrupted by nocturia (0: No, 1: Yes)
10. `Uknown_Keeps_Patient_from_Sleeping`: Other sleep disruptors (0: No, 1: Yes)
11. `Trouble_Sleeping`: Frequency of sleep difficulty (0: None to 3: Severe)
12. `Prescription_Sleep_Medication`: Sleep aid intake (1: Regular, 2: Occasional, 3: None)
13. `Race`: Demographic race/ethnicity code (1 to 5)
14. `Gender`: Gender identity (1: Male, 2: Female)

### Engineered Features (Geriatric Domain Informed)
- `Sleep_Disturbance_Score`: Cumulative score of binary sleep disruption indicators (0 to 5+).
- `Health_Deficit_Score`: Sum of physical, mental, and dental health self-assessments (3 to 15).
- `High_Pain_Flag`: Binary marker indicating pain-induced sleep disruption.

---

## 5. Directory Structure

```
elderly-health-risk-mlops/
├── data/
│   ├── raw/                        # Ingested immutable raw survey CSV & metadata
│   │   ├── npha_elderly_health_risk.csv
│   │   └── dataset_metadata.json
│   ├── interim/                    # Cleaned dataset with survey refusal mapping
│   │   └── cleaned_health_risk.csv
│   ├── processed/                  # Leakage-free train/test splits
│   │   ├── train.csv
│   │   └── test.csv
│   └── monitoring/                 # Live inference logs and drift reports
│       ├── api_requests.jsonl
│       └── drift_report.json
├── models/                         # Serialized pipeline artifacts & figures
│   ├── health_risk_model.joblib    # Production champion Random Forest model
│   ├── preprocessor.joblib         # Fitted scikit-learn preprocessor pipeline
│   ├── model_registry.json         # Lightweight registry version metadata
│   ├── evaluation_metrics.json     # Empirical metrics for all model runs
│   ├── confusion_matrix.png        # Confusion matrix diagnostic heatmap
│   ├── feature_importance.png      # Gini feature importance bar chart
│   └── model_comparison.png        # Baseline vs Champion model comparison chart
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py          # Dataset download, hash validation, inspection
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── validator.py            # Schema, duplicate & out-of-domain validation
│   │   └── preprocessor.py         # Leakage-free transformation pipeline
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py                # Model training, MLflow tracking & registry
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py             # Metrics calculation & visual generation
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI inference microservice
│   │   └── schemas.py              # Pydantic request/response schemas
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── logger.py               # Asynchronous inference telemetry logger
│   │   └── drift_detector.py       # Population Stability Index (PSI) detector
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # YAML configuration loader & path resolver
│       └── logger.py               # Standardized logging formatter
├── tests/
│   ├── __init__.py
│   ├── test_data.py                # Raw dataset & metadata integrity tests (4 tests)
│   ├── test_preprocessing.py       # Preprocessing & schema validation tests (6 tests)
│   ├── test_model.py               # Training, evaluation & registry tests (6 tests)
│   ├── test_api.py                 # FastAPI endpoints & error tests (8 tests)
│   └── test_monitoring.py          # PSI drift calculation & telemetry tests (5 tests)
├── notebooks/                      # Exploratory Data Analysis & visual experiments
├── configs/
│   └── config.yaml                 # Central project configuration parameters
├── mlruns/                         # Local MLflow SQLite tracking database & artifacts
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI/CD automation workflow
├── Dockerfile                      # Production container definition
├── docker-compose.yml              # Service orchestration (API + MLflow UI)
├── .dockerignore
├── .gitignore
├── requirements.txt                # Pinned production Python dependencies
├── run_pipeline.py                 # Master CLI pipeline orchestrator
├── viva_preparation.md             # Comprehensive faculty viva questions & answers
└── presentation/
    └── first_review_presentation.md# 20-Slide College First Review PPT Outline
```

---

## 6. Installation & Environment Setup

### 6.1 Prerequisites
- Python 3.11, 3.12, or 3.14
- Git
- (Optional) Docker & Docker Compose

### 6.2 Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/user/elderly-health-risk-mlops.git
cd elderly-health-risk-mlops

# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Or on Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Execution Guide

### 7.1 Running the Complete Master Pipeline (Single Command)
To execute the complete end-to-end pipeline (Data Ingestion -> Preprocessing -> Model Training -> MLflow Logging -> Model Registry -> Telemetry Simulation -> Drift Detection):
```bash
python run_pipeline.py --all
```

### 7.2 Running Individual Stages

#### Phase 1: Ingest and Validate Raw Data
```bash
python -m src.data.data_loader
```

#### Phase 2: Run Data Validation & Feature Engineering
```bash
python -m src.preprocessing.validator
python -m src.preprocessing.preprocessor
```

#### Phase 3, 4 & 5: Train Models, Track Experiments & Register Champion
```bash
python -m src.training.train
```

#### Launch MLflow UI to Inspect Experiments & Model Registry
```bash
# Run local MLflow UI pointing to SQLite database:
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```
Open your browser at `http://127.0.0.1:5000` to visualize run comparisons, loss curves, confusion matrices, and the registered champion model.

---

## 8. REST API Deployment (FastAPI)

### 8.1 Launching the API Locally
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Once started, access:
- **Interactive Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

### 8.2 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API discovery, status, and endpoint directory |
| `GET` | `/health` | Liveness & readiness check (model loaded status, active version) |
| `GET` | `/model-info` | Metadata of currently active production model from registry |
| `POST` | `/predict` | Single patient health risk prediction (returns tier, confidence, probabilities) |
| `POST` | `/batch-predict` | Batch predictions for multiple senior patients |
| `GET` | `/monitoring/metrics` | Operational telemetry (request counts, latency, class breakdown) |
| `GET` | `/monitoring/drift` | Real-time Population Stability Index (PSI) drift report |

### 8.3 Example Request & Response

#### Single Prediction Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 2,
       "Physical_Health": 4,
       "Mental_Health": 3,
       "Dental_Health": 4,
       "Employment": 4,
       "Stress_Keeps_Patient_from_Sleeping": 1,
       "Medication_Keeps_Patient_from_Sleeping": 0,
       "Pain_Keeps_Patient_from_Sleeping": 1,
       "Bathroom_Needs_Keeps_Patient_from_Sleeping": 1,
       "Uknown_Keeps_Patient_from_Sleeping": 0,
       "Trouble_Sleeping": 2,
       "Prescription_Sleep_Medication": 2,
       "Race": 1,
       "Gender": 2
     }'
```

#### JSON Response:
```json
{
  "predicted_class": 2,
  "risk_tier": "High Risk",
  "confidence": 0.5218,
  "class_probabilities": {
    "Low Risk": 0.1245,
    "Moderate Risk": 0.3537,
    "High Risk": 0.5218
  },
  "model_name": "ElderlyHealthRiskClassifier",
  "model_version": "v2",
  "inference_latency_ms": 14.85,
  "timestamp_utc": "2026-09-30T09:51:30.123456+00:00"
}
```

---

## 9. Actual Empirical Results

> [!NOTE]
> All results below are authentic, reproducible empirical outputs obtained by running the pipeline on the real UCI NPHA test split (143 unseen senior instances). No values are fabricated.

### 9.1 Model Benchmark Comparison

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1: Logistic Regression** | 38.46% | 0.3912 | 0.3886 | 0.3886 | 0.3896 | Archived Baseline |
| **Baseline 2: Decision Tree (Depth 4)** | 22.38% | 0.2871 | 0.2541 | 0.2002 | 0.1462 | Underfitting Baseline |
| **Candidate: Random Forest v1 (Shallow)** | 41.26% | 0.4289 | 0.4184 | 0.4158 | 0.4044 | Candidate Run |
| **Champion: Random Forest v2 (Tuned)** | **42.66%** | **0.4351** | **0.4280** | **0.4240** | **0.4337** | **Active Production Champion** |

### 9.2 Key Diagnostic Insights
- **Why Random Forest Outperforms Baselines:** Elderly survey responses exhibit non-linear interactions (e.g. chronic pain interacting with sleep disruption and age bracket). Logistic regression assumes linear decision boundaries, while a single decision tree overfits discrete splits. Random Forest averages decorrelated decision trees, mitigating variance and producing well-calibrated class probabilities.
- **Top 5 Predictive Features (Gini Importance):**
  1. `Health_Deficit_Score` (Engineered composite of Physical + Mental + Dental self-assessments)
  2. `Sleep_Disturbance_Score` (Cumulative count of sleep disruptions)
  3. `Physical_Health` (Direct subjective physical status)
  4. `Mental_Health` (Psychological resilience)
  5. `Dental_Health` (Oral health status, an established biomarker of nutritional and functional decline)

---

## 10. Monitoring & Data Drift Engine

### 10.1 Telemetry Logging
Every inference call is tracked in `data/monitoring/api_requests.jsonl` with:
- Request timestamp (UTC)
- Response latency in milliseconds
- Sanitized input feature vector
- Predicted risk tier and confidence score
- Execution status (`SUCCESS`, `VALIDATION_FAILED`, `INFERENCE_ERROR`)

### 10.2 Population Stability Index (PSI)
Data drift evaluates incoming production distributions against the training baseline reference using PSI:

$$\text{PSI} = \sum_{i=1}^{k} (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right)$$

- **PSI < 0.10:** **STABLE** (No significant shift; model operating normally).
- **0.10 <= PSI < 0.25:** **WARNING** (Moderate distribution shift; monitor outputs).
- **PSI >= 0.25:** **DRIFT_DETECTED** (Action required; pipeline automatically flags for model retraining).

Inspect drift in real time via `GET /monitoring/drift` or `python -m src.monitoring.drift_detector`.

---

## 11. Automated Testing Suite

The project includes a comprehensive 29-test suite implemented with `pytest` covering all lifecycle components:

```bash
python -m pytest tests/ -v
```

### Test Coverage Summary:
- **`tests/test_data.py` (4 tests):** Raw dataset presence, metadata structure, checksum validity, absence of raw nulls.
- **`tests/test_preprocessing.py` (6 tests):** Schema validation, missing column detection, out-of-domain value rejection, single input validation, transformer artifact serialization.
- **`tests/test_model.py` (6 tests):** Model artifact presence, evaluation metrics calculation, diagnostic plots, inference probability validation, MLflow database runs verification, registry metadata integrity.
- **`tests/test_api.py` (8 tests):** Root discovery, `/health` readiness check, `/model-info`, valid single prediction, high-risk patient prediction, invalid payload rejection (422), missing feature rejection (422), batch prediction.
- **`tests/test_monitoring.py` (5 tests):** PSI calculation on identical distributions, PSI on shifted distributions, telemetry logging, simulated drift alerting, FastAPI monitoring endpoints.

---

## 12. Containerization (Docker & Compose)

### 12.1 Building and Running via Docker
```bash
# Build the Docker image
docker build -t elderly-health-risk-api:latest .

# Run the container
docker run -p 8000:8000 --name elderly_api elderly-health-risk-api:latest
```

### 12.2 Orchestration with Docker Compose
To launch both the FastAPI prediction service and the MLflow UI together:
```bash
docker-compose up --build
```
- API available at: `http://localhost:8000`
- MLflow UI available at: `http://localhost:5000`

---

## 13. CI/CD Automation (GitHub Actions)

Located at `.github/workflows/ci.yml`, the pipeline runs on every push and pull request to `main`:
1. **Job 1: Code & Data Schema Validation:**
   - Installs dependencies on `ubuntu-latest`.
   - Runs data ingestion and checks dataset schema constraints.
2. **Job 2: Unit & Integration Test Suite:**
   - Runs feature engineering and model training.
   - Executes the 29-test Pytest suite and asserts artifact generation (`health_risk_model.joblib`, `preprocessor.joblib`, `model_registry.json`).
3. **Job 3: Docker Image Build & Test:**
   - Verifies the Dockerfile builds cleanly on standard runners.

---

## 14. Academic Limitations & Future Improvements

### Current Limitations:
1. **Dataset Volume:** The NPHA dataset comprises 714 respondents. While ideal for rapid local training on student laptops, larger multi-center clinical trials would allow deeper architectures.
2. **Subjective Survey Inputs:** Health indicators are self-assessed; continuous sensor streams (e.g. smart watch heart rate, accelerometer actigraphy) would complement survey inputs.

### Future Enhancements:
1. **IoT Edge Deployment:** Deploying the quantized preprocessor and model onto Raspberry Pi / ESP32 wearable edge devices.
2. **Continuous Active Learning:** Automatically queuing high-uncertainty predictions (`confidence < 0.40`) for clinical expert annotation and automated retraining triggers.
3. **SHAP Explainability Integration:** Serving per-patient feature contribution explanations directly in the API response.

---

## 15. Citation & Acknowledgments
- **Dataset:** Malani, P. N., Kullgren, J., & Solway, E. (2017). *National Poll on Healthy Aging (NPHA)*. University of Michigan / AARP / Michigan Medicine. Distributed by the UCI Machine Learning Repository (DOI: 10.3886/ICPSR37305.v1).
- **Architecture:** Implemented adhering to production MLOps standards for academic and industrial machine learning lifecycles.
