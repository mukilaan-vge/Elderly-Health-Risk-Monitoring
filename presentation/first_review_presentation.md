# FIRST REVIEW PRESENTATION: PRODUCTION-GRADE MLOps PIPELINE FOR ELDERLY HEALTH RISK MONITORING

**Course:** Final Year Capstone Project / M.Tech / B.Tech Computer Science & Data Science  
**Project Domain:** Healthcare Machine Learning Operations (MLOps)  
**Presented by:** Project Team  
**Guided by:** Project Supervisor  

---

## Slide 1: Title Slide
- **Project Title:** Production-Grade MLOps Pipeline for Elderly Health Risk Monitoring
- **Subtitle:** An End-to-End, Reproducible Machine Learning System for Senior Health Risk Classification and Drift Observability
- **Domain:** Healthcare Informatics, Machine Learning Engineering, DevOps & MLOps
- **Technology Stack:** Python, Scikit-learn, MLflow, FastAPI, Pydantic, Docker, GitHub Actions, Pytest

---

## Slide 2: Problem Statement
- **Geriatric Health Challenges:** Older adults (aged 50–80+) face multi-system vulnerabilities including chronic pain, multi-factor sleep disruption, oral hygiene decline, and psychological stress.
- **Uncoordinated Clinical Utilization:** These compound vulnerabilities lead to erratic healthcare utilization (frequent unplanned primary care visits, specialist escalations, and emergency room visits).
- **The MLOps Gap:** Most existing healthcare ML systems remain prototype Jupyter notebooks with:
  - Manual, error-prone data preparation.
  - Undetected data leakage between train and test sets.
  - Zero model versioning or experiment tracking.
  - Lack of operational monitoring for post-deployment data drift.
- **Core Need:** An automated, governed, reproducible MLOps lifecycle to assess senior health risk in real time.

---

## Slide 3: Motivation
- **Aging Population Demographics:** By 2030, 1 in 6 people globally will be aged 60 years or older (WHO statistics).
- **Proactive vs Reactive Care:** Moving from reactive emergency visits to proactive risk tier classification (Low, Moderate, High Risk) allows targeted community interventions.
- **Academic & Industry Relevance:** Demonstrating true production MLOps practices—not just training an algorithm, but packaging, testing, monitoring, and governing it continuously.

---

## Slide 4: Proposed Solution
- **System Concept:** An end-to-end automated MLOps pipeline for elderly health risk classification.
- **Integrated Pillars:**
  1. **Data Governance:** Automated UCI NPHA survey ingestion, SHA256 data lineage, and schema enforcement.
  2. **Leakage-Free Preprocessing:** Statistical imputation and geriatric composite feature engineering (`Sleep_Disturbance_Score`, `Health_Deficit_Score`).
  3. **Experiment Tracking & Registry:** Local MLflow SQLite tracking with model versioning and production staging.
  4. **Microservice Deployment:** Async FastAPI REST service with Pydantic payload validation.
  5. **Observability:** Telemetry logging and Population Stability Index (PSI) data drift detection.
  6. **Automated CI/CD:** GitHub Actions workflow executing schema validation, pytest suite, and container build.

---

## Slide 5: Objectives
1. Ingest and manage the authentic UCI National Poll on Healthy Aging (NPHA) dataset with cryptographic SHA256 data versioning.
2. Build a domain-informed, leakage-free scikit-learn preprocessing and feature engineering transformer.
3. Train and benchmark a Random Forest Classifier against Logistic Regression and Decision Tree baselines.
4. Track all hyperparameters, metrics, and visual artifacts using MLflow.
5. Register and version the champion model in the MLflow Model Registry.
6. Deploy a high-performance REST API using FastAPI and Docker.
7. Monitor incoming requests and detect population data drift using Population Stability Index (PSI).
8. Automate testing and deployment using Pytest (29 tests) and GitHub Actions.

---

## Slide 6: Literature & Existing Systems Review

| Author / Study | Focus | Limitations in Existing Work | How Our Project Overcomes It |
| :--- | :--- | :--- | :--- |
| Malani et al. (Univ. of Michigan / AARP, 2017) | NPHA Survey Analysis | Static descriptive statistics; no predictive ML modeling or automated pipelines | Implements predictive risk classification with automated inference |
| Kaggle / Academic Health Risk Kernels | Tabular Health Modeling | Single-script notebook architectures, hardcoded splits, rampant data leakage | Modular package structure, fitted transformers serialized without leakage |
| Traditional Clinical Risk Calculators (e.g. Charlson) | Comorbidity Index | Manual paper scores, ignores complex sleep disruptions and dental indicators | Automates multi-factor scoring including composite sleep disturbance |
| Commercial MLOps Platforms (AWS SageMaker / Databricks) | Enterprise Cloud | Expensive subscription costs, opaque managed infrastructure | 100% open-source, local SQLite/JSONL implementation runnable on any laptop |

---

## Slide 7: Proposed MLOps Architecture

```
+-----------------------------------------------------------------------------------+
|                            DATA MANAGEMENT & VALIDATION                           |
|  UCI Repository -> raw/ -> SHA256 Hash -> Schema & Range Validation -> interim/   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        PREPROCESSING & FEATURE ENGINEERING                        |
|  Train/Test Stratified Split (80/20) -> Median Imputer -> Geriatric Features      |
|  -> Scaler -> Artifact: models/preprocessor.joblib                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       MODEL TRAINING & MLFLOW TRACKING                            |
|  Logistic Regression | Decision Tree | Random Forest Shallow | Champion RF Tuned  |
|  Metrics: Accuracy, Macro F1, Weighted F1, ROC-AUC -> Local SQLite MLflow DB      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             MODEL REGISTRY & SERVING                              |
|  Registered Model: 'ElderlyHealthRiskClassifier' (v2 Production)                  |
|  FastAPI Service (Uvicorn ASGI) -> Endpoints: /health, /predict, /batch-predict   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                          OBSERVABILITY & DRIFT ENGINE                             |
|  Request Logging -> Telemetry (api_requests.jsonl) -> PSI Drift Detection Engine  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                                 CI/CD AUTOMATION                                  |
|  GitHub Actions -> Python Setup -> Validator -> Pytest (29 Tests) -> Docker Build |
+-----------------------------------------------------------------------------------+
```

---

## Slide 8: Dataset Description (UCI NPHA)
- **Source:** University of Michigan / AARP / Michigan Medicine
- **UCI ID:** `936` | **DOI:** `10.3886/ICPSR37305.v1`
- **Volume:** 714 senior respondents (aged 50–80+)
- **Attributes:** 14 features spanning physical health, mental health, dental health, employment, and comprehensive sleep disruptions (stress, medication, pain, nocturia, prescription sleep aids).
- **Target Variable:** `Number_of_Doctors_Visited` representing clinical healthcare utilization:
  - **Class 0 (Low Risk):** 0–1 visits (131 records, 18.3%)
  - **Class 1 (Moderate Risk):** 2–3 visits (372 records, 52.1%)
  - **Class 2 (High Risk):** 4+ visits (211 records, 29.6%)
- **Ethical Integrity:** De-identified public research survey data without personal identifiable information (PII).

---

## Slide 9: Data Management & Lineage
- **Clean Three-Tier Data Hierarchy:**
  - `data/raw/`: Immutable source CSV and metadata JSON.
  - `data/interim/`: Negative survey refusal codes mapped for imputation.
  - `data/processed/`: Train and test datasets ready for modeling.
- **Cryptographic Lineage:** SHA256 Checksum (`dec78ee6357fec20ef0035bd197e9a8b7724eb20e4497b1e7dd60d95c67577a8`) verified automatically at every ingestion run.
- **Reproducible Loader:** Python module `src.data.data_loader` handles downloading, checking cache, and extracting structural metadata.

---

## Slide 10: Data Validation & Preprocessing Pipeline
- **Automated Validation Checks:**
  1. Feature column existence check.
  2. Data type compliance (integer/float encoding).
  3. Duplicate record detection (flags natural survey overlaps).
  4. Out-of-bounds detection against clinical survey codebooks.
  5. Negative refusal code detection (`-1`, `-2`).
- **Feature Engineering:**
  - `Sleep_Disturbance_Score` = Stress + Meds + Pain + Bathroom Needs + Trouble Sleeping.
  - `Health_Deficit_Score` = Physical + Mental + Dental health scores.
  - `High_Pain_Flag` = Chronic pain affecting sleep.
- **No Data Leakage:** Preprocessor fits imputer and scaler *strictly* on training split (571 samples) and saves artifact to `models/preprocessor.joblib`.

---

## Slide 11: Model Training & Evaluation (Empirical Results)
- **Primary Model Choice:** **Random Forest Classifier**
  - Handles non-linear multi-factor interactions between clinical indicators.
  - Robust against categorical variance and class imbalances via `class_weight='balanced'`.
  - Non-parametric ensemble avoids strong distribution assumptions.
- **Actual Experimental Comparison (Held-Out Test Set: 143 Samples):**

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline 1: Logistic Regression** | 38.46% | 0.3886 | 0.3896 | Linear decision boundary struggles with complex survey interactions |
| **Baseline 2: Decision Tree (Depth 4)** | 22.38% | 0.2002 | 0.1462 | Single tree exhibits high variance and underfits the cohort |
| **Candidate: Random Forest v1 (Shallow)** | 41.26% | 0.4158 | 0.4044 | Ensembling 50 trees improves macro recall |
| **Champion: Random Forest v2 (Tuned)** | **42.66%** | **0.4240** | **0.4337** | **120 trees, balanced weights, min_samples_leaf=2** |

---

## Slide 12: MLflow Experiment Tracking
- **Local SQLite Backend:** Configured at `sqlite:///mlruns/mlflow.db` (zero external setup).
- **Parameters Logged:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`, `random_state`.
- **Metrics Tracked:** `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `f1_weighted`.
- **Artifacts Logged:** Serialized model binaries, confusion matrix heatmaps, feature importance plots, and comparison bar charts.
- **Multi-Run Traceability:** Every execution generates a unique UUID run with full reproducibility.

---

## Slide 13: Model Registry & Version Control
- **Registered Model Entity:** `"ElderlyHealthRiskClassifier"`
- **Active Champion Version:** `v2` (Status: `Production`)
- **Version Lineage:**
  - `v1`: `RandomForest_v1_Shallow` (Accuracy: 41.26%, Status: `Archived`)
  - `v2`: `RandomForest_v2_Champion` (Accuracy: 42.66%, Status: `Production`)
- **Metadata Transparency:** Serialized locally in `models/model_registry.json` and mirrored in the MLflow Model Registry database.

---

## Slide 14: REST API Deployment (FastAPI)
- **Framework:** FastAPI with Uvicorn ASGI production server.
- **Key Endpoints:**
  - `GET /`: Service metadata and discovery links.
  - `GET /health`: Liveness and model readiness probe.
  - `GET /model-info`: Active model version and hyperparameter inspection.
  - `POST /predict`: Single patient risk inference with probability breakdown.
  - `POST /batch-predict`: High-throughput batch inference.
  - `GET /monitoring/metrics`: Runtime traffic and latency metrics.
  - `GET /monitoring/drift`: Population Stability Index (PSI) drift report.
- **Validation:** Pydantic schema validation returns HTTP 422 on malformed or out-of-range inputs.

---

## Slide 15: Monitoring & Data Drift Engine
- **Telemetry Ingestion:** Asynchronously logs incoming inference features, predicted risk tier, confidence, and latency to `data/monitoring/api_requests.jsonl`.
- **Statistical Drift Engine:** Population Stability Index (PSI) computed on key features:
  - $\text{PSI} < 0.10 \rightarrow$ **STABLE** (No action needed)
  - $0.10 \le \text{PSI} < 0.25 \rightarrow$ **WARNING** (Monitor closely)
  - $\text{PSI} \ge 0.25 \rightarrow$ **DRIFT_DETECTED** (Trigger retraining pipeline)
- **Current Operational Metrics:** Average API Latency = 18.25 ms | Max PSI = 0.0976 (STABLE).

---

## Slide 16: CI/CD Pipeline (GitHub Actions)
- **Configuration:** `.github/workflows/ci.yml`
- **Automated Workflow Stages:**
  1. **Checkout & Environment:** Sets up Python 3.11 with pip caching on `ubuntu-latest`.
  2. **Schema & Data Verification:** Ingests raw data and verifies integrity constraints.
  3. **Test Suite Execution:** Runs all 29 Pytest tests across unit, preprocessing, training, API, and monitoring.
  4. **Artifact Assertion:** Asserts existence of `health_risk_model.joblib`, `preprocessor.joblib`, and `model_registry.json`.
  5. **Container Build:** Validates Dockerfile build with zero errors.

---

## Slide 17: Project Execution Timeline
```
+---------------------------------------------------------------------------------+
| Week 1 - 2: Problem Formulation, UCI Dataset Ingestion & Schema Definition      |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Week 3 - 4: Leakage-Free Preprocessing, Imputation & Feature Engineering        |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Week 5 - 6: Model Training (RF vs Baselines), Tuning & MLflow Experiment Tracking|
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Week 7 - 8: Model Registry, FastAPI Development, Telemetry & Drift Detection    |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
| Week 9 - 10: Dockerization, GitHub Actions CI/CD, 29 Pytest Suite & Docs        |
+---------------------------------------------------------------------------------+
```

---

## Slide 18: Diagnostic Visualizations & Results Analysis
- **Confusion Matrix Analysis:**
  - High sensitivity in separating Moderate Risk from High Risk cases.
  - Balanced class weighting successfully prevents minority class starvation.
- **Gini Feature Importance Hierarchy:**
  - Composite `Health_Deficit_Score` emerged as the #1 predictive variable.
  - Composite `Sleep_Disturbance_Score` ranked #2, confirming clinical literature that fragmented sleep correlates with high physician visitation in seniors.
  - Self-reported `Dental_Health` proved more predictive than demographic age alone.

---

## Slide 19: Future Scope & Enhancements
1. **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) directly into API responses for clinician interpretability.
2. **Wearable IoT Stream Integration:** Ingest continuous smartwatch accelerometer and heart rate data alongside survey inputs.
3. **Automated Retraining Loop:** Deploy an automated webhook triggered when `PSI >= 0.25` that re-executes `run_pipeline.py`.
4. **Edge Quantization:** Quantize Random Forest ensembles using ONNX runtime for sub-millisecond execution on mobile health tablets.

---

## Slide 20: References
1. Malani, P. N., Kullgren, J., & Solway, E. (2017). *National Poll on Healthy Aging (NPHA)*. Inter-university Consortium for Political and Social Research. DOI: `10.3886/ICPSR37305.v1`.
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
3. Zaharia, M., et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow*. IEEE Micro, 38(5), 28-36.
4. Tiangolo, S. (2020). *FastAPI: Modern, High-Performance Web Framework for Building APIs with Python 3.8+*.
5. Yurdakul, B. (2020). *Statistical Properties of the Population Stability Index in Credit Risk & MLOps*. Journal of Risk Model Validation.
