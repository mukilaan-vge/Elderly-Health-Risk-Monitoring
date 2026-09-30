# COMPREHENSIVE FACULTY VIVA PREPARATION GUIDE
## Production-Grade MLOps Pipeline for Elderly Health Risk Monitoring

This guide provides technically precise, viva-ready answers for all 18 core questions based directly on the actual codebase, architecture, and empirical results of this project.

---

### Question 1: What is MLOps?
**Answer:**  
MLOps (Machine Learning Operations) is an engineering discipline that unifies Machine Learning, DevOps, and Data Engineering to automate, govern, and monitor the entire machine learning lifecycle in production. 

Unlike traditional software engineering where only code changes, ML systems depend on three moving dimensions: **Code, Data, and Models**. MLOps establishes standard practices for automated data ingestion, validation, continuous training (CT), experiment tracking, model registry, continuous integration and deployment (CI/CD), operational telemetry, and post-deployment data drift detection to ensure models remain reliable, reproducible, and explainable over time.

---

### Question 2: Why is this project an MLOps project and not just an ML project?
**Answer:**  
A conventional Machine Learning project typically ends at model evaluation inside an exploratory Jupyter notebook, focusing solely on offline metrics (such as accuracy or loss).

In contrast, our project implements the **complete end-to-end MLOps operational lifecycle**:
1. **Automated Data Governance:** Lineage tracking via SHA256 checksums and automated UCI repository fetching.
2. **Schema & Domain Validation:** Catching out-of-bounds inputs, missing features, and survey refusal codes before pipeline entry.
3. **Leakage-Free Preprocessing:** Encapsulated in a serialized Scikit-learn transformer (`preprocessor.joblib`) fitted solely on training splits.
4. **Experiment Tracking:** Systematically logging runs, hyperparameters, and artifacts with local SQLite MLflow.
5. **Model Registry & Versioning:** Promoting candidate models to a versioned registry (`ElderlyHealthRiskClassifier` v1 -> v2) with Production status tags.
6. **Microservice Serving:** Packaging the model as an async REST API using FastAPI with strict Pydantic validation.
7. **Production Telemetry & Drift Monitoring:** Real-time request logging and Population Stability Index (PSI) distribution drift detection.
8. **Automated CI/CD:** GitHub Actions workflow executing schema validation, a 29-test Pytest suite, and Docker containerization.

---

### Question 3: Why did you select Random Forest as the primary model?
**Answer:**  
We selected Random Forest Classifier based on the empirical nature of the elderly health risk dataset:
1. **Non-Linear Multi-Factor Interactions:** Geriatric health involves complex interactions (e.g. chronic pain combined with nocturia and psychological stress). Linear models like Logistic Regression assume additive linear relationships and scored only 38.46% accuracy.
2. **Robustness to Overfitting:** A single Decision Tree suffered extreme variance and underfitting (22.38% accuracy). Random Forest aggregates 120 decorrelated bootstrap decision trees, drastically reducing prediction variance via bagging.
3. **Imbalance Handling:** The NPHA dataset has natural class imbalances (Class 1: 52%, Class 2: 30%, Class 0: 18%). By configuring `class_weight='balanced'`, Random Forest dynamically adjusts sample weights inversely proportional to class frequencies.
4. **Well-Calibrated Probabilities & Interpretability:** It outputs reliable class probability distributions (vital for clinical triage) and provides native Gini feature importance scores.
5. **Resource Efficiency:** Trains in less than 2 seconds on standard CPU hardware without GPU requirements.

---

### Question 4: Why is data validation necessary?
**Answer:**  
In production, machine learning models will produce flawed predictions without failing loudly if corrupted data is fed into them ("Garbage In, Garbage Out"). Data validation prevents this silent failure by enforcing strict quality gates before any data reaches the model.

In our system, `src/preprocessing/validator.py`:
- Checks for missing columns.
- Flags and quantifies survey refusal codes (`-1`, `-2`) so they can be imputed properly.
- Verifies that categorical and numeric features fall strictly within clinical domain constraints (e.g. `Physical_Health` must be between 1 and 5).
- Rejects malformed requests at the API boundary with HTTP 422 before unhandled exceptions occur.

---

### Question 5: What is data preprocessing?
**Answer:**  
Data preprocessing is the process of cleaning, transforming, and formatting raw, unstructured, or heterogeneous data into a numerical representation suitable for machine learning algorithms.

In our pipeline (`src/preprocessing/preprocessor.py`), preprocessing entails:
1. **Refusal Code Cleaning:** Converting survey codes `-1` (Refused) and `-2` (Not Asked) to `NaN`.
2. **Median Imputation:** Replacing `NaN` values with statistical medians computed strictly from the training cohort.
3. **Feature Engineering:** Synthesizing clinical domain indicators (`Sleep_Disturbance_Score`, `Health_Deficit_Score`, `High_Pain_Flag`).
4. **Feature Standardization:** Applying `StandardScaler` to normalize feature magnitudes for balanced gradient and tree split computation.

---

### Question 6: What is data leakage, and how did your pipeline prevent it?
**Answer:**  
**Data leakage** occurs when information from outside the training dataset (such as the validation/test set or future production data) inadvertently influences model training, leading to overly optimistic evaluation metrics that fail to generalize in production.

**How we strictly prevented data leakage:**
1. **Split-Before-Transform:** The raw dataset was first split into training (80%) and testing (20%) subsets using stratified sampling *before* computing any statistics.
2. **Isolated Transformer Fitting:** The imputer and scaler were fitted **strictly on `X_train`**. The test split (`X_test`) was strictly transformed using the parameters learned from `X_train`.
3. **Encapsulated Serialization:** The fitted preprocessor was saved to `models/preprocessor.joblib`. During FastAPI inference, the exact same transformer is loaded and calls `.transform()` on incoming requests without refitting.

---

### Question 7: What is experiment tracking?
**Answer:**  
Experiment tracking is the practice of systematically recording all inputs, configurations, code versions, parameters, evaluation metrics, and output artifacts associated with each machine learning training run. 

Without experiment tracking, data scientists rely on scattered notebooks or console logs, making it impossible to know which exact dataset version, hyperparameter combination, or seed produced a specific model binary.

---

### Question 8: Why did you choose MLflow?
**Answer:**  
We chose MLflow because:
1. **Open Source & Lightweight:** Requires zero paid cloud accounts or subscriptions.
2. **Local Portability:** Can be backed by a simple SQLite database (`sqlite:///mlruns/mlflow.db`) and local folder storage.
3. **Comprehensive Lifecycle Coverage:** MLflow natively integrates **Tracking** (parameters, metrics, loss curves), **Artifact Logging** (confusion matrices, models), and **Model Registry** (versioning and staging) in a single platform.
4. **Framework Agnostic:** Seamlessly logs Scikit-learn models using `mlflow.sklearn.log_model`.
5. **Interactive UI:** Provides an intuitive web interface (`mlflow ui`) for comparing experiment runs side-by-side during viva and project review.

---

### Question 9: What is a Model Registry?
**Answer:**  
A Model Registry is a centralized, governed catalog for storing, managing, and collaborating on trained machine learning models. It bridges the gap between model development and operational deployment.

While experiment tracking logs dozens of experimental candidates, the Model Registry stores only verified models, managing their lifecycle states (e.g. `None`, `Staging`, `Production`, `Archived`), documenting metadata, and tracking which exact artifact is currently serving live inference.

---

### Question 10: Why is model versioning necessary?
**Answer:**  
Model versioning allows an organization to track how a model evolves over time in response to hyperparameter tuning, code updates, or newly arrived retraining data.

**Key benefits demonstrated in our project:**
1. **Traceability:** We can directly identify that active model `v2` (`RandomForest_v2_Champion`) achieves 42.66% accuracy, while predecessor `v1` achieved 41.26%.
2. **Safe Rollbacks:** If a newly deployed model behaves erratically or exhibits performance degradation, the registry allows instantaneous rollback to the previous stable version (`v1`) without re-running code.
3. **Auditability:** Meets healthcare compliance standards by recording exactly which model version generated a specific patient prediction.

---

### Question 11: Why did you choose FastAPI for model serving?
**Answer:**  
We selected FastAPI over alternatives like Flask or Django because:
1. **Asynchronous Architecture:** Built on ASGI (Starlette and Uvicorn), offering significantly higher throughput and lower request latency.
2. **Native Pydantic Data Validation:** Incoming JSON payloads are automatically validated against Python type hints, instantly rejecting malformed requests with detailed 422 HTTP responses.
3. **Automatic OpenAPI / Swagger Documentation:** Automatically generates interactive API documentation at `/docs`, enabling instant testing without writing custom frontend interfaces.
4. **Modern Lifespan Management:** Uses async lifespan context managers to load heavy model artifacts into memory once at startup rather than per request.

---

### Question 12: Why is Docker used in this MLOps project?
**Answer:**  
Docker encapsulates the application code, Python interpreter, exact dependency versions (`requirements.txt`), and serialized model artifacts inside an isolated, lightweight Linux container.

**Core MLOps Benefits:**
- **Eliminates "It works on my machine" syndrome:** Guarantees that the API behaves identically across developer Windows laptops, Linux test environments, and cloud servers.
- **Portability:** Can be deployed to any container orchestration engine (Kubernetes, AWS ECS, Google Cloud Run) with a single command.
- **Reproducibility:** Fixes the runtime environment, preventing breaking changes caused by OS-level updates.

---

### Question 13: What is monitoring in an MLOps system?
**Answer:**  
Monitoring in MLOps is the continuous tracking of a deployed machine learning service across two critical dimensions:
1. **Operational Metrics:** System health, request count, HTTP status codes, error rate, memory consumption, and inference latency (measured at ~18 ms in our API).
2. **Machine Learning Metrics:** Distribution of predicted risk classes, prediction confidence scores, and input feature statistics.

In our system, `src/monitoring/logger.py` records every prediction event to `data/monitoring/api_requests.jsonl` to ensure complete observability.

---

### Question 14: What is data drift, and how does your project detect it?
**Answer:**  
**Data drift** (or covariate shift) occurs when the statistical distribution of input features in production changes significantly from the reference data used during model training:

$$P_{\text{production}}(X) \ne P_{\text{training}}(X)$$

In elderly healthcare, drift might occur if a clinic starts admitting significantly older or more frail patients whose pain levels or sleep disturbances are much more severe than the baseline survey cohort.

**Our Detection Mechanism:**
We implement the **Population Stability Index (PSI)** in `src/monitoring/drift_detector.py`:
$$\text{PSI} = \sum_{i=1}^{k} (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right)$$
- $\text{PSI} < 0.10 \rightarrow$ **STABLE:** No significant shift.
- $0.10 \le \text{PSI} < 0.25 \rightarrow$ **WARNING:** Moderate distribution change.
- $\text{PSI} \ge 0.25 \rightarrow$ **DRIFT_DETECTED:** Significant distribution shift requiring pipeline retraining.

---

### Question 15: How does the retraining workflow work in your system?
**Answer:**  
When data drift is detected ($\text{PSI} \ge 0.25$) or new survey data is acquired:
1. The drift detector sets `"retrain_recommended": True` in `data/monitoring/drift_report.json`.
2. The orchestrator executes `python run_pipeline.py --stage train`.
3. The new dataset is validated, preprocessed without leakage, and the model is retrained.
4. If the newly trained candidate model outperforms the current production champion on held-out metrics, it is registered as a new version in the MLflow Model Registry and promoted to `Production` status.
5. The API reloads the new artifact cleanly.

---

### Question 16: What is CI/CD, and what does your GitHub Actions pipeline do?
**Answer:**  
CI/CD stands for **Continuous Integration and Continuous Deployment/Delivery**. In MLOps, it automates testing, validation, and container building whenever new code or pipeline updates are pushed to the repository.

Our workflow (`.github/workflows/ci.yml`) runs on GitHub-hosted runners and executes:
1. **Code & Data Validation Job:** Validates data schema constraints and checks for dataset corruption.
2. **Testing & Evaluation Job:** Executes all 29 Pytest tests (verifying data loaders, preprocessor, model training, API endpoints, and drift calculations).
3. **Artifact Verification:** Confirms that model binaries and registry files were generated properly.
4. **Docker Build Job:** Tests and builds the container image to ensure production deployability.

---

### Question 17: How does your pipeline improve reproducibility?
**Answer:**  
Our pipeline guarantees reproducibility through six core mechanisms:
1. **Cryptographic Data Hashing:** Checking SHA256 checksums on raw CSVs.
2. **Configuration Centralization:** Storing all hyperparameters, random seeds (`42`), and paths in `configs/config.yaml`.
3. **Hermetic Preprocessing:** Saving fitted transformer objects in `models/preprocessor.joblib`.
4. **Deterministic Seeds:** Enforcing fixed seeds across train/test splits, NumPy, and Scikit-learn estimators.
5. **Pinned Dependencies:** Specifying explicit version ranges in `requirements.txt`.
6. **Containerization:** Freezing the execution environment via Docker.

---

### Question 18: What happens if model performance decreases in production?
**Answer:**  
If production monitoring indicates declining performance (e.g. high prediction uncertainty, low confidence, or significant data drift):
1. **Automated Alerting:** The monitoring module flags the alert in `drift_report.json` and via the `/monitoring/drift` endpoint.
2. **Instant Version Rollback:** Because the system maintains version history in `models/model_registry.json` and the MLflow Registry, the API can instantaneously revert to a previous verified model version (e.g. `v1`) by changing the active version pointer.
3. **Root-Cause Telemetry Inspection:** Engineers inspect `data/monitoring/api_requests.jsonl` to isolate which specific input features drifted.
4. **Triggered Retraining:** The pipeline is re-executed with updated training data and re-validated before promotion.
