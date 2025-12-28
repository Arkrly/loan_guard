# 🔧 LoanGuard - Development Guide

> Complete development documentation for contributing to and extending the LoanGuard AI Risk Assessment System.

---

## 📋 Table of Contents

- [Development Setup](#-development-setup)
- [Project Architecture](#-project-architecture)
- [ML Pipeline](#-ml-pipeline)
- [Notebooks Guide](#-notebooks-guide)
- [Model Training](#-model-training)
- [API Development](#-api-development)
- [Testing](#-testing)
- [Code Style](#-code-style)
- [Debugging](#-debugging)

---

## 🚀 Development Setup

### Prerequisites

- Python 3.8+ (3.10 recommended)
- Git
- VS Code (recommended) or your preferred IDE
- Jupyter Notebook/Lab

### Clone & Setup

```bash
# Clone the repository
git clone https://github.com/Arkrly/loan_guard.git
cd loan_guard

# Switch to dev branch
git checkout dev

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install all development dependencies
pip install -r requirements-dev.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=loanguard --display-name="LoanGuard Python"
```

> **Note:** `requirements-dev.txt` includes all production dependencies plus development tools (Jupyter, testing, ML training, visualization, etc.).

### Environment Variables

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Server Configuration
HOST=127.0.0.1
PORT=8080
DEBUG=true

# Model Configuration
MODEL_PATH=models/best_model_optimized.pkl
THRESHOLD=0.3129

# Logging
LOG_LEVEL=DEBUG
```

---

## 🏗️ Project Architecture

```
loan_guard/
├── 📂 data/
│   ├── processed/               # Cleaned & transformed datasets
│   │   ├── X_train_smote.csv    # SMOTE-balanced training data
│   │   └── feature_columns.txt  # Feature names list
│   └── raw/                     # Original datasets (gitignored)
│
├── 📂 docs/
│   ├── CHANGELOG.md             # Version history
│   └── problem_statement.pdf    # Project requirements
│
├── 📂 frontend/
│   ├── index.html               # Main UI (Tailwind CSS)
│   ├── app.js                   # Frontend JavaScript logic
│   └── style.css                # Custom CSS overrides
│
├── 📂 models/
│   ├── best_model_optimized.pkl # Production model (Logistic Regression)
│   ├── best_model.pkl           # Base trained model
│   ├── logistic_regression_tuned.pkl
│   ├── random_forest_tuned.pkl
│   ├── xgboost_tuned.pkl
│   ├── model_info.pkl           # Model metadata
│   └── model_performance_comparison.csv
│
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb      # EDA & data understanding
│   ├── 02_feature_engineering.ipynb   # Feature creation & selection
│   ├── 03_model_building.ipynb        # Model training & comparison
│   └── 04_model_fine_tuning.ipynb     # Hyperparameter optimization
│
├── 📂 src/
│   └── app.py                   # FastAPI application
│
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── CHANGELOG.md                 # Project changelog
├── CONTRIBUTING.md              # Contribution guidelines
├── DEVELOPMENT.md               # This file
├── LICENSE                      # MIT License
├── README.md                    # Project overview
├── requirements.txt             # Production dependencies (minimal)
└── requirements-dev.txt         # Development dependencies (full)
```

---

## 🤖 ML Pipeline

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Preprocessing  │───▶│    Training     │
│  (CSV files)    │    │  (Cleaning,     │    │  (Model fit,    │
│                 │    │   Encoding)     │    │   Validation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Deployment    │◀───│   Serialization │◀───│  Evaluation     │
│  (FastAPI)      │    │   (.pkl files)  │    │  (Metrics)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Feature Engineering

The model uses the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `Gender` | Categorical | Male/Female |
| `Married` | Categorical | Yes/No |
| `Dependents` | Categorical | 0/1/2/3+ |
| `Education` | Categorical | Graduate/Not Graduate |
| `Self_Employed` | Categorical | Yes/No |
| `ApplicantIncome` | Numeric | Monthly income (INR) |
| `CoapplicantIncome` | Numeric | Co-applicant income (INR) |
| `LoanAmount` | Numeric | Loan amount (in thousands) |
| `Loan_Amount_Term` | Numeric | Term in months |
| `Credit_History` | Binary | 1.0 = good, 0.0 = bad |
| `Property_Area` | Categorical | Urban/Semiurban/Rural |

### Class Imbalance Handling

We use **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

---

## 📓 Notebooks Guide

### 1. Data Exploration (`01_data_exploration.ipynb`)

**Purpose:** Understand the dataset structure, distributions, and identify data quality issues.

**Key Tasks:**
- Load and inspect raw data
- Check for missing values
- Analyze feature distributions
- Identify correlations
- Visualize target variable balance

### 2. Feature Engineering (`02_feature_engineering.ipynb`)

**Purpose:** Transform raw features into model-ready inputs.

**Key Tasks:**
- Handle missing values (imputation)
- Encode categorical variables (one-hot, label encoding)
- Create derived features (income ratios, etc.)
- Feature scaling (StandardScaler)
- Train-test split

### 3. Model Building (`03_model_building.ipynb`)

**Purpose:** Train and compare multiple ML models.

**Models Evaluated:**
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

### 4. Model Fine-Tuning (`04_model_fine_tuning.ipynb`)

**Purpose:** Optimize hyperparameters and finalize the production model.

**Techniques:**
- GridSearchCV / RandomizedSearchCV
- Cross-validation
- Threshold optimization
- Model serialization

---

## 🔬 Model Training

### Train a New Model

```python
# Load processed data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
X = pd.read_csv('data/processed/X_train_smote.csv')
y = X.pop('target')  # Assuming target is in the CSV

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.4f}")

# Save
joblib.dump(model, 'models/new_model.pkl')
```

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 82% | 0.84 | 0.93 | 0.88 | 0.85 |
| Random Forest | 80% | 0.82 | 0.91 | 0.86 | 0.83 |
| XGBoost | 81% | 0.83 | 0.92 | 0.87 | 0.84 |

**Selected Model:** Logistic Regression (best balance of performance and interpretability)

---

## 🔌 API Development

### Running the Dev Server

```bash
# With auto-reload for development
uvicorn src.app:app --reload --host 127.0.0.1 --port 8080
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend UI |
| `GET` | `/health` | Health check & model status |
| `POST` | `/predict` | Get loan risk prediction |
| `GET` | `/docs` | Swagger API documentation |
| `GET` | `/redoc` | ReDoc API documentation |

### Adding New Endpoints

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class NewFeatureRequest(BaseModel):
    feature_name: str
    value: float

@app.post("/new-endpoint")
async def new_endpoint(request: NewFeatureRequest):
    # Your logic here
    return {"result": "success"}
```

---

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Structure

```
tests/
├── test_api.py          # API endpoint tests
├── test_model.py        # Model prediction tests
└── test_validation.py   # Input validation tests
```

### Example Test

```python
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 50000,
        "CoapplicantIncome": 0,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
```

---

## 🎨 Code Style

### Python

We follow **PEP 8** with the following tools:

```bash
# Install formatters
pip install black isort flake8

# Format code
black src/
isort src/

# Lint
flake8 src/
```

### Configuration (pyproject.toml)

```toml
[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 88
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

---

## 🐛 Debugging

### Common Issues

#### 1. Model Not Loading

```python
# Check model file exists
import os
print(os.path.exists('models/best_model_optimized.pkl'))

# Try loading manually
import joblib
model = joblib.load('models/best_model_optimized.pkl')
print(type(model))
```

#### 2. Feature Mismatch

```python
# Check expected features
with open('data/processed/feature_columns.txt', 'r') as f:
    expected_features = f.read().splitlines()
print(expected_features)
```

#### 3. CORS Issues (Frontend)

Ensure CORS is configured in `app.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

**Branch Strategy:**
- `dev` → Development branch (this branch)
- `master` → Production/deployment branch

```bash
# Create feature branch from dev
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name

# After development, push and create PR to dev
git push origin feature/your-feature-name
```

---

<p align="center">
  <strong>Happy Coding! 🚀</strong>
</p>
