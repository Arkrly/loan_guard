# 🛡️ LoanGuard

**AI-Powered Loan Risk Assessment System**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>Predict loan approval risk in real-time with machine learning</strong>
</p>

---

## What is LoanGuard?

LoanGuard is an intelligent loan risk assessment application that uses machine learning to predict whether a loan application is likely to be approved or rejected. It helps financial institutions make data-driven decisions by analyzing applicant profiles and providing instant risk scores.

### Key Features

- **🤖 AI-Powered Predictions** — Logistic Regression model with 82% accuracy
- **⚡ Real-Time Analysis** — Instant risk assessment via REST API
- **🎨 Modern UI** — Beautiful glassmorphism interface with dark theme
- **💰 Indian Market Ready** — Designed for INR-based loan applications
- **📊 Risk Visualization** — Animated risk score gauge with color coding
- **📜 History Tracking** — Keep track of recent predictions

---

## Live Demo

🌐 **[Try LoanGuard Live](https://loanguard.onrender.com)** *(Hosted on Render)*

---

## How It Works

1. **Enter Applicant Details** — Fill in income, loan amount, credit history, and more
2. **Get Instant Prediction** — AI analyzes the profile in milliseconds
3. **View Risk Score** — See approval probability and risk level
4. **Make Decisions** — Use insights to guide lending decisions

### Input Parameters

| Parameter | Description |
|-----------|-------------|
| Applicant Income | Monthly income in INR |
| Co-applicant Income | Partner's monthly income |
| Loan Amount | Requested loan amount (in thousands) |
| Loan Term | Repayment period in months |
| Credit History | Credit score status (Good/Bad) |
| Property Area | Urban / Semi-Urban / Rural |
| Personal Details | Gender, Marital Status, Dependents, Education |

### Output

```json
{
  "prediction": "Approved",
  "probability_approved": 0.85,
  "risk_score": 15.0,
  "model_version": "1.0"
}
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, Python 3.8+ |
| **ML Model** | scikit-learn (Logistic Regression) |
| **Frontend** | HTML, Tailwind CSS, Vanilla JS |
| **Deployment** | Render |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check & model status |
| `/predict` | POST | Get loan risk prediction |
| `/docs` | GET | API documentation (Swagger) |

### Quick API Test

```bash
curl -X POST "https://loanguard.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 50000,
    "CoapplicantIncome": 0,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
  }'
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 82% |
| Precision | 0.84 |
| Recall | 0.93 |
| F1-Score | 0.88 |

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/Arkrly/loan_guard.git
cd loan_guard

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.app:app --host 127.0.0.1 --port 8080
```

Open **http://127.0.0.1:8080** in your browser.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

> **Note:** All development work happens on the `dev` branch. Switch to `dev` for the full development environment including ML models, datasets, notebooks, and development documentation.

```bash
git checkout dev
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Arkrly">Arkrly</a>
</p>
