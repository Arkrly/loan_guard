# 🛡️ LoanGuard - AI Risk Assessment System

> 🔧 **Development Branch** | Full development workspace with ML models, datasets, and notebooks

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="DEVELOPMENT.md">Development Guide</a> •
  <a href="CONTRIBUTING.md">Contributing</a> •
  <a href="#-api-endpoints">API Docs</a>
</p>

---

> 🎓 **Portfolio Project** | Full-Stack ML Application with Modern UI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikitlearn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Tailwind-3.0+-06B6D4?style=flat-square&logo=tailwindcss" alt="Tailwind">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="License">
</p>

A modern machine learning system that predicts loan approval risk in real-time. Features a beautiful glassmorphism UI, robust REST API, and production-ready validation.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML-Powered Predictions** | Logistic Regression model with 82% accuracy |
| 🎨 **Modern Glassmorphism UI** | Dark theme with cyan accents, smooth animations |
| 💰 **INR Currency Support** | Designed for Indian market loan assessment |
| ✅ **Input Validation** | Comprehensive Pydantic validation |
| 📊 **Animated Risk Score** | Visual circular progress with color coding |
| 📜 **Analysis History** | Local storage persists recent predictions |
| 🎚️ **Interactive Controls** | Slider presets, toggle buttons |
| 📖 **API Documentation** | Auto-generated Swagger/OpenAPI docs |

---

## 🖼️ UI Preview

### Modern Dashboard
- **Sidebar** with history tracking
- **Form sections** with glassmorphism cards
- **Interactive loan term** slider with year presets
- **Real-time validation** and toast notifications

### Result Panel
- **Status chip** - Approved (green) / Rejected (red)
- **Risk gauge** - Animated circular progress
- **Stats cards** - Approval probability, threshold, model version

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/loan_risk_sys.git
cd loan_risk_sys

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8080
```

Open your browser at: **http://127.0.0.1:8080**

---

## 📁 Project Structure

```
loan_risk_sys/
├── 📂 data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Cleaned & transformed data
│   └── sample/              # Sample data for testing
├── 📂 frontend/
│   ├── index.html           # Main UI (Tailwind CSS)
│   ├── app.js               # Frontend logic
│   └── style.css            # Custom styles
├── 📂 model/
│   ├── model.pkl            # Trained model
│   ├── scaler.pkl           # Feature scaler
│   └── features.json        # Feature configuration
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── 📂 src/
│   └── app.py               # FastAPI application
├── .env.example             # Environment variables template
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend UI |
| `GET` | `/health` | Health check & model status |
| `POST` | `/predict` | Get loan risk prediction |
| `GET` | `/docs` | Swagger API documentation |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
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

### Example Response

```json
{
  "prediction": "Approved",
  "probability_approved": 0.85,
  "risk_score": 15.0,
  "threshold_used": 0.3129,
  "model_version": "1.0"
}
```

---

## 🚀 Deployment (Koyeb)

Deploy this app for **free** on [Koyeb](https://koyeb.com) with faster cold starts than alternatives.

### One-Click Deploy

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&repository=github.com/YOUR_USERNAME/loan_risk_sys&branch=master&name=loanguard)

### Manual Deployment

1. **Create a Koyeb account** at [koyeb.com](https://koyeb.com)

2. **Create a new App** → Choose **GitHub** as source

3. **Configure the service:**
   | Setting | Value |
   |---------|-------|
   | **Repository** | `your-username/loan_risk_sys` |
   | **Branch** | `master` |
   | **Builder** | Buildpack |
   | **Run command** | `uvicorn src.app:app --host 0.0.0.0 --port 8080` |
   | **Port** | `8080` |
   | **Health check path** | `/health` |
   | **Instance type** | Nano (free tier) |
   | **Region** | Frankfurt (or closest to you) |

4. **Deploy** → Your app will be live at `https://loanguard-YOUR_ID.koyeb.app`

### Environment Variables (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `ALLOWED_ORIGINS` | CORS allowed origins | `*` |
| `PORT` | Server port | `8080` |

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern async Python web framework
- **Pydantic** - Data validation
- **scikit-learn** - Machine learning
- **joblib** - Model serialization

### Frontend
- **Tailwind CSS** - Utility-first CSS framework
- **Vanilla JavaScript** - No framework dependencies
- **Material Symbols** - Google's icon library

### ML Pipeline
- **SMOTE** - Handling class imbalance
- **Logistic Regression** - Classification model
- **StandardScaler** - Feature normalization

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 82% |
| Precision | 0.84 |
| Recall | 0.93 |
| F1-Score | 0.88 |

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Your Name**
- Portfolio: [your-portfolio.com](https://your-portfolio.com)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)
- GitHub: [@yourusername](https://github.com/yourusername)

---

<p align="center">
  Made with ❤️ for learning ML & Full-Stack Development
</p>
