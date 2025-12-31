import pandas as pd
import numpy as np
import joblib
import os
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.preprocessing import StandardScaler

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model_optimized.pkl')
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
FEATURE_COLS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'feature_columns.txt')
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train_u6lujuX_CVtuZ9i.csv')

# --- Globals ---
model = None
scaler = None
imputation_values = {}
feature_columns = []

# --- Schemas with Proper Validation ---
# FIXES: CRITICAL-001, CRITICAL-002, CRITICAL-003, MEDIUM-001, MEDIUM-004
class LoanApplication(BaseModel):
    """
    Loan application input schema with comprehensive validation.
    All categorical fields use Literal types to prevent invalid values.
    All numeric fields use Field constraints to prevent edge cases.
    """
    Gender: Literal["Male", "Female"] = Field(
        ..., 
        description="Applicant's gender"
    )
    Married: Literal["Yes", "No"] = Field(
        ..., 
        description="Marital status"
    )
    Dependents: Literal["0", "1", "2", "3+"] = Field(
        ..., 
        description="Number of dependents"
    )
    Education: Literal["Graduate", "Not Graduate"] = Field(
        ..., 
        description="Education level"
    )
    Self_Employed: Literal["Yes", "No"] = Field(
        ..., 
        description="Self-employment status"
    )
    ApplicantIncome: float = Field(
        ..., 
        ge=0, 
        description="Applicant's monthly income (must be >= 0)"
    )
    CoapplicantIncome: float = Field(
        ..., 
        ge=0, 
        description="Co-applicant's monthly income (must be >= 0)"
    )
    LoanAmount: float = Field(
        ..., 
        gt=0, 
        le=10000,
        description="Loan amount in thousands (must be > 0 and <= 10000)"
    )
    Loan_Amount_Term: float = Field(
        ..., 
        ge=12, 
        le=480,
        description="Loan term in months (12 to 480)"
    )
    Credit_History: Literal[0.0, 1.0] = Field(
        ..., 
        description="Credit history (1.0 = good, 0.0 = bad)"
    )
    Property_Area: Literal["Urban", "Rural", "Semiurban"] = Field(
        ..., 
        description="Property location"
    )
    
    @model_validator(mode='after')
    def validate_total_income(self):
        """
        CRITICAL-001 FIX: Ensure total income is greater than 0 
        to prevent division by zero errors in feature engineering.
        """
        total_income = self.ApplicantIncome + self.CoapplicantIncome
        if total_income <= 0:
            raise ValueError(
                "Total income (ApplicantIncome + CoapplicantIncome) must be greater than 0. "
                "At least one income source is required for loan assessment."
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "2",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 2000,
                "LoanAmount": 150,
                "Loan_Amount_Term": 360,
                "Credit_History": 1.0,
                "Property_Area": "Urban"
            }
        }


# --- API Response Schema ---
class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: Literal["Approved", "Rejected"]
    probability_approved: float = Field(..., ge=0, le=1)
    approval_rate: float = Field(..., ge=0, le=100, description="Loan approval rate as percentage")
    threshold_used: float
    model_version: str = "1.0"


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    features_count: int


# --- Lifespan Context Manager (MEDIUM-003 FIX: Replace deprecated on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - loads resources on startup."""
    load_resources()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown")


# --- App Instance ---
app = FastAPI(
    title="LoanIQ - Smart Risk Intelligence API", 
    version="1.0",
    description="AI-powered loan risk analysis and prediction system",
    lifespan=lifespan
)

# --- CORS & Static Files ---
# NOTE: For production, replace "*" with specific allowed origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def load_resources():
    """
    Load all required resources on application startup.
    MEDIUM-002 FIX: Added proper logging for fallback values.
    """
    global model, scaler, imputation_values, feature_columns
    
    # 1. Load Model
    logger.info("Loading model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    
    # 2. Load Pre-fitted Scaler
    logger.info("Loading scaler...")
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded from {SCALER_PATH}")
    else:
        # Fallback: Try to fit from training data (for dev environment)
        train_smote_path = os.path.join(DATA_PROCESSED_PATH, 'X_train_smote.csv')
        if os.path.exists(train_smote_path):
            df_smote = pd.read_csv(train_smote_path)
            scaler = StandardScaler()
            scaler.fit(df_smote)
            logger.info("Scaler fitted on training data (fallback)")
        else:
            logger.error("Neither scaler.pkl nor X_train_smote.csv found! Predictions will fail.")
            scaler = None
        
    # 3. Get Feature Columns (Constraint)
    # The model expects specific columns in specific order
    cols_path = os.path.join(DATA_PROCESSED_PATH, 'feature_columns.txt')
    if os.path.exists(cols_path):
        with open(cols_path, 'r') as f:
            feature_columns = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(feature_columns)} feature columns")
    else:
        # Fallback based on known notebook columns if file missing
        feature_columns = df_smote.columns.tolist() if 'df_smote' in locals() else []
        logger.warning("feature_columns.txt not found. Using scaler columns as fallback.")
         
    # 4. Get Imputation Stats (Modes/Medians)
    # derived from Raw Data or hardcoded based on notebook summary
    logger.info("Calculating imputation stats...")
    if os.path.exists(RAW_DATA_PATH):
        df_raw = pd.read_csv(RAW_DATA_PATH)
        imputation_values['LoanAmount'] = df_raw['LoanAmount'].median()
        imputation_values['Loan_Amount_Term'] = df_raw['Loan_Amount_Term'].mode()[0]
        imputation_values['Credit_History'] = df_raw['Credit_History'].mode()[0]
        imputation_values['ApplicantIncome'] = df_raw['ApplicantIncome'].median()
        imputation_values['CoapplicantIncome'] = df_raw['CoapplicantIncome'].median()
        logger.info("Imputation values calculated from raw data")
    else:
        # Fallback defaults (MEDIUM-002: Log warning about using fallback values)
        logger.warning(f"Raw data file not found at {RAW_DATA_PATH}. Using hardcoded imputation values.")
        imputation_values = {
            'LoanAmount': 128.0,
            'Loan_Amount_Term': 360.0,
            'Credit_History': 1.0,
            'ApplicantIncome': 3812.5,
            'CoapplicantIncome': 1188.5
        }

def preprocess_features(input_data: LoanApplication):
    # Convert input to DataFrame
    data = input_data.dict()
    df = pd.DataFrame([data])
    
    # 1. Handle Missing Values (Imputation)
    # In live prediction, we fill if user sends specific "null" indicators, 
    # but Pydantic ensures types. We'll ensure logical fallbacks if needed.
    # (Assuming input is complete for now based on Schema)

    # 2. Feature Creation
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['LogApplicantIncome'] = np.log1p(df['ApplicantIncome'])
    df['LogCoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])
    df['LogTotalIncome'] = np.log1p(df['TotalIncome'])
    df['LogLoanAmount'] = np.log1p(df['LoanAmount'])
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['EMI'] = df['EMI'].replace([np.inf, -np.inf], 0)
    df['BalanceIncome'] = df['TotalIncome'] - (df['EMI'] * 1000)
    df['LoanIncomeRatio'] = df['LoanAmount'] / df['TotalIncome']
    # 'IncomePerDependent' setup
    dep_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
    dep_num = dep_map.get(df['Dependents'][0], 0)
    df['IncomePerDependent'] = df['TotalIncome'] / (dep_num + 1)

    # 3. Binning
    # IncomeBin
    income_bins = [0, 2500, 4000, 6000, 10000, np.inf]
    income_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
    df['IncomeBin'] = pd.cut(df['TotalIncome'], bins=income_bins, labels=income_labels)
    
    # LoanAmountBin
    loan_bins = [0, 100, 150, 200, 300, np.inf]
    loan_labels = ['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large']
    df['LoanAmountBin'] = pd.cut(df['LoanAmount'], bins=loan_bins, labels=loan_labels)
    
    # LoanTermBin
    term_bins = [0, 180, 300, 360, np.inf]
    term_labels = ['Short', 'Medium', 'Standard', 'Long']
    df['LoanTermBin'] = pd.cut(df['Loan_Amount_Term'], bins=term_bins, labels=term_labels)
    
    # EMIBin
    emi_bins = [0, 0.25, 0.4, 0.6, np.inf]
    emi_labels = ['Low_EMI', 'Medium_EMI', 'High_EMI', 'Very_High_EMI']
    df['EMIBin'] = pd.cut(df['EMI'], bins=emi_bins, labels=emi_labels)

    # 4. Encoding
    # Binary
    df['Gender_Encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married_Encoded'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education_Encoded'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed_Encoded'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Dependents_Encoded'] = dep_num
    
    # One-Hot Encoding
    # We need to manually create the OHE columns to match training structure
    # List of all OHE prefixes and possible values (hardcoded from notebook logic)
    ohe_configs = {
        'Property': ['Rural', 'Semiurban', 'Urban'],
        'IncomeBin': income_labels,
        'LoanAmountBin': loan_labels,
        'LoanTermBin': term_labels,
        'EMIBin': emi_labels
    }
    
    for prefix, categories in ohe_configs.items():
        if prefix == 'Property':
            val = df['Property_Area'][0]
        else:
            val = df[prefix][0] # e.g. 'Very_Low'
            
        for cat in categories:
            col_name = f"{prefix}_{cat}"
            df[col_name] = 1 if val == cat else 0

    # 6. Select & Order Columns
    # Ensure dataframe matches the exact feature set of the model (20 columns)
    # The previous code was generating extra OHE columns that aren't in the model.
    # We must only populate the columns running in feature_columns.
    
    final_df = pd.DataFrame(index=[0])
    
    for col in feature_columns:
        if col in df.columns:
            final_df[col] = df[col]
        else:
            # Handle One-Hot Encoded columns manually map input to expected column
            # e.g. "Property_Urban" -> 1 if Property_Area == 'Urban'
            if col.startswith('Property_'):
                val = col.split('_')[1]
                final_df[col] = 1 if df['Property_Area'][0] == val else 0
            else:
                 # Logic for other potential missing columns or just init as 0
                final_df[col] = 0
            
    # 6. Scaling
    final_scaled = scaler.transform(final_df)
    
    return final_scaled

# --- Health Check Endpoint ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are ready."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        features_count=len(feature_columns)
    )


# --- Prediction Endpoint ---
# MEDIUM-014 FIX: Better error handling without leaking internal details
@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_risk(application: LoanApplication):
    """
    Predict loan approval risk based on applicant information.
    
    Uses a HYBRID approach:
    - ML model prediction (60% weight)
    - Manual factor scoring (40% weight) - allows bad credit to be compensated
    
    Returns:
        - prediction: "Approved" or "Rejected"
        - probability_approved: Probability (0-1) of approval
        - approval_rate: Approval rate percentage (0-100)
        - threshold_used: Decision threshold used
        - model_version: Model version identifier
    """
    try:
        # Validate model is loaded
        if model is None:
            logger.error("Model not loaded when predict was called")
            raise HTTPException(
                status_code=503, 
                detail="Model not ready. Please try again in a moment."
            )
        
        processed_data = preprocess_features(application)
        
        # 1. ML Model Prediction (base)
        probs = model.predict_proba(processed_data)[0]
        ml_prob = probs[1]
        
        # 2. Manual Factor Scoring (allows compensation)
        total_income = application.ApplicantIncome + application.CoapplicantIncome
        loan_to_income = application.LoanAmount / total_income if total_income > 0 else 999
        
        # Calculate manual score (0 to 1)
        manual_score = 0.0
        
        # Credit History: 35% of manual score
        manual_score += application.Credit_History * 0.35
        
        # Income strength: 30% of manual score (scaled by income level)
        # High income (100k+) = full score, Low income (5k) = minimal
        income_score = min(total_income / 100000, 1.0)
        manual_score += income_score * 0.30
        
        # Low debt ratio: 20% of manual score
        # Ratio < 0.1 = full score, Ratio > 0.5 = no score
        ratio_score = max(0, 1 - (loan_to_income * 2))
        manual_score += ratio_score * 0.20
        
        # Education: 10%
        education_score = 1.0 if application.Education == "Graduate" else 0.5
        manual_score += education_score * 0.10
        
        # Employment stability: 5%
        stability_score = 0.7 if application.Self_Employed == "No" else 0.5
        manual_score += stability_score * 0.05
        
        # 3. Combine ML and Manual scores
        # ML: 60%, Manual: 40% - this allows compensation
        combined_prob = (ml_prob * 0.60) + (manual_score * 0.40)
        
        # Ensure bounds
        combined_prob = max(0.0, min(1.0, combined_prob))
        
        # Threshold (Optimized from training)
        THRESHOLD = 0.3129
        prediction = 1 if combined_prob >= THRESHOLD else 0
        
        result = "Approved" if prediction == 1 else "Rejected"
        approval_rate = combined_prob * 100  # Approval rate as percentage
        
        logger.info(f"Prediction made: {result} (ml={ml_prob:.4f}, manual={manual_score:.4f}, combined={combined_prob:.4f})")
        
        return PredictionResponse(
            prediction=result,
            probability_approved=round(float(combined_prob), 4),
            approval_rate=round(float(approval_rate), 2),
            threshold_used=THRESHOLD,
            model_version="1.0"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # Validation errors - return 400 with user-friendly message
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=400, 
            detail=str(e)
        )
    except Exception as e:
        # MEDIUM-014 FIX: Log full error but return generic message
        logger.error(f"Prediction error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing your request. Please check your input and try again."
        )


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    Health check endpoint for Koyeb and monitoring.
    Returns status, model state, and version info.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0",
        "message": "LoanGuard API is running"
    }


# --- API Info Endpoint ---
@app.get("/api")
async def api_info():
    """
    API information endpoint.
    Note: Frontend is mounted separately, this won't be reached if frontend exists.
    """
    return {
        "name": "LoanGuard - AI Risk Assessment API",
        "version": "1.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


# Mount Frontend (Must be last to avoid overriding API routes)
FRONTEND_PATH = os.path.join(PROJECT_ROOT, 'frontend')
if os.path.exists(FRONTEND_PATH):
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
    logger.info(f"Frontend mounted from {FRONTEND_PATH}")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_PATH}")
