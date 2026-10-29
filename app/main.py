"""
Bharat Banking - Enterprise Analytics Backend API
FastAPI backend providing historical branch telemetry, ML efficiency forecasting,
HuggingFace NLP sentiment analysis, and Google Gemini AI orchestration.
"""

import logging
import sqlite3
import pickle
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.config import DATABASE_PATH, MODEL_PATH, GEMINI_API_KEY, GEMINI_MODEL
from app.nlp_service import analyze_feedback
from app.gemini_service import query_gemini, build_banking_context_summary

# Configure structured enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bharat_banking.api")

# Cached ML model reference
ml_model = None

def get_ml_model():
    """Retrieves cached ML model or loads it lazily from disk."""
    global ml_model
    if ml_model is None and MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                ml_model = pickle.load(f)
            logger.info(f"Loaded Random Forest Efficiency Model from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load ML model from {MODEL_PATH}: {e}")
            ml_model = None
    return ml_model


def ensure_database_and_model():
    """Auto-generates synthetic database and model if missing on disk."""
    if not DATABASE_PATH.exists() or not MODEL_PATH.exists():
        logger.info("Initializing baseline database and ML model...")
        try:
            import subprocess
            import sys
            setup_script = DATABASE_PATH.parent / "setup_ml.py"
            if setup_script.exists():
                subprocess.run([sys.executable, str(setup_script)], check=True, capture_output=True)
                logger.info("Baseline setup completed successfully.")
        except Exception as e:
            logger.warning(f"Could not auto-generate baseline: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager to initialize and cache the ML model at startup.
    """
    logger.info("Initializing Bharat Banking Enterprise API...")
    
    # 0. Ensure Database & Model Exist
    ensure_database_and_model()

    # 1. Load ML Model
    get_ml_model()

    # 2. Verify Database Connection
    if DATABASE_PATH.exists():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM branch_performance")
                count = cursor.fetchone()[0]
            logger.info(f"Connected to database {DATABASE_PATH.name} ({count} records in branch_performance)")
        except Exception as e:
            logger.error(f"Database verification warning: {e}")
    else:
        logger.warning(f"Database not found at {DATABASE_PATH}")

    yield
    logger.info("Bharat Banking Enterprise API shutting down.")


app = FastAPI(
    title="Bharat Banking | Enterprise AI Command Center API",
    description="High-performance backend API powering the Bharat Banking Command Center.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local dashboards and authorized clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC DATA MODELS
# ==============================================================================

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    database_records: int
    model_loaded: bool
    gemini_configured: bool
    version: str = "2.0.0"


class BranchRecord(BaseModel):
    branch_name: str
    region: str
    month: str
    lat: float
    lon: float
    staff_count: int
    daily_transactions: int
    customer_satisfaction: float
    efficiency_score: float


class HistoricalDataResponse(BaseModel):
    status: str
    count: int
    data: List[Dict[str, Any]]


class PredictionRequest(BaseModel):
    staff_count: int = Field(..., ge=5, le=300, description="Active branch staff headcount (5 - 300)")
    daily_transactions: int = Field(..., ge=100, le=50000, description="Projected daily transactions (100 - 50,000)")
    customer_satisfaction: float = Field(..., ge=1.0, le=5.0, description="Target customer satisfaction rating (1.0 - 5.0)")

    @field_validator("customer_satisfaction")
    @classmethod
    def validate_satisfaction(cls, v: float) -> float:
        return round(v, 2)


class PredictionResponse(BaseModel):
    status: str
    predicted_efficiency_score: float
    input_features: Dict[str, Any]
    operational_assessment: str


class FeedbackRequest(BaseModel):
    branch_id: str = Field(..., min_length=1, description="Unique branch identifier or name")
    feedback_text: str = Field(..., min_length=3, max_length=2000, description="Customer review text")


class SentimentAnalysisResult(BaseModel):
    label: str
    confidence: float
    explanation: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    branch_id: str
    feedback: str
    sentiment_analysis: Dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's query for the AI banking assistant")
    api_key: Optional[str] = Field(None, description="Optional override API key")
    model: Optional[str] = Field(None, description="Optional model identifier")
    branch_context: Optional[List[Dict[str, Any]]] = Field(None, description="Optional branch metrics for grounding")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous messages")


class ChatResponse(BaseModel):
    success: bool
    error_type: Optional[str] = None
    response: Optional[str] = None
    model_used: Optional[str] = None
    message: str


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/", tags=["General"])
def read_root():
    """
    Root discovery endpoint providing service overview.
    """
    return {
        "service": "Bharat Banking Enterprise Command Center API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    System health check verifying database connectivity and model availability.
    """
    db_connected = False
    records_count = 0
    if DATABASE_PATH.exists():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM branch_performance")
                records_count = cursor.fetchone()[0]
                db_connected = True
        except Exception as e:
            logger.error(f"Health check database error: {e}")
            db_connected = False

    model = get_ml_model()
    is_model_loaded = model is not None
    is_gemini_ready = bool(GEMINI_API_KEY)

    system_status = "healthy" if (db_connected and is_model_loaded) else "degraded"

    return HealthResponse(
        status=system_status,
        database_connected=db_connected,
        database_records=records_count,
        model_loaded=is_model_loaded,
        gemini_configured=is_gemini_ready
    )


@app.get("/api/branches/historical", response_model=HistoricalDataResponse, tags=["Analytics"])
def get_historical_performance():
    """
    Retrieves the complete historical branch performance dataset from SQLite.
    """
    if not DATABASE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database file '{DATABASE_PATH.name}' is not accessible."
        )

    try:
        import pandas as pd
        with sqlite3.connect(DATABASE_PATH) as conn:
            # Query securely using parameterized statement
            df = pd.read_sql_query(
                "SELECT branch_name, region, month, lat, lon, staff_count, daily_transactions, customer_satisfaction, efficiency_score FROM branch_performance",
                conn
            )
        return HistoricalDataResponse(
            status="success",
            count=len(df),
            data=df.to_dict(orient="records")
        )
    except Exception as e:
        logger.error(f"Error fetching historical branch data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query branch data: {str(e)}"
        )


@app.post("/api/ml/predict-efficiency", response_model=PredictionResponse, tags=["Machine Learning"])
def predict_efficiency(request: PredictionRequest):
    """
    Forecasts branch operational efficiency using the trained Random Forest Regressor.
    """
    active_model = get_ml_model()

    if active_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive Machine Learning model is currently unavailable."
        )

    try:
        import pandas as pd
        input_df = pd.DataFrame([{
            "staff_count": request.staff_count,
            "daily_transactions": request.daily_transactions,
            "customer_satisfaction": request.customer_satisfaction
        }])

        raw_pred = float(active_model.predict(input_df)[0])
        # Bounded between realistic operational thresholds
        bounded_score = round(max(40.0, min(99.9, raw_pred)), 2)

        # Operational assessment narrative
        txns_per_staff = request.daily_transactions / max(1, request.staff_count)
        if bounded_score >= 80.0:
            assessment = f"Optimal branch capacity. Workload ratio of {txns_per_staff:.0f} txns/staff aligns with high operational throughput."
        elif bounded_score >= 65.0:
            assessment = f"Moderate branch efficiency. Balanced staff allocation ({txns_per_staff:.0f} txns/staff) with steady throughput."
        else:
            assessment = f"Sub-optimal operational efficiency. High risk of counter bottleneck ({txns_per_staff:.0f} txns/staff) or under-utilized staffing."

        return PredictionResponse(
            status="success",
            predicted_efficiency_score=bounded_score,
            input_features={
                "staff_count": request.staff_count,
                "daily_transactions": request.daily_transactions,
                "customer_satisfaction": request.customer_satisfaction,
                "transactions_per_staff": round(txns_per_staff, 1)
            },
            operational_assessment=assessment
        )
    except Exception as e:
        logger.error(f"Error running ML prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction execution failed: {str(e)}"
        )


@app.post("/api/nlp/analyze-feedback", response_model=FeedbackResponse, tags=["NLP"])
def analyze_customer_feedback(request: FeedbackRequest):
    """
    Evaluates customer review text using HuggingFace sentiment analysis.
    """
    sentiment = analyze_feedback(request.feedback_text)
    if sentiment.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sentiment.get("error", "Unable to process sentiment analysis.")
        )

    return FeedbackResponse(
        status="success",
        branch_id=request.branch_id,
        feedback=request.feedback_text,
        sentiment_analysis=sentiment
    )


@app.post("/api/ai/chat", response_model=ChatResponse, tags=["Generative AI"])
def chat_banking_assistant(request: ChatRequest):
    """
    Direct API endpoint to interact with the Google Gemini AI Banking Assistant.
    """
    context_str = None
    if request.branch_context:
        context_str = build_banking_context_summary(request.branch_context)
    elif DATABASE_PATH.exists():
        # Fallback to load current dataset from SQLite so direct API callers receive grounded context
        try:
            import pandas as pd
            with sqlite3.connect(DATABASE_PATH) as conn:
                db_df = pd.read_sql_query("SELECT * FROM branch_performance", conn)
                if not db_df.empty:
                    context_str = build_banking_context_summary(db_df.to_dict(orient="records"))
        except Exception:
            pass

    result = query_gemini(
        user_prompt=request.message,
        api_key=request.api_key,
        model_name=request.model,
        context_summary=context_str,
        conversation_history=request.conversation_history
    )

    return ChatResponse(
        success=result.get("success", False),
        error_type=result.get("error_type"),
        response=result.get("response"),
        model_used=result.get("model_used"),
        message=result.get("message", "")
    )


if __name__ == "__main__":
    import uvicorn
    from app.config import API_HOST, API_PORT
    logger.info(f"Starting Bharat Banking API on http://{API_HOST}:{API_PORT}")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)
