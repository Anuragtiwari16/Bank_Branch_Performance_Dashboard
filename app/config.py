import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Application Settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}").rstrip("/")

db_env = os.getenv("DATABASE_PATH")
if db_env:
    p = Path(db_env)
    DATABASE_PATH = p if p.is_absolute() else (BASE_DIR / p)
else:
    DATABASE_PATH = BASE_DIR / "bharat_banking.db"

model_env = os.getenv("MODEL_PATH")
if model_env:
    mp = Path(model_env)
    MODEL_PATH = mp if mp.is_absolute() else (BASE_DIR / mp)
else:
    MODEL_PATH = BASE_DIR / "efficiency_model.pkl"

# Gemini Configuration
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
raw_key = os.getenv("GEMINI_API_KEY", "").strip()
# Sanitize template/placeholder keys if copied blindly from .env.example
if raw_key in ["your_gemini_api_key_here", "your_actual_gemini_api_key_here", "your_api_key_here", "None", "null"]:
    GEMINI_API_KEY = ""
else:
    GEMINI_API_KEY = raw_key

GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
