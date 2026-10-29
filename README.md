# 🏦 Bharat Banking | Enterprise Command Center

[![Python](https://img.shields.io/badge/Python-3.9+-0B2545.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Google GenAI](https://img.shields.io/badge/Google%20GenAI-SDK%202.0+-4285F4.svg)](https://aistudio.google.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E.svg)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-DistilBERT%20NLP-FFD21E.svg)](https://huggingface.co/)

**Bharat Banking Command Center** is a production-style, executive-grade banking analytics and intelligence platform designed to monitor, forecast, and optimize branch performance across commercial banking networks in India.

The platform combines a **FastAPI backend**, an **interactive Streamlit UI** with corporate light and executive dark palettes, **Scikit-Learn predictive machine learning**, **HuggingFace sentiment analysis**, and a **Google Gemini Generative AI Assistant**.

---

## 🌟 Key Features

* **📊 Executive National Overview**:
  * 4 core KPIs: Active Branch Network, National Average Efficiency, Customer Satisfaction, Total Daily Transactions.
  * Interactive Plotly charts: Efficiency rankings, regional performance, transaction volume distributions, and historical trends (Jan–Jun).
  * Interactive geographic map showing metropolitan hub locations across India.

* **🏢 Branch Deep-Dive**:
  * Granular telemetry for individual branches (Mumbai, Delhi, Bangalore, Kolkata, Chennai, Pune, Ahmedabad, Hyderabad).
  * Multi-axis Operational Radar charts (Efficiency, Satisfaction, Staff Capacity Index).
  * Historical 6-month performance trend vs. National Average.
  * Complete monthly tabular records.

* **🔮 Predictive ML Efficiency Forecasting**:
  * Powered by a trained Scikit-Learn **Random Forest Regressor** (`efficiency_model.pkl`).
  * Simulates staffing, transaction load, and customer satisfaction changes.
  * Calculates transactions-per-staff ratios and delivers actionable operational assessments.

* **🗣️ Customer Sentiment Analysis**:
  * Powered by **HuggingFace Transformers** (`distilbert-base-uncased-finetuned-sst-2-english`).
  * Evaluates customer feedback text with confidence ratings and executive takeaways.
  * Pre-loaded sample reviews for quick banking scenarios.

* **🤖 Enterprise Gemini AI Assistant**:
  * Integrated with the official `google-genai` Python SDK.
  * Grounded in active platform metrics (benchmarks, top/bottom branches, regional averages) without dumping raw databases or exposing customer PII.
  * Persistent environment-based configuration via `.env` (`GEMINI_API_KEY`, `GEMINI_MODEL`).
  * Chat history persistence across Streamlit reruns.
  * Suggested banking inquiry pills for instant one-click analysis.
  * Clear error handling for missing keys, quota limits (429), invalid keys, and network timeouts.

* **🔐 Role-Based Access Control (RBAC)**:
  * **National Head (CEO)**: Full nationwide access across all commercial branches and executive views.
  * **Branch Manager**: Scoped authorization restricted to assigned branch deep-dive and certified monthly data ingestion.

* **🎨 Dual-Theme Visual Design System**:
  * **Light Mode (Theme D — Pearl White & Teal)**: Pearl white canvas (`#F2F9FA`), soft teal surfaces (`#E4F1F4`), crisp white cards (`#FFFFFF`), primary teal highlights (`#008C95`), secondary teal (`#20B8C1`), deep slate headings (`#12343B`), and soft body text (`#526B73`).
  * **Dark Mode (Deep Navy & Charcoal with Neon Teal)**: Deep navy background (`#06111D`), charcoal-blue sidebar (`#071522`), dark blue-gray cards (`#0D1E2D`), input background (`#142A3C`), neon teal accents (`#22D3E2`), and crisp light typography (`#F1F7FC`).
  * **Centralized Theme Engine**: Toggle between Light and Dark modes seamlessly with Plotly charts, maps, inputs, and tables automatically adapting in real time.

---

## 🏗️ System Architecture

```text
[ Browser / Executive Client ]
              │
       (HTTP / Port 8501)
              ▼
    [ Streamlit Frontend ] (dashboard.py)
              │
       (REST / Port 8000)
              ▼
    [ FastAPI Backend API ] (app/main.py)
      ├── SQLite Database (bharat_banking.db) ─── Telemetry storage
      ├── ML Engine (efficiency_model.pkl)    ─── Random Forest Regressor
      ├── NLP Pipeline (app/nlp_service.py)   ─── DistilBERT Sentiment
      └── GenAI Engine (app/gemini_service.py) ─── Google Gemini API
```

---

## 📁 Project Directory Structure

```text
Bank_Branch_Performance_Dashboard/
├── app/
│   ├── __init__.py           # Application package init
│   ├── config.py             # Centralized environment & settings loader
│   ├── gemini_service.py     # Official google-genai integration & banking prompts
│   ├── main.py               # FastAPI backend with health check & Pydantic models
│   └── nlp_service.py        # HuggingFace sentiment analysis pipeline
├── .env.example              # Template for API keys and environment variables
├── .gitignore                # Safeguards secrets, local data, and caches
├── dashboard.py              # Streamlit command center frontend
├── migrate_to_sql.py         # CSV to SQLite migration utility
├── README.md                 # Complete system documentation
├── requirements.txt          # Production Python package requirements
└── setup_ml.py               # Synthetic benchmark data & model training pipeline
```

The generated database, model, and benchmark CSV are excluded from Git. Generate them locally before starting the application:
```bash
python setup_ml.py
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
* Python 3.9, 3.10, 3.11, 3.12, or 3.13
* Virtual environment (recommended)

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Configure Gemini AI API Key (Required for live AI Chat)
1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/).
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and set your key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```
> **Security Note:** The `.env` file is excluded in `.gitignore` and must never be committed to Git. The dashboard also permits entering an optional session key directly in the UI settings expander.

---

## 🚀 Running the Application

### Step 1: Start the FastAPI Backend (Terminal 1)
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
* **API Root**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **System Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Step 2: Start the Streamlit Dashboard (Terminal 2)
```bash
python -m streamlit run dashboard.py
```
* **Dashboard URL**: [http://localhost:8501](http://localhost:8501)

---

## 🔑 Demo Access Credentials

| Role | Username | Password | Scope & Privileges |
| :--- | :--- | :--- | :--- |
| **National Head (CEO)** | `ceo` | `admin123` | Nationwide overview, rankings, all branch deep dives |
| **Branch Manager** | `manager` | `admin123` | Assigned branch deep dive, monthly data upload portal |

---

## 🗄️ Database & Machine Learning Details

### SQLite Database (`bharat_banking.db`)
* **Table**: `branch_performance`
* **Schema**:
  * `branch_name` (TEXT): Metropolitan branch title (e.g. `SBI - Mumbai`, `HDFC - Delhi`)
  * `region` (TEXT): Geographic quadrant (`North`, `South`, `East`, `West`)
  * `month` (TEXT): Reporting period (`Jan` to `Jun`)
  * `lat`, `lon` (REAL): Coordinates for spatial mapping
  * `staff_count` (INTEGER): Active branch personnel
  * `daily_transactions` (INTEGER): Daily counter and digital clearing count
  * `customer_satisfaction` (REAL): Survey score (1.0 – 5.0)
  * `efficiency_score` (REAL): Composite operational efficiency percentage (50.0% – 99.9%)

### Machine Learning Model (`efficiency_model.pkl`)
* **Algorithm**: `RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)`
* **Input Features**:
  1. `staff_count` (int)
  2. `daily_transactions` (int)
  3. `customer_satisfaction` (float)
* **Target Output**: `efficiency_score` (%)

To retrain the model and regenerate benchmark data at any time:
```bash
python setup_ml.py
```

---

## 🛠️ Troubleshooting & FAQ

1. **"Google Gemini API Key is not configured" message in chatbot:**
   * Ensure `GEMINI_API_KEY` is saved in your `.env` file in the project root, or paste the key into the chatbot's configuration expander in the UI.
2. **Backend API connection error in Streamlit:**
   * Verify that FastAPI is running on `http://127.0.0.1:8000`. You can test this by visiting `http://127.0.0.1:8000/api/health` in your browser.
3. **Port 8000 or 8501 already in use:**
   * On Windows PowerShell, check running processes with `netstat -ano | findstr "8000 8501"`. You can specify alternate ports using `uvicorn app.main:app --port 8001` and updating `API_PORT` in `.env`.

---

## 📋 Known Limitations

* **Synthetic Benchmark Baseline**: The historical records in `bharat_banking.db` represent synthetic operational telemetry designed for enterprise demonstration and training purposes.
* **External AI Rate Limits**: Live Gemini API responses are governed by your Google AI Studio account quota and tier. Free-tier keys may experience temporary rate limits during burst requests.
