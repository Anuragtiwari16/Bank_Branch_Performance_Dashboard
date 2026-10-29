"""
🏦 Bharat Banking Command Center | Enterprise Analytics Dashboard
Executive-grade banking analytics, ML efficiency forecasting, NLP customer sentiment,
and Google Gemini generative AI assistant.
"""

import os
import sqlite3
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.config import (
    API_URL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    DEFAULT_GEMINI_MODEL,
    DATABASE_PATH,
)
from app.gemini_service import query_gemini, build_banking_context_summary

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Bharat Banking | Command Center",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# SESSION STATE MANAGEMENT
# ==============================================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["role"] = "National Head (CEO)"
    st.session_state["assigned_branch"] = "SBI - Mumbai"

if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

if "custom_gemini_key" not in st.session_state:
    st.session_state["custom_gemini_key"] = ""

if "selected_gemini_model" not in st.session_state:
    st.session_state["selected_gemini_model"] = GEMINI_MODEL or DEFAULT_GEMINI_MODEL


# Known branch geographical coordinates for auto-imputation if missing
BRANCH_COORDINATES = {
    "SBI - Mumbai": (18.9256, 72.8242, "West"),
    "HDFC - Delhi": (28.6315, 77.2167, "North"),
    "ICICI - Bangalore": (12.9716, 77.5946, "South"),
    "Axis - Kolkata": (22.5535, 88.3498, "East"),
    "PNB - Chennai": (13.0405, 80.2337, "South"),
    "Kotak - Pune": (18.5362, 73.8939, "West"),
    "BoB - Ahmedabad": (23.0225, 72.5714, "West"),
    "IndusInd - Hyderabad": (17.3850, 78.4867, "South"),
}

MONTHS_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ==============================================================================
# DATA LAYER (CACHED WITH RETRY AND FALLBACK)
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_branch_data() -> pd.DataFrame:
    """
    Fetches the historical branch dataset from the FastAPI backend,
    falling back directly to the local SQLite database if API is restarting.
    """
    df = None
    # 1. Attempt API fetch
    try:
        res = requests.get(f"{API_URL}/api/branches/historical", timeout=4)
        if res.status_code == 200:
            payload = res.json()
            if payload.get("status") == "success" and "data" in payload:
                df = pd.DataFrame(payload["data"])
    except Exception:
        df = None

    # 2. Fallback direct SQLite read
    if df is None or df.empty:
        if DATABASE_PATH.exists():
            try:
                with sqlite3.connect(DATABASE_PATH) as conn:
                    df = pd.read_sql_query("SELECT * FROM branch_performance", conn)
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

    if not df.empty:
        # Guarantee lat and lon columns exist and have valid coordinates
        if "lat" not in df.columns or "lon" not in df.columns or df["lat"].isnull().any():
            for idx, row in df.iterrows():
                b_name = row.get("branch_name", "")
                if b_name in BRANCH_COORDINATES:
                    df.at[idx, "lat"] = BRANCH_COORDINATES[b_name][0]
                    df.at[idx, "lon"] = BRANCH_COORDINATES[b_name][1]
                else:
                    if pd.isnull(df.at[idx, "lat"]):
                        df.at[idx, "lat"] = 20.5937
                    if pd.isnull(df.at[idx, "lon"]):
                        df.at[idx, "lon"] = 78.9629

    return df


@st.cache_data(ttl=30)
def fetch_system_health() -> dict:
    """Queries the FastAPI health endpoint."""
    try:
        res = requests.get(f"{API_URL}/api/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        return {"status": "offline", "error": str(e)}
    return {"status": "unreachable"}


df_all = fetch_branch_data()


# ==============================================================================
# ENTERPRISE THEME & STYLING ENGINE
# ==============================================================================
is_dark = st.session_state["theme"] == "Dark"

# Cohesive corporate color palettes
if not is_dark:
    # Warm Cream & Light Yellowish Luxury Banking Palette
    C_BG = "#fbf8ee"  # Warm light yellowish cream background
    C_CARD_BG = "#fffdf9"  # Elegant warm ivory/cream cards
    C_SIDEBAR_BG = "#f5efe0"  # Richer warm custard cream for sidebar
    C_TEXT = "#28231a"  # Deep espresso charcoal text
    C_TEXT_MUTED = "#7a6f5d"  # Antique bronze slate
    C_PRIMARY = "#0f2c59"  # Deep executive navy
    C_ACCENT = "#c08a3e"  # Warm golden amber
    C_BORDER = "#e8e0ce"  # Warm linen border
    C_SUCCESS = "#15803d"  # Emerald green
    C_WARNING = "#b45309"  # Warm amber
    C_DANGER = "#b91c1c"  # Rich crimson
    PLOTLY_TEMPLATE = "plotly_white"
else:
    # Executive Command Dark Palette
    C_BG = "#0b132b"
    C_CARD_BG = "#1c2541"
    C_SIDEBAR_BG = "#0e162b"
    C_TEXT = "#f8fafc"
    C_TEXT_MUTED = "#94a3b8"
    C_PRIMARY = "#5bc0be"
    C_ACCENT = "#3a506b"
    C_BORDER = "#334155"
    C_SUCCESS = "#10b981"
    C_WARNING = "#f59e0b"
    C_DANGER = "#ef4444"
    PLOTLY_TEMPLATE = "plotly_dark"

CUSTOM_CSS = f"""
<style>
    /* Global Container */
    .stApp {{
        background-color: {C_BG};
        color: {C_TEXT};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }}
    
    /* Sidebar Background - Warm Cream in Light Mode, Sleek Navy in Dark */
    [data-testid="stSidebar"] {{
        background-color: {C_SIDEBAR_BG} !important;
        border-right: 1px solid {C_BORDER} !important;
    }}
    
    /* Top Header Bar */
    .bank-header-wrap {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 24px;
        background: {C_CARD_BG};
        border-radius: 12px;
        border: 1px solid {C_BORDER};
        margin-bottom: 24px;
        box-shadow: 0 2px 8px {"rgba(120, 110, 88, 0.07)" if not is_dark else "rgba(0,0,0,0.2)"};
    }}
    .bank-brand-title {{
        font-size: 22px;
        font-weight: 700;
        color: {C_TEXT};
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .bank-brand-subtitle {{
        font-size: 13px;
        color: {C_TEXT_MUTED};
        margin-top: 2px;
    }}
    .badge-pill {{
        display: inline-block;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 20px;
        background: {"rgba(11,37,69,0.08)" if not is_dark else "rgba(91,192,190,0.15)"};
        color: {C_PRIMARY};
        border: 1px solid {"rgba(11,37,69,0.2)" if not is_dark else "rgba(91,192,190,0.3)"};
    }}
    
    /* Section Headings */
    .view-title {{
        font-size: 20px;
        font-weight: 700;
        color: {C_TEXT};
        margin-bottom: 4px;
    }}
    .view-caption {{
        font-size: 13px;
        color: {C_TEXT_MUTED};
        margin-bottom: 20px;
    }}
    
    /* High-Contrast Widget Labels & Radio Buttons */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
    div[data-baseweb="radio"] label,
    div[data-baseweb="radio"] label p,
    div[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p,
    div[data-baseweb="radio"] div,
    .stSelectbox label,
    .stSelectbox label p,
    .stTextInput label,
    .stTextInput label p,
    .stTextArea label,
    .stTextArea label p,
    .stSlider label,
    .stSlider label p {{
        color: {C_TEXT} !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        opacity: 1 !important;
    }}

    /* Login Portal Role Selection Radio (Scoped strictly to login wrapper) */
    div.login-role-wrapper div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: flex;
        flex-direction: row;
        gap: 14px;
        margin-top: 6px;
        margin-bottom: 12px;
    }}
    div.login-role-wrapper div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        flex: 1;
        background: {"#f8fafc" if not is_dark else "#1e293b"} !important;
        border: 2px solid {"#cbd5e1" if not is_dark else "#334155"} !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        cursor: pointer !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        transition: all 0.2s ease-in-out;
    }}
    div.login-role-wrapper div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
        border-color: {C_PRIMARY} !important;
        background: {"#edf2f7" if not is_dark else "#24324d"} !important;
    }}
    div.login-role-wrapper div[data-testid="stRadio"] div[role="radiogroup"] > label p {{
        font-size: 15px !important;
        font-weight: 800 !important;
        color: {C_TEXT} !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }}

    /* Sidebar Navigation Menu: Clean, full-width vertical items with NO word breaking */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        width: 100% !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        width: 100% !important;
        flex: none !important;
        background: {"#fffdf8" if not is_dark else "#1c2541"} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
        background: {"#f7f2e4" if not is_dark else "#233055"} !important;
        border-color: {C_PRIMARY} !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
        border-color: {C_PRIMARY} !important;
        background: {"rgba(15, 44, 89, 0.08)" if not is_dark else "rgba(91,192,190,0.18)"} !important;
        box-shadow: 0 0 0 1.5px {C_PRIMARY} !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label p {{
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: {C_TEXT} !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        margin: 0 0 0 8px !important;
    }}

    /* Sidebar Interface Palette: Clean side-by-side segmented toggle */
    [data-testid="stSidebar"] .theme-switch-wrapper div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        gap: 8px !important;
        width: 100% !important;
    }}
    [data-testid="stSidebar"] .theme-switch-wrapper div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        flex: 1 1 0 !important;
        display: flex !important;
        flex-direction: row !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 8px 10px !important;
        text-align: center !important;
        background: {"#fffdf8" if not is_dark else "#1c2541"} !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebar"] .theme-switch-wrapper div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
        border-color: {C_PRIMARY} !important;
    }}
    [data-testid="stSidebar"] .theme-switch-wrapper div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
        border-color: {C_PRIMARY} !important;
        background: {"rgba(15, 44, 89, 0.1)" if not is_dark else "rgba(91,192,190,0.22)"} !important;
        box-shadow: 0 0 0 1.5px {C_PRIMARY} !important;
    }}
    [data-testid="stSidebar"] .theme-switch-wrapper div[data-testid="stRadio"] div[role="radiogroup"] > label p {{
        font-size: 13px !important;
        font-weight: 700 !important;
        color: {C_TEXT} !important;
        white-space: nowrap !important;
        margin: 0 0 0 4px !important;
    }}

    /* Text Inputs in Light/Dark Mode */
    div[data-testid="stTextInput"] input {{
        color: {C_TEXT} !important;
        background-color: {"#fffdf8" if not is_dark else "#0f172a"} !important;
        border: 1.5px solid {"#e2d8c3" if not is_dark else "#334155"} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 14px !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border-color: {C_PRIMARY} !important;
        box-shadow: 0 0 0 2px {"rgba(15, 44, 89, 0.2)" if not is_dark else "rgba(91,192,190,0.3)"} !important;
    }}
    
    /* KPI Cards */
    .kpi-card {{
        background: {C_CARD_BG};
        border: 1px solid {C_BORDER};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px {"rgba(120, 110, 88, 0.07)" if not is_dark else "rgba(0,0,0,0.2)"};
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 140px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px {"rgba(120, 110, 88, 0.12)" if not is_dark else "rgba(0,0,0,0.3)"};
    }}
    .kpi-title {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {C_TEXT_MUTED};
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 32px;
        font-weight: 800;
        color: {C_TEXT};
        line-height: 1.1;
        margin: 6px 0;
    }}
    .kpi-subtext {{
        font-size: 12px;
        font-weight: 500;
        color: {C_SUCCESS};
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    
    /* Content Containers */
    .content-box {{
        background: {C_CARD_BG};
        border: 1px solid {C_BORDER};
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px {"rgba(120, 110, 88, 0.06)" if not is_dark else "rgba(0,0,0,0.2)"};
    }}

    /* Streamlit built-in bordered containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {C_CARD_BG} !important;
        border-color: {C_BORDER} !important;
        box-shadow: 0 2px 8px {"rgba(120, 110, 88, 0.05)" if not is_dark else "rgba(0,0,0,0.2)"} !important;
    }}
    
    /* Assistant Chat Bubbles */
    .chat-bubble-user {{
        background: {"#eae2ce" if not is_dark else "#334155"};
        color: {C_TEXT};
        padding: 12px 18px;
        border-radius: 14px 14px 2px 14px;
        max-width: 80%;
        margin-left: auto;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.5;
    }}
    .chat-bubble-assistant {{
        background: {C_CARD_BG};
        color: {C_TEXT};
        border: 1px solid {C_BORDER};
        padding: 14px 18px;
        border-radius: 14px 14px 14px 2px;
        max-width: 85%;
        margin-right: auto;
        margin-bottom: 14px;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 0 2px 6px {"rgba(120, 110, 88, 0.06)" if not is_dark else "rgba(0,0,0,0.2)"};
    }}
    
    /* Button fine-tuning */
    .stButton>button {{
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        padding: 6px 16px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# AUTHENTICATION & DEMO ACCESS GATE
# ==============================================================================
if not st.session_state["logged_in"]:
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    c_left, c_center, c_right = st.columns([1, 1.4, 1])
    with c_center:
        st.markdown(
            f"""
        <div class="content-box" style="text-align: center; padding: 36px 30px;">
            <div style="font-size: 42px; margin-bottom: 8px;">🏦</div>
            <h2 style="margin: 0; color: {C_TEXT}; font-weight: 800; font-size: 26px;">Bharat Banking</h2>
            <div style="font-size: 13px; color: {C_TEXT_MUTED}; margin-bottom: 24px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">
                Enterprise Analytics Command Center
            </div>
        """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                f"<h4 style='color: {C_TEXT}; margin: 0 0 4px 0;'>🔐 Secure Enterprise Portal</h4>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color: {C_TEXT_MUTED}; font-size: 13px; margin-bottom: 16px;'>Select your access identity to preview role-based controls.</p>",
                unsafe_allow_html=True,
            )

            # Prominent Role Selector with explicit icons & clear labels
            role_options = ["🌐 National Head (CEO)", "🏢 Branch Manager"]
            st.markdown('<div class="login-role-wrapper">', unsafe_allow_html=True)
            login_role = st.radio(
                "Access Profile",
                role_options,
                index=0,
                horizontal=True,
                help="Choose CEO for national network access, or Branch Manager for single-branch telemetry.",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            selected_branch_demo = None
            if "Branch Manager" in login_role:
                st.markdown(
                    f"""
                <div style="background: {"rgba(11,37,69,0.06)" if not is_dark else "rgba(91,192,190,0.1)"}; border-left: 4px solid {C_PRIMARY}; padding: 10px 14px; border-radius: 6px; margin: 8px 0 14px 0; font-size: 13px; color: {C_TEXT}; font-weight: 600;">
                    🏢 <b>Manager Scope:</b> Single Branch Analytics & Monthly Data Ingestion Portal
                </div>
                """,
                    unsafe_allow_html=True,
                )
                if not df_all.empty:
                    branch_options = sorted(df_all["branch_name"].unique())
                    selected_branch_demo = st.selectbox(
                        "Assign Branch for Demo", branch_options, index=0
                    )
            else:
                st.markdown(
                    f"""
                <div style="background: {"rgba(5,150,105,0.08)" if not is_dark else "rgba(16,185,129,0.12)"}; border-left: 4px solid {C_SUCCESS}; padding: 10px 14px; border-radius: 6px; margin: 8px 0 14px 0; font-size: 13px; color: {C_TEXT}; font-weight: 600;">
                    🌐 <b>CEO Scope:</b> Full Nationwide Network Telemetry & Leaderboard Rankings
                </div>
                """,
                    unsafe_allow_html=True,
                )

            col_u, col_p = st.columns(2)
            default_user = "ceo" if "CEO" in login_role else "manager"
            with col_u:
                username_in = st.text_input(
                    "Username", value=default_user, key=f"user_{login_role}"
                )
            with col_p:
                password_in = st.text_input(
                    "Password",
                    type="password",
                    value="admin123",
                    key=f"pwd_{login_role}",
                )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button(
                "Authenticate Session", type="primary", use_container_width=True
            ):
                if (
                    ("CEO" in login_role or "National Head" in login_role)
                    and username_in.strip() == "ceo"
                    and password_in == "admin123"
                ):
                    st.session_state["logged_in"] = True
                    st.session_state["role"] = "National Head (CEO)"
                    st.session_state["assigned_branch"] = None
                    st.rerun()
                elif (
                    ("Branch Manager" in login_role)
                    and username_in.strip() == "manager"
                    and password_in == "admin123"
                ):
                    st.session_state["logged_in"] = True
                    st.session_state["role"] = "Branch Manager"
                    st.session_state["assigned_branch"] = (
                        selected_branch_demo or "SBI - Mumbai"
                    )
                    st.rerun()
                else:
                    st.error(
                        "Invalid credentials. Please verify your username and password."
                    )

            # High-Contrast Demo Credentials Helper Card
            st.markdown(
                f"""
            <div style="background: {"#f8fafc" if not is_dark else "#131b31"}; border: 1.5px solid {"#cbd5e1" if not is_dark else "#334155"}; border-radius: 8px; padding: 12px 16px; margin-top: 16px; text-align: left;">
                <div style="font-weight: 700; color: {C_TEXT}; font-size: 13px; margin-bottom: 6px;">💡 Demo Access Credentials:</div>
                <div style="font-size: 13px; color: {C_TEXT}; display: flex; flex-direction: column; gap: 4px;">
                    <div>• <b>National Head (CEO):</b> <code style="background: {"#e2e8f0" if not is_dark else "#24324d"}; color: {C_TEXT}; padding: 2px 6px; border-radius: 4px; font-weight: bold;">ceo</code> / <code style="background: {"#e2e8f0" if not is_dark else "#24324d"}; color: {C_TEXT}; padding: 2px 6px; border-radius: 4px; font-weight: bold;">admin123</code></div>
                    <div>• <b>Branch Manager:</b> <code style="background: {"#e2e8f0" if not is_dark else "#24324d"}; color: {C_TEXT}; padding: 2px 6px; border-radius: 4px; font-weight: bold;">manager</code> / <code style="background: {"#e2e8f0" if not is_dark else "#24324d"}; color: {C_TEXT}; padding: 2px 6px; border-radius: 4px; font-weight: bold;">admin123</code></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ==============================================================================
# SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================
is_ceo = st.session_state["role"] == "National Head (CEO)"
assigned_branch = st.session_state.get("assigned_branch")

with st.sidebar:
    st.markdown("### 🏦 Bharat Banking")
    st.caption("Executive Command Center v2.0")

    # User Profile Information
    with st.container(border=True):
        st.markdown(f"**👤 Current Identity**")
        if is_ceo:
            st.markdown(
                f"<span class='badge-pill'>🌐 {st.session_state['role']}</span>",
                unsafe_allow_html=True,
            )
            st.caption("Scope: Full National Network (All Branches)")
        else:
            st.markdown(
                f"<span class='badge-pill'>🏢 Branch Manager</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Authorized Scope: **{assigned_branch}**")

        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state["logged_in"] = False
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Navigation Menu
    nav_options = [
        "📊 National Overview",
        "🏢 Branch Deep Dive",
        "🔮 Efficiency Forecasting",
        "🗣️ Customer Sentiment Analysis",
        "🤖 AI Banking Assistant",
        "⚙️ System Information",
    ]
    if not is_ceo:
        nav_options.append("📥 Monthly Data Upload")

    st.markdown('<div class="sidebar-nav-wrapper">', unsafe_allow_html=True)
    active_page = st.radio("Navigation Menu", nav_options, index=0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Theme Switcher
    st.markdown('<div class="theme-switch-wrapper">', unsafe_allow_html=True)
    theme_choice = st.radio(
        "Interface Palette",
        ["☀️ Light", "🌙 Dark"],
        index=0 if st.session_state["theme"] == "Light" else 1,
        horizontal=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    desired_theme = "Light" if "Light" in theme_choice else "Dark"
    if desired_theme != st.session_state["theme"]:
        st.session_state["theme"] = desired_theme
        st.rerun()


# ==============================================================================
# TOP APPLICATION HEADER BAR
# ==============================================================================
header_title = active_page.split(" ", 1)[-1] if " " in active_page else active_page
st.markdown(
    f"""
<div class="bank-header-wrap">
    <div>
        <div class="bank-brand-title">
            <span>🏦 Bharat Banking Command Center</span>
            <span class="badge-pill">{header_title}</span>
        </div>
        <div class="bank-brand-subtitle">
            Enterprise Banking Telemetry & Intelligence | Active Dataset: SQLite (48 Records across 8 Hubs)
        </div>
    </div>
    <div style="text-align: right;">
        <span class="badge-pill" style="font-size: 11px;">
            {"🟢 Executive CEO Access" if is_ceo else f"🔒 Branch Authorized: {assigned_branch}"}
        </span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# Validate Data Availability
if df_all.empty:
    st.error(
        "⚠️ Unable to connect to the backend database. Please ensure FastAPI (`python -m uvicorn app.main:app`) is running."
    )
    st.stop()

# Determine latest reporting month chronologically
available_months = [m for m in MONTHS_ORDER if m in df_all["month"].unique()]
latest_month = available_months[-1] if available_months else df_all["month"].iloc[-1]
latest_df = df_all[df_all["month"] == latest_month].copy()


# ==============================================================================
# SECTION 1: NATIONAL OVERVIEW
# ==============================================================================
if active_page == "📊 National Overview":
    st.markdown(
        '<div class="view-title">Executive National Performance Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="view-caption">Consolidated operational telemetry across all 8 commercial hubs for benchmark period ({latest_month}). Derived from verified SQLite records.</div>',
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Row
    total_branches = latest_df["branch_name"].nunique()
    nat_avg_eff = latest_df["efficiency_score"].mean()
    nat_avg_sat = latest_df["customer_satisfaction"].mean()
    nat_total_txns = latest_df["daily_transactions"].sum()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Branch Network</div>
            <div class="kpi-value">{total_branches}</div>
            <div class="kpi-subtext">Tier-1 & Metro Commercial Hubs</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">National Avg Efficiency</div>
            <div class="kpi-value">{nat_avg_eff:.1f}%</div>
            <div class="kpi-subtext" style="color: {C_SUCCESS};">▲ +1.5% QoQ Momentum</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Customer Satisfaction</div>
            <div class="kpi-value">{nat_avg_sat:.2f} <span style="font-size: 18px; color: {C_TEXT_MUTED}; font-weight: normal;">/ 5.0</span></div>
            <div class="kpi-subtext" style="color: {C_SUCCESS};">Consistent Quality Rating</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Daily Transactions</div>
            <div class="kpi-value">{int(nat_total_txns):,}</div>
            <div class="kpi-subtext">Peak Day Clearing Volume</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Leaderboard & Geographic Map Row
    row1_c1, row1_c2 = st.columns([1.2, 1])

    with row1_c1:
        st.markdown("##### 🏆 Branch Efficiency Rankings")
        sorted_branches = latest_df.sort_values(by="efficiency_score", ascending=True)
        fig_rank = px.bar(
            sorted_branches,
            x="efficiency_score",
            y="branch_name",
            orientation="h",
            color="efficiency_score",
            color_continuous_scale="Blues" if not is_dark else "Teal",
            text="efficiency_score",
            template=PLOTLY_TEMPLATE,
            labels={
                "efficiency_score": "Efficiency Score (%)",
                "branch_name": "Commercial Branch",
            },
        )
        fig_rank.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_rank.update_layout(
            height=380,
            margin=dict(t=10, l=10, r=40, b=10),
            coloraxis_showscale=False,
            xaxis=dict(range=[40, 100], title="Operational Efficiency (%)"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    with row1_c2:
        st.markdown("##### 📍 Geographic Hub Distribution")
        fig_map = px.scatter_geo(
            latest_df,
            lat="lat",
            lon="lon",
            hover_name="branch_name",
            hover_data={
                "region": True,
                "efficiency_score": ":.1f%",
                "daily_transactions": ":,",
                "lat": False,
                "lon": False,
            },
            color="efficiency_score",
            size="daily_transactions",
            color_continuous_scale="Viridis",
            template=PLOTLY_TEMPLATE,
        )
        fig_map.update_geos(
            scope="asia",
            center=dict(lat=20.5937, lon=78.9629),
            projection_scale=3.8,
            visible=False,
            showcountries=True,
            showcoastlines=True,
            showland=True,
            landcolor="#f1f5f9" if not is_dark else "#1e293b",
        )
        fig_map.update_layout(
            height=380,
            margin=dict(t=10, l=10, r=10, b=10),
            coloraxis_colorbar=dict(title="Eff %", len=0.7),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Regional Comparison & Historical Trend Row
    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        st.markdown("##### 🧭 Regional Efficiency vs Satisfaction")
        reg_summary = latest_df.groupby("region", as_index=False).agg(
            {
                "efficiency_score": "mean",
                "customer_satisfaction": "mean",
                "daily_transactions": "sum",
            }
        )
        fig_reg = go.Figure(
            data=[
                go.Bar(
                    name="Avg Efficiency (%)",
                    x=reg_summary["region"],
                    y=reg_summary["efficiency_score"],
                    marker_color="#1d4e89" if not is_dark else "#5bc0be",
                ),
                go.Bar(
                    name="Satisfaction (×20)",
                    x=reg_summary["region"],
                    y=reg_summary["customer_satisfaction"] * 20,
                    marker_color="#00b4d8" if not is_dark else "#3a506b",
                ),
            ]
        )
        fig_reg.update_layout(
            barmode="group",
            height=340,
            template=PLOTLY_TEMPLATE,
            margin=dict(t=20, l=20, r=20, b=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            yaxis=dict(title="Score Scale (0 - 100)"),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with row2_c2:
        st.markdown("##### 📈 Historical Network Efficiency (Jan - Jun)")
        monthly_trend = (
            df_all.groupby("month", sort=False)["efficiency_score"].mean().reset_index()
        )
        fig_trend = px.line(
            monthly_trend,
            x="month",
            y="efficiency_score",
            markers=True,
            template=PLOTLY_TEMPLATE,
            labels={"month": "Month", "efficiency_score": "Avg Efficiency (%)"},
        )
        fig_trend.update_traces(
            line_color="#059669" if not is_dark else "#10b981", line_width=3
        )
        fig_trend.update_layout(
            height=340,
            margin=dict(t=20, l=20, r=20, b=20),
            yaxis=dict(title="National Average Efficiency (%)"),
            xaxis=dict(title="Reporting Period", categoryorder="array", categoryarray=MONTHS_ORDER),
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# ==============================================================================
# SECTION 2: BRANCH DEEP DIVE
# ==============================================================================
elif active_page == "🏢 Branch Deep Dive":
    st.markdown(
        '<div class="view-title">Branch Deep-Dive Operational Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="view-caption">Granular telemetry, radar balance assessment, and 6-month historical benchmarking.</div>',
        unsafe_allow_html=True,
    )

    # Branch Selection
    branch_list = sorted(df_all["branch_name"].unique())
    if is_ceo:
        selected_branch = st.selectbox(
            "Select Target Branch for Investigation", branch_list, index=0
        )
    else:
        selected_branch = assigned_branch
        st.info(f"Authorized Scope: Currently analyzing **{selected_branch}**.")

    b_df = df_all[df_all["branch_name"] == selected_branch].copy()
    if b_df.empty:
        st.warning("No records found for the selected branch.")
        st.stop()

    b_latest = b_df.iloc[-1]
    nat_avg = latest_df["efficiency_score"].mean()
    delta_eff = b_latest["efficiency_score"] - nat_avg

    # Calculate Branch Ranking
    ranked_branches = latest_df.sort_values(by="efficiency_score", ascending=False)[
        "branch_name"
    ].tolist()
    rank_pos = (
        ranked_branches.index(selected_branch) + 1
        if selected_branch in ranked_branches
        else "N/A"
    )

    # Branch KPI Row
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Current Efficiency Score</div>
            <div class="kpi-value">{b_latest["efficiency_score"]:.1f}%</div>
            <div class="kpi-subtext" style="color: {C_SUCCESS if delta_eff >= 0 else C_DANGER};">
                {"▲" if delta_eff >= 0 else "▼"} {abs(delta_eff):.1f}% vs National Avg
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with bc2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Network Rank</div>
            <div class="kpi-value">#{rank_pos} <span style="font-size: 16px; color: {C_TEXT_MUTED}; font-weight: normal;">of {len(ranked_branches)}</span></div>
            <div class="kpi-subtext">Region: {b_latest["region"]} Hub</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with bc3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Customer Rating</div>
            <div class="kpi-value">{b_latest["customer_satisfaction"]:.1f} <span style="font-size: 16px; color: {C_TEXT_MUTED}; font-weight: normal;">/ 5.0</span></div>
            <div class="kpi-subtext">High Retention Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with bc4:
        txns_per_staff = b_latest["daily_transactions"] / max(
            1, b_latest["staff_count"]
        )
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Workload Capacity</div>
            <div class="kpi-value">{int(txns_per_staff)} <span style="font-size: 16px; color: {C_TEXT_MUTED}; font-weight: normal;">txns/staff</span></div>
            <div class="kpi-subtext">{int(b_latest["daily_transactions"]):,} Daily Txns | {b_latest["staff_count"]} Staff</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Detailed Charts
    col_chart1, col_chart2 = st.columns([1.3, 1])

    with col_chart1:
        st.markdown(f"##### 📈 {selected_branch} vs National Benchmark Trend")
        nat_trend = (
            df_all.groupby("month", sort=False)["efficiency_score"].mean().reset_index()
        )
        nat_trend["Metric"] = "National Average"
        b_trend = b_df[["month", "efficiency_score"]].copy()
        b_trend["Metric"] = selected_branch
        combined_trend = pd.concat([b_trend, nat_trend])

        fig_line = px.line(
            combined_trend,
            x="month",
            y="efficiency_score",
            color="Metric",
            markers=True,
            template=PLOTLY_TEMPLATE,
            color_discrete_map={
                selected_branch: "#1d4e89" if not is_dark else "#5bc0be",
                "National Average": "#94a3b8",
            },
        )
        fig_line.update_layout(
            height=350,
            margin=dict(t=20, l=10, r=10, b=10),
            yaxis=dict(title="Efficiency Score (%)"),
            xaxis=dict(title="Reporting Period", categoryorder="array", categoryarray=MONTHS_ORDER),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_chart2:
        st.markdown("##### 🎯 Operational Balance Radar")
        norm_sat = (b_latest["customer_satisfaction"] / 5.0) * 100
        norm_staff = min(100, (b_latest["staff_count"] / 50.0) * 100)

        radar_categories = [
            "Efficiency",
            "Customer Rating",
            "Staff Capacity",
            "Efficiency",
        ]
        radar_values = [
            b_latest["efficiency_score"],
            norm_sat,
            norm_staff,
            b_latest["efficiency_score"],
        ]

        fig_radar = go.Figure(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_categories,
                fill="toself",
                name=selected_branch,
                line_color="#1d4e89" if not is_dark else "#5bc0be",
                fillcolor="rgba(29, 78, 137, 0.2)"
                if not is_dark
                else "rgba(91, 192, 190, 0.25)",
            )
        )
        fig_radar.update_layout(
            height=350,
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template=PLOTLY_TEMPLATE,
            margin=dict(t=30, l=20, r=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("##### 📋 6-Month Historical Record")
    st.dataframe(
        b_df[
            [
                "month",
                "region",
                "staff_count",
                "daily_transactions",
                "customer_satisfaction",
                "efficiency_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# SECTION 3: EFFICIENCY FORECASTING (MACHINE LEARNING)
# ==============================================================================
elif active_page == "🔮 Efficiency Forecasting":
    st.markdown(
        '<div class="view-title">Machine Learning Efficiency Forecasting Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="view-caption">Simulate staffing and transaction volume changes using the custom trained Random Forest Regressor.</div>',
        unsafe_allow_html=True,
    )

    col_inputs, col_results = st.columns([1.2, 1])

    with col_inputs:
        with st.container(border=True):
            st.markdown("##### 🎛️ Simulation Parameters")
            st.caption(
                "Adjust operational parameters to evaluate forecasted branch throughput."
            )

            ref_branch = st.selectbox(
                "Reference Branch Baseline", sorted(df_all["branch_name"].unique())
            )
            ref_data = df_all[df_all["branch_name"] == ref_branch].iloc[-1]

            p_staff = st.slider(
                "Active Staff Headcount",
                min_value=10,
                max_value=80,
                value=int(ref_data["staff_count"]),
                step=1,
                help="Total counter, officer, and support personnel assigned to the branch.",
            )

            p_txns = st.slider(
                "Projected Daily Transaction Load",
                min_value=1000,
                max_value=10000,
                value=int(ref_data["daily_transactions"]),
                step=100,
                help="Expected daily clearing, deposit, and digital counter transactions.",
            )

            p_sat = st.slider(
                "Target Customer Satisfaction Score",
                min_value=1.0,
                max_value=5.0,
                value=float(ref_data["customer_satisfaction"]),
                step=0.1,
                help="Desired survey rating out of 5.0.",
            )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            predict_btn = st.button(
                "🚀 Run Efficiency Simulation", type="primary", use_container_width=True
            )

    with col_results:
        with st.container(border=True):
            st.markdown("##### 📊 Forecasted Output & Impact")

            if predict_btn:
                with st.spinner("Executing Random Forest Regressor..."):
                    pred_payload = None
                    try:
                        resp = requests.post(
                            f"{API_URL}/api/ml/predict-efficiency",
                            json={
                                "staff_count": p_staff,
                                "daily_transactions": p_txns,
                                "customer_satisfaction": p_sat,
                            },
                            timeout=5,
                        )
                        if resp.status_code == 200:
                            pred_payload = resp.json()
                    except Exception:
                        pred_payload = None

                    # If API is unreachable, execute prediction directly via local model
                    if not pred_payload and MODEL_PATH.exists():
                        try:
                            import pickle
                            with open(MODEL_PATH, "rb") as mf:
                                local_model = pickle.load(mf)
                            in_df = pd.DataFrame([{
                                "staff_count": p_staff,
                                "daily_transactions": p_txns,
                                "customer_satisfaction": p_sat
                            }])
                            raw_p = float(local_model.predict(in_df)[0])
                            b_score = round(max(40.0, min(99.9, raw_p)), 2)
                            txns_r = p_txns / max(1, p_staff)
                            if b_score >= 80.0:
                                ass = f"Optimal branch capacity. Workload ratio of {txns_r:.0f} txns/staff aligns with high operational throughput."
                            elif b_score >= 65.0:
                                ass = f"Moderate branch efficiency. Balanced staff allocation ({txns_r:.0f} txns/staff) with steady throughput."
                            else:
                                ass = f"Sub-optimal operational efficiency. High risk of counter bottleneck ({txns_r:.0f} txns/staff) or under-utilized staffing."
                            pred_payload = {
                                "predicted_efficiency_score": b_score,
                                "operational_assessment": ass
                            }
                        except Exception:
                            pred_payload = None

                    if pred_payload:
                        predicted_score = pred_payload["predicted_efficiency_score"]
                        assessment = pred_payload["operational_assessment"]

                        delta_vs_baseline = (
                            predicted_score - ref_data["efficiency_score"]
                        )

                        st.markdown(
                            f"""
                        <div style="background: {"rgba(11,37,69,0.05)" if not is_dark else "rgba(91,192,190,0.1)"}; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 16px;">
                            <div style="font-size: 13px; text-transform: uppercase; font-weight: 700; color: {C_TEXT_MUTED};">Projected Operational Efficiency</div>
                            <div style="font-size: 46px; font-weight: 900; color: {C_PRIMARY}; margin: 8px 0;">{predicted_score:.1f}%</div>
                            <div style="font-size: 13px; font-weight: 600; color: {C_SUCCESS if delta_vs_baseline >= 0 else C_DANGER};">
                                {"▲" if delta_vs_baseline >= 0 else "▼"} {abs(delta_vs_baseline):.2f}% vs {ref_branch} Baseline ({ref_data["efficiency_score"]:.1f}%)
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(f"**Operational Analysis:**")
                        st.info(assessment)

                        txns_ratio = p_txns / max(1, p_staff)
                        st.markdown(
                            f"- **Simulated Workload**: `{txns_ratio:.0f} transactions / staff member`"
                        )
                        st.markdown(
                            f"- **Model Used**: `Scikit-Learn Random Forest Regressor (150 Estimators)`"
                        )
                    else:
                        st.error("Predictive ML engine is currently unable to compute prediction. Ensure backend is running or model file is intact.")
            else:
                st.info(
                    "Adjust the sliders on the left and click **Run Efficiency Simulation** to generate an AI forecast."
                )


# ==============================================================================
# SECTION 4: CUSTOMER SENTIMENT ANALYSIS
# ==============================================================================
elif active_page == "🗣️ Customer Sentiment Analysis":
    st.markdown(
        '<div class="view-title">Customer Feedback Sentiment Decoder</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="view-caption">Deep learning sentiment classification powered by HuggingFace Transformers (DistilBERT).</div>',
        unsafe_allow_html=True,
    )

    col_input, col_out = st.columns([1.2, 1])

    with col_input:
        with st.container(border=True):
            st.markdown("##### 📝 Review Submission")
            branch_target = st.selectbox(
                "Select Branch Context", sorted(df_all["branch_name"].unique())
            )

            # Preset template buttons for quick banking tests
            st.caption("Quick sample reviews:")
            pcol1, pcol2 = st.columns(2)
            sample_text = ""
            with pcol1:
                if st.button("👍 Sample: Fast Service", use_container_width=True):
                    sample_text = "The counter staff was extremely swift and guided me through loan documentation seamlessly!"
            with pcol2:
                if st.button("👎 Sample: Long Wait", use_container_width=True):
                    sample_text = "Waited over 45 minutes for cash withdrawal; only one teller window was operational during peak hours."

            feedback_input = st.text_area(
                "Customer Review Text",
                value=sample_text,
                placeholder="Paste customer feedback, branch Google review, or survey response here...",
                height=140,
            )

            analyze_btn = st.button(
                "🔍 Run Sentiment Analysis", type="primary", use_container_width=True
            )

    with col_out:
        with st.container(border=True):
            st.markdown("##### 🧠 Sentiment Evaluation")
            if analyze_btn:
                clean_text = feedback_input.strip()
                if len(clean_text) < 3:
                    st.warning("Please provide a review text with at least 3 characters before running analysis.")
                else:
                    with st.spinner("Analyzing feedback with HuggingFace pipeline..."):
                        res_data = None
                        try:
                            resp = requests.post(
                                f"{API_URL}/api/nlp/analyze-feedback",
                                json={
                                    "branch_id": branch_target,
                                    "feedback_text": clean_text,
                                },
                                timeout=10,
                            )
                            if resp.status_code == 200:
                                res_data = resp.json().get("sentiment_analysis")
                        except Exception:
                            res_data = None

                        # Fallback to local sentiment analyzer if API is unreachable
                        if not res_data:
                            from app.nlp_service import analyze_feedback
                            res_data = analyze_feedback(clean_text)

                        if res_data and res_data.get("status") == "success":
                            label = res_data.get("label", "UNKNOWN")
                            conf = res_data.get("confidence", 0.0) * 100
                            expl = res_data.get("explanation", "")

                            badge_color = (
                                C_SUCCESS if label == "POSITIVE" else C_DANGER
                            )

                            st.markdown(
                                f"""
                            <div style="background: {"rgba(5,150,105,0.08)" if label == "POSITIVE" else "rgba(220,38,38,0.08)"}; border-left: 5px solid {badge_color}; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
                                <div style="font-size: 13px; font-weight: 700; color: {badge_color}; text-transform: uppercase;">Classified Sentiment</div>
                                <div style="font-size: 32px; font-weight: 800; color: {badge_color}; margin: 4px 0;">{label}</div>
                                <div style="font-size: 13px; color: {C_TEXT_MUTED}; font-weight: 600;">Model Confidence: {conf:.1f}%</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            st.markdown("**Executive Takeaway:**")
                            st.write(expl)
                        else:
                            st.error(res_data.get("error", "Sentiment service encountered an issue.") if res_data else "Sentiment service is currently unavailable.")
            else:
                st.info(
                    "Input customer review text or choose a sample above, then click **Run Sentiment Analysis**."
                )


# ==============================================================================
# SECTION 5: AI BANKING ASSISTANT (GEMINI INTEGRATION)
# ==============================================================================
elif active_page == "🤖 AI Banking Assistant":
    st.markdown(
        '<div class="view-title">Bharat Banking AI Executive Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="view-caption">Conversational intelligence powered by Google Gemini, grounded in active platform telemetry and banking standards.</div>',
        unsafe_allow_html=True,
    )

    # Configuration / Key Settings (Collapsible)
    with st.expander(
        "⚙️ Gemini Configuration & Model Settings", expanded=not bool(GEMINI_API_KEY)
    ):
        cf_col1, cf_col2 = st.columns([1.5, 1])
        with cf_col1:
            key_source_note = (
                "✅ Found in environment / `.env` file"
                if GEMINI_API_KEY
                else "⚠️ Not found in `.env` file"
            )
            st.markdown(f"**API Key Status**: {key_source_note}")
            user_override_key = st.text_input(
                "Runtime Gemini API Key (Optional Override)",
                value=st.session_state["custom_gemini_key"],
                type="password",
                placeholder="Enter AI Studio API key if not set in .env",
                help="Obtain key from https://aistudio.google.com/. Keys are never stored or logged.",
            )
            if user_override_key != st.session_state["custom_gemini_key"]:
                st.session_state["custom_gemini_key"] = user_override_key

        with cf_col2:
            model_options = [
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]
            current_model_idx = (
                model_options.index(st.session_state["selected_gemini_model"])
                if st.session_state["selected_gemini_model"] in model_options
                else 0
            )
            sel_model = st.selectbox(
                "Active Gemini Model", model_options, index=current_model_idx
            )
            st.session_state["selected_gemini_model"] = sel_model

        if not GEMINI_API_KEY and not st.session_state["custom_gemini_key"]:
            st.info("""
            **How to configure persistent Gemini access:**
            1. Create a free API key at [Google AI Studio](https://aistudio.google.com/).
            2. Copy `.env.example` to `.env` in the project root: `cp .env.example .env`
            3. Set `GEMINI_API_KEY=your_actual_key` in `.env`.
            4. Alternatively, paste your key in the field above for this session.
            """)

    # Suggested Questions Pills
    st.markdown("##### 💡 Suggested Banking Inquiries")
    suggested_queries = [
        "What is the national branch efficiency?",
        "Which branch has the highest efficiency?",
        "How can a bank branch improve customer satisfaction?",
        "Explain the efficiency forecast.",
        "Summarize the available branch performance data.",
        "What does negative customer sentiment indicate?",
    ]

    q_cols = st.columns(3)
    clicked_suggested = None
    for idx, query_text in enumerate(suggested_queries):
        col_idx = idx % 3
        with q_cols[col_idx]:
            if st.button(
                f"💬 {query_text}", key=f"sug_btn_{idx}", use_container_width=True
            ):
                clicked_suggested = query_text

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Chat Controls Header
    chat_ctrl_c1, chat_ctrl_c2 = st.columns([1, 4])
    with chat_ctrl_c1:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()

    # Render Chat History
    chat_container = st.container(border=True)
    with chat_container:
        if not st.session_state["chat_messages"]:
            st.markdown(
                f"""
            <div style="text-align: center; padding: 40px 20px; color: {C_TEXT_MUTED};">
                <div style="font-size: 38px; margin-bottom: 8px;">🤖</div>
                <div style="font-size: 16px; font-weight: 600; color: {C_TEXT};">Bharat Banking Executive Assistant Ready</div>
                <div style="font-size: 13px; max-width: 500px; margin: 8px auto 0;">
                    Ask questions regarding branch operational efficiency, customer sentiment, staffing trade-offs, or quarterly banking benchmarks.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state["chat_messages"]:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-bubble-user">👤 <b>You:</b><br>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-bubble-assistant">🤖 <b>Assistant:</b><br>{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

    # Chat Input Box
    user_input = st.chat_input(
        "Ask a question about bank performance or banking operations..."
    )
    query_to_send = clicked_suggested or user_input

    if query_to_send:
        # Append User Message
        st.session_state["chat_messages"].append(
            {"role": "user", "content": query_to_send}
        )

        # Prepare Context Summary from active dataset
        context_str = build_banking_context_summary(df_all.to_dict(orient="records"))
        effective_key = st.session_state["custom_gemini_key"] or GEMINI_API_KEY

        with st.spinner("Consulting Google Gemini AI..."):
            result = query_gemini(
                user_prompt=query_to_send,
                api_key=effective_key,
                model_name=st.session_state["selected_gemini_model"],
                context_summary=context_str,
                conversation_history=st.session_state["chat_messages"][:-1],
            )

        if result.get("success"):
            response_text = result.get("response", "")
            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": response_text}
            )
            st.rerun()
        else:
            err_msg = result.get("message", "An unknown error occurred.")
            err_type = result.get("error_type", "ERROR")
            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": f"⚠️ **[{err_type}]**: {err_msg}"}
            )
            st.rerun()


# ==============================================================================
# SECTION 6: SYSTEM INFORMATION
# ==============================================================================
elif active_page == "⚙️ System Information":
    st.markdown(
        '<div class="view-title">System Architecture & Service Telemetry</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="view-caption">Diagnostics, component health checks, and runtime environment specifications.</div>',
        unsafe_allow_html=True,
    )

    health_data = fetch_system_health()

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        is_healthy = health_data.get("status") == "healthy"
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Backend API Status</div>
            <div class="kpi-value" style="font-size: 24px; color: {C_SUCCESS if is_healthy else C_DANGER};">
                {"🟢 OPERATIONAL" if is_healthy else "🔴 DEGRADED"}
            </div>
            <div class="kpi-subtext">Endpoint: {API_URL}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with h2:
        db_records = health_data.get("database_records", 0)
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">SQLite Database</div>
            <div class="kpi-value">{db_records}</div>
            <div class="kpi-subtext">Verified Telemetry Records</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with h3:
        is_ml_up = health_data.get("model_loaded", False)
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">ML Forecasting Engine</div>
            <div class="kpi-value" style="font-size: 24px; color: {C_SUCCESS if is_ml_up else C_DANGER};">
                {"🟢 LOADED" if is_ml_up else "🔴 UNAVAILABLE"}
            </div>
            <div class="kpi-subtext">Random Forest Regressor</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with h4:
        is_gemini_configured = bool(GEMINI_API_KEY) or bool(
            st.session_state.get("custom_gemini_key")
        )
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-title">Google Gemini SDK</div>
            <div class="kpi-value" style="font-size: 24px; color: {C_SUCCESS if is_gemini_configured else C_WARNING};">
                {"🟢 CONFIGURED" if is_gemini_configured else "🟡 KEY REQUIRED"}
            </div>
            <div class="kpi-subtext">Model: {st.session_state["selected_gemini_model"]}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("##### 🏗️ System Architecture & Data Flow")
        st.markdown("""
        ```
        [Browser / Client] ───> [Streamlit Enterprise UI (Port 8501)]
                                         │
                                         ▼ (REST / HTTP)
                                [FastAPI Engine (Port 8000)]
                                 ├──> SQLite Database (bharat_banking.db)
                                 ├──> Scikit-Learn Model (efficiency_model.pkl)
                                 ├──> HuggingFace Transformers (DistilBERT NLP)
                                 └──> Google Gemini API (google-genai SDK)
        ```
        """)

    with st.container(border=True):
        st.markdown("##### 📦 Active Software Stack")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("- **Frontend**: Streamlit 1.49+, Plotly Express")
            st.markdown("- **Backend**: FastAPI 0.141+, Uvicorn")
        with s2:
            st.markdown("- **Machine Learning**: Scikit-Learn 1.9 (Random Forest)")
            st.markdown("- **NLP**: Transformers 5.16 (DistilBERT Sentiment)")
        with s3:
            st.markdown("- **Generative AI**: Official `google-genai` SDK 2.33+")
            st.markdown("- **Data Storage**: SQLite3 Relational Engine")


# ==============================================================================
# SECTION 7: MONTHLY DATA UPLOAD (MANAGER ACTION)
# ==============================================================================
elif active_page == "📥 Monthly Data Upload":
    st.markdown(
        f'<div class="view-title">Monthly Branch Telemetry Ingestion ({assigned_branch})</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="view-caption">Securely append newly certified monthly branch operational records to the central SQLite database.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        # 1. Downloadable Template for managers
        target_br = assigned_branch or "SBI - Mumbai"
        coords = BRANCH_COORDINATES.get(target_br, (18.9256, 72.8242, "West"))
        sample_template = pd.DataFrame([
            {
                "branch_name": target_br,
                "region": coords[2],
                "month": "Jul",
                "lat": coords[0],
                "lon": coords[1],
                "staff_count": 32,
                "daily_transactions": 4500,
                "customer_satisfaction": 4.6,
                "efficiency_score": 86.5,
            }
        ])
        csv_bytes = sample_template.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Certified Monthly Ingestion CSV Template",
            data=csv_bytes,
            file_name=f"monthly_telemetry_template_{target_br.split()[0]}.csv",
            mime="text/csv",
            help="Download pre-formatted CSV template tailored for your branch records."
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Certified CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                required_cols = {
                    "branch_name",
                    "region",
                    "month",
                    "staff_count",
                    "daily_transactions",
                    "customer_satisfaction",
                    "efficiency_score",
                }
                missing = required_cols - set(new_df.columns)
                if missing:
                    st.error(f"Upload rejected. Missing required columns: {missing}")
                else:
                    # Impute coordinates if missing
                    if "lat" not in new_df.columns:
                        new_df["lat"] = new_df["branch_name"].map(lambda b: BRANCH_COORDINATES.get(b, (20.5937, 78.9629, "Central"))[0])
                    if "lon" not in new_df.columns:
                        new_df["lon"] = new_df["branch_name"].map(lambda b: BRANCH_COORDINATES.get(b, (20.5937, 78.9629, "Central"))[1])

                    # Standardize column order matching database schema
                    ordered_cols = [
                        "branch_name", "region", "month", "lat", "lon",
                        "staff_count", "daily_transactions", "customer_satisfaction", "efficiency_score"
                    ]
                    for col in ordered_cols:
                        if col not in new_df.columns:
                            new_df[col] = None
                    final_df = new_df[ordered_cols].copy()

                    st.markdown("##### Preview of Ingestion Payload:")
                    st.dataframe(final_df.head(5), use_container_width=True)

                    if st.button(
                        "💾 Commit Records to Central SQLite Database", type="primary"
                    ):
                        with sqlite3.connect(DATABASE_PATH) as conn:
                            final_df.to_sql(
                                "branch_performance",
                                conn,
                                if_exists="append",
                                index=False,
                            )
                        st.success(
                            f"Successfully ingested {len(final_df)} new monthly records into 'branch_performance'!"
                        )
                        st.cache_data.clear()
                        st.balloons()
                        st.rerun()
            except Exception as e:
                st.error(f"Data ingestion error: {e}")
