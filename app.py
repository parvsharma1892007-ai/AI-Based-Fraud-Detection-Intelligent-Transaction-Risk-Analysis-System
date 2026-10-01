# ==============================================================================
# FraudGuard AI
# AI-Based Fraud Detection & Intelligent Transaction Risk Analysis Platform
#
# HOW TO RUN (recommended):
#     python -m streamlit run app.py
#
# Login: admin / admin123   or   analyst / analyst123
# ==============================================================================
import os
import sqlite3
import hashlib
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
# Optional dependencies — the app must never crash if these are missing.
# ------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ------------------------------------------------------------------------
# Streamlit version compatibility: newer versions replaced
# use_container_width=True with width="stretch".
# ------------------------------------------------------------------------
def _stretch_kwargs():
    try:
        from packaging.version import Version
        if Version(st.__version__) >= Version("1.50.0"):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


STRETCH = _stretch_kwargs()


def to_dt(series):
    """Robust timestamp parsing (handles mixed ISO formats in pandas 2.x)."""
    try:
        return pd.to_datetime(series, format="ISO8601")
    except Exception:
        return pd.to_datetime(series, errors="coerce")


# ==============================================================================
# PATHS & CONSTANTS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "fraud_detection.db")
DATASET_PATH = os.path.join(DATA_DIR, "transactions_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")
ISO_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
META_PATH = os.path.join(MODEL_DIR, "metadata.joblib")

RANDOM_SEED = 42

NUMERIC_FEATURES = [
    "amount", "transaction_hour", "account_age_days", "transaction_frequency",
    "location_risk", "device_risk", "previous_fraud_count",
    "transaction_velocity", "amount_deviation", "customer_risk_score",
]
CATEGORICAL_FEATURES = ["transaction_type", "merchant_category"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TRANSACTION_TYPES = ["ONLINE", "POS", "ATM", "TRANSFER", "MOBILE"]
MERCHANT_CATEGORIES = [
    "GROCERY", "ELECTRONICS", "TRAVEL", "FUEL", "ENTERTAINMENT",
    "JEWELRY", "UTILITY", "HEALTHCARE", "RESTAURANT", "ONLINE_RETAIL",
]

PAGE_LIST = [
    "Dashboard", "Transaction Analysis", "Batch Fraud Detection", "Fraud Alerts",
    "Transaction History", "ML Model Performance", "Anomaly Detection",
    "Explainable AI", "Analytics", "Reports", "System Settings",
]
ANALYST_PAGES = ["Dashboard", "Transaction Analysis", "Fraud Alerts",
                 "Transaction History", "Analytics", "Reports"]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================================================================
# SECURITY — PASSWORD HASHING
# ==============================================================================
def hash_password(password: str, salt: str = "fraudguard_ai_static_salt_2024") -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


# ==============================================================================
# DATABASE LAYER (SQLite3)
# ==============================================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT UNIQUE,
        amount REAL,
        transaction_type TEXT,
        merchant_category TEXT,
        transaction_hour INTEGER,
        account_age_days INTEGER,
        transaction_frequency INTEGER,
        location_risk REAL,
        device_risk REAL,
        previous_fraud_count INTEGER,
        transaction_velocity REAL,
        amount_deviation REAL,
        customer_risk_score REAL,
        prediction INTEGER,
        fraud_probability REAL,
        risk_score REAL,
        risk_level TEXT,
        anomaly_score REAL,
        anomaly_status TEXT,
        source TEXT,
        timestamp TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS fraud_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        risk_level TEXT,
        reason TEXT,
        status TEXT DEFAULT 'New',
        created_at TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        accuracy REAL,
        precision_score REAL,
        recall_score REAL,
        f1_score REAL,
        roc_auc REAL,
        is_selected INTEGER DEFAULT 0,
        trained_at TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    conn.commit()

    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        now = datetime.now().isoformat()
        c.execute("INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)",
                  ("admin", hash_password("admin123"), "admin", now))
        c.execute("INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)",
                  ("analyst", hash_password("analyst123"), "analyst", now))
        conn.commit()

    defaults = {
        "low_max": "30", "medium_max": "70",
        "selected_model": "",
        "weights_prob": "0.5", "weights_anomaly": "0.2",
        "weights_behavior": "0.2", "weights_amount": "0.1",
    }
    for k, v in defaults.items():
        c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))
    conn.commit()
    conn.close()


def get_setting(key, default=None):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone(); conn.close()
    return row[0] if row else default


def set_setting(key, value):
    conn = get_conn(); c = conn.cursor()
    c.execute("""INSERT INTO settings(key,value) VALUES(?,?)
                 ON CONFLICT(key) DO UPDATE SET value=excluded.value""", (key, str(value)))
    conn.commit(); conn.close()


def get_risk_weights():
    return {
        "prob": float(get_setting("weights_prob", 0.5)),
        "anomaly": float(get_setting("weights_anomaly", 0.2)),
        "behavior": float(get_setting("weights_behavior", 0.2)),
        "amount": float(get_setting("weights_amount", 0.1)),
    }


def authenticate(username: str, password: str):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username=?", (username,))
    row = c.fetchone(); conn.close()
    if row and row[0] == hash_password(password):
        return row[1]
    return None


def bulk_insert_transactions(df: pd.DataFrame, source: str = "manual"):
    """Insert prediction results and auto-generate alerts for medium/high
    risk predicted-fraud rows."""
    df = df.reset_index(drop=True)
    conn = get_conn(); c = conn.cursor()
    rows = []
    for _, r in df.iterrows():
        tid = r.get("transaction_id")
        if tid is None or pd.isna(tid) or tid == "":
            tid = f"TXN{int(datetime.now().timestamp()*1000)}{np.random.randint(1000,9999)}"
        ts = r.get("timestamp")
        if ts is None or pd.isna(ts) or ts == "":
            ts = datetime.now().isoformat()
        rows.append((
            str(tid), float(r["amount"]), str(r["transaction_type"]), str(r["merchant_category"]),
            int(r["transaction_hour"]), int(r["account_age_days"]), int(r["transaction_frequency"]),
            float(r["location_risk"]), float(r["device_risk"]), int(r["previous_fraud_count"]),
            float(r["transaction_velocity"]), float(r["amount_deviation"]), float(r["customer_risk_score"]),
            int(r["prediction"]), float(r["fraud_probability"]), float(r["risk_score"]),
            str(r["risk_level"]), float(r["anomaly_score"]), str(r["anomaly_status"]),
            source, str(ts),
        ))

    c.executemany("""INSERT OR REPLACE INTO transactions (
        transaction_id, amount, transaction_type, merchant_category, transaction_hour,
        account_age_days, transaction_frequency, location_risk, device_risk, previous_fraud_count,
        transaction_velocity, amount_deviation, customer_risk_score, prediction, fraud_probability,
        risk_score, risk_level, anomaly_score, anomaly_status, source, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    conn.commit()

    stats = get_dataset_stats()
    for i, r in df.iterrows():
        if r["risk_level"] in ("HIGH", "MEDIUM") and int(r["prediction"]) == 1:
            tid = rows[i][0]
            c.execute("SELECT COUNT(*) FROM fraud_alerts WHERE transaction_id=?", (tid,))
            if c.fetchone()[0] == 0:
                reasons = "; ".join(generate_reasons(r, stats)[:3])
                c.execute("""INSERT INTO fraud_alerts(transaction_id,risk_level,reason,status,created_at)
                             VALUES(?,?,?,?,?)""",
                          (tid, str(r["risk_level"]), reasons, "New", datetime.now().isoformat()))
    conn.commit()
    conn.close()


def fetch_transactions(limit=None) -> pd.DataFrame:
    conn = get_conn()
    q = "SELECT * FROM transactions ORDER BY timestamp DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    try:
        df = pd.read_sql_query(q, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def fetch_alerts() -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT fa.id, fa.transaction_id, t.amount, t.fraud_probability, t.risk_score,
                   fa.risk_level, t.anomaly_status, fa.reason, fa.created_at, fa.status
            FROM fraud_alerts fa
            LEFT JOIN transactions t ON fa.transaction_id = t.transaction_id
            ORDER BY fa.created_at DESC
        """, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def update_alert_status(alert_id: int, status: str):
    conn = get_conn(); c = conn.cursor()
    c.execute("UPDATE fraud_alerts SET status=? WHERE id=?", (status, alert_id))
    conn.commit(); conn.close()


# ==============================================================================
# SYNTHETIC DATASET GENERATION
# ==============================================================================
def generate_synthetic_dataset(n_samples: int = 6000, fraud_rate: float = 0.045,
                               seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    amount = np.round(rng.gamma(2.0, 80, n_samples) + 5, 2)
    transaction_type = rng.choice(TRANSACTION_TYPES, n_samples, p=[0.35, 0.25, 0.10, 0.15, 0.15])
    merchant_category = rng.choice(MERCHANT_CATEGORIES, n_samples)
    transaction_hour = rng.integers(0, 24, n_samples)
    account_age_days = rng.integers(1, 3650, n_samples)
    transaction_frequency = rng.integers(1, 50, n_samples)
    location_risk = np.round(rng.beta(2, 5, n_samples) * 100, 2)
    device_risk = np.round(rng.beta(2, 5, n_samples) * 100, 2)
    previous_fraud_count = rng.poisson(0.12, n_samples)
    transaction_velocity = np.round(rng.gamma(1.5, 2, n_samples), 2)

    avg_amount_baseline = 120.0
    amount_deviation = np.round((amount - avg_amount_baseline) / avg_amount_baseline, 2)
    customer_risk_score = np.round(
        np.clip(location_risk * 0.3 + device_risk * 0.3 + previous_fraud_count * 10
                + rng.normal(20, 10, n_samples), 0, 100), 2)

    df = pd.DataFrame({
        "amount": amount, "transaction_type": transaction_type,
        "merchant_category": merchant_category, "transaction_hour": transaction_hour,
        "account_age_days": account_age_days, "transaction_frequency": transaction_frequency,
        "location_risk": location_risk, "device_risk": device_risk,
        "previous_fraud_count": previous_fraud_count, "transaction_velocity": transaction_velocity,
        "amount_deviation": amount_deviation, "customer_risk_score": customer_risk_score,
    })

    risk_signal = (
        (df["amount"] > df["amount"].quantile(0.90)).astype(int) * 2.0
        + df["transaction_hour"].isin([0, 1, 2, 3, 4]).astype(int) * 1.5
        + (df["device_risk"] > 70).astype(int) * 2.0
        + (df["location_risk"] > 70).astype(int) * 2.0
        + (df["previous_fraud_count"] > 0).astype(int) * 2.5
        + (df["transaction_velocity"] > df["transaction_velocity"].quantile(0.90)).astype(int) * 1.5
        + (df["amount_deviation"] > 2).astype(int) * 1.5
        + (df["account_age_days"] < 30).astype(int) * 1.0
    )
    noise = rng.normal(0, 1.2, n_samples)
    score = risk_signal + noise
    threshold = np.quantile(score, 1 - fraud_rate)
    df["is_fraud"] = (score >= threshold).astype(int)

    now = datetime.now()
    offsets_minutes = rng.integers(0, 90 * 24 * 60, n_samples)
    df["timestamp"] = [(now - timedelta(minutes=int(m))).isoformat() for m in offsets_minutes]
    df["transaction_id"] = [f"TXN{100000+i}" for i in range(n_samples)]
    return df


@st.cache_data(show_spinner=False)
def get_dataset_stats():
    if not os.path.exists(DATASET_PATH):
        return {"amount_p90": 500.0, "velocity_p90": 8.0}
    df = pd.read_csv(DATASET_PATH)
    return {
        "amount_p90": float(df["amount"].quantile(0.9)),
        "velocity_p90": float(df["transaction_velocity"].quantile(0.9)),
    }


# ==============================================================================
# ML PIPELINE
# ==============================================================================
def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def train_models(df: pd.DataFrame):
    """Train LR, RF and (if available) XGBoost; select best by F1."""
    X = df[ALL_FEATURES].copy()
    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipelines, metrics = {}, {}

    lr = Pipeline([("pre", build_preprocessor()),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                              random_state=RANDOM_SEED))])
    lr.fit(X_train, y_train)
    pipelines["Logistic Regression"] = lr

    rf = Pipeline([("pre", build_preprocessor()),
                   ("clf", RandomForestClassifier(n_estimators=200, max_depth=12,
                                                  class_weight="balanced",
                                                  random_state=RANDOM_SEED, n_jobs=-1))])
    rf.fit(X_train, y_train)
    pipelines["Random Forest"] = rf

    if XGBOOST_AVAILABLE:
        try:
            neg, pos = (y_train == 0).sum(), max((y_train == 1).sum(), 1)
            xgb = Pipeline([("pre", build_preprocessor()),
                            ("clf", XGBClassifier(n_estimators=300, max_depth=5,
                                                  learning_rate=0.08, scale_pos_weight=neg / pos,
                                                  eval_metric="logloss", random_state=RANDOM_SEED))])
            xgb.fit(X_train, y_train)
            pipelines["XGBoost"] = xgb
        except Exception:
            pass

    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

    best_name = max(metrics, key=lambda n: metrics[n]["f1"])
    best_pipeline = pipelines[best_name]

    joblib.dump(best_pipeline, MODEL_PATH)
    meta = {
        "best_model": best_name, "features": ALL_FEATURES,
        "trained_at": datetime.now().isoformat(), "dataset_size": int(len(df)),
        "fraud_pct": float(y.mean() * 100),
    }
    joblib.dump(meta, META_PATH)

    conn = get_conn(); c = conn.cursor()
    c.execute("DELETE FROM model_metrics")
    now = datetime.now().isoformat()
    for name, m in metrics.items():
        c.execute("""INSERT INTO model_metrics(model_name,accuracy,precision_score,recall_score,
                     f1_score,roc_auc,is_selected,trained_at) VALUES(?,?,?,?,?,?,?,?)""",
                  (name, m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"],
                   1 if name == best_name else 0, now))
    conn.commit(); conn.close()
    set_setting("selected_model", best_name)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[NUMERIC_FEATURES])
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_SEED, n_estimators=200)
    iso.fit(X_scaled)
    joblib.dump({"scaler": scaler, "model": iso}, ISO_PATH)

    return best_name, metrics


def evaluate_saved_model():
    df = pd.read_csv(DATASET_PATH)
    X, y = df[ALL_FEATURES], df["is_fraud"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "y_test": y_test.values, "y_pred": y_pred, "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }


def get_feature_importance(model_pipeline) -> pd.DataFrame:
    clf = model_pipeline.named_steps["clf"]
    pre = model_pipeline.named_steps["pre"]
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)


# ==============================================================================
# ANOMALY DETECTION + RISK SCORING ENGINE
# ==============================================================================
def anomaly_score_for(df_features: pd.DataFrame):
    iso_bundle = joblib.load(ISO_PATH)
    scaler, model = iso_bundle["scaler"], iso_bundle["model"]
    X = scaler.transform(df_features[NUMERIC_FEATURES])
    raw_score = model.decision_function(X)
    pred = model.predict(X)
    anomaly_pct = np.clip((0.5 - raw_score) * 100, 0, 100)
    status = np.where(pred == -1, "Anomalous", "Normal")
    return anomaly_pct, status


def compute_risk_score(fraud_probability: float, anomaly_pct: float, row, w=None) -> float:
    if w is None:
        w = get_risk_weights()
    behavioral = float(np.clip(
        row["location_risk"] * 0.4 + row["device_risk"] * 0.4
        + min(row["previous_fraud_count"], 5) * 4, 0, 100))
    amount_component = float(np.clip(abs(row["amount_deviation"]) * 20, 0, 100))
    score = (w["prob"] * fraud_probability * 100 + w["anomaly"] * anomaly_pct
             + w["behavior"] * behavioral + w["amount"] * amount_component)
    return float(np.clip(score, 0, 100))


def risk_level_from_score(score: float, low_max=None, medium_max=None) -> str:
    if low_max is None:
        low_max = float(get_setting("low_max", 30))
    if medium_max is None:
        medium_max = float(get_setting("medium_max", 70))
    if score <= low_max:
        return "LOW"
    elif score <= medium_max:
        return "MEDIUM"
    return "HIGH"


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Core prediction: classifier probability + anomaly + behavior + amount."""
    df = df.reset_index(drop=True)
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(df[ALL_FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    anomaly_pct, anomaly_status = anomaly_score_for(df)

    # Read settings from DB ONCE (not once per row) — big speed-up
    w = get_risk_weights()
    low_max = float(get_setting("low_max", 30))
    medium_max = float(get_setting("medium_max", 70))

    risk_scores = np.array([
        compute_risk_score(proba[i], anomaly_pct[i], df.iloc[i], w) for i in range(len(df))
    ])
    risk_levels = np.array([risk_level_from_score(s, low_max, medium_max) for s in risk_scores])

    out = df.copy()
    out["prediction"] = pred
    out["fraud_probability"] = proba
    out["anomaly_score"] = anomaly_pct
    out["anomaly_status"] = anomaly_status
    out["risk_score"] = risk_scores
    out["risk_level"] = risk_levels
    return out


def generate_reasons(row, stats) -> list:
    reasons = []
    try:
        if row["amount"] > stats["amount_p90"]:
            reasons.append(f"Transaction amount (₹{float(row['amount']):.2f}) is unusually high (above the 90th percentile)")
        if int(row["transaction_hour"]) in [0, 1, 2, 3, 4]:
            reasons.append(f"Transaction occurred at an unusual hour ({int(row['transaction_hour'])}:00)")
        if float(row["device_risk"]) > 70:
            reasons.append(f"Elevated device risk score ({float(row['device_risk']):.1f}/100)")
        if float(row["location_risk"]) > 70:
            reasons.append(f"Elevated location risk score ({float(row['location_risk']):.1f}/100)")
        if int(row["previous_fraud_count"]) > 0:
            reasons.append(f"Account has {int(row['previous_fraud_count'])} previous fraud flag(s)")
        if float(row["transaction_velocity"]) > stats["velocity_p90"]:
            reasons.append("Unusually high transaction velocity detected")
        if abs(float(row["amount_deviation"])) > 2:
            reasons.append("Transaction amount deviates significantly from typical spending pattern")
        if int(row["account_age_days"]) < 30:
            reasons.append("Account is relatively new (less than 30 days old)")
    except Exception:
        pass
    if not reasons:
        reasons.append("No strong individual risk indicators; flagged based on combined model probability")
    return reasons


# ==============================================================================
# INITIALIZATION
# ==============================================================================
def populate_initial_transactions(df: pd.DataFrame):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM transactions")
    already = c.fetchone()[0]
    conn.close()
    if already > 0:
        return
    sample = df.sample(n=min(3000, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
    preds = predict_batch(sample)
    bulk_insert_transactions(preds, source="synthetic")


def ensure_initialized():
    init_db()
    if not os.path.exists(DATASET_PATH):
        df = generate_synthetic_dataset()
        df.to_csv(DATASET_PATH, index=False)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ISO_PATH):
        df = pd.read_csv(DATASET_PATH)
        train_models(df)
    populate_initial_transactions(pd.read_csv(DATASET_PATH))


def reset_demo_data():
    conn = get_conn(); c = conn.cursor()
    c.execute("DELETE FROM transactions")
    c.execute("DELETE FROM fraud_alerts")
    c.execute("DELETE FROM model_metrics")
    conn.commit(); conn.close()
    for p in (DATASET_PATH, MODEL_PATH, ISO_PATH, META_PATH):
        if os.path.exists(p):
            os.remove(p)
    get_dataset_stats.clear()
    df = generate_synthetic_dataset()
    df.to_csv(DATASET_PATH, index=False)
    train_models(df)
    populate_initial_transactions(df)


# ==============================================================================
# UI — CUSTOM CSS
# ==============================================================================
def apply_custom_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0f1420; }
    #MainMenu, footer {visibility: hidden;}
    h1, h2, h3, h4 { color: #eef1f8; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #131a2b; border-right: 1px solid #232b40; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #161d30, #1a2238);
        border: 1px solid #2a3350; border-radius: 12px; padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: #9aa5c0 !important; }
    .stButton>button {
        background: linear-gradient(90deg, #4f6df5, #6a4ff5); color: white;
        border: none; border-radius: 8px; font-weight: 600; padding: 0.55em 1.2em;
    }
    .stButton>button:hover { opacity: 0.9; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stExpander"] {
        background-color: #161d30; border: 1px solid #2a3350; border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE: LOGIN
# ==============================================================================
def page_login():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown("## 🛡️ FraudGuard AI")
        st.markdown("##### AI-Based Fraud Detection & Intelligent Transaction Risk Analysis Platform")
        st.markdown("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", **STRETCH):
            role = authenticate(username.strip(), password)
            if role:
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.session_state.role = role
                st.session_state.page = "Dashboard"
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown("---")
        st.info("**Demo credentials**\n\nAdmin → `admin` / `admin123`\n\nAnalyst → `analyst` / `analyst123`")


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ FraudGuard AI")
        st.caption("Fraud Detection & Risk Analysis")
        st.markdown(f"**User:** {st.session_state.username}  \n**Role:** {st.session_state.role.upper()}")
        st.markdown("---")
        pages = PAGE_LIST if st.session_state.role == "admin" else ANALYST_PAGES
        current = st.session_state.get("page", "Dashboard")
        idx = pages.index(current) if current in pages else 0
        choice = st.radio("Navigation", pages, index=idx, label_visibility="collapsed")
        st.session_state.page = choice
        st.markdown("---")
        st.caption(f"XGBoost: {'✅ available' if XGBOOST_AVAILABLE else '⚠️ not installed (using Random Forest)'}")
        st.caption(f"SHAP: {'✅ available' if SHAP_AVAILABLE else '⚠️ not installed (using feature importance)'}")
        st.markdown("---")
        if st.button("Logout", **STRETCH):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()


# ==============================================================================
# PAGE: DASHBOARD
# ==============================================================================
def page_dashboard():
    st.markdown("# Dashboard")
    st.caption("Overview of transaction monitoring and fraud detection activity")
    df = fetch_transactions()
    if df.empty:
        st.warning("No transaction data available yet.")
        return

    total = len(df)
    fraud_count = int((df["prediction"] == 1).sum())
    fraud_rate = fraud_count / total * 100 if total else 0
    high_risk = int((df["risk_level"] == "HIGH").sum())
    avg_amount = df["amount"].mean()
    alerts = fetch_alerts()
    active_alerts = int((alerts["status"] != "Resolved").sum()) if not alerts.empty else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Detected", f"{fraud_count:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("High-Risk Txns", f"{high_risk:,}")
    c5.metric("Avg Txn Value", f"₹{avg_amount:,.2f}")
    c6.metric("Active Alerts", f"{active_alerts:,}")
    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        fig = px.pie(names=df["prediction"].map({0: "Legitimate", 1: "Fraud"}),
                     title="Fraud vs Legitimate", hole=0.45)
        st.plotly_chart(fig, **STRETCH)
    with colB:
        cat = df.groupby("merchant_category")["prediction"].sum().reset_index()
        fig = px.bar(cat, x="merchant_category", y="prediction", title="Fraud by Merchant Category")
        st.plotly_chart(fig, **STRETCH)

    colC, colD = st.columns(2)
    with colC:
        tt = df.groupby("transaction_type")["prediction"].sum().reset_index()
        fig = px.bar(tt, x="transaction_type", y="prediction", color="transaction_type",
                     title="Fraud by Transaction Type")
        st.plotly_chart(fig, **STRETCH)
    with colD:
        rl = df["risk_level"].value_counts().reset_index()
        rl.columns = ["risk_level", "count"]
        fig = px.pie(rl, names="risk_level", values="count", title="Risk Distribution")
        st.plotly_chart(fig, **STRETCH)

    colE, colF = st.columns(2)
    with colE:
        d = df.copy()
        d["date"] = to_dt(d["timestamp"]).dt.date
        trend = d.groupby("date").size().reset_index(name="count")
        fig = px.line(trend, x="date", y="count", title="Transactions Over Time")
        st.plotly_chart(fig, **STRETCH)
    with colF:
        fig = px.histogram(df, x="fraud_probability", nbins=30, title="Fraud Probability Distribution")
        st.plotly_chart(fig, **STRETCH)

    fig = px.histogram(df, x="amount", nbins=40, title="Transaction Amount Distribution")
    st.plotly_chart(fig, **STRETCH)
    st.caption("Dashboard populated from synthetic/demo data plus any transactions you analyze or upload.")


# ==============================================================================
# PAGE: TRANSACTION ANALYSIS
# ==============================================================================
def page_transaction_analysis():
    st.markdown("# Transaction Analysis")
    st.caption("Analyze a single transaction for fraud risk in real time")

    with st.form("txn_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=500.0, step=10.0)
            ttype = st.selectbox("Transaction Type", TRANSACTION_TYPES)
            mcat = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
            hour = st.slider("Transaction Hour", 0, 23, 14)
        with c2:
            acc_age = st.number_input("Account Age (days)", min_value=0, value=365)
            freq = st.number_input("Transaction Frequency (per month)", min_value=0, value=10)
            loc_risk = st.slider("Location Risk (0-100)", 0, 100, 20)
            dev_risk = st.slider("Device Risk (0-100)", 0, 100, 20)
        with c3:
            prev_fraud = st.number_input("Previous Fraud Count", min_value=0, value=0)
            velocity = st.number_input("Transaction Velocity", min_value=0.0, value=2.0)
            amt_dev = st.number_input("Amount Deviation (from average)", value=0.0)
        submitted = st.form_submit_button("Analyze Transaction", **STRETCH)

    if submitted:
        try:
            cust_risk = float(np.clip(loc_risk * 0.3 + dev_risk * 0.3 + prev_fraud * 10 + 20, 0, 100))
            row = pd.DataFrame([{
                "amount": amount, "transaction_type": ttype, "merchant_category": mcat,
                "transaction_hour": hour, "account_age_days": acc_age, "transaction_frequency": freq,
                "location_risk": loc_risk, "device_risk": dev_risk, "previous_fraud_count": prev_fraud,
                "transaction_velocity": velocity, "amount_deviation": amt_dev,
                "customer_risk_score": cust_risk,
            }])
            result = predict_batch(row)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        result["transaction_id"] = [f"TXN{int(datetime.now().timestamp()*1000)}"]
        result["timestamp"] = [datetime.now().isoformat()]
        bulk_insert_transactions(result, source="manual")
        r = result.iloc[0]

        st.markdown("## Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction", "FRAUD" if r["prediction"] == 1 else "LEGITIMATE")
        c2.metric("Fraud Probability", f"{r['fraud_probability']*100:.1f}%")
        c3.metric("Risk Score", f"{r['risk_score']:.0f} / 100")
        c4.metric("Risk Level", r["risk_level"])
        st.metric("Anomaly Status", r["anomaly_status"])

        st.markdown("### Why was this transaction flagged?")
        for reason in generate_reasons(r, get_dataset_stats()):
            st.markdown(f"- {reason}")
        st.info("Risk thresholds are application-defined for demonstration and are not universal banking standards.")


# ==============================================================================
# PAGE: BATCH FRAUD DETECTION
# ==============================================================================
def page_batch_detection():
    st.markdown("# Batch Fraud Detection")
    st.caption("Upload a CSV file of transactions for bulk fraud analysis")
    st.markdown(f"**Required columns:** `{', '.join(ALL_FEATURES)}`")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
            return

        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return

        try:
            for col in NUMERIC_FEATURES:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            before = len(df)
            df = df.dropna(subset=ALL_FEATURES).reset_index(drop=True)
            if len(df) < before:
                st.warning(f"{before - len(df)} row(s) had invalid/missing values and were dropped.")
            if df.empty:
                st.error("No valid rows to process.")
                return
            with st.spinner("Running fraud detection..."):
                result = predict_batch(df)
            if "transaction_id" not in result.columns:
                result["transaction_id"] = ""
            result["transaction_id"] = result["transaction_id"].fillna("").astype(str)
            stamp = int(datetime.now().timestamp())
            result["transaction_id"] = [
                t if t != "" else f"BATCH{stamp}{i}"
                for i, t in enumerate(result["transaction_id"])
            ]
            if "timestamp" not in result.columns:
                result["timestamp"] = datetime.now().isoformat()
            bulk_insert_transactions(result, source="batch_csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return

        total = len(result)
        fraud = int((result["prediction"] == 1).sum())
        high = int((result["risk_level"] == "HIGH").sum())
        med = int((result["risk_level"] == "MEDIUM").sum())
        low = int((result["risk_level"] == "LOW").sum())

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total", total); c2.metric("Fraud", fraud); c3.metric("Legitimate", total - fraud)
        c4.metric("High Risk", high); c5.metric("Medium Risk", med); c6.metric("Low Risk", low)

        st.dataframe(result[["transaction_id", "amount", "transaction_type", "merchant_category",
                             "prediction", "fraud_probability", "risk_score", "risk_level",
                             "anomaly_status"]], **STRETCH)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(names=result["prediction"].map({0: "Legitimate", 1: "Fraud"}),
                         title="Fraud vs Legitimate")
            st.plotly_chart(fig, **STRETCH)
        with col2:
            vc = result["risk_level"].value_counts().reset_index()
            vc.columns = ["risk_level", "count"]
            fig = px.bar(vc, x="risk_level", y="count", title="Risk Level Breakdown")
            st.plotly_chart(fig, **STRETCH)

        st.download_button("Download Results as CSV", result.to_csv(index=False).encode("utf-8"),
                           "fraud_detection_results.csv", "text/csv")


# ==============================================================================
# PAGE: FRAUD ALERTS
# ==============================================================================
def page_alerts():
    st.markdown("# Fraud Alerts")
    df = fetch_alerts()
    if df.empty:
        st.info("No alerts generated yet.")
        return

    status_filter = st.multiselect("Filter by Status", ["New", "Investigating", "Reviewed", "Resolved"],
                                   default=["New", "Investigating"])
    view = df[df["status"].isin(status_filter)] if status_filter else df
    st.dataframe(view[["transaction_id", "amount", "fraud_probability", "risk_score", "risk_level",
                       "anomaly_status", "created_at", "status"]], **STRETCH)

    st.markdown("### Update Alert Status")
    if not view.empty:
        options = view["transaction_id"].tolist()
        selected = st.selectbox("Select Alert (Transaction ID)", options)
        new_status = st.selectbox("New Status", ["New", "Investigating", "Reviewed", "Resolved"])
        if st.button("Update Status"):
            row = df[df["transaction_id"] == selected].iloc[0]
            update_alert_status(int(row["id"]), new_status)
            st.success(f"Alert for {selected} updated to {new_status}.")
            st.rerun()


# ==============================================================================
# PAGE: TRANSACTION HISTORY
# ==============================================================================
def page_history():
    st.markdown("# Transaction History")
    df = fetch_transactions()
    if df.empty:
        st.info("No transactions found.")
        return

    col1, col2, col3 = st.columns(3)
    ttype = col1.selectbox("Transaction Type", ["All"] + TRANSACTION_TYPES)
    mcat = col2.selectbox("Merchant Category", ["All"] + MERCHANT_CATEGORIES)
    risklvl = col3.selectbox("Risk Level", ["All", "LOW", "MEDIUM", "HIGH"])

    view = df.copy()
    if ttype != "All": view = view[view["transaction_type"] == ttype]
    if mcat != "All": view = view[view["merchant_category"] == mcat]
    if risklvl != "All": view = view[view["risk_level"] == risklvl]

    st.dataframe(view, height=450, **STRETCH)
    st.download_button("Download as CSV", view.to_csv(index=False).encode("utf-8"),
                       "transaction_history.csv", "text/csv")


# ==============================================================================
# PAGE: ML MODEL PERFORMANCE
# ==============================================================================
def page_model_performance():
    st.markdown("# ML Model Performance")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        st.warning("No trained model found.")
        return

    meta = joblib.load(META_PATH)
    st.markdown(f"**Selected Model:** {meta['best_model']}  \n"
                f"**Trained At:** {meta['trained_at']}  \n"
                f"**Dataset Size:** {meta['dataset_size']:,}  \n"
                f"**Fraud %:** {meta['fraud_pct']:.2f}%")

    conn = get_conn()
    metrics_df = pd.read_sql_query("SELECT * FROM model_metrics ORDER BY trained_at DESC", conn)
    conn.close()
    st.markdown("### Model Comparison")
    st.dataframe(metrics_df, **STRETCH)

    try:
        ev = evaluate_saved_model()
    except Exception as e:
        st.error(f"Could not evaluate model: {e}")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{ev['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{ev['precision']*100:.2f}%")
    c3.metric("Recall", f"{ev['recall']*100:.2f}%")
    c4.metric("F1 Score", f"{ev['f1']*100:.2f}%")
    c5.metric("ROC-AUC", f"{ev['roc_auc']:.3f}")

    st.info("Precision and Recall matter more than raw accuracy here because fraud is rare: a model "
            "that predicts 'legitimate' for every transaction could still score high accuracy while "
            "catching zero fraud. Recall measures how many actual frauds are caught; precision measures "
            "how many flagged transactions are truly fraud (avoiding false alarms).")

    col1, col2 = st.columns(2)
    with col1:
        cm = ev["cm"]
        fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred: Legit", "Pred: Fraud"],
                                        y=["Actual: Legit", "Actual: Fraud"],
                                        text=cm, texttemplate="%{text}", colorscale="Blues"))
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, **STRETCH)
    with col2:
        fpr, tpr, _ = roc_curve(ev["y_test"], ev["y_proba"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig.update_layout(title=f"ROC Curve (AUC={ev['roc_auc']:.3f})", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, **STRETCH)

    prec, rec, _ = precision_recall_curve(ev["y_test"], ev["y_proba"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR Curve"))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, **STRETCH)

    model = joblib.load(MODEL_PATH)
    fi = get_feature_importance(model)
    fig = px.bar(fi.head(15), x="importance", y="feature", orientation="h", title="Feature Importance")
    st.plotly_chart(fig, **STRETCH)

    with st.expander("Full Classification Report"):
        st.text(ev["report"])

    if st.session_state.role == "admin":
        st.markdown("---")
        if st.button("Retrain Models Now", key="retrain_perf"):
            with st.spinner("Retraining all models..."):
                train_models(pd.read_csv(DATASET_PATH))
            st.success("Models retrained successfully.")
            st.rerun()


# ==============================================================================
# PAGE: ANOMALY DETECTION
# ==============================================================================
def page_anomaly_detection():
    st.markdown("# Anomaly Detection")
    st.caption("Unsupervised outlier detection using Isolation Forest — an additional signal, "
               "not a standalone fraud verdict.")
    df = fetch_transactions()
    if df.empty:
        st.info("No transaction data.")
        return

    total = len(df)
    anomalous = int((df["anomaly_status"] == "Anomalous").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", total)
    c2.metric("Anomalous", anomalous)
    c3.metric("Normal", total - anomalous)

    fig = px.scatter(df, x="amount", y="anomaly_score", color="anomaly_status",
                     title="Anomaly Score vs Transaction Amount", hover_data=["transaction_id"])
    st.plotly_chart(fig, **STRETCH)

    fig2 = px.histogram(df, x="anomaly_score", color="anomaly_status", nbins=30,
                        title="Anomaly Score Distribution")
    st.plotly_chart(fig2, **STRETCH)

    st.markdown("### Top Anomalous Transactions")
    st.dataframe(df[df["anomaly_status"] == "Anomalous"].sort_values("anomaly_score", ascending=False).head(25),
                 **STRETCH)


# ==============================================================================
# PAGE: EXPLAINABLE AI
# ==============================================================================
def page_explainable_ai():
    st.markdown("# Explainable AI")
    st.caption("Understand what drives the model's fraud predictions.")
    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found.")
        return

    model = joblib.load(MODEL_PATH)
    fi = get_feature_importance(model)
    st.markdown("### Global Feature Importance")
    fig = px.bar(fi.head(15), x="importance", y="feature", orientation="h")
    st.plotly_chart(fig, **STRETCH)

    if SHAP_AVAILABLE:
        st.markdown("### SHAP Feature Impact (sample)")
        try:
            clf = model.named_steps["clf"]
            pre = model.named_steps["pre"]
            sample_df = pd.read_csv(DATASET_PATH).sample(200, random_state=RANDOM_SEED)
            Xt = pre.transform(sample_df[ALL_FEATURES])
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()
            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(Xt)
                if isinstance(shap_values, list):
                    sv = shap_values[1]
                elif getattr(shap_values, "ndim", 2) == 3:
                    sv = shap_values[:, :, 1]
                else:
                    sv = shap_values
                mean_abs = np.abs(sv).mean(axis=0)
                names = NUMERIC_FEATURES + list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
                shap_df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values(
                    "mean_abs_shap", ascending=False)
                fig = px.bar(shap_df.head(15), x="mean_abs_shap", y="feature", orientation="h",
                             title="SHAP Mean |Impact| on Model Output")
                st.plotly_chart(fig, **STRETCH)
            else:
                st.info("SHAP TreeExplainer applies to tree-based models; showing feature importance above instead.")
        except Exception as e:
            st.info(f"SHAP explanation unavailable ({e}). Showing model feature importance instead.")
    else:
        st.info("SHAP library not installed — showing model-based feature importance instead.")

    st.markdown("### Inspect a Transaction")
    df_txn = fetch_transactions(limit=300)
    if not df_txn.empty:
        tid = st.selectbox("Select Transaction ID", df_txn["transaction_id"].tolist())
        row = df_txn[df_txn["transaction_id"] == tid].iloc[0]
        st.markdown(f"**Prediction:** {'FRAUD' if row['prediction']==1 else 'LEGITIMATE'} | "
                    f"**Probability:** {row['fraud_probability']*100:.1f}% | **Risk:** {row['risk_level']}")
        for reason in generate_reasons(row, get_dataset_stats()):
            st.markdown(f"- {reason}")


# ==============================================================================
# PAGE: ANALYTICS
# ==============================================================================
def page_analytics():
    st.markdown("# Analytics")
    df = fetch_transactions()
    if df.empty:
        st.info("No data.")
        return

    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        ttype = c1.selectbox("Transaction Type", ["All"] + TRANSACTION_TYPES)
        mcat = c2.selectbox("Merchant Category", ["All"] + MERCHANT_CATEGORIES)
        risklvl = c3.selectbox("Risk Level", ["All", "LOW", "MEDIUM", "HIGH"])
        pred = c4.selectbox("Prediction", ["All", "Fraud", "Legitimate"])

    view = df.copy()
    if ttype != "All": view = view[view["transaction_type"] == ttype]
    if mcat != "All": view = view[view["merchant_category"] == mcat]
    if risklvl != "All": view = view[view["risk_level"] == risklvl]
    if pred == "Fraud": view = view[view["prediction"] == 1]
    elif pred == "Legitimate": view = view[view["prediction"] == 0]

    st.markdown(f"**Filtered Records:** {len(view):,}")
    if view.empty:
        st.warning("No records match the selected filters.")
        return
    view = view.copy()
    view["date"] = to_dt(view["timestamp"]).dt.date

    colA, colB = st.columns(2)
    with colA:
        trend = view.groupby("date").agg(total=("id", "count"), fraud=("prediction", "sum")).reset_index()
        fig = px.line(trend, x="date", y=["total", "fraud"], title="Transaction & Fraud Trend")
        st.plotly_chart(fig, **STRETCH)
    with colB:
        by_hour = view.groupby("transaction_hour")["prediction"].sum().reset_index()
        fig = px.bar(by_hour, x="transaction_hour", y="prediction", title="Fraud by Hour of Day")
        st.plotly_chart(fig, **STRETCH)

    colC, colD = st.columns(2)
    with colC:
        merch = view.groupby("merchant_category").agg(fraud_rate=("prediction", "mean")).reset_index()
        merch["fraud_rate"] *= 100
        fig = px.bar(merch, x="merchant_category", y="fraud_rate", title="Fraud Rate % by Merchant Category")
        st.plotly_chart(fig, **STRETCH)
    with colD:
        fig = px.box(view, x="risk_level", y="amount", title="Amount Distribution by Risk Level")
        st.plotly_chart(fig, **STRETCH)

    st.markdown("### High-Risk Transactions")
    st.dataframe(view[view["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False).head(50),
                 **STRETCH)

    st.markdown("### Anomaly Analysis")
    anomaly_rate = (view["anomaly_status"] == "Anomalous").mean() * 100
    st.metric("Anomaly Rate (filtered view)", f"{anomaly_rate:.2f}%")


# ==============================================================================
# PAGE: REPORTS
# ==============================================================================
def page_reports():
    st.markdown("# Reports")
    df = fetch_transactions()
    if df.empty:
        st.info("No data to report on.")
        return

    meta = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}
    conn = get_conn()
    metrics_df = pd.read_sql_query(
        "SELECT * FROM model_metrics WHERE is_selected=1 ORDER BY trained_at DESC LIMIT 1", conn)
    conn.close()

    total = len(df)
    fraud = int((df["prediction"] == 1).sum())
    fraud_rate = fraud / total * 100 if total else 0
    high = int((df["risk_level"] == "HIGH").sum())

    lines = [
        "FRAUDGUARD AI - FRAUD DETECTION SUMMARY REPORT",
        f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Total Transactions Analyzed: {total}",
        f"Fraudulent Transactions: {fraud}",
        f"Fraud Rate: {fraud_rate:.2f}%",
        f"High-Risk Transactions: {high}",
        f"Model Used: {meta.get('best_model', 'N/A')}",
    ]
    if not metrics_df.empty:
        m = metrics_df.iloc[0]
        lines += [
            f"Model Accuracy: {m['accuracy']*100:.2f}%",
            f"Model Precision: {m['precision_score']*100:.2f}%",
            f"Model Recall: {m['recall_score']*100:.2f}%",
            f"Model F1 Score: {m['f1_score']*100:.2f}%",
            f"Model ROC-AUC: {m['roc_auc']:.3f}",
        ]
    lines.append("=" * 60)
    lines.append("Major Findings:")
    if fraud > 0:
        top_merchant = df[df["prediction"] == 1]["merchant_category"].value_counts().idxmax()
        top_type = df[df["prediction"] == 1]["transaction_type"].value_counts().idxmax()
    else:
        top_merchant = top_type = "N/A"
    lines.append(f"- Highest fraud concentration in merchant category: {top_merchant}")
    lines.append(f"- Highest fraud concentration in transaction type: {top_type}")
    lines.append(f"- Anomaly detection flagged {(df['anomaly_status']=='Anomalous').sum()} transactions as outliers.")
    lines.append("- Risk thresholds used are application-defined for demonstration purposes.")
    report_text = "\n".join(lines)

    st.text_area("Report Preview", report_text, height=380)
    c1, c2 = st.columns(2)
    c1.download_button("Download Report (TXT)", report_text.encode("utf-8"),
                       "fraud_summary_report.txt", "text/plain")
    c2.download_button("Download Transactions (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "fraud_transactions_export.csv", "text/csv")


# ==============================================================================
# PAGE: SYSTEM SETTINGS
# ==============================================================================
def page_settings():
    st.markdown("# System Settings")
    if st.session_state.role != "admin":
        st.error("Access restricted to Administrator role.")
        return

    st.markdown("### Risk Thresholds")
    low_max = st.slider("LOW risk max score", 0, 50, int(float(get_setting("low_max", 30))))
    medium_max = st.slider("MEDIUM risk max score", low_max + 1, 100,
                           max(int(float(get_setting("medium_max", 70))), low_max + 1))
    if st.button("Save Risk Thresholds"):
        set_setting("low_max", low_max)
        set_setting("medium_max", medium_max)
        st.success("Risk thresholds updated.")
    st.caption("These thresholds are application-defined for demonstration and are not universal banking standards.")

    st.markdown("---")
    st.markdown("### Model Selection")
    conn = get_conn()
    models_df = pd.read_sql_query("SELECT DISTINCT model_name FROM model_metrics", conn)
    conn.close()
    if not models_df.empty:
        st.write(f"Currently deployed model: **{get_setting('selected_model', 'N/A')}**")
        st.caption("The deployed model is automatically chosen by best F1-score during training/retraining.")

    st.markdown("---")
    st.markdown("### Retrain Model")
    if st.button("Retrain Models Now", key="retrain_settings"):
        with st.spinner("Retraining..."):
            train_models(pd.read_csv(DATASET_PATH))
        st.success("Models retrained.")

    st.markdown("---")
    st.markdown("### Database Statistics")
    conn = get_conn()
    n_txn = pd.read_sql_query("SELECT COUNT(*) c FROM transactions", conn)["c"][0]
    n_alerts = pd.read_sql_query("SELECT COUNT(*) c FROM fraud_alerts", conn)["c"][0]
    n_users = pd.read_sql_query("SELECT COUNT(*) c FROM users", conn)["c"][0]
    conn.close()
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", int(n_txn)); c2.metric("Alerts", int(n_alerts)); c3.metric("Users", int(n_users))

    st.markdown("---")
    st.markdown("### ⚠️ Reset Demo Data")
    st.warning("This permanently deletes all transactions, alerts, and models, then regenerates fresh "
               "synthetic demo data. This action cannot be undone.")
    confirm = st.checkbox("I understand this action is irreversible")
    if st.button("Reset Demo Data", disabled=not confirm):
        with st.spinner("Resetting..."):
            reset_demo_data()
        st.success("Demo data reset successfully.")
        st.rerun()


# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================
def main():
    st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide",
                       initial_sidebar_state="expanded")
    apply_custom_css()

    try:
        with st.spinner("Initializing (first run trains the models — please wait ~1 minute)..."):
            ensure_initialized()
    except Exception as e:
        st.error(f"Application failed to initialize: {e}")
        st.info("Try deleting 'fraud_detection.db', the 'data' folder and the 'models' folder, then reload.")
        st.stop()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None

    if not st.session_state.logged_in:
        page_login()
        return

    render_sidebar()
    page = st.session_state.get("page", "Dashboard")

    try:
        if page == "Dashboard":
            page_dashboard()
        elif page == "Transaction Analysis":
            page_transaction_analysis()
        elif page == "Batch Fraud Detection":
            page_batch_detection()
        elif page == "Fraud Alerts":
            page_alerts()
        elif page == "Transaction History":
            page_history()
        elif page == "ML Model Performance":
            page_model_performance()
        elif page == "Anomaly Detection":
            page_anomaly_detection()
        elif page == "Explainable AI":
            page_explainable_ai()
        elif page == "Analytics":
            page_analytics()
        elif page == "Reports":
            page_reports()
        elif page == "System Settings":
            page_settings()
    except Exception as e:
        st.error(f"An error occurred while loading this page: {e}")
        st.exception(e)


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _ctx = get_script_run_ctx()
    except Exception:
        _ctx = None

    if _ctx is None:
        # Launched with "python app.py" — relaunch under Streamlit
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
    else:
        main()