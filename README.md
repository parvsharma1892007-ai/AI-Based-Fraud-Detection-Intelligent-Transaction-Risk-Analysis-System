# 🛡️ FraudGuard AI

**AI-Based Fraud Detection & Intelligent Transaction Risk Analysis Platform**

A single-file Streamlit application that detects fraudulent transactions with machine learning, scores risk, flags anomalies, and explains its decisions. Built as a final-year B.Tech (AI & Data Science) project.

> ⚠️ **Note:** The app trains on a **synthetic dataset** generated at first run. It is a demonstration and learning project, not a production banking system.

---

## ✨ Features

| Module | What it does |
|---|---|
| **Dashboard** | KPIs and charts: fraud rate, risk distribution, trends, amount distribution |
| **Transaction Analysis** | Analyze one transaction in real time with fraud probability, risk score and reasons |
| **Batch Fraud Detection** | Upload a CSV, get bulk predictions, download results |
| **Fraud Alerts** | Auto-generated alerts with a workflow: New → Investigating → Reviewed → Resolved |
| **Transaction History** | Filterable, exportable transaction log |
| **ML Model Performance** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC and PR curves |
| **Anomaly Detection** | Unsupervised outlier detection with Isolation Forest |
| **Explainable AI** | Feature importance, SHAP (if installed), and per-transaction reasons |
| **Analytics / Reports** | Filtered analysis and downloadable TXT/CSV reports |
| **System Settings** | Admin-only risk thresholds, retraining, and demo data reset |
| **Role-based login** | Admin and Analyst roles with hashed passwords |

---

## 🧠 How it works

1. **Data:** a synthetic, imbalanced dataset (about 4.5% fraud) is generated on first run.
2. **Models:** Logistic Regression, Random Forest, and XGBoost (if installed) are trained.
3. **Model selection:** the best model is chosen by **F1-score**, not accuracy, because fraud is rare and accuracy is misleading on imbalanced data.
4. **Anomaly signal:** an Isolation Forest adds an unsupervised outlier score.
5. **Final risk score (0-100)** is a weighted blend of:
   - fraud probability (50%)
   - anomaly score (20%)
   - behavioral risk: location, device, previous fraud (20%)
   - amount deviation (10%)
6. **Risk levels:** LOW / MEDIUM / HIGH using admin-configurable thresholds.
7. **Explanations:** rule-based reasons compared against dataset percentiles.

---

## 🧰 Tech Stack

Python · Streamlit · scikit-learn · XGBoost (optional) · SHAP (optional) · Pandas · NumPy · Plotly · SQLite · Joblib

---

## 🚀 Getting Started

```bash
# 1. Clone
git clone https://github.com/parvsharma1892007-ai/FraudGuard-AI.git
cd FraudGuard-AI

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python -m streamlit run app.py
```

Open `http://localhost:8501`. The first run takes about a minute while it generates data and trains the models.

### Demo credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| Analyst | `analyst` | `analyst123` |

---

## 📁 Project Structure

```
FraudGuard-AI/
├── app.py                    # Entire application (DB, ML, UI)
├── sample_transactions.csv   # Sample file for Batch Fraud Detection
├── requirements.txt
├── .gitignore
└── README.md

# Auto-created at runtime (git-ignored):
├── fraud_detection.db
├── data/transactions_dataset.csv
└── models/*.joblib
```

---

## 📄 Batch CSV Format

Required columns:

```
amount, transaction_type, merchant_category, transaction_hour, account_age_days,
transaction_frequency, location_risk, device_risk, previous_fraud_count,
transaction_velocity, amount_deviation, customer_risk_score
```

`transaction_id` is optional. See `sample_transactions.csv`.

---

## ⚠️ Limitations

- Trained on **synthetic data**, so real-world performance is not implied.
- Risk thresholds and score weights are **application-defined**, not banking standards.
- Password hashing (salted SHA-256) and the built-in demo accounts are for demonstration only. Use bcrypt or Argon2 and remove default accounts for any real deployment.
- SQLite is suitable for a demo, not for high-volume production.

## 🔮 Future Work

- Train on a real dataset (for example, the Kaggle credit card fraud data)
- Real-time streaming with Kafka
- REST API (FastAPI) for model serving
- Stronger authentication and an audit log
- Hyperparameter tuning and probability calibration

---

## 👤 Author

**Parv Sharma** · B.Tech AI & Data Science
[GitHub](https://github.com/parvsharma1892007-ai) · [LinkedIn](https://www.linkedin.com/in/parv-sharma-051413331/)

## 📜 License

MIT License. See `LICENSE` (optional).
