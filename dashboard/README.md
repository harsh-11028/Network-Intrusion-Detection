# Phase 8: Interactive Network Intrusion Detection Dashboard

Welcome to the interactive web application for the **Machine Learning-Based Network Intrusion Detection and Attack Classification** project using the UNSW-NB15 dataset.

## 1. Overview
The dashboard is a decision-support and exploratory tool powered by models and preprocessing pipelines strictly finalized in **Phase 7.1** (leakage-free internal validation and retraining).

### Core Features
- **Dashboard (Overview)**: Summary metric cards, Normal vs. Attack distribution donut chart, attack probability histogram, and a preview table of top suspicious flows.
- **Traffic Analyzer**: Offline CSV ingestion, model selection (Logistic Regression, Decision Tree, Random Forest, XGBoost), interactive threshold tuning with validation-optimized presets (Strategies A–D), paginated table with probability progress bars, filtering by class, keyword search, and CSV export. When ground-truth labels (`label`) are present, it computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
- **Model Performance**: Comprehensive comparison of all four models against official test set metrics (55,945 samples) across all decision strategies, multi-metric bar charts, and validation threshold sweep curves.
- **Model Insights**: Interactive feature importance visualizations (XGBoost Gain-based, RF/DT Impurity decrease, Logistic Regression signed coefficient weights) with top-N filtering (10, 15, 20, 30) and domain caveats.
- **Methodology & Audit**: Transparent documentation of Phases 1 through 7.1, details on the Phase 7.1 preprocessing leakage correction, threshold optimization criteria, and system limitations.

---

## 2. Directory Structure
```
Network-Intrusion-Detection/
├── dashboard/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── config.py             # Path configuration & feature list
│   │   │   ├── main.py               # FastAPI application setup & CORS
│   │   │   ├── routes/
│   │   │   │   └── api.py            # API endpoints (/health, /models, /predict, etc.)
│   │   │   └── services/
│   │   │       ├── model_loader.py   # Artifact loader for Phase 7.1 models & preprocessor
│   │   │       └── prediction_service.py # Feature extraction, transform, & inference
│   │   └── requirements.txt          # Python dependencies
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── components/           # Reusable UI widgets
│   │   │   ├── pages/                # 5 primary application views
│   │   │   ├── services/             # Axios API client & Analysis Context
│   │   │   ├── App.tsx               # Main routing & layout shell
│   │   │   └── index.css             # Tailwind design styles (SOC Dark Theme)
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── vite.config.ts            # Vite proxy configuration
│   ├── sample_data/
│   │   ├── sample_traffic.csv        # 200 unlabeled network flows (inference demo)
│   │   └── sample_traffic_labeled.csv# 200 labeled flows (evaluation demo)
│   └── README.md
├── outputs/phase7_1/                 # Source of truth for Phase 7.1 artifacts
│   ├── final_models/
│   │   ├── final_preprocessor.joblib # 194-feature preprocessor (never refit)
│   │   ├── logistic_regression_final.joblib
│   │   ├── decision_tree_final.joblib
│   │   ├── random_forest_final.joblib
│   │   └── xgboost_final.joblib
│   ├── validation/                   # Validation threshold analysis & metrics
│   └── final_test/                   # Official test set evaluations
└── run_dashboard.sh                  # One-click startup script
```

---

## 3. Requirements & Prerequisites

### System Requirements
- **Operating System**: macOS (Apple Silicon / Intel) or Linux.
- **Python**: 3.10+ (using project virtual environment `.venv`).
- **Node.js**: v18+ with `npm` (v9+).
- **Network Ports**: Port `8000` (Backend) and Port `5173` (Frontend) must be free.

---

## 4. Quick Start Guide

To launch the complete application with a single command from the project root:
```bash
./run_dashboard.sh
```

### What this script does:
1. Validates the existence of the Python virtual environment (`.venv`).
2. Checks that Node.js and npm are installed.
3. Automatically installs Python and npm dependencies if missing.
4. Checks that ports `8000` and `5173` are unallocated.
5. Launches the FastAPI backend (`http://localhost:8000`) and waits for health check confirmation.
6. Launches the Vite development server (`http://localhost:5173`).
7. Sets up graceful signal traps so pressing `Ctrl+C` cleanly terminates both backend and frontend processes.

### Accessing the Dashboard:
- **Frontend Web UI**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 5. Manual Execution (Alternative)

If you prefer running services in separate terminal windows:

### Terminal 1: Backend
```bash
cd dashboard/backend
../../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: Frontend
```bash
cd dashboard/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

---

## 6. How to Use the Dashboard

### 1. Ingesting Traffic
1. Navigate to the **Traffic Analyzer** page (`/analyze`).
2. Drag and drop or browse for a CSV file.
   - For a quick demo without ground truth: use `dashboard/sample_data/sample_traffic.csv`.
   - For a demo with ground-truth verification: use `dashboard/sample_data/sample_traffic_labeled.csv`.
3. Choose one of the 4 models: **XGBoost** (recommended), **Random Forest**, **Decision Tree**, or **Logistic Regression**.
4. Adjust the threshold slider or click any validation-selected preset:
   - **Default (0.50)**
   - **Strategy A (Max Attack F1)**
   - **Strategy B (Max Macro F1)**
   - **Strategy C (Security Priority, Recall ≥ 97%)**
   - **Strategy D (Operational Target)**
5. Click **Analyze Traffic**.
6. Review the results in the table below or switch to the **Dashboard** overview page for visual charts.
7. Click **Export CSV** to download a CSV containing individual flow probabilities and binary classifications.

### 2. Inspecting Model Performance
Navigate to **Model Performance** (`/performance`) to review official Phase 7.1 test-set evaluations across strategies, multi-metric comparisons, and validation threshold sweep curves.

### 3. Reviewing Model Insights
Navigate to **Model Insights** (`/insights`) to inspect top predictive features per model and review model explanations and caveats.

---

## 7. Troubleshooting & FAQ

| Problem | Likely Cause | Solution |
| :--- | :--- | :--- |
| **Port 8000 or 5173 in use** | An existing background server is holding the port. | Run `lsof -i :8000` or `lsof -i :5173` to identify the PID and terminate it with `kill <PID>`. |
| **Missing column error on upload** | Uploaded CSV lacks one of the 42 required features. | Check that the input CSV contains all 42 raw feature columns as detailed in Section 8. |
| **Empty or malformed CSV error** | The uploaded file has zero bytes or cannot be parsed. | Ensure the file is a valid standard CSV format. |
| **Decision Tree threshold has no effect** | Tree leaf probabilities are discrete. | This is expected behavior documented in Phase 4 & Phase 7.1. Leaf boundary thresholds produce step changes. |
| **CORS errors in browser console** | Accessing the frontend via an unauthorized origin. | Access the dashboard strictly via `http://localhost:5173` or `http://127.0.0.1:5173`. |

---

## 8. Expected Input Features (42 Raw Columns)
The uploaded CSV file must contain the following 42 features (case-sensitive):
```
dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload,
sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin,
tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len,
ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm,
ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm,
ct_srv_dst, is_sm_ips_ports, proto, service, state
```
*Optional columns*: `label` (0 = Normal, 1 = Attack) and `attack_cat` (attack category name). If present, the dashboard automatically extracts them for ground-truth performance evaluation and removes them from the model input features.
