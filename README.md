# Machine Learning-Based Network Intrusion Detection and Attack Classification

A comprehensive, end-to-end cybersecurity machine learning project designed, implemented, and evaluated on the **UNSW-NB15** network intrusion dataset. The project features leakage-free validation, threshold optimization across four distinct model architectures, and an interactive offline Threat Intelligence/SOC dashboard.

---

## 📌 Project Architecture & Highlights

- **Dataset**: UNSW-NB15 (107,740 training flows, 55,945 testing flows) with 42 raw network traffic features (39 numerical, 3 categorical).
- **Binary Target**: `label` (0 = Normal, 1 = Attack).
- **Leakage-Free Validation (Phase 7.1)**:
  - 80:20 stratified split strictly on raw training data (86,192 internal train / 21,548 validation).
  - Validation preprocessor fitted exclusively on internal train split (192 features; rare categories handled via `handle_unknown='ignore'`).
  - Production preprocessor fitted on full training set (194 transformed features).
  - Single final evaluation on untouched official test partition (55,945 flows).
- **Models Evaluated**:
  - **Logistic Regression**: Linear baseline with directional log-odds feature weights.
  - **Decision Tree**: CART decision tree classifier (max depth 15).
  - **Random Forest**: 300-estimator bagging ensemble with balanced precision/recall.
  - **XGBoost (Top Performer)**: Gradient-boosted ensemble achieving **0.9719 ROC-AUC**, **0.9624 PR-AUC**, and **97.33% Attack Recall**.
- **Interactive SOC Dashboard (Phase 8 & 8.1)**:
  - Built with **FastAPI** backend and **React 18 + Vite + TypeScript + Tailwind CSS + Recharts** frontend.
  - Features dual-state overview, offline CSV ingestion, threshold tuning, and feature interpretability consoles.

---

## 📊 Final Model Benchmark Results (Official Test Set: 55,945 Flows)

Evaluated at the standard threshold of $0.50$:

| Model | Accuracy | Attack Recall | Attack Precision | Attack F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Recommended)** | **83.53%** | **97.33%** | **71.02%** | **0.8212** | **0.9719** | **0.9624** |
| **Random Forest** | 82.73% | 97.51% | 69.92% | 0.8144 | 0.9655 | 0.9502 |
| **Decision Tree** | 82.21% | 92.72% | 70.66% | 0.8020 | 0.8432 | 0.6868 |
| **Logistic Regression** | 74.45% | 95.81% | 60.88% | 0.7445 | 0.9193 | 0.8912 |

---

## 📁 Repository Structure

```
Network-Intrusion-Detection/
├── data/
│   ├── UNSW_NB15_training_cleaned.csv  # Cleaned training dataset
│   ├── UNSW_NB15_testing_cleaned.csv   # Cleaned test dataset
│   └── processed/                      # Preprocessed arrays and feature metadata
├── dashboard/                          # Interactive SOC Web Dashboard
│   ├── backend/                        # FastAPI REST API
│   │   ├── app/
│   │   │   ├── main.py                 # App entry point & CORS
│   │   │   ├── config.py               # Path configurations
│   │   │   ├── routes/api.py           # Endpoints (/health, /models, /predict, etc.)
│   │   │   └── services/               # Model loader & prediction services
│   │   └── requirements.txt
│   ├── frontend/                       # React + TypeScript + Tailwind UI
│   │   ├── src/
│   │   │   ├── pages/                  # Dashboard, Analyzer, Performance, Insights, Methodology
│   │   │   └── services/               # Axios API client & Analysis Context
│   │   └── package.json
│   ├── sample_data/                    # Sample CSVs for offline demo
│   └── README.md
├── outputs/                            # Experiment results, plots, metrics & models
│   ├── eda/                            # EDA reports and correlation heatmaps
│   ├── models/                         # Baseline training outputs & figures
│   ├── phase7/                         # Phase 7 comparative analysis & threshold sweeps
│   └── phase7_1/                       # Leakage-free validation & retrained final models
│       ├── final_models/               # Deployed production models & preprocessor
│       ├── validation/                 # Threshold analysis & validation curves
│       ├── final_test/                 # Official test evaluations
│       └── reports/                    # Phase 7.1 audit documentation
├── inspect_data.py                     # Initial dataset inspection script
├── eda.py                              # Exploratory data analysis pipeline
├── preprocessing.py                    # Reusable ColumnTransformer module
├── prepare_data.py                     # Feature matrix generation script
├── train_logistic_regression.py        # Logistic regression training script
├── train_decision_tree.py              # Decision tree training script
├── train_random_forest.py              # Random forest training script
├── train_xgboost.py                    # XGBoost training script
├── run_phase7.py                       # Phase 7 threshold sweep pipeline
├── run_phase7_1.py                     # Phase 7.1 leakage-free retraining pipeline
├── run_dashboard.sh                    # Unified startup script for backend & frontend
└── requirements.txt                    # Project dependencies
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (using `.venv`)
- Node.js (v18+) & npm

### 2. Run the Interactive Dashboard
Launch both the FastAPI backend (`http://localhost:8000`) and the React frontend (`http://localhost:5173`) with a single command:

```bash
./run_dashboard.sh
```

- **Frontend Dashboard**: [http://localhost:5173](http://localhost:5173)
- **FastAPI OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Demo Data**: Test inference using sample CSVs located in `dashboard/sample_data/`.

---

## 🛡️ License & Acknowledgments
Dataset provided by the Cyber Range Lab of UNSW Canberra (Moustafa & Slay, 2015).
Developed as an academic Machine Learning for Cybersecurity project.
