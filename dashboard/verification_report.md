# Phase 8 Verification & Audit Report

**Project**: Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Dataset**: UNSW-NB15  
**Execution Date**: November 24, 2026  
**Artifacts Source of Truth**: Phase 7.1 Final Models & Leakage-Free Validation (`outputs/phase7_1/`)  

---

## 1. Executive Summary
Phase 8 has been finalized, integrated, and verified end-to-end. The interactive Network Intrusion Detection Dashboard delivers an offline traffic analyzer, model performance benchmarks, and feature interpretability insights powered by the models retrained and evaluated in Phase 7.1.

Both the FastAPI backend (`http://localhost:8000`) and the React + Vite frontend (`http://localhost:5173`) are fully operational and managed cleanly via `run_dashboard.sh`.

---

## 2. Test Execution & Verification Log

All tests listed below were executed directly in the project environment and confirmed passing.

### Test 1: Startup Script & Process Isolation (`run_dashboard.sh`)
- **Port Collision Prevention**: The script checks whether ports `8000` or `5173` are occupied before launching processes and reports informative warnings rather than crashing silently.
- **Dependency Auto-Resolution**: Confirmed detection of missing backend Python packages and automatic invocation of npm package validation.
- **Process Cleanup**: Verified that SIGINT/SIGTERM/EXIT traps cleanly terminate both the child Uvicorn server and Vite process without killing unrelated host processes.
- **Readiness Verification**: Confirms backend health via polling `http://127.0.0.1:8000/api/health` before triggering the frontend launcher.

### Test 2: Source of Truth & Preprocessing Integrity
- **Model Path Verification**: Confirmed all four models are strictly loaded from:
  - `outputs/phase7_1/final_models/logistic_regression_final.joblib`
  - `outputs/phase7_1/final_models/decision_tree_final.joblib`
  - `outputs/phase7_1/final_models/random_forest_final.joblib`
  - `outputs/phase7_1/final_models/xgboost_final.joblib`
- **Preprocessor Verification**: Verified that the preprocessor is loaded strictly from:
  - `outputs/phase7_1/final_models/final_preprocessor.joblib` (194 transformed features)
- **Zero-Refit Assurance**: Verified via code inspection and AST checks that `fit()` and `fit_transform()` are never invoked during dashboard execution. The preprocessor strictly runs `.transform()` on inference data.

### Test 3: Backend Health & Metadata Endpoints
- `GET /api/health`: Returned status `200 OK` with 4 models loaded, 42 expected raw features, and 194 transformed features.
- `GET /api/models`: Returned 4 model configurations along with leakage-free validation threshold presets (Strategies A–D).
- `GET /api/metrics/official`: Returned untouched official test-set metrics (55,945 samples) across all models.
- `GET /api/metrics/thresholds`: Returned 364 validation threshold analysis points for comparative visualization.
- `GET /api/insights/features`: Returned top-50 predictive features for all 4 models (XGBoost Gain-based, RF/DT Impurity decrease, Logistic Regression signed coefficients).

### Test 4: Ingestion & Inference Testing
- **Unlabeled CSV Ingestion (`dashboard/sample_data/sample_traffic.csv`)**:
  - Ingested 200 rows without ground-truth labels.
  - Successfully produced predictions across all 4 models:
    - XGBoost (threshold 0.50): 95 Attacks, 105 Normal (47.5% attack rate).
    - Random Forest (threshold 0.50): 94 Attacks, 106 Normal (47.0% attack rate).
    - Decision Tree (threshold 0.50): 80 Attacks, 120 Normal (40.0% attack rate).
    - Logistic Regression (threshold 0.50): 132 Attacks, 68 Normal (66.0% attack rate).
- **Labeled CSV Ingestion & Evaluation (`dashboard/sample_data/sample_traffic_labeled.csv`)**:
  - Ingested 200 rows containing `label` and `attack_cat`.
  - Confirmed labels are removed from feature input to prevent leakage.
  - Successfully calculated live ground-truth metrics:
    - XGBoost (threshold 0.49): Accuracy = 64.0%, Recall = 100.0%, Precision = 24.21%, F1 = 0.3898, ROC-AUC = 0.9838, Confusion Matrix = `[[105, 72], [0, 23]]`.

### Test 5: Exporting Predictions
- `POST /api/predict/export`: Verified generation of RFC 4180-compliant CSV containing `row_index`, `probability`, `prediction`, and `label_text`. Verified 201 lines (header + 200 records).

### Test 6: Error Handling & Input Validation
- **Non-CSV file**: Rejected with `400 Bad Request`.
- **Empty file**: Rejected with `400 Bad Request`.
- **Missing feature columns**: Injected malformed CSV missing 39 columns; rejected with `422 Unprocessable Entity` listing missing column names.
- **Invalid threshold**: Tested threshold `1.5` (out of range `(0, 1)`); rejected with `422 Unprocessable Entity`.

### Test 7: Frontend Build & UI Integrity
- Ran TypeScript compilation (`tsc`) and Vite bundling (`npm run build`). Successfully generated production bundle in `dashboard/frontend/dist/` without errors or warnings.
- Verified all 5 routes:
  1. `/` (Dashboard Overview)
  2. `/analyze` (Traffic Analyzer)
  3. `/performance` (Model Performance & Threshold Analysis)
  4. `/insights` (Feature Importances & Interpretability)
  5. `/methodology` (Project Audit & Methodology)

---

## 3. Issues Identified and Resolved

1. **TypeScript Type Indexing in Model Performance Table**:
   - *Problem*: In `ModelPerformancePage.tsx`, indexing `row[k]` with strict string union types caused `TS7053` compilation errors during `tsc`.
   - *Resolution*: Declared explicit type annotation `Array<{ model: string; [key: string]: unknown }>` for `tableData`.
2. **PostCSS Browserslist Sandbox Issue**:
   - *Problem*: Build step tried to stat `/Users/harshmac/package.json` due to missing `browserslist` specification.
   - *Resolution*: Added `"browserslist": ["defaults"]` directly to `dashboard/frontend/package.json`.
3. **Out of Range Float / NaN in JSON Serialization**:
   - *Problem*: For Decision Tree Strategy D (which is infeasible), metrics in `threshold_selection_summary.csv` were empty/NaN, causing `ValueError: Out of range float values are not JSON compliant: nan` when calling `GET /api/models`.
   - *Resolution*: Enhanced `_safe_float()` in `model_loader.py` to check for `pd.isna()`, `np.isnan()`, and `np.isinf()`, returning clean `None` (serialized as `null` in JSON).
4. **Child Process Clean Termination in `run_dashboard.sh`**:
   - *Problem*: Unconditional process killing could leave orphan processes or terminate unintended services.
   - *Resolution*: Stored specific process IDs (`BACKEND_PID`, `FRONTEND_PID`) and checked process existence via `kill -0` before issuing graceful `SIGTERM` signals.

---

## 4. Operational Notes & Limitations
- **Offline CSV Ingestion Only**: The application performs file-based batch flow analysis; it does not capture live network packets.
- **Decision Tree Threshold Behavior**: Decision Tree produces discrete leaf probabilities; adjusting thresholds between leaf boundaries has step-function characteristics. This is explicitly noted in the UI and documentation.
- **Academic Context**: Flagged traffic represents "model-flagged suspicious flows" that require human analyst triage, not confirmed intrusions.
