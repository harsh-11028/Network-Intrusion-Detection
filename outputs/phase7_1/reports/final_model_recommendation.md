# Phase 7.1: Defensible Final Model Selection & Operational Recommendations

**Project:** Machine Learning-Based Network Intrusion Detection and Attack Classification  
**Phase:** 7.1  
**Dataset:** UNSW-NB15  
**Date:** November 2026  

---

## 1. Executive Recommendation Matrix

Rather than prescribing an arbitrary single model, we evaluate operational trade-offs across four realistic deployment contexts:

| Operational Deployment Scenario | Recommended Architecture | Operating Threshold | Expected Attack Detection (Recall) | Expected False Alarm Rate (FPR) | Test Accuracy | Test Precision | Test F1-Score | Operational Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Scenario 1: Critical Infrastructure & High Security** | **XGBoost (Strategy C)** or **Random Forest (Strategy C)** | **$\tau = 0.42$** | **98.35% – 98.81%** | 29.67% – 31.94% | 80.01% – 81.22% | 66.28% – 67.81% | 0.7934 – 0.8027 | Zero-tolerance for network intrusion. Random Forest misses only **258 attacks** out of 21,739. XGBoost misses only **359 attacks**. False alarms are accepted as necessary defense overhead. |
| **Scenario 2: Commercial Enterprise SOC (Workload Management)** | **XGBoost (Strategy A / B)** | **$\tau = 0.49$** | **97.48%** | **25.83%** | **83.23%** | **70.57%** | **0.8187** | Reduces alert fatigue for human analysts. Detects 21,192 attacks while suppressing thousands of false alarms compared to linear baselines. |
| **Scenario 3: Balanced General-Purpose Cyber Defense** | **XGBoost (Default)** | **$\tau = 0.50$** | **97.33%** | **25.24%** | **83.53%** | **71.02%** | **0.8212** | Maximizes overall test accuracy (83.53%) and Attack F1 (0.8212), capturing 21,158 intrusions with 8,632 false positives. |
| **Scenario 4: Resource-Constrained IoT Gateway / Edge Router** | **Logistic Regression (Default)** | **$\tau = 0.50$** | **95.81%** | **39.13%** | **74.45%** | **60.88%** | **0.7445** | Sub-millisecond inference latency, sub-kilobyte memory footprint, linear execution with zero tree ensemble overhead. |

---

## 2. Multi-Metric Comparative Evaluation

| Evaluated Metric | Logistic Regression | Decision Tree | Random Forest | XGBoost | Best Architecture |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy (Default 0.50)** | 74.45% | 82.21% | 82.73% | **83.53%** | **XGBoost (+0.80%)** |
| **Attack Recall (Default 0.50)** | 95.81% | 92.72% | **97.51%** | 97.33% | **Random Forest (+0.18%)** |
| **Attack Precision (Default 0.50)** | 60.88% | 70.66% | 69.92% | **71.02%** | **XGBoost (+0.36%)** |
| **Attack F1-Score (Default 0.50)** | 0.7445 | 0.8020 | 0.8144 | **0.8212** | **XGBoost (+0.0068)** |
| **ROC-AUC** | 0.9193 | 0.8432 | 0.9655 | **0.9719** | **XGBoost (+0.0064)** |
| **PR-AUC (Average Precision)** | 0.8912 | 0.6868 | 0.9502 | **0.9624** | **XGBoost (+0.0122)** |
| **Inference Latency (55k samples)** | **0.02s** | 0.04s | 0.82s | 0.07s | **Logistic Regression** |
| **Model Size on Disk** | **2.4 KB** | 879 KB | 314 MB | 958 KB | **Logistic Regression** |
| **Threshold Tunability** | High (Smooth) | None (Discrete) | High (Smooth) | High (Smooth) | **Ensembles (RF / XGB)** |

---

## 3. Definitive Benchmark Champion

**XGBoost** is confirmed as the definitive overall champion of this benchmark research study because:
1. **Highest Global Performance:** Outperforms all competitors across Accuracy (83.53%), Attack Precision (71.02%), Attack F1 (0.8212), ROC-AUC (0.9719), and PR-AUC (0.9624).
2. **Smooth Threshold Calibration:** Continuous posterior probabilities permit precise threshold selection between high-recall security modes ($	au = 0.42$) and high-precision analyst modes ($	au = 0.49 - 0.53$).
3. **Computational Efficiency:** Serialized model footprint is under 1 MB (compared to Random Forest's 314 MB) with inference speed exceeding 750,000 flows/second.
4. **Generalization Stability:** Generalization gap remains tighter ($12.03\%$) than Random Forest ($16.94\%$) and Decision Tree ($17.33\%$).
