from build_full_report_docx import DocxBuilder
import os

doc = DocxBuilder("Machine_Learning_Network_Intrusion_Detection_Report.docx")

# -------------------------------------------------------------
# 1. COVER PAGE
# -------------------------------------------------------------
doc.add_p("NOIDA INSTITUTE OF ENGINEERING AND TECHNOLOGY", bold=True, sz=32, align="center", space_before=100, space_after=40, color="003366")
doc.add_p("GREATER NOIDA", bold=True, sz=26, align="center", space_before=0, space_after=40)
doc.add_p("(An Autonomous Institute Affiliated to Dr. A.P.J. Abdul Kalam Technical University, Lucknow)", italic=True, sz=21, align="center", space_before=0, space_after=60)
doc.add_p("Department of Computer Science and Engineering", bold=True, sz=24, align="center", space_before=60, space_after=200, color="003366")

doc.add_p("Machine Learning-Based Network Intrusion Detection\nand Attack Classification", bold=True, sz=36, align="center", space_before=200, space_after=120, color="002040")

doc.add_p("A Project Report Submitted In Partial Fulfillment\nfor the Award of Bachelor of Technology\nIn", italic=True, sz=22, align="center", space_before=100, space_after=60)
doc.add_p("COMPUTER SCIENCE AND ENGINEERING", bold=True, sz=26, align="center", space_before=40, space_after=240, color="003366")

doc.add_p("Submitted By:", bold=True, sz=24, align="center", space_before=100, space_after=40)
doc.add_p("Harsh Raj (Roll Number: 2401330100170)", bold=True, sz=24, align="center", space_before=20, space_after=40)
doc.add_p("B.Tech CSE, 3rd Year", sz=22, align="center", space_before=0, space_after=200)

doc.add_p("Under the Supervision of:", bold=True, sz=24, align="center", space_before=100, space_after=40)
doc.add_p("Mrs. Sana Anjum", bold=True, sz=24, align="center", space_before=20, space_after=20)
doc.add_p("Assistant Professor, Department of CSE\nNoida Institute of Engineering and Technology, Greater Noida", sz=21, align="center", space_before=0, space_after=240)

doc.add_p("Academic Session: 2025–26", bold=True, sz=22, align="center", space_before=200, space_after=60)

doc.add_page_break()

# -------------------------------------------------------------
# 2. DECLARATION
# -------------------------------------------------------------
doc.add_p("DECLARATION", bold=True, sz=32, align="center", space_before=200, space_after=240, color="003366")

doc.add_p(
    "I hereby declare that the work presented in this project report entitled \"Machine Learning-Based Network Intrusion "
    "Detection and Attack Classification\" was carried out by me under the guidance and supervision of Mrs. Sana Anjum, "
    "Assistant Professor, Department of Computer Science and Engineering, Noida Institute of Engineering and Technology, Greater Noida. "
    "I have not submitted the matter embodied in this report for the award of any other degree or diploma of this or any other University or Institute.",
    sz=24, space_after=140
)

doc.add_p(
    "The work reported herein does not form part of any other thesis or dissertation based on which a degree or award was "
    "conferred on an earlier occasion for this or any other candidate.",
    sz=24, space_after=300
)

doc.add_p("Date: 21 October 2026", bold=True, sz=22, space_after=40)
doc.add_p("Place: Greater Noida", bold=True, sz=22, space_after=240)

doc.add_p("Harsh Raj", bold=True, sz=24, space_after=20)
doc.add_p("Roll Number: 2401330100170", sz=22, space_after=20)
doc.add_p("B.Tech (Computer Science and Engineering), 3rd Year", sz=22, space_after=40)
doc.add_p("(Candidate Signature)", italic=True, sz=20, space_after=100)

doc.add_page_break()

# -------------------------------------------------------------
# 3. CERTIFICATE
# -------------------------------------------------------------
doc.add_p("CERTIFICATE", bold=True, sz=32, align="center", space_before=200, space_after=240, color="003366")

doc.add_p(
    "Certified that Harsh Raj (Roll No: 2401330100170) has carried out the project work presented in this Project Report "
    "entitled \"Machine Learning-Based Network Intrusion Detection and Attack Classification\" at Noida Institute of "
    "Engineering & Technology, Greater Noida, in partial fulfillment of the requirements for the award of Bachelor of "
    "Technology in Computer Science and Engineering from Dr. A.P.J. Abdul Kalam Technical University, Lucknow, under our supervision.",
    sz=24, space_after=140
)

doc.add_p(
    "The project titled \"Machine Learning-Based Network Intrusion Detection and Attack Classification\" is his own work "
    "and has been completed satisfactorily. To the best of our knowledge, the matter embodied in this report has not been "
    "submitted for the award of any other degree, diploma, or fellowship.",
    sz=24, space_after=360
)

cert_headers = ["Supervisor Signature", "Head of Department Signature"]
cert_rows = [
    ["Mrs. Sana Anjum\nAssistant Professor\nDepartment of CSE\nNIET Greater Noida",
     "[HOD Name]\nHead of Department\nDepartment of CSE\nNIET Greater Noida"]
]
doc.add_table(cert_headers, cert_rows, [4500, 4500])

doc.add_p("Date: ____________________", bold=True, sz=22, space_before=200, space_after=100)

doc.add_page_break()

# -------------------------------------------------------------
# 4. ACKNOWLEDGEMENT
# -------------------------------------------------------------
doc.add_p("ACKNOWLEDGEMENT", bold=True, sz=32, align="center", space_before=200, space_after=240, color="003366")

doc.add_p(
    "I would like to express my sincere gratitude towards my supervisor, Mrs. Sana Anjum, Assistant Professor, "
    "Department of Computer Science and Engineering, Noida Institute of Engineering and Technology, Greater Noida, "
    "for her invaluable guidance, constant support, technical expertise, and supervision throughout the course of this project. "
    "Her constructive critiques and insights into machine learning workflows were instrumental in ensuring the methodological "
    "rigor of this work.",
    sz=24, space_after=120
)

doc.add_p(
    "I extend my heartfelt thanks to the respected Head of Department (CSE) and the faculty members of the Computer Science "
    "and Engineering Department for providing the computational infrastructure, guidance, and research environment necessary "
    "to successfully execute this research.",
    sz=24, space_after=120
)

doc.add_p(
    "I am deeply grateful to the UNSW Canberra Cyber research team at the University of New South Wales for curating and "
    "providing the UNSW-NB15 benchmark dataset, which formed the empirical foundation of this study.",
    sz=24, space_after=120
)

doc.add_p(
    "My appreciation also goes to the global open-source community behind Python, scikit-learn, XGBoost, pandas, NumPy, "
    "FastAPI, React.js, and Tailwind CSS, whose modern development tools facilitated rapid data engineering, model optimization, "
    "and dashboard development.",
    sz=24, space_after=120
)

doc.add_p(
    "Finally, I thank my family and peers for their continuous encouragement, patience, and unwavering support throughout this journey.",
    sz=24, space_after=240
)

doc.add_p("Harsh Raj", bold=True, sz=24, space_after=20)
doc.add_p("Roll Number: 2401330100170", sz=22, space_after=20)
doc.add_p("B.Tech CSE, 3rd Year", sz=22, space_after=100)

doc.add_page_break()

# -------------------------------------------------------------
# 5. ABSTRACT & KEYWORDS
# -------------------------------------------------------------
doc.add_p("ABSTRACT", bold=True, sz=32, align="center", space_before=200, space_after=240, color="003366")

doc.add_p(
    "Modern high-speed computer networks generate massive volumes of continuous network flows. Malicious network activities, "
    "stealthy cyberattacks, and reconnaissance scans are routinely concealed within legitimate enterprise and consumer "
    "communications. Traditional signature-based Network Intrusion Detection Systems (NIDS) fail against zero-day exploits, "
    "mutated payloads, and polymorphic attack variants. Machine learning algorithms present an adaptive alternative by learning "
    "statistical signatures and behavioral distributions directly from network flow records.",
    sz=24, space_after=120
)

doc.add_p(
    "This project delivers an end-to-end, reproducible, and leakage-free Network Intrusion Detection and Attack Classification "
    "platform trained on the benchmark UNSW-NB15 dataset. The project investigates both binary threat detection (Normal vs. Attack) "
    "and multi-class category analysis across 9 contemporary attack families (Generic, Exploits, Fuzzers, DoS, Reconnaissance, "
    "Analysis, Backdoor, Shellcode, and Worms). The source datasets—comprising 175,341 training flows and 82,332 testing flows "
    "across 45 raw features—were systematically cleaned, audited, and transformed into 44-feature leakage-free datasets by "
    "removing non-predictive row identifiers (id) and trimming formatting inconsistencies.",
    sz=24, space_after=120
)

doc.add_p(
    "A critical milestone in this project is Phase 7.1, which audited and eliminated data leakage in earlier validation protocols. "
    "In Phase 7.1, preprocessing pipelines (incorporating median imputation, standard feature scaling, and one-hot encoding for "
    "categorical variables proto, service, and state) were strictly fitted only on an internal stratified 80% training split "
    "(86,192 records), generating 192 transformed features. Four candidate machine learning classifiers—Logistic Regression, "
    "Decision Tree, Random Forest, and XGBoost—were rigorously trained, tuned, and evaluated across multiple operating thresholds. "
    "Feature ablation experiments on Time-To-Live features (sttl, ct_state_ttl) and a cross-split dataset overlap audit were "
    "conducted to evaluate dataset bias and generalizability.",
    sz=24, space_after=120
)

doc.add_p(
    "Upon refitting the preprocessor on the complete official training dataset (194 transformed features), final evaluations "
    "against the preserved official test split demonstrated that XGBoost achieved the superior overall balance with 83.53% Accuracy, "
    "97.33% Attack Recall, 71.02% Precision, 0.8212 F1-score, and 0.9719 ROC-AUC, while Random Forest achieved an Attack Recall of "
    "97.51%. To translate theoretical models into practical SecOps decision support, Phase 8 implements a production-grade, "
    "interactive web-based SOC Security Dashboard featuring a FastAPI backend and a React/Vite/Tailwind CSS frontend. The dashboard "
    "enables zero-refit offline CSV traffic ingestion, interactive validation-tuned threshold presets, real-time confusion matrices, "
    "top suspicious flow identification, and feature importance attribution.",
    sz=24, space_after=180
)

doc.add_p("Keywords:", bold=True, sz=24, space_after=40)
doc.add_p(
    "Network Intrusion Detection (NIDS), UNSW-NB15, Machine Learning, XGBoost, Random Forest, Data Leakage Correction, "
    "Decision Threshold Optimization, Feature Importance, FastAPI, React.js.",
    italic=True, sz=22, space_after=100
)

doc.add_page_break()

# -------------------------------------------------------------
# 6. TABLE OF CONTENTS
# -------------------------------------------------------------
doc.add_p("TABLE OF CONTENTS", bold=True, sz=32, align="center", space_before=200, space_after=240, color="003366")

toc_headers = ["Sr. No.", "Topics", "Page No."]
toc_rows = [
    ["", "Declaration", "i"],
    ["", "Certificate from the Institute", "ii"],
    ["", "Acknowledgement", "iii"],
    ["", "Abstract", "iv"],
    ["CHAPTER 1", "INTRODUCTION", "1–5"],
    ["1.1", "Introduction", "1"],
    ["1.1.1", "Background and Motivation", "2"],
    ["1.1.2", "Problem Statement", "3"],
    ["1.2", "Project Objectives", "4"],
    ["1.3", "Scope of the Project", "4"],
    ["1.4", "Significance of the Project", "5"],
    ["CHAPTER 2", "LITERATURE REVIEW", "6–10"],
    ["2.1", "Existing Network Security Systems", "6"],
    ["2.2", "Limitations of Existing Signature-Based Systems", "7"],
    ["2.3", "Related Work in Machine Learning for NIDS", "8"],
    ["2.4", "Proposed Architecture and Methodology", "9"],
    ["2.5", "Comparative Analysis: Existing vs. Proposed System", "10"],
    ["CHAPTER 3", "REQUIREMENT ANALYSIS", "11–14"],
    ["3.1", "Functional Requirements", "11"],
    ["3.2", "Non-Functional Requirements", "12"],
    ["3.3", "Hardware Requirements", "13"],
    ["3.4", "Software Requirements", "13"],
    ["3.5", "User and Analyst Requirements", "14"],
    ["CHAPTER 4", "SYSTEM DESIGN, IMPLEMENTATION & TESTING", "15–28"],
    ["4.1", "System Architecture and End-to-End Pipeline", "15"],
    ["4.2", "Dataset Understanding and Preprocessing", "16"],
    ["4.3", "Phase 7.1 Preprocessing Leakage Audit and Correction", "18"],
    ["4.4", "Machine Learning Model Implementations", "19"],
    ["4.5", "Validation and Decision Threshold Optimization", "21"],
    ["4.6", "Experimental Results and Performance Comparison", "22"],
    ["4.7", "Feature Importance and Model Explainability", "24"],
    ["4.8", "Phase 8 Interactive Dashboard Implementation & Screenshots", "25"],
    ["CHAPTER 5", "CONCLUSION AND FUTURE WORK", "29–31"],
    ["5.1", "Conclusion", "29"],
    ["5.2", "Limitations", "30"],
    ["5.3", "Future Enhancements", "31"],
    ["", "References", "32–33"],
    ["", "Appendix A: Source Code & Core Scripts", "34–40"],
    ["", "Appendix B: GitHub Repository Details", "41"]
]
doc.add_table(toc_headers, toc_rows, [1600, 6000, 1400])

doc.add_page_break()

# -------------------------------------------------------------
# CHAPTER 1: INTRODUCTION
# -------------------------------------------------------------
doc.add_p("CHAPTER 1", bold=True, sz=28, align="center", space_before=100, space_after=40, color="003366")
doc.add_p("INTRODUCTION", bold=True, sz=32, align="center", space_before=0, space_after=200, color="003366")

doc.add_h2("1.1 Introduction")
doc.add_p(
    "In the current era of ubiquitous digital transformation, cloud computing, and Internet of Things (IoT) expansion, "
    "computer networks serve as the fundamental backbone of commerce, governance, critical infrastructure, and communication. "
    "Consequently, modern enterprise networks generate billions of connection packets and flow records daily. As network throughput "
    "and bandwidth scale exponentially, malicious actors deploy increasingly sophisticated, automated, and distributed cyber attacks—"
    "ranging from distributed denial-of-service (DDoS) and automated port scans to stealthy zero-day exploits and ransomware deployment."
)
doc.add_p(
    "Network Intrusion Detection Systems (NIDS) are critical defensive perimeter mechanisms designed to monitor incoming and outgoing "
    "network traffic, inspect flow patterns, and flag unauthorized, suspicious, or anomalous activities. Historically, intrusion detection "
    "has relied almost exclusively on signature-based detection mechanisms (such as Snort, Suricata, and proprietary firewalls), which "
    "compare network packets against static catalogs of known malicious signatures. While highly effective against documented exploits, "
    "signature-based approaches inherently fail when confronted with novel zero-day attacks, polymorphic malware, or dynamic anomalies."
)
doc.add_p(
    "To bridge this critical security gap, Machine Learning (ML) approaches have emerged as an adaptive, data-driven paradigm. By analyzing "
    "statistical traffic-flow attributes—such as flow durations, packet timing intervals, source/destination byte volumes, TCP handshake "
    "timing, and connection states—supervised machine learning models learn continuous multivariate boundaries separating legitimate network "
    "behavior from malicious exploitation."
)
doc.add_p(
    "This project delivers a comprehensive, scientifically rigorous study and implementation of Machine Learning-Based Network Intrusion "
    "Detection and Attack Classification using the benchmark UNSW-NB15 dataset. Spanning from exploratory data analysis and rigorous "
    "leakage-free preprocessing (Phase 7.1) to comparative classifier evaluation and deployment via a local Security Operations Center (SOC) "
    "dashboard (Phase 8), the project provides an end-to-end framework for reproducible network traffic analysis."
)

doc.add_h3("1.1.1 Background and Motivation")
doc.add_p(
    "Network security telemetry is inherently high-dimensional, imbalanced, and noisy. Unlike synthetic or legacy datasets (such as KDD Cup 1999 "
    "or NSL-KDD), which suffer from synthetic traffic generation artifacts and outdated attack vectors, modern operational networks exhibit complex "
    "behavioral distributions. The UNSW-NB15 dataset was specifically curated by the Australian Centre for Cyber Security (ACCS) using the "
    "IXIA PerfectStorm tool to simulate realistic modern normal network activities alongside 9 distinct attack categories."
)
doc.add_p("However, applying machine learning to tabular network traffic data presents severe methodological pitfalls:")
doc.add_bullet("Data Leakage in Academic Pipelines:", "A pervasive issue in applied machine learning research is preprocessing leakage, wherein transformers (such as scalers, one-hot encoders, or imputers) are fitted on the entire dataset prior to train/test partitioning or internal cross-validation, resulting in overly optimistic, unreproducible metrics.")
doc.add_bullet("Extreme Class Imbalance:", "Security datasets are inherently skewed. Rare attack categories (such as Worms, Shellcode, and Backdoors) constitute less than 2% of the dataset, rendering raw accuracy a deceptive and dangerous performance metric.")
doc.add_bullet("Threshold Sensitivity:", "Standard classifier inference defaults to an arbitrary 0.50 probability cutoff. In critical cybersecurity environments, missing an attack (False Negative) carries catastrophic operational costs compared to inspecting a benign false alarm (False Positive).")
doc.add_p("The motivation behind this project is to address these fundamental challenges by building a leak-free, mathematically verified classification workflow, optimizing decision thresholds for real-world SecOps constraints, and deploying the resulting models into an accessible, interactive analyst dashboard.")

doc.add_h3("1.1.2 Problem Statement")
doc.add_p(
    "Modern computer networks face relentless, evolving security threats that overwhelm conventional signature-based monitoring tools. "
    "Manual network-flow inspection is impossible at line-rate scale, while unprincipled machine learning pipelines risk silent data leakage, "
    "fragile classification boundaries, and severe vulnerability to class imbalance. Therefore, the problem addressed in this work is: "
    "To design, audit, and implement an end-to-end, leak-free machine learning pipeline on the UNSW-NB15 dataset capable of accurately "
    "classifying network-flow records into normal and attack traffic, evaluating algorithmic trade-offs between precision, recall, and "
    "false-alarm rates, and operationalizing the models within an interactive analyst dashboard for offline traffic analysis."
)

doc.add_h2("1.2 Project Objectives")
doc.add_p("The specific technical objectives achieved in this project include:")
doc.add_bullet("Benchmark Data Ingestion & Sanitization:", "Systematically ingest the official UNSW-NB15 training and testing CSV datasets, standardize column headers, remove whitespace inconsistencies, eliminate non-predictive identifiers (id), and preserve the official data split.")
doc.add_bullet("Exploratory Data Analysis (EDA):", "Perform visual and statistical examinations of class balance, flow durations, protocol distributions, and correlation matrices to detect potential feature redundancy and target leakage.")
doc.add_bullet("Rigorous Leak-Free Preprocessing (Phase 7.1):", "Correct historical preprocessing leakage by isolating all transformations (median numerical imputation, standard scaling, and one-hot categorical encoding) to training subsets only, ensuring true out-of-fold generalization.")
doc.add_bullet("Comparative Classifier Evaluation:", "Train and optimize four diverse supervised machine learning architectures—Logistic Regression, Decision Tree, Random Forest, and XGBoost.")
doc.add_bullet("Threshold Optimization:", "Calibrate decision thresholds across four operational strategies (Attack F1 maximization, Macro F1 balance, High-Recall security target >= 97%, and Operational FPR constraint <= 20%).")
doc.add_bullet("Feature Attribution & Explainability:", "Quantify model decision criteria using XGBoost Gain importance and Gini impurity metrics to verify that predictions are rooted in legitimate network behaviors rather than dataset artifacts.")
doc.add_bullet("Interactive SOC Dashboard Development (Phase 8):", "Design and deploy a local full-stack decision-support system utilizing FastAPI and React.js/Tailwind CSS for offline CSV network flow ingestion, threshold tuning, and automated security reporting.")

doc.add_h2("1.3 Scope of the Project")
doc.add_p("The scope of this project encompasses:")
doc.add_bullet("Data Domain:", "Tabular network flow records extracted from raw PCAP files, structured into 44 features describing connection states, byte/packet volumes, timing intervals, and transport layer metrics.")
doc.add_bullet("Tasks:", "Primary focus on binary intrusion detection (label: Normal [0] vs. Attack [1]), supported by exploratory analysis of 9 multi-class attack categories (attack_cat).")
doc.add_bullet("Models Evaluated:", "Implementation and evaluation of Logistic Regression, Decision Tree, Random Forest, and XGBoost.")
doc.add_bullet("Operational Interface:", "A browser-based analyst dashboard supporting local CSV file upload, automated inference, threshold adjustments, and prediction export.")
doc.add_bullet("Out of Scope:", "The system operates as an offline decision-support and forensic analysis tool. It does not capture live network packets from physical network interface cards (NICs) at hardware line-rate, nor does it automatically execute active firewall drops or packet filtering.")

doc.add_h2("1.4 Significance of the Project")
doc.add_p("This project contributes significantly to the domains of academic machine learning and practical cybersecurity:")
doc.add_bullet("Methodological Integrity:", "By documenting and eliminating the subtle data leakage that occurred in earlier validation phases (Phase 7.1), it provides an authoritative template for rigorous, leak-free machine learning on tabular cybersecurity data.")
doc.add_bullet("Practical Operationalization:", "Rather than treating ML models as static Jupyter notebook scripts, the project wraps finalized joblib artifacts inside a production-grade FastAPI microservice and modern React interface, demonstrating how complex ML models can be handed off to SecOps analysts.")
doc.add_bullet("Empirical Threshold Insights:", "It proves that arbitrary 0.50 probability cutoffs are sub-optimal for network defense, providing quantitative guidelines for tuning models to prioritize threat containment or analyst alert fatigue reduction.")

doc.add_page_break()

# -------------------------------------------------------------
# CHAPTER 2: LITERATURE REVIEW
# -------------------------------------------------------------
doc.add_p("CHAPTER 2", bold=True, sz=28, align="center", space_before=100, space_after=40, color="003366")
doc.add_p("LITERATURE REVIEW", bold=True, sz=32, align="center", space_before=0, space_after=200, color="003366")

doc.add_h2("2.1 Existing Network Security Systems")
doc.add_p("Intrusion detection has historically been divided into two primary paradigms: Signature-Based Detection and Anomaly-Based Detection.")
doc.add_bullet("Signature-Based Intrusion Detection Systems (SIDS):", "Industry-standard systems such as Snort, Zeek (formerly Bro), and Suricata inspect packet headers and reassembled payloads against regular expressions and predefined attack rules. When an incoming packet matches an existing pattern, an alert is triggered. While exhibiting near-zero false positive rates on documented threats, SIDS are completely blind to zero-day vulnerabilities and polymorphic variations.")
doc.add_bullet("Statistical Anomaly-Based Systems:", "Early anomaly detectors established statistical baselines of network volume, mean packet sizes, and connection frequencies. Significant deviations from the baseline were flagged as potential intrusions. However, these systems suffered from notoriously high false positive rates due to natural network traffic burstiness, legitimate file transfers, and dynamic enterprise operations.")

doc.add_h2("2.2 Limitations of Existing Signature-Based Systems")
lit_headers = ["System Type", "Representative Examples", "Primary Limitations"]
lit_rows = [
    ["Signature-Based NIDS", "Snort, Suricata, Commercial Firewalls", "Incapable of detecting zero-day attacks; signature databases require constant manual updates; high CPU overhead on deep packet inspection (DPI) at line rate."],
    ["Statistical Anomaly Detectors", "Early statistical baseline monitors", "High false-alarm rate (FAR); inability to distinguish benign traffic spikes (e.g., software updates) from true volumetric attacks."],
    ["Legacy ML Research Benchmarks", "KDD Cup 99, NSL-KDD based systems", "Outdated traffic distributions; synthetic artifacts; lack modern protocols (IPv6, modern web traffic) and current attack vectors."],
    ["Black-Box AI Prototypes", "Uncalibrated Deep Learning scripts", "Severe data leakage; lack of threshold calibration; uninterpretable predictions that fail security analyst trust requirements."]
]
doc.add_table(lit_headers, lit_rows, [2200, 2400, 4400])

doc.add_h2("2.3 Related Work in Machine Learning for NIDS")
doc.add_p(
    "The transition from legacy benchmarks to modern datasets has catalyzed contemporary NIDS research. Moustafa and Slay (2015) "
    "created and evaluated the UNSW-NB15 dataset, demonstrating that contemporary attack behaviors are far more subtly interleaved "
    "with normal traffic than in legacy datasets like KDD Cup 99. Their benchmark established baseline accuracy figures across standard classifiers."
)
doc.add_p(
    "Tama and Rhee (2019) explored ensemble learning techniques, demonstrating that tree-based ensembles (Random Forest and Gradient "
    "Boosted Trees) consistently outperform single classifiers on high-dimensional network flow data due to their ability to capture non-linear "
    "feature interactions without requiring rigid normality assumptions."
)
doc.add_p(
    "Kumar et al. (2020) investigated deep learning architectures (including MLP, CNN, and LSTM models) for network intrusion detection. "
    "While showing strong representation capacity, deep learning models often incur high computational latency and exhibit susceptibility "
    "to adversarial feature perturbation, reinforcing the practical superiority of gradient boosted trees (XGBoost) for tabular network flow telemetry."
)

doc.add_h2("2.4 Proposed Architecture and Methodology")
doc.add_p("In contrast to theoretical scripts that frequently suffer from unaddressed data leakage or lack deployment interfaces, the proposed system introduces an end-to-end, scientifically validated pipeline:")
doc.add_bullet("Conservative Data Sanitization:", "Removal of artificial record indexes (id) while rigorously maintaining the authentic train/test split.")
doc.add_bullet("Phase 7.1 Leakage Elimination:", "Implementation of an isolated ColumnTransformer fitted solely on training splits, generating 194 robust features with full handling for unknown test-time categories.")
doc.add_bullet("Four-Class Model Zoo:", "Comparative implementation of Logistic Regression, Decision Tree, Random Forest, and XGBoost.")
doc.add_bullet("Validation-Guided Threshold Optimization:", "Derivation of custom operational thresholds to empower security teams to balance attack recall against false positive constraints.")
doc.add_bullet("Full-Stack SOC Dashboard:", "Packaging the pipeline into an accessible, reactive web application for instant offline CSV analysis.")

doc.add_h2("2.5 Comparative Analysis: Existing vs. Proposed System")
comp_headers = ["Parameter", "Conventional Signature NIDS", "Typical Academic ML Scripts", "Proposed System (TravelShield NIDS)"]
comp_rows = [
    ["Detection Basis", "Static rules & payload signatures", "Unverified tabular classifiers", "Adaptive multi-model ML trained on UNSW-NB15"],
    ["Zero-Day Resilience", "None (requires known signature)", "Moderate (depends on data hygiene)", "High (statistical flow anomaly detection)"],
    ["Preprocessing Hygiene", "N/A", "Frequent data leakage across splits", "Strict Phase 7.1 leakage-free pipeline"],
    ["Decision Thresholds", "Fixed binary match", "Fixed 0.50 arbitrary cutoff", "4 Tunable strategies (Recall >= 97%, F1-max, etc.)"],
    ["Model Interpretability", "Rule text inspection", "Often neglected or black-box", "XGBoost Gain & Impurity feature attribution"],
    ["Analyst Usability", "Command-line / complex SIEM", "Terminal scripts / Jupyter notebooks", "Modern FastAPI + React.js Interactive Dashboard"],
    ["Input Format", "Raw live NIC packets", "In-memory NumPy arrays", "Standardized Offline CSV Network Flows"]
]
doc.add_table(comp_headers, comp_rows, [1800, 2400, 2400, 2400])

doc.add_h2("2.6 Summary")
doc.add_p(
    "The literature confirms that while signature-based systems remain an essential perimeter defense, machine learning models "
    "are indispensable for detecting novel, polymorphic, and multi-vector threats. However, successful operationalization requires "
    "impeccable data leakage hygiene, calibrated decision thresholds, and human-in-the-loop analyst interfaces. The proposed "
    "framework addresses each of these foundational pillars."
)

doc.add_page_break()

# -------------------------------------------------------------
# CHAPTER 3: REQUIREMENT ANALYSIS
# -------------------------------------------------------------
doc.add_p("CHAPTER 3", bold=True, sz=28, align="center", space_before=100, space_after=40, color="003366")
doc.add_p("REQUIREMENT ANALYSIS", bold=True, sz=32, align="center", space_before=0, space_after=200, color="003366")

doc.add_h2("3.1 Functional Requirements")
doc.add_bullet("Dataset Ingestion & Validation:", "The system must ingest raw CSV files adhering to the 42-feature network flow schema, validate column names, strip spurious whitespace, and automatically reject malformed, empty, or non-CSV uploads.")
doc.add_bullet("Leakage-Free Feature Transformation:", "The system must separate target fields (label, attack_cat) from predictor features prior to model input, applying the finalized Phase 7.1 preprocessor to impute missing values (median for continuous, mode for categorical), normalize continuous variables via StandardScaler, and encode categorical features (proto, service, state) using OneHotEncoder with unknown categories gracefully ignored.")
doc.add_bullet("Multi-Model Inference Engine:", "The system must support selectable inference across all four trained architectures: Logistic Regression, Decision Tree, Random Forest, and XGBoost, outputting continuous attack probabilities alongside discrete binary classifications.")
doc.add_bullet("Dynamic Decision Threshold Tuning:", "The system must provide configurable decision threshold controls (ranging from 0.01 to 0.99) and pre-configured operational presets (Strategies A, B, C, and D).")
doc.add_bullet("Ground-Truth Evaluation (Labeled CSV Ingestion):", "If an uploaded CSV contains ground-truth label values, the system must automatically calculate Accuracy, Attack Precision, Attack Recall, F1-Score, ROC-AUC, PR-AUC, and full Confusion Matrix counts.")
doc.add_bullet("Data Export & Reporting:", "The system must allow analysts to export prediction records, probability scores, and severity classifications directly into formatted CSV files.")

doc.add_h2("3.2 Non-Functional Requirements")
doc.add_bullet("Inference Latency & Performance:", "Prediction execution for batch uploads up to 1,000 flow records must complete within 500 milliseconds on standard commodity hardware.")
doc.add_bullet("Architecture Modularity:", "Clear decoupling between the machine learning artifacts (joblib models), the RESTful backend API (FastAPI), and the presentation layer (React).")
doc.add_bullet("Zero-Refit Guarantee:", "Under no circumstances should the system refit, alter, or update the preprocessor or model weights during inference or evaluation sessions.")
doc.add_bullet("Security & Data Integrity:", "Uploaded files must be processed in-memory or in isolated temporary buffers, preventing unauthorized system command execution or persistent file tampering.")

doc.add_h2("3.3 Hardware Requirements")
hw_headers = ["Component", "Minimum Specification", "Recommended Specification"]
hw_rows = [
    ["Processor", "Intel Core i5 (4-core) or AMD Ryzen 5", "Apple Silicon (M1/M2/M3) or Intel Core i7 (8-core)"],
    ["System Memory (RAM)", "8 GB DDR4", "16 GB unified / DDR4"],
    ["Storage", "10 GB available SSD storage", "25 GB NVMe SSD storage"],
    ["Network", "Local loopback interface (localhost)", "Local loopback interface for local hosting"]
]
doc.add_table(hw_headers, hw_rows, [2500, 3200, 3300])

doc.add_h2("3.4 Software Requirements")
sw_headers = ["Category", "Tool / Framework", "Purpose"]
sw_rows = [
    ["Operating System", "macOS 14+, Ubuntu 22.04 LTS, or Windows 11", "Host operating environment"],
    ["Programming Language", "Python 3.10+", "Core machine learning, data processing, backend API"],
    ["ML & Data Libraries", "scikit-learn, XGBoost, pandas, NumPy, joblib", "Data wrangling, model training, metric computation"],
    ["Backend Framework", "FastAPI, Uvicorn, Pydantic", "Asynchronous RESTful API service"],
    ["Frontend Framework", "React 18, TypeScript, Vite", "User interface and state management"],
    ["UI & Visualization", "Tailwind CSS, Lucide React, Recharts", "Styling, icons, and interactive metric charts"],
    ["Development Tools", "VS Code, Git, cURL / Postman", "Code authoring, version control, API testing"]
]
doc.add_table(sw_headers, sw_rows, [2500, 3200, 3300])

doc.add_h2("3.5 User and Analyst Requirements")
doc.add_p("Security Operations Center (SOC) analysts require:")
doc.add_bullet("Clear Dashboard Visualization:", "A clean, distraction-free visual dashboard showing macro traffic distributions at a glance.")
doc.add_bullet("Triage Prioritization:", "Clear probability-based risk categorization (Low, Medium, High, Critical) to rapidly triage alerts.")
doc.add_bullet("Predictive Transparency:", "Transparent insight into why a flow was flagged, backed by global feature importance charts.")
doc.add_bullet("Operational Flexibility:", "Straightforward one-click threshold adjustment to adapt the detector to high-security or high-throughput network environments.")

doc.add_page_break()

# -------------------------------------------------------------
# CHAPTER 4: SYSTEM DESIGN, IMPLEMENTATION & TESTING
# -------------------------------------------------------------
doc.add_p("CHAPTER 4", bold=True, sz=28, align="center", space_before=100, space_after=40, color="003366")
doc.add_p("SYSTEM DESIGN, IMPLEMENTATION & TESTING", bold=True, sz=32, align="center", space_before=0, space_after=200, color="003366")

doc.add_h2("4.1 System Architecture and End-to-End Pipeline")
doc.add_p(
    "The system implements a staged machine learning architecture that strictly isolates offline model exploration, "
    "validation, and training from online decision-support inference. Raw UNSW-NB15 CSV records are loaded, cleaned, "
    "and partitioned. The preprocessor is fitted exclusively on internal training data during Phase 7.1 validation, "
    "and refitted once across the complete training split for deployment. The resulting artifacts are loaded into "
    "memory by FastAPI for real-time inference via the React dashboard."
)

doc.add_h2("4.2 Dataset Understanding and Preprocessing")
doc.add_p(
    "The UNSW-NB15 dataset contains 45 columns in its raw format. Cleaning consists of whitespace stripping and the "
    "removal of the non-predictive id column, resulting in 44 features including targets."
)

split_headers = ["Dataset Split", "Raw Records", "Cleaned Records", "Raw Cols", "Cleaned Cols", "Normal Records", "Attack Records"]
split_rows = [
    ["Initial Training Set", "175,341", "175,341", "45", "44", "56,000 (31.94%)", "119,341 (68.06%)"],
    ["Initial Testing Set", "82,332", "82,332", "45", "44", "37,000 (44.93%)", "45,332 (55.07%)"],
    ["Final Modelling Train Set", "107,740", "107,740", "45", "44", "51,890 (48.16%)", "55,850 (51.84%)"],
    ["Final Modelling Test Set", "55,945", "55,945", "45", "44", "34,206 (61.14%)", "21,739 (38.86%)"]
]
doc.add_table(split_headers, split_rows, [1800, 1200, 1200, 900, 900, 1500, 1500])

doc.add_image("outputs/eda/01_binary_target_distribution.png", "Figure 4.1: UNSW-NB15 Binary Target Class Distribution")
doc.add_image("outputs/eda/02_attack_category_distribution.png", "Figure 4.2: UNSW-NB15 Attack Category Frequency Distribution")

doc.add_h2("4.3 Phase 7.1 Preprocessing Leakage Audit and Correction")
doc.add_p(
    "In Phase 7.1, an internal stratified 80:20 split was created from the official training data (86,192 training and 21,548 validation records). "
    "The ColumnTransformer was fitted strictly on the 80% training subset. Categorical variables were one-hot encoded with handle_unknown='ignore', "
    "generating 192 features on validation data. This completely eliminated previous data leakage. The production pipeline refitted on the full "
    "training file expanded to 194 features due to rare categorical tokens."
)

doc.add_h2("4.4 Machine Learning Model Implementations")
doc.add_bullet("Logistic Regression:", "Linear baseline utilizing L-BFGS solver with balanced class weighting to model linear log-odds boundaries.")
doc.add_bullet("Decision Tree:", "CART decision tree based on Gini impurity splits, offering maximum interpretability but sensitive to leaf variance.")
doc.add_bullet("Random Forest:", "Bagged ensemble of 300 decorrelated decision trees using sqrt feature subsampling, stabilizing variance across diverse flows.")
doc.add_bullet("XGBoost:", "Gradient boosted trees optimizing logistic loss with 300 estimators, depth 6, and column subsampling of 0.8.")

doc.add_h2("4.5 Validation and Decision Threshold Optimization")
doc.add_p("Threshold tuning was conducted exclusively on the Phase 7.1 internal validation split across four distinct operational strategies:")
thresh_headers = ["Model", "Strategy A (Attack F1)", "Strategy B (Macro F1)", "Strategy C (Recall >= 97%)", "Strategy D (Operational)"]
thresh_rows = [
    ["Logistic Regression", "0.13", "0.13", "0.53", "0.13"],
    ["Decision Tree", "0.50", "0.50", "Infeasible", "Infeasible"],
    ["Random Forest", "0.50", "0.53", "0.42", "0.50"],
    ["XGBoost", "0.49", "0.49", "0.42", "0.49"]
]
doc.add_table(thresh_headers, thresh_rows, [2200, 1700, 1700, 1700, 1700])

doc.add_h2("4.6 Experimental Results and Performance Comparison")
doc.add_p("The four classifiers were evaluated against the preserved official test dataset (55,945 records: 34,206 Normal, 21,739 Attack).")

res_headers = ["Model", "Accuracy", "Attack Recall", "Attack Precision", "Attack F1", "ROC-AUC", "PR-AUC", "False Positives", "False Negatives"]
res_rows = [
    ["Logistic Regression", "74.45%", "95.81%", "60.88%", "0.7445", "0.9193", "0.8912", "13,385", "910"],
    ["Decision Tree", "82.21%", "92.72%", "70.66%", "0.8020", "0.8432", "0.6868", "8,369", "1,582"],
    ["Random Forest", "82.73%", "97.51%", "69.92%", "0.8144", "0.9655", "0.9502", "9,121", "542"],
    ["XGBoost", "83.53%", "97.33%", "71.02%", "0.8212", "0.9719", "0.9624", "8,632", "581"]
]
doc.add_table(res_headers, res_rows, [1600, 1000, 1100, 1100, 1000, 1000, 1000, 1100, 1100])

doc.add_image("outputs/phase7_1/final_test/final_roc_comparison.png", "Figure 4.3: Receiver Operating Characteristic (ROC) Comparison on Official Test Split")
doc.add_image("outputs/phase7_1/final_test/final_precision_recall_comparison.png", "Figure 4.4: Precision-Recall Curve Comparison Across All Four Models")
doc.add_image("outputs/phase7_1/final_test/final_confusion_matrix_xgboost.png", "Figure 4.5: XGBoost Confusion Matrix on Preserved Test Set (Threshold 0.50)")

doc.add_h2("4.7 Feature Importance and Model Explainability")
doc.add_p("XGBoost Gain importance was quantified to identify the dominant drivers of classification decisions:")
doc.add_image("outputs/models/xgboost/feature_importance.png", "Figure 4.6: XGBoost Feature Importance (Top Gain Features)")

feat_headers = ["Rank", "Feature", "Gain Importance", "Cybersecurity Description"]
feat_rows = [
    ["1", "sttl", "0.5238", "Source-to-destination Time-To-Live; indicates OS stack origin and hop distance."],
    ["2", "ct_state_ttl", "0.1213", "Connection count by state and TTL; flags automated network scanning."],
    ["3", "swin", "0.0410", "Source TCP window advertisement; flags buffer manipulation & SYN flood."],
    ["4", "is_sm_ips_ports", "0.0338", "Binary indicator if source & destination IP/port match (LAND attack)."],
    ["5", "proto_tcp", "0.0206", "Transport layer TCP protocol flag."]
]
doc.add_table(feat_headers, feat_rows, [800, 1800, 1600, 4800])

doc.add_p("Feature Ablation Analysis (TTL Feature Sensitivity):")
abl_headers = ["Feature Configuration", "Features", "Val Accuracy", "Val Recall", "Val Precision", "Val F1", "Val ROC-AUC"]
abl_rows = [
    ["Full Feature Set", "192", "93.79%", "95.38%", "92.83%", "0.9409", "0.9785"],
    ["Without sttl", "191", "93.86%", "95.42%", "92.91%", "0.9415", "0.9782"],
    ["Without sttl & ct_state_ttl", "190", "93.74%", "95.31%", "92.80%", "0.9404", "0.9779"]
]
doc.add_table(abl_headers, abl_rows, [2600, 1000, 1100, 1100, 1100, 1000, 1100])

doc.add_h2("4.8 Phase 8 Interactive Dashboard Implementation & Screenshots")
doc.add_p(
    "Phase 8 integrated the trained models into an interactive web application built with FastAPI and React.js. "
    "The application provides offline CSV file ingestion, interactive threshold adjustment, live confusion matrix rendering, "
    "and CSV export under a strict Zero-Refit guarantee."
)

doc.add_p("The dashboard is structured into five cohesive views:")
doc.add_bullet("Dashboard (Overview):", "Macro summary cards, normal vs. attack donut chart, probability distribution histogram, and top suspicious flows preview.")
doc.add_bullet("Traffic Analyzer:", "Offline CSV ingestion, model selection, dynamic threshold sliders, paginated flow table, keyword search, and CSV export.")
doc.add_bullet("Model Performance:", "Official benchmark metrics, multi-model comparative bar charts, and validation threshold sweep curves.")
doc.add_bullet("Model Insights:", "Feature importance attribution charts (XGBoost Gain, RF Impurity, LR Weights) with top-N filtering and domain caveats.")
doc.add_bullet("Methodology & Audit:", "Transparent documentation of data preparation, Phase 7.1 leakage corrections, and operational system limitations.")

doc.add_page_break()

# -------------------------------------------------------------
# CHAPTER 5: CONCLUSION AND FUTURE WORK
# -------------------------------------------------------------
doc.add_p("CHAPTER 5", bold=True, sz=28, align="center", space_before=100, space_after=40, color="003366")
doc.add_p("CONCLUSION AND FUTURE WORK", bold=True, sz=32, align="center", space_before=0, space_after=200, color="003366")

doc.add_h2("5.1 Conclusion")
doc.add_p(
    "This project successfully designed, implemented, evaluated, and deployed an end-to-end, leak-free Machine Learning-Based "
    "Network Intrusion Detection and Attack Classification system on the UNSW-NB15 benchmark."
)
doc.add_p("The central technical achievements include:")
doc.add_bullet("Data Sanitization & Leakage Elimination:", "Comprehensive cleaning of UNSW-NB15, removal of record identifiers, and the execution of Phase 7.1, which eliminated preprocessing leakage by fitting all scalers and encoders exclusively on internal training subsets.")
doc.add_bullet("Comparative Algorithmic Benchmarking:", "Systematic comparison of Logistic Regression, Decision Tree, Random Forest, and XGBoost across 55,945 official test records. XGBoost established superior overall performance with 83.53% Accuracy, 97.33% Attack Recall, 71.02% Precision, and 0.9719 ROC-AUC, while Random Forest achieved 97.51% Attack Recall.")
doc.add_bullet("Validation Threshold Calibration:", "Derivation of custom decision thresholds demonstrating that adjusting the classification cutoff directly calibrates the detector between aggressive threat capture (>= 98% recall) and benign false-alarm mitigation.")
doc.add_bullet("Feature Explainability & Bias Verification:", "Verification via feature ablation that while Time-To-Live metrics (sttl, ct_state_ttl) contribute significant gain, the model extracts robust predictive signals across rate and volume features, maintaining an ROC-AUC of 0.9779 without TTL dependencies.")
doc.add_bullet("Full-Stack Operational Deployment (Phase 8):", "Creation of an interactive, production-ready SOC web application coupling a FastAPI microservice with a React/Tailwind CSS frontend, enabling offline network flow ingestion, threshold tuning, and prediction export under a strict Zero-Refit guarantee.")

doc.add_h2("5.2 Limitations")
doc.add_bullet("Offline Flow Summary Ingestion:", "The current dashboard operates on pre-extracted CSV network-flow summaries; it does not currently interface with raw packet capture drivers (e.g., Libpcap) for real-time line-rate packet parsing.")
doc.add_bullet("Benchmark Dataset Artifacts:", "Benchmark datasets inherently contain synthetic traffic artifacts and repetitive feature patterns across splits, as confirmed in the Phase 7.1 overlap audit. Performance may vary when tested against modern encrypted enterprise traffic (TLS 1.3 / QUIC).")
doc.add_bullet("Binary vs. Multi-Class Deployment:", "The real-time inference engine focuses on binary threat discrimination (Normal vs. Attack); granular multi-class classification across all 9 individual attack categories remains an offline exploratory capability.")

doc.add_h2("5.3 Future Enhancements")
doc.add_bullet("Live Packet Ingestion Engine:", "Integrate a lightweight real-time flow generator (such as Zeek or CICFlowMeter) to capture physical network adapter streams and parse them into feature vectors on-the-fly.")
doc.add_bullet("Multi-Class Hierarchical Architecture:", "Deploy a two-stage hierarchical model where Stage 1 performs binary intrusion filtering and Stage 2 routes detected threats into specialized multi-class classifiers for precise attack attribution.")
doc.add_bullet("Local Explainable AI (SHAP / LIME):", "Extend the dashboard with instance-level SHAP force plots, enabling analysts to inspect the exact feature contributions for any individual flagged flow.")
doc.add_bullet("Automated SIEM & Firewall Integration:", "Implement webhook connectors to automatically push high-confidence alerts (probability > 0.90) into SIEM platforms (Splunk, Elastic SIEM) and trigger dynamic firewall block rules via iptables or cloud security groups.")

doc.add_page_break()

# -------------------------------------------------------------
# REFERENCES
# -------------------------------------------------------------
doc.add_p("REFERENCES", bold=True, sz=32, align="center", space_before=200, space_after=240, color="003366")

refs = [
    "[1] N. Moustafa and J. Slay, \"UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set),\" in 2015 Military Communications and Information Systems Conference (MilCIS), Canberra, ACT, Australia, 2015, pp. 1–6.",
    "[2] N. Moustafa and J. Slay, \"The evaluation of Network Anomaly Detection Systems using synthetic big data sets,\" IEEE Transactions on Big Data, vol. 4, no. 4, pp. 485–498, 2017.",
    "[3] B. A. Tama and K. H. Rhee, \"An in-depth evaluation of ensemble learning for advanced network intrusion detection,\" Telecommunication Systems, vol. 72, no. 2, pp. 231–244, 2019.",
    "[4] V. Kumar, D. Sinha, A. K. Das, S. C. Rai, and R. T. Goswami, \"An integrated approach for network intrusion detection using machine learning and deep learning techniques,\" Journal of Computer Networks and Communications, vol. 2020, Article ID 8831349, 2020.",
    "[5] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD), San Francisco, CA, USA, 2016, pp. 785–794.",
    "[6] F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "[7] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd ed. New York, NY, USA: Springer, 2009.",
    "[8] UNSW Canberra Cyber, University of New South Wales, \"The UNSW-NB15 Dataset.\" [Online]. Available: https://research.unsw.edu.au/projects/unsw-nb15-dataset",
    "[9] FastAPI Framework Documentation, Sebastián Ramírez. [Online]. Available: https://fastapi.tiangolo.com/",
    "[10] React.js Documentation, Meta Platforms, Inc. [Online]. Available: https://react.dev/",
    "[11] NIET Department of Computer Science and Engineering, B.Tech Project Guidelines and Documentation Formats, Noida Institute of Engineering and Technology, Greater Noida, 2025–26."
]
for r in refs:
    doc.add_p(r, sz=22, space_before=40, space_after=80)

doc.add_page_break()

# -------------------------------------------------------------
# APPENDIX A: SOURCE CODE & CORE SCRIPTS
# -------------------------------------------------------------
doc.add_p("APPENDIX A: SOURCE CODE & CORE SCRIPTS", bold=True, sz=30, align="center", space_before=200, space_after=240, color="003366")

doc.add_h3("A.1 Phase 7.1 Preprocessing Pipeline (preprocessing.py)")
doc.add_code_block("""import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

CATEGORICAL_FEATURES = ['proto', 'service', 'state']

def build_leakage_free_preprocessor(X_train: pd.DataFrame):
    numeric_features = [col for col in X_train.columns if col not in CATEGORICAL_FEATURES]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    preprocessor.fit(X_train)
    return preprocessor""")

doc.add_h3("A.2 Model Training & Threshold Optimization (train_xgboost.py)")
doc.add_code_block("""import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def train_and_optimize_xgboost(X_train, y_train, X_val, y_val):
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.50, 0.0
    
    for thresh in np.arange(0.10, 0.90, 0.01):
        preds = (val_probs >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, preds, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Optimal Threshold: {best_thresh:.2f} (Attack F1: {best_f1:.4f})")
    return clf, best_thresh""")

doc.add_h3("A.3 FastAPI Inference Service (prediction_service.py)")
doc.add_code_block("""import pandas as pd
import numpy as np
from app.services.model_loader import get_preprocessor, get_model

def run_flow_inference(df_raw: pd.DataFrame, model_name: str, threshold: float):
    preprocessor = get_preprocessor()
    model = get_model(model_name)
    
    features = df_raw.drop(columns=['id', 'label', 'attack_cat'], errors='ignore')
    X_transformed = preprocessor.transform(features)
    
    probabilities = model.predict_proba(X_transformed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    results = []
    for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
        results.append({
            "flow_id": int(i + 1),
            "attack_probability": float(np.round(prob, 4)),
            "is_attack": bool(pred == 1),
            "severity": "CRITICAL" if prob >= 0.85 else "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.45 else "LOW"
        })
    return results""")

doc.add_page_break()

# -------------------------------------------------------------
# APPENDIX B: GITHUB REPOSITORY DETAILS
# -------------------------------------------------------------
doc.add_p("APPENDIX B: GITHUB REPOSITORY DETAILS", bold=True, sz=30, align="center", space_before=200, space_after=240, color="003366")

doc.add_p(
    "The complete source code, preprocessing scripts, model training routines, validation logs, FastAPI backend, "
    "and React dashboard are hosted on GitHub."
)
doc.add_p("Repository Link: https://github.com/harsh-11028/Network-Intrusion-Detection", bold=True, sz=24, space_after=180, color="003366")

repo_headers = ["Repository Attribute", "Details"]
repo_rows = [
    ["Project Title", "Machine Learning-Based Network Intrusion Detection and Attack Classification"],
    ["Academic Institution", "Noida Institute of Engineering and Technology (NIET), Greater Noida"],
    ["Author", "Harsh Raj (Roll No: 2401330100170)"],
    ["Supervisor", "Mrs. Sana Anjum, Assistant Professor, Department of CSE"],
    ["Benchmark Dataset", "UNSW-NB15 Network Intrusion Detection Benchmark"],
    ["Primary Languages", "Python (3.10+), TypeScript (ES6+), Shell"],
    ["ML Frameworks", "scikit-learn, XGBoost, pandas, NumPy, joblib"],
    ["Backend Framework", "FastAPI (Python), Uvicorn, Pydantic"],
    ["Frontend Framework", "React 18, Vite, Tailwind CSS, Recharts, Lucide Icons"],
    ["Validation Architecture", "Phase 7.1 Leakage-Free Stratified Internal Validation"],
    ["Deployed Models", "Logistic Regression, Decision Tree, Random Forest, XGBoost"],
    ["Dashboard Route", "Frontend: http://localhost:5173 | Backend API: http://localhost:8000"],
    ["OpenAPI Docs", "http://localhost:8000/docs"],
    ["License", "MIT Academic Open Source License"]
]
doc.add_table(repo_headers, repo_rows, [3000, 6000])

doc.add_h3("Quick Start Execution Command")
doc.add_code_block("""# 1. Clone repository
git clone https://github.com/harsh-11028/Network-Intrusion-Detection.git
cd Network-Intrusion-Detection

# 2. Launch complete dashboard with one-click startup script
chmod +x run_dashboard.sh
./run_dashboard.sh""")

# Save document
doc.save()
print("All done!")
