import { CheckCircle2, AlertTriangle, Info, ShieldCheck, Database, Layers, GitBranch, Cpu } from 'lucide-react'

const PHASES = [
  {
    phase: '1',
    title: 'Dataset Validation & Exploratory Data Analysis',
    desc: 'Audited cleaned UNSW-NB15 datasets (107,740 training flows, 55,945 testing flows). Checked column types, missingness, binary class balance (48.2% Normal / 51.8% Attack in train), and high cardinality protocols.'
  },
  {
    phase: '2',
    title: 'Reusable Preprocessing Pipeline',
    desc: 'Constructed an sklearn ColumnTransformer applying StandardScaler to 39 numerical attributes and OneHotEncoder(handle_unknown="ignore") to 3 categorical attributes (proto, service, state). Fitted exclusively on training partitions to avoid contamination.'
  },
  {
    phase: '3',
    title: 'Logistic Regression Baseline',
    desc: 'Trained linear L2-regularized logistic regression (max_iter=1000). Established standard baseline with test ROC-AUC of 0.9193, identifying key directional linear coefficients.'
  },
  {
    phase: '4',
    title: 'Decision Tree Classifier',
    desc: 'Constructed standalone CART tree classifier (max_depth=15). Evaluated discrete leaf outputs, observing high test accuracy (82.21%) but restricted continuous threshold flexibility.'
  },
  {
    phase: '5',
    title: 'Random Forest Ensemble',
    desc: 'Trained 300-estimator bagging ensemble (max_features="sqrt"). Achieved 97.51% test attack recall and 0.9655 ROC-AUC, providing strong resistance to overfitting.'
  },
  {
    phase: '6',
    title: 'XGBoost Gradient Boosting',
    desc: 'Optimized gradient boosted tree ensemble (n_estimators=300, max_depth=6, lr=0.1, subsample=0.8, colsample_bytree=0.8). Emerged as top performer with 0.9719 ROC-AUC and 0.9624 PR-AUC.'
  },
  {
    phase: '7',
    title: 'Comparative Analysis & Threshold Optimization',
    desc: 'Systematic study of decision threshold trade-offs across Recall, Precision, and False Positive Rate on internal validation data.'
  },
  {
    phase: '7.1',
    title: 'Leakage-Free Validation, Retraining & Audit',
    desc: 'Discovered and resolved preprocessor leakage in Phase 7. Performed stratified 80:20 raw split on training data (86,192 train / 21,548 validation), re-selected thresholds on leakage-free validation data, refit final preprocessor and models on full training data (107,740 rows), and executed a single final evaluation on the untouched official test set (55,945 rows).'
  },
  {
    phase: '8 & 8.1',
    title: 'Interactive Dashboard & SOC Console Refinement',
    desc: 'Engineered a modern offline network analysis interface using FastAPI, React, TypeScript, and Tailwind CSS. Implemented dual-state dashboard overview, traffic analyzer, model performance benchmarks, and feature interpretability consoles.'
  },
]

const STRATEGIES = [
  {
    name: 'Default (0.50)',
    desc: 'Standard unbiased 50% posterior probability cutoff. Serves as baseline reference across all 4 models.'
  },
  {
    name: 'Strategy A — Max Attack F1',
    desc: 'Optimizes harmonic mean of Attack Precision and Recall on validation data (LR: 0.13, DT: 0.50, RF: 0.50, XGB: 0.49).'
  },
  {
    name: 'Strategy B — Max Macro F1',
    desc: 'Balances performance evenly between Normal and Attack classes without class frequency bias (LR: 0.13, DT: 0.50, RF: 0.53, XGB: 0.49).'
  },
  {
    name: 'Strategy C — Security Priority (Recall ≥ 97%)',
    desc: 'Minimizes false negatives by selecting the lowest threshold achieving ≥97% attack detection rate. Feasible for LR (0.53), RF (0.42), XGB (0.42); infeasible for DT (max validation recall 91.36%).'
  },
  {
    name: 'Strategy D — Operational Target (Recall ≥ 95% & FPR ≤ 20%)',
    desc: 'Simultaneously enforces high attack catch rate with manageable analyst alarm burden. Feasible for LR (0.13), RF (0.50), XGB (0.49); infeasible for DT.'
  },
]

const DATASET_FACTS = [
  { label: 'Full Training Partition', value: '107,740 flows', sub: '51,890 Normal / 55,850 Attack' },
  { label: 'Official Test Partition', value: '55,945 flows', sub: '34,206 Normal / 21,739 Attack' },
  { label: 'Raw Feature Columns', value: '42 attributes', sub: '39 numerical, 3 categorical' },
  { label: 'Transformed Features', value: '194 features', sub: 'After One-Hot Encoding' },
  { label: 'Internal Train Subset', value: '86,192 flows', sub: '80% raw split (Phase 7.1)' },
  { label: 'Internal Validation Subset', value: '21,548 flows', sub: '20% raw split (Phase 7.1)' },
]

export default function MethodologyPage() {
  return (
    <div className="space-y-6 max-w-5xl">
      {/* Overview Banner */}
      <div className="card">
        <div className="flex items-center gap-2 mb-2">
          <ShieldCheck className="text-cyan-400" size={18} />
          <h2 className="text-sm font-semibold text-text-primary uppercase tracking-wide">
            Project Architecture & Methodological Foundations
          </h2>
        </div>
        <p className="text-xs text-text-secondary leading-relaxed">
          The <strong>Machine Learning-Based Network Intrusion Detection and Attack Classification</strong> project establishes an end-to-end research and deployment pipeline on the UNSW-NB15 benchmark.
          The primary objective is rigorous, leakage-free evaluation of statistical machine learning algorithms for binary network intrusion detection (0 = Normal, 1 = Attack), followed by validation-based classification threshold tuning.
        </p>
      </div>

      {/* Dataset Verification Grid */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">
            Verified Dataset Statistics (Source of Truth)
          </span>
          <span className="text-[11px] text-cyan-400 font-mono">UNSW-NB15 Cleaned</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {DATASET_FACTS.map(f => (
            <div key={f.label} className="metric-card bg-bg-card border-bg-border/80">
              <span className="metric-label text-[10px]">{f.label}</span>
              <span className="text-base font-bold font-mono text-cyan-400 mt-0.5">{f.value}</span>
              <span className="text-[10px] text-text-muted">{f.sub}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Methodological Deep Dive: Phase 7.1 Leakage-Free Correction */}
      <div className="card border-cyan-500/30 bg-bg-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Layers className="text-cyan-400" size={18} />
          <h3 className="text-xs font-semibold text-cyan-300 uppercase tracking-wider">
            Phase 7.1 Methodological Audit: Preprocessing Leakage Elimination
          </h3>
        </div>
        <p className="text-xs text-text-secondary leading-relaxed mb-3">
          During the Phase 7 audit, a subtle data leakage flaw was identified: the preprocessing pipeline had been fitted on the entire training set (107,740 rows) <em>prior</em> to partitioning into internal train and validation subsets.
          While the official test set (55,945 rows) remained strictly isolated, fitting StandardScaler and OneHotEncoder before internal validation allowed validation distribution statistics (means, standard deviations, categorical levels) to bleed into the internal training set.
        </p>

        <div className="p-3 rounded-lg bg-bg-secondary border border-bg-border space-y-2 text-xs">
          <div className="font-semibold text-text-primary flex items-center gap-1.5">
            <CheckCircle2 size={14} className="text-emerald-400" />
            <span>Strict Phase 7.1 Correction Protocol:</span>
          </div>
          <ol className="list-decimal list-inside space-y-1 text-text-muted text-[11px] pl-1 font-mono">
            <li><strong>Raw Stratified Split:</strong> Partitioned raw training data 80:20 into internal training (86,192 rows) and validation (21,548 rows).</li>
            <li><strong>Isolated Validation Fitting:</strong> Fitted a validation preprocessor exclusively on the 86,192 internal training rows (yielding 192 features; rare categories <code className="text-cyan-400">proto='rtp'</code> and <code className="text-cyan-400">state='URN'</code> were unobserved).</li>
            <li><strong>Leakage-Free Validation Predictions:</strong> Applied <code className="text-cyan-400">handle_unknown='ignore'</code> on validation data to re-evaluate decision thresholds under authentic zero-leakage conditions.</li>
            <li><strong>Final Production Retraining:</strong> Fitted the final production preprocessor on the complete official training set (107,740 rows $\to$ 194 features) and retrained final models with validated hyperparameters.</li>
            <li><strong>Single Official Evaluation:</strong> Final retrained models were evaluated strictly once on the untouched official test set (55,945 samples) across all selected thresholds.</li>
          </ol>
        </div>
      </div>

      {/* Complete Project Timeline */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <GitBranch className="text-cyan-400" size={17} />
          <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
            Sequential Engineering Roadmap (Phases 1 through 8.1)
          </h3>
        </div>
        <div className="space-y-4">
          {PHASES.map((p, idx) => (
            <div key={p.phase} className="flex gap-3.5">
              <div className="flex flex-col items-center">
                <div className="w-7 h-7 rounded-full bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 font-bold font-mono text-xs flex items-center justify-center flex-shrink-0">
                  {p.phase}
                </div>
                {idx < PHASES.length - 1 && <div className="w-px flex-1 bg-bg-border mt-1.5" />}
              </div>
              <div className="pb-3 text-xs">
                <div className="font-semibold text-text-primary flex items-center gap-2">
                  <span>{p.title}</span>
                  <CheckCircle2 size={13} className="text-emerald-400" />
                </div>
                <p className="text-text-secondary mt-1 leading-relaxed text-[11px]">{p.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Threshold Optimization Strategies */}
      <div className="card">
        <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider mb-3">
          Validation-Based Threshold Selection Criteria
        </h3>
        <div className="space-y-2.5">
          {STRATEGIES.map(s => (
            <div key={s.name} className="p-3 rounded-lg bg-bg-secondary/60 border border-bg-border text-xs">
              <span className="font-semibold text-cyan-300 block mb-0.5">{s.name}</span>
              <p className="text-text-secondary text-[11px] leading-relaxed">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Real-world Operational Disclaimers */}
      <div className="card border-amber-500/30 bg-amber-500/5 p-4">
        <div className="flex items-start gap-2.5">
          <AlertTriangle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1.5 text-xs text-text-secondary">
            <h4 className="font-semibold text-amber-300 uppercase tracking-wide">
              Engineering & Deployment Boundaries
            </h4>
            <ul className="space-y-1 list-disc list-inside text-text-muted text-[11px]">
              <li><strong>Offline Inference Guarantee:</strong> The dashboard application only performs batch inference against user-provided CSV records. It does not bind to network interfaces or sniff live network sockets.</li>
              <li><strong>Zero Runtime Refitting:</strong> The FastAPI backend strictly executes <code className="text-cyan-400">final_preprocessor.transform()</code>. The preprocessor and models are immutable runtime artifacts loaded from disk.</li>
              <li><strong>Advisory Classification:</strong> High probability scores indicate statistical anomalies resembling known UNSW-NB15 attack signatures. Human analysts must verify indicators of compromise (IoCs) before firewall policy enforcement.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
