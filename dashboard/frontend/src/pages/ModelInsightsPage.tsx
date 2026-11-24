import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer
} from 'recharts'
import { getFeatureImportances, FeatureImportance } from '../services/api'
import { Info, HelpCircle, ShieldAlert, Award } from 'lucide-react'

const MODEL_COLORS: Record<string, string> = {
  logistic_regression: '#6366f1',
  decision_tree: '#f59e0b',
  random_forest: '#10b981',
  xgboost: '#06b6d4',
}

const MODEL_NAMES: Record<string, string> = {
  logistic_regression: 'Logistic Regression',
  decision_tree: 'Decision Tree',
  random_forest: 'Random Forest',
  xgboost: 'XGBoost',
}

const MODEL_EXPLANATIONS: Record<string, { title: string; desc: string; math: string }> = {
  xgboost: {
    title: 'Gain-Based Feature Importance',
    desc: 'Measures the relative improvement in loss (Gain) brought by each feature when creating split branches across gradient-boosted trees.',
    math: 'Higher Gain indicates the feature frequently provides the sharpest discriminative split in reducing log-loss.'
  },
  random_forest: {
    title: 'Mean Decrease in Impurity (MDI)',
    desc: 'Calculated as the total decrease in Gini impurity accumulated across all 300 decision trees in the forest.',
    math: 'Reflects how consistently the feature partitions normal flows from attacks across random sub-samples.'
  },
  decision_tree: {
    title: 'Total Gini Impurity Decrease',
    desc: 'Total Gini split improvement across all internal nodes in the single CART decision tree (max depth 15).',
    math: 'Identifies the primary top-level root decision boundaries used by the standalone tree.'
  },
  logistic_regression: {
    title: 'Linear Model Log-Odds Coefficients',
    desc: 'Directional weights assigned to standardized features in the logistic link function.',
    math: 'Positive weights (+w) increase the predicted attack log-odds; negative weights (-w) favor normal traffic.'
  },
}

// Domain definitions for notable UNSW-NB15 features
const FEATURE_GLOSSARY: Record<string, string> = {
  sttl: 'Source-to-destination Time To Live (TTL). Strong indicator of OS origin and routing distance.',
  ct_state_ttl: 'Count of connections possessing specific state and TTL combinations. Distinguishes automated scanning.',
  swin: 'Source TCP window advertisement value. Anomalous window sizes often characterize SYN floods or port probes.',
  dwin: 'Destination TCP window advertisement value.',
  dttl: 'Destination-to-source Time To Live (TTL).',
  sbytes: 'Volume of transaction bytes transferred from source to destination.',
  dbytes: 'Volume of transaction bytes transferred from destination to source.',
  sload: 'Source bits per second transmission rate during the flow.',
  dload: 'Destination bits per second transmission rate during the flow.',
  smean: 'Mean packet size transmitted by the source.',
  dmean: 'Mean packet size transmitted by the destination.',
  rate: 'Total packets per second across the bidirectional connection.',
  dur: 'Total flow duration in seconds.',
  tcprtt: 'TCP Round Trip Time (sum of synack and ackdat).',
  synack: 'TCP handshake duration between SYN and SYN-ACK.',
  ackdat: 'TCP handshake duration between SYN-ACK and ACK.',
  ct_srv_src: 'Count of concurrent connections to the same service from the source IP in 100-record window.',
  ct_dst_ltm: 'Count of connections to the destination IP in the last 100 records.',
  ct_srv_dst: 'Count of connections to the same service arriving at the destination IP.',
}

export default function ModelInsightsPage() {
  const [importances, setImportances] = useState<Record<string, FeatureImportance[]>>({})
  const [selectedModel, setSelectedModel] = useState('xgboost')
  const [topN, setTopN] = useState(15)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getFeatureImportances()
      .then(r => setImportances(r.data.feature_importances))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-text-muted">Loading feature importance profiles…</p>
        </div>
      </div>
    )
  }

  const rawData: FeatureImportance[] = importances[selectedModel] ?? []
  const isLR = selectedModel === 'logistic_regression'

  const chartData = rawData.slice(0, topN).map(d => ({
    feature: d.feature,
    importance: isLR ? Math.abs(d.importance) : d.importance,
    rawImportance: d.importance,
    description: FEATURE_GLOSSARY[d.feature] || (d.feature.startsWith('proto_') ? `Protocol one-hot indicator: ${d.feature.replace('proto_', '')}` : (d.feature.startsWith('state_') ? `TCP state one-hot: ${d.feature.replace('state_', '')}` : 'Transformed network traffic attribute'))
  })).sort((a, b) => b.importance - a.importance)

  const maxVal = Math.max(...chartData.map(d => d.importance), 0.0001)
  const explanation = MODEL_EXPLANATIONS[selectedModel]

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header and Model Switcher */}
      <div className="card">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-text-primary uppercase tracking-wide">
              Model Interpretability & Feature Significance
            </h2>
            <p className="text-xs text-text-muted mt-0.5">
              Inspecting the internal decision signals learned by Phase 7.1 models across 194 features.
            </p>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {Object.keys(MODEL_NAMES).map(mk => {
              const active = selectedModel === mk
              const color = MODEL_COLORS[mk] || '#94a3b8'
              return (
                <button
                  key={mk}
                  onClick={() => setSelectedModel(mk)}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all flex items-center gap-1.5 ${
                    active
                      ? 'border shadow-sm'
                      : 'border border-bg-border bg-bg-card text-text-secondary hover:text-text-primary hover:border-cyan-500/30'
                  }`}
                  style={active ? {
                    backgroundColor: `${color}15`,
                    borderColor: `${color}60`,
                    color: color
                  } : {}}
                >
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                  {MODEL_NAMES[mk]}
                </button>
              )
            })}
          </div>
        </div>

        {/* Dynamic Metric Explanation Banner */}
        <div className="mt-4 p-3 rounded-lg bg-bg-secondary/60 border border-bg-border/80 flex items-start gap-3">
          <Info size={16} className="text-cyan-400 flex-shrink-0 mt-0.5" />
          <div className="text-xs space-y-1">
            <div className="font-semibold text-text-primary flex items-center gap-2">
              <span>{explanation.title}</span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-bg-card border border-bg-border text-cyan-400">
                {selectedModel.toUpperCase()}
              </span>
            </div>
            <p className="text-text-secondary leading-relaxed">{explanation.desc}</p>
            <p className="text-[11px] text-text-muted font-mono">{explanation.math}</p>
          </div>
        </div>
      </div>

      {/* Feature Importance Chart */}
      {chartData.length > 0 ? (
        <div className="card">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <div>
              <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
                Top {topN} Predictive Features — {MODEL_NAMES[selectedModel]}
              </h3>
              <p className="text-[11px] text-text-muted">
                {isLR ? 'Ranked by absolute coefficient weight |w|' : 'Ranked in descending order of feature importance score'}
              </p>
            </div>
            <div className="flex items-center gap-1.5 bg-bg-secondary p-1 rounded-lg border border-bg-border text-xs">
              <span className="text-text-muted px-2">Display:</span>
              {[10, 15, 20, 30].map(n => (
                <button
                  key={n}
                  onClick={() => setTopN(n)}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    topN === n
                      ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                      : 'text-text-secondary hover:text-text-primary'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          <ResponsiveContainer width="100%" height={Math.max(340, topN * 26)}>
            <BarChart
              data={chartData}
              layout="vertical"
              barSize={14}
              margin={{ left: 130, right: 20, top: 5, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#64748b', fontSize: 10 }}
                tickFormatter={v => v.toFixed(3)}
              />
              <YAxis
                dataKey="feature"
                type="category"
                width={125}
                tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'monospace' }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #1e2d3d', borderRadius: 8 }}
                itemStyle={{ color: '#e2e8f0', fontSize: 11 }}
                formatter={(v: number, _: string, props: any) => [
                  isLR
                    ? `${(props?.payload?.rawImportance ?? v).toFixed(6)} (signed weight)`
                    : v.toFixed(6),
                  isLR ? 'Coefficient' : 'Importance'
                ]}
                labelFormatter={(label: string) => {
                  const desc = FEATURE_GLOSSARY[label]
                  return desc ? `${label}: ${desc}` : label
                }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {chartData.map((entry, idx) => {
                  let fillColor = MODEL_COLORS[selectedModel] || '#06b6d4'
                  if (isLR) {
                    fillColor = entry.rawImportance > 0 ? '#ef4444' : '#10b981'
                  }
                  return <Cell key={idx} fill={fillColor} />
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {isLR && (
            <div className="flex items-center gap-4 mt-3 pt-3 border-t border-bg-border/60 text-xs">
              <span className="flex items-center gap-1.5 text-red-400 font-mono">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500" /> Positive Weight (+w) \u2192 Drives Attack Decision
              </span>
              <span className="flex items-center gap-1.5 text-emerald-400 font-mono">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Negative Weight (-w) \u2192 Drives Normal Decision
              </span>
            </div>
          )}
        </div>
      ) : (
        <div className="card text-text-muted text-xs p-6 text-center">
          No feature importance records found for {MODEL_NAMES[selectedModel]}.
        </div>
      )}

      {/* Interactive Feature Table with Cyber Domain Explanations */}
      {chartData.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-semibold text-text-primary uppercase tracking-wider">
              Network Feature Attribution Dictionary (Top {topN})
            </h3>
            <span className="text-[11px] text-text-muted font-mono">UNSW-NB15 Flow Features</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-bg-border text-text-muted font-medium uppercase">
                  <th className="py-2 px-3">Rank</th>
                  <th className="py-2 px-3">Feature Name</th>
                  <th className="py-2 px-3">Importance Value</th>
                  <th className="py-2 px-3">Relative Scale</th>
                  <th className="py-2 px-3">Domain Role in Traffic Analysis</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-bg-border/40">
                {chartData.map((row, i) => (
                  <tr key={row.feature} className="hover:bg-bg-secondary/40 transition-colors">
                    <td className="py-2.5 px-3 text-text-muted font-mono">#{i + 1}</td>
                    <td className="py-2.5 px-3 font-mono font-semibold" style={{ color: MODEL_COLORS[selectedModel] }}>
                      {row.feature}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-text-primary">
                      {isLR ? row.rawImportance.toFixed(6) : row.importance.toFixed(6)}
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full bg-bg-border overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${(row.importance / maxVal) * 100}%`,
                              backgroundColor: MODEL_COLORS[selectedModel]
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-text-muted font-mono">
                          {((row.importance / maxVal) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-text-secondary text-[11px] leading-relaxed">
                      {row.description}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Methodological Caveats & Boundaries */}
      <div className="card border-bg-border bg-bg-secondary/40 p-4">
        <div className="flex items-start gap-2.5">
          <ShieldAlert size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1.5 text-xs text-text-secondary">
            <h4 className="font-semibold text-text-primary uppercase tracking-wide">
              Interpretability Guidance & Research Integrity
            </h4>
            <ul className="space-y-1 list-disc list-inside text-text-muted">
              <li><strong>Association vs. Causation:</strong> High feature significance reflects statistical correlation within UNSW-NB15 training distribution; it does not constitute deterministic proof of malicious intrusion.</li>
              <li><strong>TTL Influence:</strong> Features like <code className="text-cyan-400">sttl</code> and <code className="text-cyan-400">ct_state_ttl</code> represent strong synthetic distribution signals in UNSW-NB15. Phase 7.1 ablation verified that removing them still preserves over 94% validation F1 score across tree models.</li>
              <li><strong>Model Specificity:</strong> Logistic Regression weights, Random Forest MDI, and XGBoost Gain represent fundamentally different mathematical properties and should not be compared as equivalent numerical quantities.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
