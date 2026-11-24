/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#0a0f1a',
          secondary: '#0f1623',
          card: '#111827',
          border: '#1e2d3d',
        },
        accent: {
          cyan: '#06b6d4',
          green: '#10b981',
          red: '#ef4444',
          amber: '#f59e0b',
          purple: '#6366f1',
          blue: '#3b82f6',
        },
        text: {
          primary: '#e2e8f0',
          secondary: '#94a3b8',
          muted: '#64748b',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [],
}
