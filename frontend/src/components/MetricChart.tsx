import type { ReactElement } from 'react';
import Plot from 'react-plotly.js';

import type { MetricPoint } from '../types';

interface MetricChartProps {
  readonly points: readonly MetricPoint[];
  readonly baseline: number;
}

const CHART_CONFIG = {
  displayModeBar: false,
  responsive: true,
} as const;

export function MetricChart({ points, baseline }: MetricChartProps): ReactElement {
  const iterations = points.map((point) => point.iteration);

  if (points.length === 0) {
    return <p className="empty-panel">No scores have been recorded yet.</p>;
  }

  return (
    <section className="panel min-h-[290px] p-3" aria-labelledby="metrics-title">
      <p className="section-kicker">Performance</p>
      <h2 id="metrics-title" className="section-title">
        Scores by experiment
      </h2>
      <Plot
        className="mt-1 h-[230px] w-full"
        useResizeHandler
        config={CHART_CONFIG}
        data={[
          {
            x: iterations,
            y: points.map((point) => point.primary),
            name: 'Primary score',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#0284c7', width: 3 },
            marker: { size: 7 },
          },
          {
            x: iterations,
            y: points.map((point) => point.gauc),
            name: 'GAUC',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#7c3aed', width: 2, dash: 'dot' },
          },
          {
            x: iterations,
            y: points.map((point) => point.ndcgAt5),
            name: 'nDCG@5',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#d97706', width: 2 },
          },
          {
            x: iterations,
            y: iterations.map(() => baseline),
            name: 'Baseline',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#64748b', width: 1, dash: 'dash' },
          },
        ]}
        layout={{
          autosize: true,
          margin: { l: 48, r: 10, t: 14, b: 34 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#475569', family: 'IBM Plex Mono, monospace', size: 11 },
          hovermode: 'x unified',
          legend: { orientation: 'h', y: 1.15, x: 0 },
          xaxis: {
            title: { text: 'Experiment' },
            dtick: 1,
            gridcolor: 'rgba(148,163,184,0.28)',
            zeroline: false,
          },
          yaxis: {
            title: { text: 'Score' },
            tickformat: '.4f',
            gridcolor: 'rgba(148,163,184,0.28)',
            zeroline: false,
          },
        }}
      />
    </section>
  );
}
