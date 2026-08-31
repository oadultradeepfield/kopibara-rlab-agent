import type { ReactElement } from 'react';

import { formatExperimentStatus, formatScore } from '../format';
import type { Experiment } from '../types';

interface ExperimentTableProps {
  readonly experiments: readonly Experiment[];
  readonly selectedExperimentId: string | null;
  readonly onSelectExperiment: (experimentId: string) => void;
}

export function ExperimentTable({
  experiments,
  selectedExperimentId,
  onSelectExperiment,
}: ExperimentTableProps): ReactElement {
  return (
    <section className="panel overflow-hidden" aria-labelledby="experiments-title">
      <div className="border-b border-white/7 p-3">
        <p className="section-kicker">All experiments</p>
        <h2 id="experiments-title" className="section-title">
          Experiment results
        </h2>
      </div>
      <div className="max-h-[400px] overflow-auto">
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead className="sticky top-0 z-10 bg-slate-800 text-xs text-slate-400">
            <tr>
              <th className="px-3 py-2 font-medium">Experiment</th>
              <th className="px-2 py-2 font-medium">Status</th>
              <th className="px-2 py-2 text-right font-medium">Primary</th>
              <th className="px-2 py-2 font-medium">Hypothesis</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/6">
            {experiments.map((experiment) => (
              <tr
                key={experiment.id}
                className={
                  selectedExperimentId === experiment.id
                    ? 'bg-blue-400/5'
                    : 'hover:bg-white/[0.025]'
                }
              >
                <td className="px-3 py-2 font-mono text-xs text-slate-200">
                  <button
                    type="button"
                    className="text-left hover:text-blue-300 hover:underline"
                    aria-pressed={selectedExperimentId === experiment.id}
                    onClick={() => {
                      onSelectExperiment(experiment.id);
                    }}
                  >
                    {experiment.id}
                  </button>
                  <p className="mt-1 max-w-56 font-sans text-xs text-slate-400">
                    {experiment.title}
                  </p>
                </td>
                <td className="px-2 py-2">
                  <span className={`experiment-status status-${experiment.status}`}>
                    {formatExperimentStatus(experiment.status)}
                  </span>
                </td>
                <td className="px-2 py-2 text-right font-mono text-xs text-slate-200">
                  {formatScore(experiment.score.primary)}
                </td>
                <td className="max-w-[34rem] px-2 py-2 text-xs leading-5 text-slate-400">
                  {experiment.hypothesis}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
