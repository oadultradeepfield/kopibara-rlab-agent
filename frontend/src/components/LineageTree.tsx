import type { ReactElement } from 'react';

import { formatExperimentStatus, formatScore } from '../format';
import type { Experiment } from '../types';

const BASELINE_ID = 'baseline';

interface LineageTreeProps {
  readonly experiments: readonly Experiment[];
  readonly baseline: number;
  readonly selectedExperimentId: string | null;
  readonly onSelectExperiment: (experimentId: string | null) => void;
}

interface LineageNode {
  readonly experiment: Experiment | null;
  readonly children: readonly LineageNode[];
}

interface MutableLineageNode {
  readonly experiment: Experiment | null;
  readonly children: MutableLineageNode[];
}

export function LineageTree({
  experiments,
  baseline,
  selectedExperimentId,
  onSelectExperiment,
}: LineageTreeProps): ReactElement {
  return (
    <section className="panel overflow-hidden" aria-labelledby="lineage-title">
      <div className="flex items-end justify-between border-b border-white/7 p-3">
        <div>
          <p className="section-kicker">Experiment decisions</p>
          <h2 id="lineage-title" className="section-title">
            How experiments branched
          </h2>
        </div>
        <span className="font-mono text-xs text-slate-400">
          {experiments.length} experiments
        </span>
      </div>
      <div className="lineage-scroll" tabIndex={0} aria-label="Experiment decision tree">
        <ul className="lineage-root">
          <LineageBranch
            node={buildLineageTree(experiments)}
            baseline={baseline}
            selectedExperimentId={selectedExperimentId}
            onSelectExperiment={onSelectExperiment}
          />
        </ul>
      </div>
    </section>
  );
}

function buildLineageTree(experiments: readonly Experiment[]): LineageNode {
  const root: MutableLineageNode = { experiment: null, children: [] };
  const nodes = new Map(
    experiments.map((experiment) => [
      experiment.id,
      { experiment, children: [] } satisfies MutableLineageNode,
    ]),
  );

  for (const experiment of experiments) {
    const node = nodes.get(experiment.id);
    const parent = experiment.parentId === null ? null : nodes.get(experiment.parentId);

    if (node !== undefined) {
      (parent ?? root).children.push(node);
    }
  }

  return root;
}

interface LineageBranchProps {
  readonly node: LineageNode;
  readonly baseline: number;
  readonly selectedExperimentId: string | null;
  readonly onSelectExperiment: (experimentId: string | null) => void;
}

function LineageBranch({
  node,
  baseline,
  selectedExperimentId,
  onSelectExperiment,
}: LineageBranchProps): ReactElement {
  const experiment = node.experiment;
  const experimentId = experiment?.id ?? null;
  const isSelected = selectedExperimentId === experimentId;

  return (
    <li className="lineage-branch">
      <button
        type="button"
        className={`lineage-node ${getLineageTone(experiment)} ${isSelected ? 'lineage-selected' : ''}`}
        aria-pressed={isSelected}
        aria-label={getNodeLabel(experiment, baseline)}
        onClick={() => {
          onSelectExperiment(experimentId);
        }}
      >
        <span className="truncate font-mono text-xs font-semibold text-slate-200">
          {experimentId ?? 'Baseline'}
        </span>
        <span className="mt-1 font-mono text-sm text-slate-100">
          {formatScore(experiment?.score.primary ?? baseline)}
        </span>
        <span className="mt-1 text-xs text-slate-400">
          {experiment === null
            ? 'Starting point'
            : formatExperimentStatus(experiment.status)}
        </span>
      </button>

      {node.children.length > 0 && (
        <ul className="lineage-children">
          {node.children.map((child) => (
            <LineageBranch
              key={child.experiment?.id ?? BASELINE_ID}
              node={child}
              baseline={baseline}
              selectedExperimentId={selectedExperimentId}
              onSelectExperiment={onSelectExperiment}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function getLineageTone(experiment: Experiment | null): string {
  if (experiment === null) return 'lineage-baseline';
  if (experiment.status === 'accepted') return 'lineage-champion';
  if (experiment.status === 'rejected') return 'lineage-rejected';
  return 'lineage-completed';
}

function getNodeLabel(experiment: Experiment | null, baseline: number): string {
  if (experiment === null) return `Baseline, score ${formatScore(baseline)}`;
  return `${experiment.id}, ${formatExperimentStatus(experiment.status)}, score ${formatScore(experiment.score.primary)}`;
}
