import type {
  Experiment,
  ManifestMetrics,
  ManifestNode,
  MetricPoint,
  RunEvent,
  RunManifest,
  RunSnapshot,
  ScoreMetrics,
} from './types';

const RUN_JSON_PATH = '/run.json';
const DEFAULT_BENCHMARK = 'KuaiRand-Pure';
const SEED_NODE_ID = '000-root';

export async function loadRunSnapshot(url = RUN_JSON_PATH): Promise<RunSnapshot> {
  const response = await fetch(url, { cache: 'no-store' });

  if (!response.ok) {
    throw new Error(
      `Run artifact unavailable (${String(response.status)}). Run the agent first.`,
    );
  }

  return buildSnapshot(await response.json());
}

export function buildSnapshot(payload: unknown): RunSnapshot {
  const manifest = readManifest(payload);
  const baseline = toScoreMetrics(manifest.baseline_validation);
  const best = toScoreMetrics(manifest.best_validation ?? manifest.baseline_validation);
  const experiments = manifest.nodes.flatMap((node) => {
    if (node.validation === null) {
      return [];
    }

    return [
      toExperiment({ ...node, validation: node.validation }, manifest.best_node_id),
    ];
  });

  return {
    detail: {
      runId: manifest.run_id ?? 'latest',
      benchmark: manifest.benchmark ?? DEFAULT_BENCHMARK,
      phase: 'completed',
      baseline,
      best,
      iterations: manifest.iterations,
      wallClockSeconds: manifest.wall_clock_seconds,
      inputTokens: manifest.total_input_tokens,
      outputTokens: manifest.total_output_tokens,
      manualInterventions: manifest.manual_interventions,
      hiddenTestAccess: manifest.hidden_test_access,
      submissionPath: manifest.final_submission,
    },
    experiments,
    metrics: toMetricPoints(baseline, experiments),
    events: toEvents(manifest, experiments),
  };
}

function readManifest(payload: unknown): RunManifest {
  if (!isRecord(payload)) {
    throw new Error('Run artifact must contain a JSON object.');
  }

  if (!isRecord(payload['baseline_validation']) || !Array.isArray(payload['nodes'])) {
    throw new Error('Run artifact is missing baseline or node data.');
  }

  return payload as unknown as RunManifest;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function toScoreMetrics(metrics: ManifestMetrics): ScoreMetrics {
  return {
    gauc: metrics.gauc,
    ndcgAt5: metrics.ndcg_at_5,
    primary: metrics.primary,
  };
}

function toExperiment(
  node: ManifestNode & { readonly validation: ManifestMetrics },
  bestNodeId: string,
): Experiment {
  return {
    id: node.node_id,
    parentId: node.parent_id,
    title:
      node.title || (node.node_id === SEED_NODE_ID ? 'Initial candidate' : node.node_id),
    hypothesis:
      node.hypothesis ||
      (node.node_id === SEED_NODE_ID
        ? 'Initial candidate copied from the audited history ranker.'
        : 'No hypothesis was recorded.'),
    status:
      node.node_id === bestNodeId
        ? 'accepted'
        : node.node_id === SEED_NODE_ID
          ? 'seed'
          : 'rejected',
    score: toScoreMetrics(node.validation),
  };
}

function toMetricPoints(
  baseline: ScoreMetrics,
  experiments: readonly Experiment[],
): MetricPoint[] {
  return [
    {
      iteration: 0,
      label: 'Official baseline',
      primary: baseline.primary,
      gauc: baseline.gauc,
      ndcgAt5: baseline.ndcgAt5,
    },
    ...experiments.map((experiment, index) => ({
      iteration: index + 1,
      label: experiment.id,
      primary: experiment.score.primary,
      gauc: experiment.score.gauc,
      ndcgAt5: experiment.score.ndcgAt5,
    })),
  ];
}

function toEvents(manifest: RunManifest, experiments: readonly Experiment[]): RunEvent[] {
  const events: RunEvent[] = [
    {
      id: 'run-started',
      experimentId: null,
      step: 1,
      label: 'Run started',
      message: `${manifest.benchmark ?? DEFAULT_BENCHMARK} autonomous search opened.`,
      tone: 'cyan',
    },
    {
      id: 'baseline-verified',
      experimentId: null,
      step: 2,
      label: 'Baseline verified',
      message: `Official validation reference recorded at primary ${manifest.baseline_validation.primary.toFixed(4)}.`,
      tone: 'cyan',
    },
  ];

  for (const [index, experiment] of experiments.entries()) {
    const accepted = experiment.status === 'accepted';
    events.push({
      id: `${experiment.id}-measured`,
      experimentId: experiment.id,
      step: index + 3,
      label: accepted ? 'New best accepted' : 'Candidate measured',
      message: `${experiment.title} reached primary ${experiment.score.primary.toFixed(4)}.`,
      tone: accepted ? 'green' : experiment.status === 'rejected' ? 'red' : 'cyan',
    });
  }

  events.push({
    id: 'run-finished',
    experimentId: null,
    step: events.length + 1,
    label: manifest.stopped_reason === 'converged' ? 'Run converged' : 'Run finished',
    message: `Search stopped: ${manifest.stopped_reason}.`,
    tone: 'green',
  });
  return events;
}
