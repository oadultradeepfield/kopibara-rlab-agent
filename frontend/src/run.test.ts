import { describe, expect, it } from 'vitest';

import { buildSnapshot, datasetPath } from './run';
import type { RunManifest } from './types';

const MANIFEST: RunManifest = {
  status: 'completed',
  stopped_reason: 'converged',
  run_id: '20260831T092157Z',
  benchmark: 'KuaiRand-Pure',
  baseline_validation: { gauc: 0.6674, ndcg_at_5: 0.5357, primary: 0.6016 },
  best_validation: { gauc: 0.7, ndcg_at_5: 0.55, primary: 0.625 },
  best_node_id: '001-child',
  iterations: 1,
  wall_clock_seconds: 10,
  total_input_tokens: 20,
  total_output_tokens: 5,
  manual_interventions: 0,
  hidden_test_access: false,
  nodes: [
    {
      node_id: '000-root',
      parent_id: null,
      status: 'kept',
      title: '',
      hypothesis: '',
      validation: { gauc: 0.69, ndcg_at_5: 0.54, primary: 0.615 },
    },
    {
      node_id: '001-child',
      parent_id: '000-root',
      status: 'evaluated',
      title: 'Top-five objective',
      hypothesis: 'Focus on the scored list head.',
      validation: { gauc: 0.7, ndcg_at_5: 0.55, primary: 0.625 },
    },
  ],
  final_submission: 'runs/example/submission.csv',
};

describe('run artifact adapter', () => {
  it('maps each supported dataset to its saved artifact', () => {
    expect(datasetPath('KuaiRand-Pure')).toBe('/benchmarks/pure.json');
    expect(datasetPath('KuaiRand-1K')).toBe('/benchmarks/1k.json');
  });

  it('maps the saved manifest into the dashboard view', () => {
    const snapshot = buildSnapshot(MANIFEST);

    expect(snapshot.detail.runId).toBe('20260831T092157Z');
    expect(snapshot.detail.best.primary).toBe(0.625);
    expect(snapshot.experiments[1]?.status).toBe('accepted');
    expect(snapshot.metrics).toHaveLength(3);
  });

  it('rejects malformed artifacts at the input boundary', () => {
    expect(() => buildSnapshot({})).toThrow('missing baseline or node data');
  });
});
