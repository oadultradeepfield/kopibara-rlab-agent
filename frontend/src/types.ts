export type RunPhase = 'completed';

export interface ScoreMetrics {
  readonly gauc: number;
  readonly ndcgAt5: number;
  readonly primary: number;
}

export interface RunDetail {
  readonly runId: string;
  readonly benchmark: string;
  readonly phase: RunPhase;
  readonly baseline: ScoreMetrics;
  readonly best: ScoreMetrics;
  readonly iterations: number;
  readonly wallClockSeconds: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly manualInterventions: number;
  readonly hiddenTestAccess: boolean;
  readonly submissionPath: string | null;
}

export type ExperimentStatus = 'seed' | 'accepted' | 'rejected';

export interface Experiment {
  readonly id: string;
  readonly parentId: string | null;
  readonly title: string;
  readonly hypothesis: string;
  readonly status: ExperimentStatus;
  readonly score: ScoreMetrics;
}

export interface MetricPoint {
  readonly iteration: number;
  readonly label: string;
  readonly primary: number;
  readonly gauc: number;
  readonly ndcgAt5: number;
}

export type EventTone = 'cyan' | 'green' | 'amber' | 'red';

export interface RunEvent {
  readonly id: string;
  readonly experimentId: string | null;
  readonly step: number;
  readonly label: string;
  readonly message: string;
  readonly tone: EventTone;
}

export interface RunSnapshot {
  readonly detail: RunDetail;
  readonly experiments: readonly Experiment[];
  readonly metrics: readonly MetricPoint[];
  readonly events: readonly RunEvent[];
}

export interface ManifestMetrics {
  readonly gauc: number;
  readonly ndcg_at_5: number;
  readonly primary: number;
}

export interface ManifestNode {
  readonly node_id: string;
  readonly parent_id: string | null;
  readonly status: string;
  readonly title: string;
  readonly hypothesis: string;
  readonly validation: ManifestMetrics | null;
}

export interface RunManifest {
  readonly status: string;
  readonly stopped_reason: string;
  readonly run_id?: string;
  readonly benchmark?: string;
  readonly baseline_validation: ManifestMetrics;
  readonly best_validation: ManifestMetrics | null;
  readonly best_node_id: string;
  readonly iterations: number;
  readonly wall_clock_seconds: number;
  readonly total_input_tokens: number;
  readonly total_output_tokens: number;
  readonly manual_interventions: number;
  readonly hidden_test_access: boolean;
  readonly nodes: readonly ManifestNode[];
  readonly final_submission: string | null;
}
