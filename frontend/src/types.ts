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
  readonly submissionRows: number;
}

export type ExperimentStatus = 'seed' | 'accepted' | 'rejected';

export interface Experiment {
  readonly id: string;
  readonly parentId: string | null;
  readonly title: string;
  readonly hypothesis: string;
  readonly status: ExperimentStatus;
  readonly score: ScoreMetrics;
  readonly seconds: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly recoveryEvents: readonly string[];
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
