import type { ExperimentStatus } from './types';

const SCORE_DIGITS = 4;
const PERCENT_MULTIPLIER = 100;
const SECONDS_PER_MINUTE = 60;
const SECONDS_PER_HOUR = 3_600;

export function formatScore(value: number): string {
  return value.toFixed(SCORE_DIGITS);
}

export function formatPercent(value: number): string {
  return `${(value * PERCENT_MULTIPLIER).toFixed(0)}%`;
}

export function formatDuration(seconds: number): string {
  if (seconds >= SECONDS_PER_HOUR) {
    return `${(seconds / SECONDS_PER_HOUR).toFixed(1)}h`;
  }

  return `${String(Math.round(seconds / SECONDS_PER_MINUTE))}m`;
}

const EXPERIMENT_STATUS_LABELS: Readonly<Record<ExperimentStatus, string>> = {
  seed: 'Seed',
  accepted: 'New best',
  rejected: 'Rejected',
};

export function formatExperimentStatus(status: ExperimentStatus): string {
  return EXPERIMENT_STATUS_LABELS[status];
}
