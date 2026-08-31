import type { ReactElement } from 'react';

import { formatDuration, formatScore } from '../format';
import type { RunDetail } from '../types';

interface OverviewProps {
  readonly detail: RunDetail;
}

interface EvidenceCardProps {
  readonly label: string;
  readonly value: string;
  readonly detail: string;
}

export function Overview({ detail }: OverviewProps): ReactElement {
  const delta = detail.best.primary - detail.baseline.primary;
  const totalTokens = detail.inputTokens + detail.outputTokens;

  return (
    <section aria-labelledby="overview-title">
      <div className="mb-2 flex items-end justify-between gap-3">
        <div>
          <p className="section-kicker">Completed run</p>
          <h2 id="overview-title" className="section-title">
            Results so far
          </h2>
        </div>
        <p className="font-mono text-xs text-slate-400">
          {detail.iterations} autonomous iterations
        </p>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <EvidenceCard
          label="Official baseline"
          value={formatScore(detail.baseline.primary)}
          detail={`GAUC ${formatScore(detail.baseline.gauc)} · nDCG@5 ${formatScore(detail.baseline.ndcgAt5)}`}
        />
        <EvidenceCard
          label="Best validation"
          value={formatScore(detail.best.primary)}
          detail={`GAUC ${formatScore(detail.best.gauc)} · nDCG@5 ${formatScore(detail.best.ndcgAt5)}`}
        />
        <EvidenceCard
          label="Delta vs baseline"
          value={`${delta >= 0 ? '+' : ''}${formatScore(delta)}`}
          detail="Primary score improvement"
        />
        <EvidenceCard
          label="Manual interventions"
          value={String(detail.manualInterventions)}
          detail="Human input during the run"
        />
      </div>

      <div className="panel mt-2 p-3">
        <p className="section-kicker">Run evidence</p>
        <p className="mt-1 max-w-5xl text-sm leading-5 text-slate-200">
          The agent searched history-aware LambdaRank variants and retained the checkpoint
          that optimized the evaluated top-five ranking region.
        </p>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 border-t border-white/10 pt-2 font-mono text-xs text-slate-400">
          <span>Benchmark: {detail.benchmark}</span>
          <span>Runtime: {formatDuration(detail.wallClockSeconds)}</span>
          <span>LLM tokens: {totalTokens.toLocaleString()}</span>
          <span>Hidden test access: {detail.hiddenTestAccess ? 'Yes' : 'No'}</span>
          <span>Submission rows checked: {detail.submissionRows.toLocaleString()}</span>
        </div>
      </div>
    </section>
  );
}

function EvidenceCard({ label, value, detail }: EvidenceCardProps): ReactElement {
  return (
    <article className="evidence-card">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-1 font-mono text-xl font-semibold tracking-tight text-slate-100">
        {value}
      </p>
      <p className="text-xs text-slate-400">{detail}</p>
    </article>
  );
}
