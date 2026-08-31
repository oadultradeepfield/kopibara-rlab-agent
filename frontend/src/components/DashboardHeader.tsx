import type { ReactElement } from 'react';

import type { RunDetail } from '../types';

interface DashboardHeaderProps {
  readonly detail: RunDetail;
}

export function DashboardHeader({ detail }: DashboardHeaderProps): ReactElement {
  return (
    <header className="border-b border-slate-700 bg-[#111827] px-3 py-2 md:px-4">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 items-center gap-2">
          <div className="grid size-8 shrink-0 place-items-center rounded border border-slate-600 bg-slate-800 text-xs font-semibold text-slate-200">
            KR
          </div>
          <div className="min-w-0">
            <p className="text-xs text-slate-400">Recommender system research</p>
            <h1 className="truncate text-base font-semibold text-slate-100">
              Kopibara Research Dashboard
            </h1>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="status-pill phase-frozen">
            <span className="status-dot" aria-hidden="true" />
            Finished
          </span>
          <span className="status-pill border-white/10 text-slate-400">
            {detail.benchmark} · saved snapshot
          </span>
          <span className="font-mono text-xs text-slate-500">Run {detail.runId}</span>
        </div>
      </div>
    </header>
  );
}
