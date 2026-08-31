import type { ReactElement } from 'react';

import type { RunDetail } from '../types';
import { DATASET_OPTIONS, type DatasetName } from '../run';

interface DashboardHeaderProps {
  readonly detail: RunDetail;
  readonly dataset: DatasetName;
  readonly onDatasetChange: (dataset: DatasetName) => void;
}

export function DashboardHeader({
  detail,
  dataset,
  onDatasetChange,
}: DashboardHeaderProps): ReactElement {
  return (
    <header className="border-b border-slate-200 bg-white px-3 py-2 md:px-4">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 items-center gap-2">
          <div className="grid size-8 shrink-0 place-items-center rounded border border-slate-300 bg-slate-50 text-xs font-semibold text-slate-700">
            KR
          </div>
          <div className="min-w-0">
            <p className="text-xs text-slate-500">Recommender system research</p>
            <h1 className="truncate text-base font-semibold text-slate-900">
              Kopibara RLab Agent Dashboard
            </h1>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-slate-500">
            Dataset
            <select
              aria-label="Dataset"
              className="rounded border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700"
              value={dataset}
              onChange={(event) => {
                onDatasetChange(event.target.value as DatasetName);
              }}
            >
              {DATASET_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <span className="status-pill phase-frozen">
            <span className="status-dot" aria-hidden="true" />
            Finished
          </span>
          <span className="status-pill border-slate-200 text-slate-600">
            {detail.benchmark} · saved snapshot
          </span>
          <span className="font-mono text-xs text-slate-500">Run {detail.runId}</span>
        </div>
      </div>
    </header>
  );
}
