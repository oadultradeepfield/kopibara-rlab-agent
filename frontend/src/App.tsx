import { useEffect, useState } from 'react';
import type { ReactElement } from 'react';

import { DashboardHeader } from './components/DashboardHeader';
import { EventTimeline } from './components/EventTimeline';
import { ExperimentTable } from './components/ExperimentTable';
import { LineageTree } from './components/LineageTree';
import { MetricChart } from './components/MetricChart';
import { Overview } from './components/Overview';
import { loadRunSnapshot } from './run';
import type { RunSnapshot } from './types';

type ViewState =
  | { readonly status: 'loading' }
  | { readonly status: 'ready'; readonly snapshot: RunSnapshot }
  | { readonly status: 'failed'; readonly message: string };

export function App(): ReactElement {
  const [state, setState] = useState<ViewState>({ status: 'loading' });
  const [reload, setReload] = useState(0);
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading' });

    void loadRunSnapshot().then(
      (snapshot) => {
        if (!cancelled) {
          setState({ status: 'ready', snapshot });
        }
      },
      (error: unknown) => {
        if (!cancelled) {
          setState({ status: 'failed', message: formatLoadError(error) });
        }
      },
    );

    return () => {
      cancelled = true;
    };
  }, [reload]);

  if (state.status === 'loading') {
    return (
      <StatusScreen
        title="Loading research run"
        detail="Reading the latest JSON artifact saved by the agent…"
      />
    );
  }

  if (state.status === 'failed') {
    return (
      <StatusScreen
        title="No saved research run"
        detail={state.message}
        onRetry={() => {
          setReload((value) => value + 1);
        }}
      />
    );
  }

  const { snapshot } = state;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-700">
      <DashboardHeader detail={snapshot.detail} />
      <main className="mx-auto grid max-w-[1600px] gap-3 px-3 py-3 md:px-4">
        <Overview detail={snapshot.detail} />
        <LineageTree
          experiments={snapshot.experiments}
          baseline={snapshot.detail.baseline.primary}
          selectedExperimentId={selectedExperimentId}
          onSelectExperiment={setSelectedExperimentId}
        />
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1.3fr)_minmax(360px,0.7fr)]">
          <MetricChart
            points={snapshot.metrics}
            baseline={snapshot.detail.baseline.primary}
          />
          <EventTimeline
            events={snapshot.events}
            selectedExperimentId={selectedExperimentId}
            onClearSelection={() => {
              setSelectedExperimentId(null);
            }}
          />
        </div>
        <ExperimentTable
          experiments={snapshot.experiments}
          selectedExperimentId={selectedExperimentId}
          onSelectExperiment={setSelectedExperimentId}
        />
      </main>
    </div>
  );
}

interface StatusScreenProps {
  readonly title: string;
  readonly detail: string;
  readonly onRetry?: () => void;
}

function StatusScreen({ title, detail, onRetry }: StatusScreenProps): ReactElement {
  return (
    <main className="grid min-h-screen place-items-center bg-slate-50 p-6 text-center text-slate-700">
      <div className="max-w-md">
        <div className="mx-auto mb-5 grid size-12 place-items-center rounded border border-slate-300 bg-white font-semibold text-slate-700 shadow-sm">
          KR
        </div>
        <h1 className="text-xl font-semibold text-slate-900">{title}</h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">{detail}</p>
        {onRetry !== undefined && (
          <button
            type="button"
            className="mt-5 rounded border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm hover:border-blue-500 hover:text-blue-700"
            onClick={onRetry}
          >
            Try again
          </button>
        )}
      </div>
    </main>
  );
}

function formatLoadError(error: unknown): string {
  return error instanceof Error
    ? error.message
    : 'The JSON run artifact could not be read.';
}
