import { lazy, Suspense, useState } from 'react';
import type { ReactElement } from 'react';

import { DashboardHeader } from './components/DashboardHeader';
import { EventTimeline } from './components/EventTimeline';
import { ExperimentTable } from './components/ExperimentTable';
import { LineageTree } from './components/LineageTree';
import { Overview } from './components/Overview';
import { RUN_SNAPSHOT } from './data';

const MetricChart = lazy(() =>
  import('./components/MetricChart').then((module) => ({ default: module.MetricChart })),
);

export function App(): ReactElement {
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-[#111827] text-slate-300">
      <DashboardHeader detail={RUN_SNAPSHOT.detail} />
      <main className="mx-auto grid max-w-[1600px] gap-3 px-3 py-3 md:px-4">
        <Overview detail={RUN_SNAPSHOT.detail} />
        <LineageTree
          experiments={RUN_SNAPSHOT.experiments}
          baseline={RUN_SNAPSHOT.detail.baseline.primary}
          selectedExperimentId={selectedExperimentId}
          onSelectExperiment={setSelectedExperimentId}
        />
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1.3fr)_minmax(360px,0.7fr)]">
          <Suspense fallback={<div className="empty-panel">Loading score chart…</div>}>
            <MetricChart
              points={RUN_SNAPSHOT.metrics}
              baseline={RUN_SNAPSHOT.detail.baseline.primary}
            />
          </Suspense>
          <EventTimeline
            events={RUN_SNAPSHOT.events}
            selectedExperimentId={selectedExperimentId}
            onClearSelection={() => {
              setSelectedExperimentId(null);
            }}
          />
        </div>
        <ExperimentTable
          experiments={RUN_SNAPSHOT.experiments}
          selectedExperimentId={selectedExperimentId}
          onSelectExperiment={setSelectedExperimentId}
        />
      </main>
    </div>
  );
}
