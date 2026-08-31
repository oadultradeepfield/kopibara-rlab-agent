import type { ReactElement } from 'react';

import type { RunEvent } from '../types';

interface EventTimelineProps {
  readonly events: readonly RunEvent[];
  readonly selectedExperimentId: string | null;
  readonly onClearSelection: () => void;
}

export function EventTimeline({
  events,
  selectedExperimentId,
  onClearSelection,
}: EventTimelineProps): ReactElement {
  const visibleEvents =
    selectedExperimentId === null
      ? events
      : events.filter((event) => event.experimentId === selectedExperimentId);
  const ordered = [...visibleEvents].sort((left, right) => right.step - left.step);

  return (
    <section className="panel overflow-hidden" aria-labelledby="timeline-title">
      <div className="flex items-end justify-between border-b border-white/7 p-3">
        <div>
          <p className="section-kicker">Run history</p>
          <h2 id="timeline-title" className="section-title">
            What happened
          </h2>
        </div>
        <div className="flex items-center gap-2">
          {selectedExperimentId !== null && (
            <button
              type="button"
              className="text-xs text-blue-300 hover:text-blue-200 hover:underline"
              onClick={onClearSelection}
            >
              Clear filter
            </button>
          )}
          <span className="font-mono text-xs text-slate-400">
            {visibleEvents.length} events
          </span>
        </div>
      </div>
      <ol className="max-h-[400px] overflow-auto px-3">
        {ordered.length === 0 && (
          <li className="py-6 text-center text-sm text-slate-400">
            No events recorded for this experiment.
          </li>
        )}
        {ordered.map((event) => (
          <li key={event.id} className="timeline-event">
            <span className={`timeline-marker event-${event.tone}`} />
            <div className="min-w-0 flex-1 py-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-medium text-slate-200">{event.label}</p>
                <span className="font-mono text-xs text-slate-500">
                  Step {event.step}
                </span>
              </div>
              <p className="mt-1 text-xs leading-5 text-slate-400">{event.message}</p>
              <div className="mt-1 font-mono text-xs text-slate-500">
                {event.experimentId ?? 'Run'}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
