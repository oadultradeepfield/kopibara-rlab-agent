import { describe, expect, it } from 'vitest';

import { RUN_SNAPSHOT } from './data';

describe('Pure run snapshot', () => {
  it('keeps the recorded best above the official baseline', () => {
    expect(RUN_SNAPSHOT.detail.best.primary).toBeGreaterThan(
      RUN_SNAPSHOT.detail.baseline.primary,
    );
  });

  it('records a submission-sized result without hidden-test access', () => {
    expect(RUN_SNAPSHOT.detail.submissionRows).toBe(170588);
    expect(RUN_SNAPSHOT.detail.hiddenTestAccess).toBe(false);
  });
});
