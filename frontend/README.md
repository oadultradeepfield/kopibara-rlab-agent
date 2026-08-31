# Kopibara Agent dashboard

Small read-only dashboard for the latest autonomous run. The agent writes its
manifest to `frontend/public/run.json`; the dashboard reads that file directly,
so no backend or credentials are needed.

```bash
npm install
npm run dev
```

From the project root, run the agent with `kopibara-agent --run`. Refresh the
dashboard after the run finishes to see the new manifest.
