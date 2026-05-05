# BlandAI-Dashboard — Project Instructions

## Purpose

Analytics dashboard for Biometrics4ALL's BlandAI voice AI support system.
Tracks call outcomes, resolution rates, and performance trends across pathways.
Deployed on Streamlit Cloud, data refreshed via a 5-stage pipeline.

---

## Dashboard

- **URL**: blandai-mhekzfr5yyd3nj2sdbr3x6.streamlit.app
- **App file**: `dashboard_drivers.py` (repo root)
- **Data**: `data/calls_drivers.csv`

Three charts:
1. **Outcomes by Pathway** — stacked horizontal bars (% resolved, transferred, abandoned, etc.)
2. **Containment vs True Resolution** — weekly trend with 48h callback validation
3. **Week Fixed Effects** — LPM regression controlling for component mix over time

---

## Data Pipeline

Run stages in order. Each reads from the previous stage's output.

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1. Pull | `scripts/pull_calls.py` | BlandAI API | `data/calls_list.json`, `data/calls_detail/*.json`, `data/calls_summary.csv` |
| 2. Classify outcomes | `scripts/classify_outcomes.py` | `data/calls_summary.csv` + `data/calls_detail/` | `data/calls_classified.csv` |
| 3. Classify drivers | `scripts/classify_drivers.py` | `data/calls_classified.csv` + `data/calls_detail/` + CDA taxonomy | `data/calls_drivers.csv` |
| 4. Callback flag | `scripts/add_callback_flag.py` | `data/calls_drivers.csv` + `data/calls_summary.csv` | `data/calls_drivers.csv` (in-place) |
| 5. Deploy | `git commit + push` | — | Streamlit Cloud auto-deploys |

**"Update the dashboard" = run all 5 stages end-to-end, not piecemeal.**

### Cross-Project Dependency

`classify_drivers.py` reads the taxonomy from `../CallDriverAnalysis/taxonomy/components.json`.
This is the 24-component taxonomy (v2.2) used to classify calls into components and symptoms.

---

## Data Source

- **API**: BlandAI REST API (`https://api.bland.ai/v1/`)
- **Auth**: `BLAND_API_KEY` in `../.env`
- **Classification model**: Claude Haiku (`claude-haiku-4-5-20251001`)
- **Classification API key**: `ANTHROPIC_API_KEY` in `../.env`

---

## Deployment

- **Deploy branch**: `master`
- **Platform**: Streamlit Cloud (auto-deploys on push)
- **Dependencies**: `requirements.txt` (streamlit, pandas, plotly, statsmodels)

---

## General Preferences

- Windows 11, files on OneDrive — always use `encoding='utf-8'` for file I/O
- Python via Anaconda
- Use `pathlib.Path` for all file paths
- `.env` is one directory above project root (`../.env`)

---

## Related Projects

- **BlandAI** (`../BlandAI/`): Pathway config, prompts, KBs, deployment, testing
- **CallDriverAnalysis** (`../CallDriverAnalysis/`): 12-stage NLP pipeline, taxonomy, KB generation
