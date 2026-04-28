# Startup Signal Pipeline

An automated intelligence pipeline that monitors startup funding activity, detects active hiring signals, and surfaces high-intent leads — delivered to Telegram and Google Sheets in near real-time.

Includes a separate **Biotech Lead Generator** Streamlit app that identifies and ranks researchers and conference speakers from PubMed and the SOT toxicology conference.

---

## Features

- Ingests funding news from six RSS feeds (TechCrunch, VentureBeat, Inc42, Entrackr, YourStory, FinSMEs)
- Extracts structured company data (name, round, amount, investors, country) via Gemini LLM
- Resolves company domains via press-release scraping, DuckDuckGo search, and slug guessing
- Finds official LinkedIn company pages with heuristic scoring
- Detects hiring signals by scraping Greenhouse, Lever, Ashby, Workable, BambooHR, and internal careers pages
- Tiers companies A/B/C by recency and volume of open tech roles
- Stores everything in a local SQLite database with upsert semantics
- Sends Tier-A alerts to a Telegram channel
- Publishes results to Google Sheets
- Biotech Lead Generator: PubMed search + LLM enrichment + propensity scoring + Streamlit UI

## Tech Stack

- Python 3.11
- Google Gemini API (`google-generativeai`)
- `feedparser`, `requests`, `beautifulsoup4`
- `duckduckgo-search`
- SQLite (via stdlib `sqlite3`)
- `gspread` + Google Sheets API
- Telegram Bot API
- Streamlit + Plotly + Biopython (biotech UI)
- Docker + Docker Compose

## Architecture

```
RSS Feeds
    │
    ▼
[rss_ingest]  ──► raw articles (title, url, date)
    │
    ▼
[llm_parse]   ──► structured data (company, round, amount, investors)
    │
    ├──► [domain_resolver]  ──► website URL
    │
    ├──► [find_linkedin]    ──► LinkedIn company URL
    │
    ▼
[detect_ats]  ──► hiring tier (A/B/C), tech role count, ATS provider
    │
    ▼
[upsert]      ──► SQLite (data/companies.db)
    │
    ├──► [telegram_alerts]  ──► Telegram channel (Tier A only)
    │
    └──► [to_gsheet]        ──► Google Sheets
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/startup-signal-pipeline.git
cd startup-signal-pipeline
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Run with Docker (recommended)

```bash
# Start the Streamlit UI
docker compose up streamlit

# Run the funding pipeline once
docker compose --profile pipeline run --rm pipeline
```

The Streamlit app will be available at http://localhost:8501.

### 3. Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the funding pipeline
python pipeline.py

# Run the Streamlit biotech UI
streamlit run biotech_main.py
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key. Get one at [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `EMAIL` | Yes (biotech UI) | Email address for PubMed Entrez API |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram bot token for Tier-A alerts |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat/channel ID to send alerts to |
| `GOOGLE_CREDS_JSON` | Optional | Filename of your Google service account JSON (default: `google_creds.json`) |

Copy `.env.example` to `.env` and fill in your values. Never commit `.env` or credential files.

## Google Sheets Setup

1. Create a Google Cloud project and enable the Google Sheets API.
2. Create a service account and download the JSON key as `google_creds.json` in the project root.
3. Create a Google Sheet named **"Recently Funded Startups"** and share it with the service account email (Editor access).

## Telegram Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) and copy the token.
2. Add the bot to your channel/group and get the chat ID.
3. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.

Test your setup:

```bash
python scripts/test_telegram_alert.py
```

## Folder Structure

```
.
├── app/
│   ├── extract/        # LLM-based article parsing
│   ├── hiring/         # ATS detection and role classification
│   ├── ingest/         # RSS feed ingestion
│   ├── publish/        # Telegram and Google Sheets output
│   ├── resolve/        # Domain and LinkedIn resolution
│   └── store/          # SQLite schema and upsert logic
├── data/               # SQLite database (gitignored)
├── scripts/            # Utility scripts
├── tests/              # Test suite
├── biotech_main.py     # Streamlit biotech lead generator
├── pipeline.py         # Main pipeline entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Hiring Tier Logic

| Tier | Meaning |
|---|---|
| A | Tech roles posted within the last 14 days |
| B | Tech roles exist but none posted recently |
| C | No tech roles found |

Only Tier-A companies trigger Telegram alerts and are written to Google Sheets.

## Scheduling

To run the pipeline on a schedule, use cron or a task scheduler:

```bash
# Run every 6 hours (cron)
0 */6 * * * cd /path/to/project && docker compose --profile pipeline run --rm pipeline
```

## Future Improvements

- Add a proper job queue (Celery / RQ) for parallel enrichment
- Persist Streamlit session state to avoid re-running the full pipeline on filter changes
- Add CI/CD with GitHub Actions (lint, test, Docker build)
- Replace SQLite with PostgreSQL for multi-instance deployments
- Add rate-limit handling and retry logic for external APIs
- Expand ATS coverage (Rippling, Jobvite, SmartRecruiters)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
