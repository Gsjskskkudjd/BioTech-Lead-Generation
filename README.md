# Biotech Lead Generation

This is a demo web agent/crawler for generating leads in the 3D in-vitro models space for therapies.

## Features

- **Identification**: Searches PubMed for recent papers on relevant keywords, extracts authors. Also includes mock conference attendees.
- **Enrichment**: Adds mock contact info and LinkedIn URLs.
- **Ranking**: Scores leads based on role fit, company funding, location, and scientific intent.
- **Dashboard**: Interactive table with search and export to CSV.

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Set up `.env` with your free Gemini API key (from Google AI Studio) and email.
3. Run: `streamlit run biotech_main.py`

## APIs Used (All Free)
- PubMed API (NCBI)
- DuckDuckGo Search
- Google Gemini (free tier)
- Web scraping (public sites)

## Architecture

Based on the existing startup pipeline, adapted for biotech leads.

- Ingest: PubMed API and mock conference scraping.
- Enrich: LLM-based extraction (mocked in demo).
- Rank: Weighted scoring.
- Publish: Streamlit dashboard.

## Demo Output

The app displays a table of ranked leads with columns: Rank Probability, Name, Title, Company, Location, Email, LinkedIn.

You can search and download as CSV.

## Extensions

- For real LinkedIn data, integrate Proxycurl API.
- For conferences, scrape actual attendee lists.
- For emails, use Hunter.io or similar.
- Add more sources like Crunchbase for funding info.
