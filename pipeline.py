"""
Startup Signal Pipeline — main entry point.

Runs the full pipeline:
  1. Ingest recent funding articles from RSS feeds
  2. Extract structured data via Gemini LLM
  3. Resolve company domains
  4. Find LinkedIn company pages
  5. Detect hiring signals (ATS / careers page)
  6. Store results in SQLite
  7. Publish Tier-A leads to Telegram and Google Sheets
"""

import os
from dotenv import load_dotenv

from app.ingest.rss_ingest import fetch_recent_articles
from app.extract.llm_parse import enrich_articles
from app.resolve.domain_resolver import resolve_company_domain
from app.resolve.find_linkedin import find_best_linkedin_url
from app.hiring.detect_ats import detect_hiring_signal
from app.store.upsert import init_db, upsert_company, check_articles_exist
from app.publish.telegram_alerts import send_telegram_alert
from app.publish.to_gsheet import save_to_sheet

load_dotenv()


def run_pipeline(days_back: int = 3) -> None:
    print("=" * 60)
    print("Startup Signal Pipeline")
    print("=" * 60)

    # 1. Initialise database
    init_db()

    # 2. Ingest
    print(f"\n[1/6] Fetching articles from the last {days_back} day(s)...")
    articles = fetch_recent_articles(days_back=days_back)
    print(f"      Found {len(articles)} candidate articles.")

    if not articles:
        print("No new articles. Exiting.")
        return

    # Skip articles already in the database
    known_urls = check_articles_exist([a["url"] for a in articles])
    new_articles = [a for a in articles if a["url"] not in known_urls]
    print(f"      {len(new_articles)} new (not yet stored).")

    if not new_articles:
        print("All articles already processed. Exiting.")
        return

    # 3. Extract structured data
    print(f"\n[2/6] Extracting structured data via LLM...")
    enriched = enrich_articles(new_articles)
    print(f"      Extracted data for {len(enriched)} articles.")

    if not enriched:
        print("No structured data extracted. Exiting.")
        return

    # 4. Resolve domains + LinkedIn
    print(f"\n[3/6] Resolving company domains and LinkedIn pages...")
    for item in enriched:
        company = item.get("company_name", "")
        article_url = item.get("url", "")

        if not item.get("website_url"):
            result = resolve_company_domain(company, article_url)
            item["domain"] = result.get("domain")
            item["domain_confidence"] = result.get("confidence", 0.0)
        else:
            item["domain"] = item["website_url"]

        if not item.get("linkedin_url"):
            item["linkedin_url"] = find_best_linkedin_url(company, item.get("domain"))

        print(f"      {company}: domain={item.get('domain')} linkedin={item.get('linkedin_url')}")

    # 5. Detect hiring signals
    print(f"\n[4/6] Detecting hiring signals...")
    for item in enriched:
        signal = detect_hiring_signal(item.get("domain"))
        item.update({
            "hiring_tier": signal["hiring_tier"],
            "tech_roles": signal["tech_roles"],
            "careers_url": signal["careers_url"],
            "ats_provider": signal["ats_provider"],
            "details": signal["details"],
        })
        print(f"      {item.get('company_name')}: tier={signal['hiring_tier']} tech_roles={signal['tech_roles']}")

    # 6. Store
    print(f"\n[5/6] Storing {len(enriched)} records...")
    for item in enriched:
        upsert_company(item)

    # 7. Publish Tier-A leads
    tier_a = [item for item in enriched if item.get("hiring_tier") == "A"]
    print(f"\n[6/6] Publishing {len(tier_a)} Tier-A lead(s)...")

    if tier_a:
        for item in tier_a:
            send_telegram_alert(item)
        save_to_sheet(tier_a)
    else:
        print("      No Tier-A leads this run.")

    print("\nPipeline complete.")
    print(f"  Total processed : {len(enriched)}")
    print(f"  Tier A (hiring) : {len(tier_a)}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
