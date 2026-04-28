import json
import os
import re

import google.generativeai as genai
import pandas as pd
import streamlit as st
from Bio import Entrez
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PREFERRED_MODELS = ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro"]
_MODEL = None
try:
    available = [
        m.name.replace("models/", "")
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]
    for pref in PREFERRED_MODELS:
        if pref in available:
            _MODEL = genai.GenerativeModel(pref)
            break
    if not _MODEL and available:
        _MODEL = genai.GenerativeModel(available[0])
except Exception:
    _MODEL = None

_quota_exceeded = False
Entrez.email = os.getenv("EMAIL", "demo@example.com")


def search_pubmed(keywords, max_results=50):
    query = f"({' OR '.join(keywords)}) AND (2023[DP] : 2025[DP])"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_paper_details(pmid):
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="text")
    records = Entrez.read(handle)
    handle.close()
    paper = records["PubmedArticle"][0]
    title = paper["MedlineCitation"]["Article"]["ArticleTitle"]
    authors = []
    for author in paper["MedlineCitation"]["Article"].get("AuthorList", []):
        if "LastName" not in author or "ForeName" not in author:
            continue
        name = f"{author['ForeName']} {author['LastName']}"
        affil_list = author.get("AffiliationInfo", [])
        affil = affil_list[0].get("Affiliation", "") if affil_list else ""
        parts = affil.split(",")
        location = (
            f"{parts[-2].strip()}, {parts[-1].strip()}" if len(parts) > 1 else "Unknown"
        )
        authors.append({"name": name, "affiliation": affil, "location": location})
    return {"pmid": pmid, "title": title, "authors": authors}


def _call_llm(prompt):
    global _quota_exceeded
    if not _MODEL or _quota_exceeded:
        return None
    try:
        response = _MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        if "429" in str(exc) or "quota" in str(exc).lower():
            _quota_exceeded = True
        print(f"LLM error: {exc}")
        return None


def scrape_conference_attendees():
    with DDGS() as ddgs:
        results = ddgs.text("SOT toxicology conference speakers 2024", max_results=10)
    snippets = [r["body"] for r in results]
    combined = " ".join(snippets)
    prompt = (
        "Extract names of speakers or attendees from the following search snippets "
        "about the SOT toxicology conference. "
        "Return a JSON list of up to 20 names.\n\nSnippets: " + combined
    )
    names = []
    raw = _call_llm(prompt)
    if raw:
        try:
            raw = raw.replace("```json", "").replace("```", "").strip()
            names = json.loads(raw)
        except Exception:
            pass
    if not names:
        names = list(set(re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", combined)))[:20]
    return [
        {"name": n, "title": "Speaker", "company": "Unknown", "location": "Unknown", "source": "Conference"}
        for n in names
    ]


def enrich_person(person):
    name = person["name"]
    company = person.get("company", "Unknown")

    def _search(query, n=5):
        with DDGS() as ddgs:
            return [r["body"] for r in ddgs.text(query, max_results=n)]

    linkedin_snippets = _search(f'"{name}" "{company}" linkedin')
    email_snippets = _search(f'"{name}" "{company}" email')
    location_snippets = _search(f'"{company}" headquarters location', n=3)

    prompt = (
        f"Extract information for {name} at {company} from the search snippets below.\n\n"
        f"LinkedIn snippets: {' '.join(linkedin_snippets)}\n"
        f"Email snippets: {' '.join(email_snippets)}\n"
        f"Location snippets: {' '.join(location_snippets)}\n\n"
        "Return JSON with keys: linkedin (URL or null), email (string or null), location (string or null)."
    )

    linkedin = email = location = None
    raw = _call_llm(prompt)
    if raw:
        try:
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            linkedin = data.get("linkedin")
            email = data.get("email")
            location = data.get("location")
        except Exception:
            pass

    person["linkedin"] = linkedin or f"https://linkedin.com/in/{name.replace(' ', '').lower()}"
    person["email"] = (
        email
        or f"{name.split()[0].lower()}.{name.split()[-1].lower()}@{company.replace(' ', '').lower()}.com"
    )
    if location:
        person["location"] = location
    return person


def calculate_score(person):
    title = person.get("title", "").lower()
    company = person.get("company", "").lower()
    location = person.get("location", "").lower()

    with DDGS() as ddgs:
        funding_snippets = [
            r["body"]
            for r in ddgs.text(f'"{company}" series funding OR raised OR IPO', max_results=5)
        ]

    prompt = (
        f"Analyze {company} in the biotech/toxicology space.\n"
        f"Title: {title}\nLocation: {location}\n"
        f"Funding snippets: {' '.join(funding_snippets)}\n\n"
        "Assign scores: Role Fit (0-30), Company Intent (0-20), Technographic (0-15), "
        "Location (0-10), Scientific Intent (0-40). Return only the total integer score (0-100)."
    )

    score = 0
    raw = _call_llm(prompt)
    if raw:
        match = re.search(r"\d+", raw)
        if match:
            score = int(match.group())

    if any(w in title for w in ["toxicology", "safety", "hepatic", "3d", "preclinical", "director", "head"]):
        score += 30
    if any("series" in s.lower() or "raised" in s.lower() for s in funding_snippets):
        score += 20
    score += 15
    if any(hub in location for hub in ["boston", "cambridge", "san francisco", "basel", "london"]):
        score += 10
    if person.get("has_recent_paper"):
        score += 40

    return min(score, 100)


def run_biotech_pipeline():
    st.set_page_config(page_title="Biotech Lead Generator", page_icon="🔬", layout="wide")
    st.title("Biotech Lead Generator")
    st.markdown("Automated pipeline for identifying, enriching, and ranking biotech leads in 3D in-vitro models.")

    if not os.getenv("GEMINI_API_KEY"):
        st.error("Set GEMINI_API_KEY in your .env file.")
        return
    if not os.getenv("EMAIL"):
        st.error("Set EMAIL in your .env file (required for PubMed API access).")
        return

    st.sidebar.header("Dashboard")

    with st.spinner("Identifying leads from PubMed and conferences..."):
        st.header("1. Identification")
        keywords = [
            "Drug-Induced Liver Injury", "3D cell culture", "Organ-on-chip",
            "Hepatic spheroids", "Investigative Toxicology",
        ]
        pmids = search_pubmed(keywords, max_results=30)
        st.write(f"Found {len(pmids)} relevant papers on PubMed.")

        leads = []
        for pmid in pmids:
            paper = fetch_paper_details(pmid)
            for author in paper["authors"]:
                leads.append({
                    "name": author["name"],
                    "title": "Researcher",
                    "company": author["affiliation"].split(",")[0] if author["affiliation"] else "Unknown",
                    "location": author["location"],
                    "source": "PubMed",
                    "has_recent_paper": True,
                })

        for lead in scrape_conference_attendees():
            lead["has_recent_paper"] = False
            leads.append(lead)

        st.write(f"Total leads identified: {len(leads)}")
        st.sidebar.metric("Total Leads", len(leads))

    with st.spinner("Enriching lead data..."):
        st.header("2. Enrichment")
        enriched_leads = []
        progress = st.progress(0)
        batch = leads[:30]
        for i, lead in enumerate(batch):
            enriched_leads.append(enrich_person(lead))
            progress.progress((i + 1) / len(batch))
        progress.empty()

    with st.spinner("Calculating propensity scores..."):
        st.header("3. Ranking")
        for lead in enriched_leads:
            lead["score"] = calculate_score(lead)
        enriched_leads.sort(key=lambda x: x["score"], reverse=True)
        avg_score = (
            sum(l["score"] for l in enriched_leads) / len(enriched_leads)
            if enriched_leads else 0
        )
        st.sidebar.metric("Average Score", f"{avg_score:.1f}")

    st.header("Lead Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Min Score", 0, 100, 0)
    with col2:
        location_filter = st.text_input("Filter by Location")
    with col3:
        company_filter = st.text_input("Filter by Company")

    filtered = [l for l in enriched_leads if l["score"] >= min_score]
    if location_filter:
        filtered = [l for l in filtered if location_filter.lower() in l.get("location", "").lower()]
    if company_filter:
        filtered = [l for l in filtered if company_filter.lower() in l.get("company", "").lower()]

    df = pd.DataFrame(filtered)[["score", "name", "title", "company", "location", "email", "linkedin"]]
    df.columns = ["Score", "Name", "Title", "Company", "Location", "Email", "LinkedIn"]
    st.dataframe(df, use_container_width=True)

    if enriched_leads:
        import plotly.express as px
        fig = px.histogram(
            pd.DataFrame({"Score": [l["score"] for l in enriched_leads]}),
            x="Score", nbins=10, title="Lead Score Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.download_button("Download CSV", df.to_csv(index=False), "leads.csv", "text/csv")

    st.header("Email Outreach")
    if filtered:
        selected_name = st.selectbox("Select a lead:", [l["name"] for l in filtered])
        lead = next(l for l in filtered if l["name"] == selected_name)
        body = (
            f"Dear {lead['name']},\n\n"
            "My name is [Your Name] from [Your Company]. We specialise in advanced 3D in-vitro "
            "models for drug safety and toxicology research.\n\n"
            f"Given your work in {lead.get('title', 'research')} at {lead.get('company', 'your institution')}, "
            "I believe our solutions could support your research.\n\n"
            "Would you be open to a brief conversation?\n\n"
            "Best regards,\n[Your Name]\n[Your Position]\n[Your Company]"
        )
        st.text_area("Email Draft:", body, height=200)
        subject = "Interest in 3D In-Vitro Models for Drug Safety Research"
        mailto = (
            f"mailto:{lead.get('email', '')}?subject={subject}"
            f"&body={body.replace(chr(10), '%0A').replace(' ', '%20')}"
        )
        st.markdown(f"[Open in Email Client]({mailto})")
    else:
        st.info("No leads match the current filters.")


if __name__ == "__main__":
    run_biotech_pipeline()