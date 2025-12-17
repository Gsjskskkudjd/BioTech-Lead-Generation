import os
import streamlit as st
import pandas as pd
from Bio import Entrez
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import re
from urllib.parse import urlparse
from ddgs import DDGS
import json

load_dotenv()

# Configure Gemini (Free Tier)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Find available model, prefer stable ones
preferred_models = ['gemini-pro', 'gemini-1.0-pro', 'gemini-1.5-pro']
MODEL = None
try:
    available = [model.name.replace('models/', '') for model in genai.list_models() if 'generateContent' in model.supported_generation_methods]
    for pref in preferred_models:
        if pref in available:
            MODEL = genai.GenerativeModel(pref)
            break
    if not MODEL and available:
        MODEL = genai.GenerativeModel(available[0])
except:
    MODEL = None

# Global flag for quota
quota_exceeded = False

Entrez.email = os.getenv("EMAIL", "demo@example.com")  # Use env var

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def search_pubmed(keywords, max_results=50):
    """Search PubMed for papers with given keywords in last 2 years."""
    query = f"({' OR '.join(keywords)}) AND (2023[DP] : 2025[DP])"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_paper_details(pmid):
    """Fetch paper details including authors and affiliations."""
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="text")
    records = Entrez.read(handle)
    handle.close()
    paper = records['PubmedArticle'][0]
    title = paper['MedlineCitation']['Article']['ArticleTitle']
    authors = []
    for author in paper['MedlineCitation']['Article']['AuthorList']:
        if 'LastName' in author and 'ForeName' in author:
            name = f"{author['ForeName']} {author['LastName']}"
            affil = author.get('AffiliationInfo', [{}])[0].get('Affiliation', '') if author.get('AffiliationInfo') else ''
            # Extract location from affiliation
            location = 'Unknown'
            if affil:
                parts = affil.split(',')
                if len(parts) > 1:
                    location = parts[-2].strip() + ', ' + parts[-1].strip()
            authors.append({'name': name, 'affiliation': affil, 'location': location})
    return {'pmid': pmid, 'title': title, 'authors': authors}

def scrape_conference_attendees():
    """Scrape conference speakers using web search and LLM."""
    with DDGS() as ddgs:
        results = ddgs.text("SOT toxicology conference speakers 2024", max_results=10)
    snippets = [r['body'] for r in results]
    
    prompt = f"""
    Extract names of speakers or attendees from the following search snippets about SOT toxicology conference.
    Snippets: {' '.join(snippets)}
    
    Return a list of up to 10 names in JSON format, e.g., ["Name1", "Name2"].
    """
    
    global quota_exceeded
    if MODEL and not quota_exceeded:
        prompt = f"""
        Extract names of speakers or attendees from the following search snippets about SOT toxicology conference.
        Snippets: {' '.join(snippets)}
        
        Return a list of up to 20 names in JSON format, e.g., ["Name1", "Name2"].
        """
        
        try:
            response = MODEL.generate_content(prompt)
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:-3].strip()
            names = json.loads(text)
        except Exception as e:
            print(f"LLM error: {e}")
            if '429' in str(e) or 'quota' in str(e).lower():
                quota_exceeded = True
            names = []
    else:
        names = []
    
    if not names:
        for snippet in snippets:
            names.extend(re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', snippet))
        names = list(set(names))[:20]
    
    attendees = []
    for name in names:
        attendees.append({
            'name': name,
            'title': 'Speaker',
            'company': 'Unknown',
            'location': 'Unknown',
            'source': 'Conference'
        })
    return attendees

def enrich_person(person):
    """Enrich person data using web search and LLM."""
    name = person['name']
    company = person.get('company', 'Unknown')
    
    # Collect search results
    linkedin_snippets = []
    email_snippets = []
    location_snippets = []
    
    with DDGS() as ddgs:
        linkedin_results = ddgs.text(f'"{name}" "{company}" linkedin', max_results=5)
        for r in linkedin_results:
            linkedin_snippets.append(r['body'])
    
    with DDGS() as ddgs:
        email_results = ddgs.text(f'"{name}" "{company}" email', max_results=5)
        for r in email_results:
            email_snippets.append(r['body'])
    
    with DDGS() as ddgs:
        location_results = ddgs.text(f'"{company}" headquarters location', max_results=3)
        for r in location_results:
            location_snippets.append(r['body'])
    
    # Use LLM to extract
    global quota_exceeded
    if MODEL and not quota_exceeded:
        prompt = f"""
        Extract information for {name} at {company} from the following search snippets.
        
        LinkedIn snippets: {' '.join(linkedin_snippets)}
        Email snippets: {' '.join(email_snippets)}
        Location snippets: {' '.join(location_snippets)}
        
        Return JSON with:
        - linkedin: LinkedIn URL if found, else null
        - email: Email if found, else null
        - location: HQ location if found, else null
        """
        
        try:
            response = MODEL.generate_content(prompt)
            text = response.text.strip()
            # Remove markdown if present
            if text.startswith('```json'):
                text = text[7:-3].strip()
            data = json.loads(text)
            linkedin = data.get('linkedin')
            email = data.get('email')
            location = data.get('location')
        except Exception as e:
            print(f"LLM error: {e}")
            if '429' in str(e) or 'quota' in str(e).lower():
                quota_exceeded = True
            linkedin = None
            email = None
            location = None
    else:
        linkedin = None
        email = None
        location = None
    
    person['linkedin'] = linkedin or f"https://linkedin.com/in/{name.replace(' ', '').lower()}"
    person['email'] = email or f"{name.split()[0].lower()}.{name.split()[-1].lower()}@{company.replace(' ', '').lower()}.com"
    if location:
        person['location'] = location
    return person

def calculate_score(person):
    """Calculate propensity score using LLM for analysis."""
    title = person.get('title', '').lower()
    company = person.get('company', '').lower()
    location = person.get('location', '').lower()
    
    # Collect funding snippets
    funding_snippets = []
    with DDGS() as ddgs:
        results = ddgs.text(f'"{company}" series funding OR raised OR IPO', max_results=5)
        for r in results:
            funding_snippets.append(r['body'])
    
    prompt = f"""
    Analyze the following for {company} in biotech/toxicology space.
    Title: {title}
    Location: {location}
    Funding snippets: {' '.join(funding_snippets)}
    
    Assign scores:
    - Role Fit (0-30): High if title contains toxicology, safety, etc.
    - Company Intent (0-20): High if recent funding (series A/B, raised money).
    - Technographic (0-15): Assume 15 for biotech.
    - Location (0-10): High if in Boston, Cambridge, etc.
    - Scientific Intent (0-40): High if has recent paper.
    
    Return total score (0-100).
    """
    
    global quota_exceeded
    if MODEL and not quota_exceeded:
        prompt = f"""
        Analyze the following for {company} in biotech/toxicology space.
        Title: {title}
        Location: {location}
        Funding snippets: {' '.join(funding_snippets)}
        
        Assign scores:
        - Role Fit (0-30): High if title contains toxicology, safety, etc.
        - Company Intent (0-20): High if recent funding (series A/B, raised money).
        - Technographic (0-15): Assume 15 for biotech.
        - Location (0-10): High if in Boston, Cambridge, etc.
        - Scientific Intent (0-40): High if has recent paper.
        
        Return total score (0-100).
        """
        
        try:
            response = MODEL.generate_content(prompt)
            text = response.text.strip()
            # Extract number
            score_match = re.search(r'\d+', text)
            if score_match:
                score = int(score_match.group())
            else:
                score = 0
        except Exception as e:
            print(f"LLM error: {e}")
            if '429' in str(e) or 'quota' in str(e).lower():
                quota_exceeded = True
            score = 0
    else:
        score = 0
    
    # Fallback
    if any(word in title for word in ['toxicology', 'safety', 'hepatic', '3d', 'preclinical', 'director', 'head']):
        score += 30
    if any('series' in s.lower() or 'raised' in s.lower() for s in funding_snippets):
        score += 20
    score += 15
    if any(hub in location.lower() for hub in ['boston', 'cambridge', 'san francisco', 'basel', 'london']):
        score += 10
    if person.get('has_recent_paper', False):
        score += 40
    
    return min(score, 100)

def run_biotech_pipeline():
    st.title("Biotech Lead Generation Demo")
    
    # Check API keys
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Please set GEMINI_API_KEY in .env file. Get it from https://aistudio.google.com/app/apikey")
        return
    if not os.getenv("EMAIL"):
        st.error("Please set EMAIL in .env file for PubMed API.")
        return

def run_biotech_pipeline():
    st.set_page_config(page_title="Biotech Lead Generator", page_icon="🔬", layout="wide")
    
    st.title("🔬 Biotech Lead Generation Demo")
    st.markdown("Automated pipeline for identifying, enriching, and ranking biotech leads in 3D in-vitro models.")
    
    # Check API keys
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Please set GEMINI_API_KEY in .env file. Get it from https://aistudio.google.com/app/apikey")
        return
    if not os.getenv("EMAIL"):
        st.error("Please set EMAIL in .env file for PubMed API.")
        return

    # Sidebar
    st.sidebar.header("📊 Dashboard")
    total_leads = 0
    avg_score = 0
    
    # Stage 1: Identification
    with st.spinner("🔍 Identifying leads from PubMed and conferences..."):
        st.header("1️⃣ Identification")
        keywords = ["Drug-Induced Liver Injury", "3D cell culture", "Organ-on-chip", "Hepatic spheroids", "Investigative Toxicology"]
        pmids = search_pubmed(keywords, max_results=30)
        st.write(f"📄 Found {len(pmids)} relevant papers on PubMed.")
        
        leads = []
        for pmid in pmids:
            paper = fetch_paper_details(pmid)
            for author in paper['authors']:
                lead = {
                    'name': author['name'],
                    'title': 'Researcher',  # Assume
                    'company': author['affiliation'].split(',')[0] if author['affiliation'] else 'Unknown',
                    'location': author['location'],
                    'source': 'PubMed',
                    'has_recent_paper': True
                }
                leads.append(lead)
        
        # Add conference attendees
        conf_leads = scrape_conference_attendees()
        for lead in conf_leads:
            lead['has_recent_paper'] = False
            leads.append(lead)
        
        total_leads = len(leads)
        st.write(f"👥 Total leads identified: {total_leads}")
        
        # Metrics
        st.sidebar.metric("Total Leads", total_leads)
        
    # Stage 2: Enrichment
    with st.spinner("🔧 Enriching lead data with web searches..."):
        st.header("2️⃣ Enrichment")
        enriched_leads = []
        progress_bar = st.progress(0)
        for i, lead in enumerate(leads[:30]):
            enriched = enrich_person(lead)
            enriched_leads.append(enriched)
            progress_bar.progress((i+1)/30)
        progress_bar.empty()
        
    # Stage 3: Ranking
    with st.spinner("📈 Calculating propensity scores..."):
        st.header("3️⃣ Ranking")
        for lead in enriched_leads:
            lead['score'] = calculate_score(lead)
        
        # Sort by score
        enriched_leads.sort(key=lambda x: x['score'], reverse=True)
        
        avg_score = sum(lead['score'] for lead in enriched_leads) / len(enriched_leads) if enriched_leads else 0
        st.sidebar.metric("Average Score", f"{avg_score:.1f}")
        
    # Output
    st.header("📋 Lead Generation Dashboard")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Min Score", 0, 100, 0)
    with col2:
        location_filter = st.text_input("Filter by Location", "")
    with col3:
        company_filter = st.text_input("Filter by Company", "")
    
    filtered_leads = [lead for lead in enriched_leads if lead['score'] >= min_score]
    if location_filter:
        filtered_leads = [lead for lead in filtered_leads if location_filter.lower() in lead.get('location', '').lower()]
    if company_filter:
        filtered_leads = [lead for lead in filtered_leads if company_filter.lower() in lead.get('company', '').lower()]
    
    df = pd.DataFrame(filtered_leads)
    df = df[['score', 'name', 'title', 'company', 'location', 'email', 'linkedin']]
    df.columns = ['Rank Probability', 'Name', 'Title', 'Company', 'Location', 'Email', 'LinkedIn']
    
    st.dataframe(df, use_container_width=True)
    
    # Chart
    if enriched_leads:
        import plotly.express as px
        score_df = pd.DataFrame({'Score': [lead['score'] for lead in enriched_leads]})
        fig = px.histogram(score_df, x='Score', nbins=10, title="Lead Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Export
    csv = df.to_csv(index=False)
    st.download_button("📥 Download as CSV", csv, "leads.csv", "text/csv")
    
    # Email Draft
    st.header("📧 Email Outreach")
    if filtered_leads:
        selected_name = st.selectbox("Select a lead to draft email:", [lead['name'] for lead in filtered_leads])
        selected_lead = next(lead for lead in filtered_leads if lead['name'] == selected_name)
        
        subject = "Interest in 3D In-Vitro Models for Drug Safety Research"
        body = f"""
Dear {selected_lead['name']},

I hope this email finds you well. My name is [Your Name], and I'm reaching out from [Your Company], where we specialize in advanced 3D in-vitro models for drug safety and toxicology research.

Given your expertise in {selected_lead.get('title', 'research')} at {selected_lead.get('company', 'your institution')}, I believe our solutions could greatly enhance your work on hepatic models and investigative toxicology.

Would you be interested in learning more about how our technology can support your research?

Best regards,
[Your Name]
[Your Position]
[Your Contact Info]
[Your Company]
"""
        
        st.text_area("Email Draft:", body, height=200)
        
        # Mailto link
        body_encoded = body.replace('\n', '%0A').replace(' ', '%20')
        mailto_link = f"mailto:{selected_lead.get('email', '')}?subject={subject}&body={body_encoded}"
        st.markdown(f"[📤 Open in Email Client]({mailto_link})")
    else:
        st.write("No leads to select.")

if __name__ == "__main__":
    run_biotech_pipeline()