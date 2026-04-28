import re
import time
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

DOMAIN_BLOCKLIST = [
    "domains.atom.com", "sedo.com", "godaddy.com", "namecheap.com",
    "dan.com", "hugedomains.com", "afternic.com", "wix.com",
    "squarespace.com", "uniregistry.com", "brandpa.com",
]

SOCIAL_DOMAINS = [
    "linkedin.com", "twitter.com", "x.com", "facebook.com",
    "instagram.com", "youtube.com", "tiktok.com", "threads.net",
    "whatsapp.com", "api.whatsapp.com",
]

# Strip common legal suffixes before slug generation
_LEGAL_SUFFIXES = re.compile(r"\b(inc|corp|co|llc|ltd|gmbh|ag|sas|bv)\b\.?", re.I)

# Detect TLD already embedded in the company name, e.g. "IndustrialMind.ai"
_TLD_IN_NAME = re.compile(r"([a-z0-9\-]+)\.([a-z]{2,})", re.I)


def _create_slug_and_tld(company_name: str):
    """Return (slug, tld_or_None) derived from the company name."""
    name = _LEGAL_SUFFIXES.sub("", company_name.strip()).strip()
    match = _TLD_IN_NAME.search(name)
    if match:
        return match.group(1).lower(), f".{match.group(2).lower()}"
    slug = re.sub(r"[^a-z0-9]", "", name.lower())
    return slug, None


def normalize_domain(url: str):
    if not url:
        return None
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parsed = urlparse(url)
    base = parsed.netloc.lower().replace("www.", "")
    if any(b in base for b in DOMAIN_BLOCKLIST):
        return None
    return f"https://{base}"


def resolve_from_press_release(article_url: str):
    """Try to extract the company website directly from the source article."""
    try:
        resp = requests.get(article_url, headers=HEADERS, timeout=10)
        if resp.status_code >= 400:
            return None, 0.0

        soup = BeautifulSoup(resp.text, "html.parser")
        article_host = urlparse(article_url).netloc.lower().replace("www.", "")

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href.startswith("http") or "mailto:" in href:
                continue
            if any(b in href for b in DOMAIN_BLOCKLIST + SOCIAL_DOMAINS):
                continue

            clean = normalize_domain(href)
            if not clean:
                continue

            candidate_host = urlparse(clean).netloc.lower().replace("www.", "")
            if candidate_host == article_host:
                continue

            return clean, 0.92
    except Exception:
        pass
    return None, 0.0


def resolve_via_duckduckgo(company_name: str):
    """Search DuckDuckGo for the company's official website."""
    try:
        time.sleep(1.0)  # polite delay
        query = f"{company_name} official site"
        resp = requests.get(
            f"https://duckduckgo.com/html/?q={quote_plus(query)}",
            headers=HEADERS,
            timeout=10,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        link = soup.select_one("a.result__a")
        if not link:
            return None, 0.0

        href = link.get("href", "")
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            href = unquote(qs.get("uddg", [href])[0])

        if any(b in href for b in ["linkedin.com", "crunchbase.com"]):
            return None, 0.0

        return normalize_domain(href), 0.85
    except Exception:
        return None, 0.0


def resolve_via_guessing(company_name: str):
    """Last-resort: probe common TLD variants for the company slug."""
    slug, tld = _create_slug_and_tld(company_name)
    tlds = [tld] if tld else [".com", ".io", ".ai", ".co"]

    for ext in tlds:
        candidate = f"https://{slug}{ext}"
        try:
            resp = requests.head(candidate, headers=HEADERS, timeout=4, allow_redirects=True)
            final = resp.url.lower()
            if resp.status_code < 400 and not any(b in final for b in DOMAIN_BLOCKLIST):
                return normalize_domain(final), 0.60
        except Exception:
            continue
    return None, 0.0


def resolve_company_domain(company_name: str, article_url: str) -> dict:
    """
    Resolve a company's website domain using a tiered strategy:
    1. Extract from the press-release article directly.
    2. DuckDuckGo search.
    3. Slug-based guessing.
    """
    domain, conf = resolve_from_press_release(article_url)
    if domain:
        return {"domain": domain, "confidence": conf, "source": "press_release"}

    domain, conf = resolve_via_duckduckgo(company_name)
    if domain:
        return {"domain": domain, "confidence": conf, "source": "search"}

    print(f"⚠️  Search failed for '{company_name}', attempting slug guessing...")
    domain, conf = resolve_via_guessing(company_name)
    if domain:
        return {"domain": domain, "confidence": conf, "source": "guess"}

    return {"domain": None, "confidence": 0.0, "source": "failed"}
