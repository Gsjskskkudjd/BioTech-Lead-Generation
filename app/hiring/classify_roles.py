"""
Role classification helpers.

Provides lightweight keyword-based classification of job titles into
broad engineering/data/ML categories. Used to filter hiring signals
before storing or alerting.
"""

from __future__ import annotations

ROLE_CATEGORIES = {
    "engineering": {
        "software", "engineer", "developer", "backend", "frontend",
        "full stack", "full-stack", "platform", "devops", "sre",
        "infra", "infrastructure", "android", "ios", "mobile",
    },
    "data": {
        "data engineer", "data scientist", "analytics", "analyst",
        "bi ", "business intelligence", "etl", "pipeline",
    },
    "ml_ai": {
        "machine learning", "ml", "mle", "ai ", "deep learning",
        "nlp", "computer vision", "llm", "generative",
    },
}


def classify_role(title: str) -> str | None:
    """
    Return the best-matching category for a job title, or None if no match.

    Categories (in priority order): ml_ai > data > engineering
    """
    t = title.lower()
    for category in ("ml_ai", "data", "engineering"):
        if any(kw in t for kw in ROLE_CATEGORIES[category]):
            return category
    return None


def is_tech_role(title: str) -> bool:
    """Return True if the title matches any known tech category."""
    return classify_role(title) is not None
