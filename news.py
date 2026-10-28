"""Pull live headlines and keep only the ones that could threaten energy supply."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

FEEDS = [
    "https://news.google.com/rss/search?q=pipeline+OR+LNG+OR+%22undersea+cable%22+OR+refinery+OR+%22gas+plant%22+when:2d&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=energy+attack+OR+sabotage+OR+drone+OR+strike+energy+when:2d&hl=en-GB&gl=GB&ceid=GB:en",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
]

KEYWORDS = [
    "pipeline", "gas", "lng", "oil", "refinery", "cable", "interconnector",
    "drone", "sabotage", "attack", "strike", "sanction", "blockade", "outage",
    "explosion", "fire", "shutdown", "nuclear", "power plant", "grid", "hormuz",
    "red sea", "tanker", "norway", "north sea", "energy",
]

SAVED_FILE = Path(__file__).parent / "saved_headlines.json"


def _parse_rss(xml_text, source):
    items = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        date = (item.findtext("pubDate") or "").strip()
        if title:
            items.append({"title": title, "link": link, "date": date, "source": source})
    return items


def fetch_live(limit_per_feed=40):
    """Fetch headlines from all feeds. Returns (headlines, errors)."""
    headlines, errors = [], []
    for url in FEEDS:
        source = "Google News" if "google" in url else "BBC"
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "GridShock/1.0"})
            r.raise_for_status()
            headlines += _parse_rss(r.text, source)[:limit_per_feed]
        except Exception as e:  # keep going if one feed fails
            errors.append(f"{source}: {e}")
    # de-duplicate by title
    seen, unique = set(), []
    for h in headlines:
        key = h["title"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique, errors


def filter_energy(headlines, max_items=20):
    """Cheap keyword filter so only likely-relevant headlines go to Gemma."""
    # word boundaries so "ceasefire" doesn't match "fire" or "turmoil" match "oil"
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in KEYWORDS) + r")s?\b", re.IGNORECASE)
    hits = [h for h in headlines if pattern.search(h["title"])]
    return hits[:max_items]


def save_headlines(headlines):
    SAVED_FILE.write_text(json.dumps(headlines, indent=2))


def load_saved():
    if SAVED_FILE.exists():
        return json.loads(SAVED_FILE.read_text())
    return []
