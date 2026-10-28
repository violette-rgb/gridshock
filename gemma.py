"""Gemma 4 does the reasoning: screen headlines, extract shocks, write briefings."""
import json
import os
import re
import time

from pathlib import Path

from google import genai
from google.genai import types

# load GEMINI_API_KEY from .env (kept out of GitHub by .gitignore)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

MODELS = ["models/gemma-4-26b-a4b-it", "models/gemma-4-31b-it"]  # fast first, big as backup
_client = None


def client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("No GEMINI_API_KEY found. Put it in the .env file.")
        # no hidden SDK retries: on a server error we want to fall back to the other model fast
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=90_000, retry_options=types.HttpRetryOptions(attempts=1)))
    return _client


def ask(prompt, temperature=0.2):
    """Call Gemma, retrying and falling back to the other model on server errors."""
    last_error = None
    thinking = types.ThinkingConfig(thinking_level="minimal")  # ~10x faster than default thinking
    for attempt in range(8):  # the API often returns instant 500s under load
        model = MODELS[attempt % len(MODELS)]
        try:
            r = client().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature, thinking_config=thinking),
            )
            if r.text:
                return r.text
            last_error = "empty reply"
        except Exception as e:
            last_error = e
            if "INVALID_ARGUMENT" in str(e):
                thinking = None  # model doesn't accept the thinking setting; retry without it
        time.sleep(1)
    raise RuntimeError(f"Gemma failed after retries: {last_error}")


def _extract_json(text):
    """Pull the first JSON object or array out of Gemma's reply."""
    text = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in reply: {text[:200]}")
    return json.loads(match.group(1))


def _asset_list(assets_df):
    return "\n".join(
        f"- {r.id}: {r.name} ({r.country}, {r.fuel}, at {r.lat:.2f}, {r.lon:.2f})" for r in assets_df.itertuples()
    )


SHOCK_FIELDS = """- "event_location": short place name where the event itself happens (e.g. "Ras Tanura, Saudi Arabia")
- "event_lat", "event_lon": approximate coordinates of that place, as numbers. If the
  event happens AT a listed asset, use that asset's coordinates from the list
- "asset_id": the best matching asset id from the list, or "other"
- "fuel": "gas", "power" or "oil"
- "lost_fraction": fraction of THAT ASSET'S OWN capacity or throughput knocked out,
  0.05 to 1.0. Do NOT multiply by the UK's share; that is done later in Python.
  A shutdown, cut cable or closure of the asset is 1.0; a partial outage or damage
  0.2-0.6; a diversion affecting some ships 0.1-0.3. Never 0 for a relevant event
- "duration_days": best estimate of how long the disruption lasts, a whole number
  from 1 to 90 (use 90 for open-ended disruptions)
- "confidence": "low", "medium" or "high"
- "summary": one short sentence on what happened"""

EXTRACT_RULES = f"""For each item, decide if it describes an event that could reduce
energy supply to Great Britain (gas, power or oil). Ignore opinion pieces,
prices-only stories, and events with no physical supply effect.

For each RELEVANT item return:
- "i": the item number
{SHOCK_FIELDS}

Be realistic and conservative. Use known facts (e.g. Russian gas transit
through Ukraine to Europe ended in January 2025, so a Ukrainian pipeline
event has little direct effect on UK supply).

Reply with ONLY a JSON array. Return [] if nothing is relevant."""


def screen_headlines(headlines, assets_df):
    """One batched Gemma call for many headlines. Returns list of shock dicts."""
    if not headlines:
        return []
    items = "\n".join(f"{i}. {h['title']}" for i, h in enumerate(headlines))
    prompt = f"""You are an energy security analyst for Great Britain.

Known infrastructure assets:
{_asset_list(assets_df)}

{EXTRACT_RULES}

Items:
{items}"""
    shocks = _extract_json(ask(prompt))
    if isinstance(shocks, dict):
        shocks = [shocks]
    return [s for s in shocks if isinstance(s, dict) and "i" in s]


def _one_object(result):
    if isinstance(result, list):
        result = result[0] if result else {}
    return result if isinstance(result, dict) else {}


def extract_scenario(text, assets_df):
    """Turn a user's hypothetical 'what if' into one structured shock (None if irrelevant)."""
    prompt = f"""You are an energy security analyst for Great Britain.
A user is running a HYPOTHETICAL what-if scenario (it has not happened):
"{text}"

Known infrastructure assets:
{_asset_list(assets_df)}

Assume the scenario happens exactly as described. Reply with ONLY one JSON object:
- "relevant": true if it would reduce energy supply to GB or move GB energy prices, else false
If relevant, also:
{SHOCK_FIELDS}"""
    shock = _one_object(_extract_json(ask(prompt)))
    return shock if shock.get("relevant") else None


def analyse_image(image_bytes, mime_type, text, assets_df):
    """Multimodal: Gemma looks at a photo/screenshot (plus optional text) and returns
    {"description", "relevant", ...shock fields}."""
    prompt = f"""You are an energy security analyst for Great Britain. Look at this image
(a photo, satellite image, map, chart or news screenshot).
{f'The user adds this context: "{text}". It may be a single keyword (who, where or what). Combine it with what you see in the image, but do not invent details that neither supports.' if text else ""}

Known infrastructure assets:
{_asset_list(assets_df)}

Reply with ONLY one JSON object:
- "description": one or two plain sentences on what you actually see in the image
- "relevant": true if it shows or reports an event that could reduce energy supply to GB, else false
If relevant, also:
{SHOCK_FIELDS}
Use lower confidence if the image is ambiguous or unverifiable. Do not invent details you cannot see."""
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    return _one_object(_extract_json(ask([part, prompt])))


def match_live(summary, location, headlines, seen=""):
    """Does any current live headline describe the same event? Returns its index or None."""
    if not headlines:
        return None
    items = "\n".join(f"{i}. {h['title']}" for i, h in enumerate(headlines))
    prompt = f"""An uploaded image reports this event: "{summary}"
Location: {location or "unknown"}
{f'What the image shows: "{seen}"' if seen else ""}

Current live news headlines:
{items}

Is any headline about the SAME incident: the same kind of event (e.g. an attack, outage,
blockade) in the same country or region, happening now? A headline can be vaguer than the
image (e.g. "Saudi refinery attack" matches a drone strike on a Saudi Aramco facility).
Price-only stories, company results and unrelated background stories do not count.
Reply with ONLY one JSON object: {{"match": <best matching headline number, or -1 if none>}}"""
    i = _one_object(_extract_json(ask(prompt))).get("match", -1)
    try:
        i = int(i)
    except (TypeError, ValueError):
        return None
    return i if 0 <= i < len(headlines) else None


EVENT_TYPE_OPENERS = {
    "live": "Live event:",
    "reported": "Reported event (unverified):",
    "reported_confirmed": "Reported event, confirmed by live news:",
    "scenario": "Hypothetical scenario:",
}
_PCT = re.compile(r"[+-−]?\d+(?:\.\d+)?\s?%")


def _norm_pct(s):
    return s.replace("−", "-").replace(" ", "").lstrip("+")


def _numbers_block(impact):
    lines = [f"- {impact['supply_label']}: {impact['supply_loss_pct']}%"]
    for p in impact["prices"]:
        lines.append(f"- {p['label']}: {p['central']:+.1f}% (range {p['low']:+.1f}% to {p['high']:+.1f}%)")
    lines.append(f"- Duration: {impact['duration_days']:.0f} days")
    lines.append(f"- Risk score: {impact['risk_score']}/100 ({impact['risk_level']})")
    return "\n".join(lines)


def _allowed_pcts(impact):
    allowed = {_norm_pct(f"{impact['supply_loss_pct']}%")}
    for p in impact["prices"]:
        for k in ("low", "central", "high"):
            allowed |= {_norm_pct(f"{p[k]:+.1f}%"), _norm_pct(f"{p[k]:.1f}%")}
    return allowed


def write_briefing(headline, shock, impact, event_type):
    """Trader briefing that quotes ONLY the model's numbers, opening with the event type.
    Any percentage not in the model output is rejected; after two tries we fall back
    to a plain template so a made-up number never reaches the screen."""
    opener = EVENT_TYPE_OPENERS[event_type]
    numbers = _numbers_block(impact)
    allowed = _allowed_pcts(impact) | {_norm_pct(f"{float(shock.get('lost_fraction') or 0) * 100:.0f}%")}
    prompt = f"""You are briefing an energy trader and a grid operator in Great Britain.

Event: {headline}
Event type: {opener[:-1]}
Gemma's extraction: {json.dumps({k: v for k, v in shock.items() if k != "i"})}

Model output (Python, simple stated assumptions). These are the ONLY numbers you may use:
{numbers}

Write a briefing of at most 90 words, plain English, no headings, no hype:
- The first sentence MUST start with exactly "{opener}"
- Say what happened and why it matters for GB supply.
- Quote the price impacts EXACTLY as written above, central value AND range,
  e.g. "+30.0% (range +13.5% to +45.0%)". Do not round, convert or invent any other
  percentage or price figure.
- If you think the model understates or overstates the risk, say so in words only, no numbers.
- End with two things to watch next."""
    for _ in range(2):
        text = ask(prompt, temperature=0.3).strip()
        bad = [m for m in _PCT.findall(text) if _norm_pct(m) not in allowed]
        if not bad:
            return text if text.startswith(opener) else f"{opener} {text}"
        prompt += f"\n\nYour last draft used numbers that are not in the model output ({', '.join(bad)}). Use only the numbers listed."
    p = impact["prices"]
    return (f"{opener} {shock.get('summary') or headline} "
            f"Model estimate: {p[0]['label'].lower()} {p[0]['central']:+.1f}% (range {p[0]['low']:+.1f}% to {p[0]['high']:+.1f}%), "
            f"{p[1]['label'].lower()} {p[1]['central']:+.1f}% (range {p[1]['low']:+.1f}% to {p[1]['high']:+.1f}%) "
            f"over about {impact['duration_days']:.0f} days. Watch for official outage updates and forward price moves.")
