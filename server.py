"""GridShock API: live geopolitical early-warning for GB energy supply.

Run with:  python server.py   (then open http://localhost:8000)
"""
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import gemma
import model
import news

ROOT = Path(__file__).parent
ASSETS = pd.read_csv(ROOT / "assets.csv")
SAVED_THREATS = ROOT / "saved_threats.json"  # last good Gemma output, for offline demo
SAVED_SCENARIOS = ROOT / "saved_scenarios.json"  # scenario text -> Gemma's shock, so presets are instant and work offline
CACHE_TTL = 600  # 10 minutes

app = FastAPI(title="GridShock")
_cache = {}           # key -> (timestamp, value)
_threats_by_id = {}   # threat id -> threat dict (for /api/briefing)
SAVED_BRIEFINGS = ROOT / "saved_briefings.json"
_briefings = json.loads(SAVED_BRIEFINGS.read_text()) if SAVED_BRIEFINGS.exists() else {}  # id -> text
_lock = threading.Lock()
_scenarios = json.loads(SAVED_SCENARIOS.read_text()) if SAVED_SCENARIOS.exists() else {}
_scenario_lock = threading.Lock()

PRESETS = [
    {"label": "Nyhamna drone strike", "text": "Drone strike shuts Nyhamna gas plant in Norway for two weeks"},
    {"label": "IFA cable cut", "text": "Ship anchor cuts the IFA power cable to France for a month"},
    {"label": "Taiwan Strait blockade", "text": "China blockades the Taiwan Strait for 30 days"},
    {"label": "Ras Tanura strike", "text": "Houthi missiles hit Saudi Arabia's Ras Tanura oil terminal"},
]


def cached(key, fn, ttl=CACHE_TTL):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


def _clean(v):
    """pandas NaN -> None so the JSON is valid."""
    return None if isinstance(v, float) and v != v else v


def asset_info(asset_id):
    row = ASSETS[ASSETS["id"] == asset_id]
    if not len(row):
        return None
    return {k: _clean(v) for k, v in row.iloc[0].to_dict().items()}


def get_mix():
    mix, live = cached("mix", model.live_power_mix)
    return mix, live


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _event_point(shock, asset):
    """Where the event happens. Falls back to the asset if Gemma's coordinates are missing or invalid."""
    lat, lon = _num(shock.get("event_lat")), _num(shock.get("event_lon"))
    if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180 and (lat, lon) != (0, 0):
        return {"name": shock.get("event_location") or "Event location", "lat": lat, "lon": lon}
    if asset:
        return {"name": shock.get("event_location") or asset["name"], "lat": asset["lat"], "lon": asset["lon"]}
    return None


def build_threat(headline, shock, kind, tid=None):
    """kind is one of "live" (news feed), "reported" (uploaded image), "scenario" (hypothetical)."""
    shock = dict(shock)
    if shock.get("asset_id") not in set(ASSETS["id"]):
        shock["asset_id"] = "other"
    # Gemma doesn't always respect the bounds in the prompt, so enforce them here
    shock["duration_days"] = max(1, min(model.MAX_DURATION_DAYS, _num(shock.get("duration_days")) or 7))
    shock["lost_fraction"] = max(0.05, min(1.0, _num(shock.get("lost_fraction")) or 0.5))
    asset = asset_info(shock["asset_id"])
    return {
        "id": tid or hashlib.md5(headline["title"].encode()).hexdigest()[:10],
        "kind": kind,
        "title": headline["title"],
        "link": headline.get("link", ""),
        "source": headline.get("source", ""),
        "date": headline.get("date", ""),
        "shock": shock,
        "impact": model.compute_impact(shock, ASSETS, get_mix()[0]),
        "asset": asset,
        "event": _event_point(shock, asset),
    }


def rebuild(t, kind="live"):
    """Re-run the Python model on a saved threat so saved data always uses the current maths."""
    new = build_threat(t, t["shock"], t.get("kind", kind), t["id"])
    return {**t, **{k: new[k] for k in ("kind", "shock", "impact", "asset", "event")}}


def dedupe(threats):
    """Merge headlines about the same asset into one card, keeping the highest-risk one."""
    best = {}
    for t in sorted(threats, key=lambda t: t["impact"]["risk_score"], reverse=True):
        aid = t["shock"].get("asset_id", "other")
        key = t["id"] if aid == "other" else aid  # unmapped events can't be matched, keep separate
        if key in best:
            best[key]["similar"].append({"title": t["title"], "link": t.get("link", "")})
        else:
            best[key] = {**t, "similar": []}
    return list(best.values())


def _package(threats, headlines, energy, mode, errors):
    flagged = {t["title"] for t in threats}
    threats = dedupe(threats)
    # energy headlines first, then general news, de-duped
    titles = list(dict.fromkeys(h["title"] for h in energy + headlines))[:50]
    for t in threats:
        _threats_by_id[t["id"]] = t
    return {
        "mode": mode,
        "threats": threats,
        "ticker": [{"title": t, "flagged": t in flagged} for t in titles],
        "screened": len(energy),
        "total_headlines": len(headlines),
        "errors": errors,
        "updated": time.strftime("%H:%M:%S"),
    }


def saved_result(mode="demo", errors=()):
    """Last saved headlines + Gemma analysis. Works fully offline."""
    headlines = news.load_saved()
    threats = [rebuild(t) for t in json.loads(SAVED_THREATS.read_text())] if SAVED_THREATS.exists() else []
    return _package(threats, headlines, news.filter_energy(headlines), mode, list(errors))


def compute_live():
    headlines, errors = news.fetch_live()
    if not headlines:
        return saved_result("demo", errors + ["Live feeds unavailable, showing saved analysis"])
    news.save_headlines(headlines)
    energy = news.filter_energy(headlines)
    try:
        shocks = gemma.screen_headlines(energy, ASSETS)
    except Exception as e:
        return saved_result("demo", errors + [f"Gemma unavailable ({str(e)[:80]}), showing saved analysis"])
    threats = [build_threat(energy[int(s["i"])], s, "live") for s in shocks if 0 <= int(s["i"]) < len(energy)]
    SAVED_THREATS.write_text(json.dumps(threats, indent=2))
    return _package(threats, headlines, energy, "live", errors)


_live = {"result": None, "at": 0.0, "running": False}


def _refresh_live():
    try:
        result = compute_live()
    except Exception as e:
        result = saved_result("demo", [f"Analysis failed: {e}"])
    with _lock:
        _live.update(result=result, at=time.time(), running=False)


# ---------------- endpoints ----------------
@app.get("/api/threats")
def threats(demo: bool = False, refresh: bool = False):
    """Returns instantly. Live analysis runs in a background thread; the page polls
    while "analysing" is true and shows the last saved analysis meanwhile."""
    if demo:
        return {**saved_result("demo"), "analysing": False}
    with _lock:
        stale = refresh or time.time() - _live["at"] > CACHE_TTL
        if stale and not _live["running"]:
            _live["running"] = True
            threading.Thread(target=_refresh_live, daemon=True).start()
        result = _live["result"] or saved_result("demo")
        return {**result, "analysing": _live["running"]}


@app.get("/api/assets")
def assets():
    return [{k: _clean(v) for k, v in r.items()} for r in ASSETS.to_dict("records")]


@app.get("/api/powermix")
def powermix():
    mix, live = get_mix()
    return {"mix": mix, "live": live}


@app.get("/api/briefing")
def briefing(id: str):
    """Returns the briefing, or 202 {"status": "pending"} while Gemma writes it in the
    background. Polling keeps each request short, so slow Gemma calls never hit a browser timeout."""
    t = _threats_by_id.get(id) or _restore_scenario(id)
    if not t:
        raise HTTPException(404, "This event is no longer on the server (it was restarted). Run the scenario or upload the image again.")
    with _brief_lock:
        job = _brief_jobs.get(id)
        if job and job["state"] == "running":
            return JSONResponse({"id": id, "status": "pending"}, status_code=202)
        if job and job["state"] == "error":
            del _brief_jobs[id]  # the next request starts a fresh attempt
            raise HTTPException(503, f"Gemma 4 could not write the briefing: {job['error']}")
    try:
        cached_text = _briefings.get(_briefing_key(t))
    except Exception:
        cached_text = None
    if cached_text:
        return {"id": id, "briefing": cached_text}
    with _brief_lock:
        _brief_jobs[id] = {"state": "running"}
    threading.Thread(target=_brief_job, args=(id, t), daemon=True).start()
    return JSONResponse({"id": id, "status": "pending"}, status_code=202)


_brief_jobs = {}  # threat id -> {"state": "running"|"error", "error": str}
_brief_lock = threading.Lock()


def _brief_job(id, t):
    try:
        get_briefing(t)
        with _brief_lock:
            _brief_jobs.pop(id, None)
    except Exception as e:
        with _brief_lock:
            _brief_jobs[id] = {"state": "error", "error": str(e)[:300]}


def _restore_scenario(id):
    """Scenario ids are derived from their text, so they can be rebuilt from the saved cache after a restart."""
    if not id.startswith("scn-"):
        return None
    for key in list(_scenarios):
        if "scn-" + hashlib.md5(key.encode()).hexdigest()[:10] == id:
            return run_scenario(key)
    return None


def event_type(t):
    if t["kind"] == "reported":
        return "reported_confirmed" if t.get("corroboration", {}).get("status") == "confirmed" else "reported"
    return t["kind"]


def _briefing_key(t):
    # includes the event type and the model numbers, so a briefing is never reused for different figures
    return f"{t['id']}:{event_type(t)}:{t['impact']['risk_score']}:{t['impact']['prices'][0]['central']}"


def get_briefing(t):
    key = _briefing_key(t)
    if key not in _briefings:
        _briefings[key] = gemma.write_briefing(t["title"], t["shock"], t["impact"], event_type(t))
        SAVED_BRIEFINGS.write_text(json.dumps(_briefings, indent=2))
    return _briefings[key]


class Scenario(BaseModel):
    text: str


def _scenario_key(text):
    return " ".join(text.lower().split())


def run_scenario(text, stage=lambda s: None):
    """Hypothetical what-if. Gemma's extraction is cached per text (and saved to disk),
    so presets are instant, repeatable and work offline."""
    key = _scenario_key(text)
    with _scenario_lock:
        shock = _scenarios.get(key)
    if shock is None:
        stage("reading")
        shock = gemma.extract_scenario(text, ASSETS)
        if not shock:
            return None
        with _scenario_lock:
            _scenarios[key] = shock
            SAVED_SCENARIOS.write_text(json.dumps(_scenarios, indent=2))
    stage("modelling")
    t = build_threat({"title": text, "source": "What-if scenario"}, shock, "scenario",
                     "scn-" + hashlib.md5(key.encode()).hexdigest()[:10])
    _threats_by_id[t["id"]] = t
    return t


@app.get("/api/presets")
def presets():
    return PRESETS


# ---- background jobs: slow Gemma work runs in a thread and the page polls /api/job,
# so no single request lasts long enough for a browser (Safari: ~60 s) to kill it ----
class JobError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


_jobs = {}  # job id -> {"state": "running"|"done"|"error", "stage", "result", "status", "detail", "at"}
_jobs_lock = threading.Lock()
JOB_TTL = 30 * 60


def start_job(work):
    """Run work(stage) in a thread; return 202 with the job id straight away."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        for old in [k for k, j in _jobs.items() if time.time() - j["at"] > JOB_TTL]:
            del _jobs[old]
        _jobs[job_id] = {"state": "running", "stage": "queued", "at": time.time()}

    def stage(name):
        with _jobs_lock:
            _jobs[job_id]["stage"] = name

    def run():
        try:
            update = {"state": "done", "result": work(stage)}
        except JobError as e:
            update = {"state": "error", "status": e.status, "detail": e.detail}
        except Exception as e:
            update = {"state": "error", "status": 503, "detail": f"Gemma 4 unavailable: {str(e)[:300]}"}
        with _jobs_lock:
            _jobs[job_id].update(update)

    threading.Thread(target=run, daemon=True).start()
    return JSONResponse({"job": job_id, "status": "pending", "stage": "queued"}, status_code=202)


@app.get("/api/job")
def job(id: str):
    with _jobs_lock:
        j = dict(_jobs.get(id) or {})
    if not j:
        raise HTTPException(404, "This request is no longer on the server (it may have restarted). Please run it again.")
    if j["state"] == "running":
        return JSONResponse({"job": id, "status": "pending", "stage": j["stage"]}, status_code=202)
    if j["state"] == "error":
        raise HTTPException(j["status"], j["detail"])
    return j["result"]


@app.post("/api/scenario")
def scenario(body: Scenario):
    """Cached scenarios (the presets, or anything run before) return instantly;
    new ones return 202 with a job id to poll."""
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Describe a scenario")
    with _scenario_lock:
        cached = _scenario_key(text) in _scenarios
    if cached:
        return run_scenario(text)

    def work(stage):
        t = run_scenario(text, stage)
        if not t:
            raise JobError(422, "Gemma 4 judged this scenario to have no effect on GB energy supply.")
        return t
    return start_job(work)


def warm_presets():
    """Run each preset (and its briefing) once in the background so the demo is instant."""
    for p in PRESETS:
        try:
            t = run_scenario(p["text"])
            if t:
                get_briefing(t)
        except Exception as e:
            print(f"Preset warm-up failed for {p['label']}: {e}")


MAX_IMAGE_BYTES = 8 * 1024 * 1024


@app.post("/api/scenario-image")
async def scenario_image(image: UploadFile = File(...), text: str = Form("")):
    """Multimodal what-if: Gemma 4 looks at an image (plus optional text) and runs the same pipeline."""
    mime = image.content_type or ""
    if not mime.startswith("image/"):
        raise HTTPException(400, "Please upload an image (JPEG, PNG, WebP…)")
    data = await image.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 8 MB)")
    return start_job(lambda stage: image_job(data, mime, text.strip(), stage))


def image_job(data, mime, text, stage):
    stage("reading_image")
    seen = gemma.analyse_image(data, mime, text, ASSETS)
    description = str(seen.get("description", "")).strip()
    if not seen.get("relevant"):
        raise JobError(422, f"Gemma 4 sees: {description.rstrip('.') or 'nothing recognisable'}. No effect on GB energy supply.")
    stage("modelling")
    shock = {k: seen.get(k) for k in ("event_location", "event_lat", "event_lon", "asset_id", "fuel",
                                       "lost_fraction", "duration_days", "confidence", "summary")}
    shock["seen"] = description  # lets the briefing refer to what was in the image
    t = build_threat({"title": text or shock.get("summary") or description, "source": "Uploaded image"}, shock,
                     "reported", "rep-" + hashlib.md5(data + text.encode()).hexdigest()[:10])
    t["vision"] = {"description": description}
    stage("checking")
    t["corroboration"] = corroborate(t)
    _threats_by_id[t["id"]] = t
    return t


def corroborate(t):
    """Ask Gemma whether a current live headline describes the same event as the image."""
    headlines = news.filter_energy(news.load_saved(), max_items=30)
    try:
        i = gemma.match_live(t["shock"].get("summary") or t["title"], t["shock"].get("event_location"), headlines,
                             t.get("vision", {}).get("description", ""))
    except Exception:
        i = None
    if i is None:
        return {"status": "unverified"}
    h = headlines[i]
    return {"status": "confirmed", "headline": {"title": h["title"], "link": h.get("link", ""), "source": h.get("source", "")}}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    print("GridShock running at http://localhost:8000")
    threading.Thread(target=warm_presets, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
