# ⚡ GridShock

**A live geopolitical early-warning globe for Great Britain's energy supply.**

GridShock reads breaking news (and photos or screenshots you drop in), spots
events that could threaten GB energy supply, traces the causal chain from the
event to the affected infrastructure to Britain, and estimates what it means for
gas, power, Brent and UK fuel prices — with an honest uncertainty range.

Built at the MLH Hacktoberfest Hack Day, Loughborough x DigiTech, 28 Oct 2026.

## Run it

```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env   # free key: https://aistudio.google.com
python server.py
```

Then open **http://localhost:8000**.

## Three event types, never mixed

| Type | Where it comes from | Badge | Arc on the globe |
| --- | --- | --- | --- |
| **Live** | Fetched automatically from Google News and BBC RSS | 🟢 `LIVE` + source, link, publish time | Solid red, broken |
| **Reported** | An image or screenshot you upload, drop or paste | 🔵 `REPORTED · UNVERIFIED`, or `REPORTED · CONFIRMED BY LIVE NEWS` (with the matching headline) once Gemma 4 finds a live headline about the same incident | Dotted red |
| **Scenario** | A typed what-if or one of the 4 preset chips | 🟠 `HYPOTHETICAL` + "Simulation. This event has not happened." | Dashed amber |

The threat list has a separate section for each type.

## How it works

1. **News** (`news.py`): pulls live headlines and keeps those matching
   energy-risk keywords (pipeline, LNG, cable, drone, sabotage, refinery, …).
2. **Gemma 4 extraction** (`gemma.py`): all filtered headlines go to Gemma 4 in
   **one batched call**. For each relevant event it returns `event_location`,
   `event_lat`, `event_lon`, `asset_id`, `fuel`, `lost_fraction` (share of the
   asset's own capacity knocked out), `duration_days` (max 90), `confidence` and
   `summary`. Uses `gemma-4-26b-a4b-it`, falling back to `gemma-4-31b-it`.
3. **Multimodal**: for an image, Gemma 4 first says what it sees ("Gemma 4
   sees: …"), extracts the same fields, then checks the current live headlines
   for the same incident.
4. **Impact model** (`model.py`): plain Python, no AI. Formulas below.
5. **Briefing**: Gemma 4 writes a trader briefing. It must open with the event
   type, and may quote only the model's numbers, exactly, with their range. Every
   percentage is checked against the model output; a draft with any other number
   is rejected and regenerated, with a plain template as a last resort. If Gemma
   thinks the model under- or overstates the risk, it says so in words only.
6. **Flow Globe** (`static/index.html`): animated energy flows from every asset
   into Britain. Selecting a threat shows the causal chain: a pulsing ring at the
   event location, a shockwave arc to the affected asset, that asset's flow
   breaking, and the shock travelling on to Britain. The camera frames the event
   and the UK together. The panel shows a **How it's calculated** section with
   every formula and this event's numbers plugged in, each step labelled
   **Gemma 4** (extraction) or **Python** (maths).

## Formulas

All numbers are **indicative estimates from stated assumptions, not a market forecast.**

**Gas events**
- Supply lost % = asset's UK share % × lost fraction
- Gas price % = supply lost × 2.0 elasticity × duration factor
- Power price % = gas price × 0.7 pass-through × (gas % of live GB mix ÷ 30)

**Power events**
- Supply lost % = capacity GW × lost fraction ÷ 30 GW GB demand
- Power price % = supply lost × 3.0 elasticity × duration factor
- Gas price % = supply lost × 0.5 (extra gas burn) × duration factor

**Oil events**
- World supply lost % = asset's share of world oil % × lost fraction
- Brent % = world supply lost × 3.0 elasticity × duration factor
- UK fuel % = Brent × 0.4 pass-through (fuel duty and VAT cushion pump prices)

**Duration factor**: under 2 days 0.5 (storage cushions it) · 2–14 days 1.0 · over 14 days 1.3

**Uncertainty range** for every price impact
- low = lost fraction × 0.6 and elasticity × 0.75
- central = as above
- high = lost fraction × 1.0 (capped at 100%) and elasticity × 1.5

Shown as e.g. `+30.0% (range +13.5% to +45.0%)` and as error bars on the chart.

**Risk score** (0–100) = (largest central price move × 2 + days, max 30) × confidence
weight (low 0.5 · medium 0.8 · high 1.0). 60+ is High, 30+ Medium.

## Assumptions (deliberately simple)

- GB demand ≈ 30 GW; rough UK gas supply split: UK North Sea 32%, Norway 35%, LNG 30%, Continent 3%
- Elasticities and pass-through rates above are round, short-run rules of thumb
- Disruptions are capped at 90 days
- Asset shares, capacities, world-oil shares and coordinates in `assets.csv` are rough public figures

It is a prototype to speed up analysts, not replace them.

## Demo presets and offline mode

The 4 preset chips under the "What if…" bar ("Nyhamna drone strike", "IFA
cable cut", "Taiwan Strait blockade", "Ras Tanura strike") are run once in the
background when the server starts. Gemma's extraction is cached in
`saved_scenarios.json` and briefings in `saved_briefings.json`, so presets are
instant, repeatable and work offline.

Every live run also saves its headlines (`saved_headlines.json`) and Gemma's
analysis (`saved_threats.json`). Tick **demo** in the header to replay them with
no network or API calls. If live feeds or Gemma fail, the globe falls back to
the saved analysis automatically.

## API

| Endpoint | What it does |
| --- | --- |
| `GET /api/threats?demo=false&refresh=false` | Live threats + ticker. Returns instantly; live analysis runs in the background (`analysing: true`) |
| `GET /api/briefing?id=…` | Gemma 4 briefing for any threat |
| `POST /api/scenario` `{"text": "…"}` | Hypothetical what-if (cached per text) |
| `GET /api/presets` | The 4 demo presets |
| `POST /api/scenario-image` (multipart: `image`, optional `text`) | Reported event from an image, with live-news corroboration |
| `GET /api/powermix` | Live GB generation mix (falls back to a typical mix) |
| `GET /api/assets` | Infrastructure assets with coordinates |

## Licence

MIT — see [LICENSE](LICENSE).
