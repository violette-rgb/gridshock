"""Pure Python impact model. No AI here: simple, stated assumptions."""
import requests

# ---- Stated assumptions (say these out loud in the pitch) ----
GB_DEMAND_GW = 30.0          # typical average GB electricity demand
GAS_PRICE_ELASTICITY = 2.0   # 1% of UK gas supply lost -> ~2% gas price rise (short run)
POWER_PRICE_ELASTICITY = 3.0 # 1% of GB power supply lost -> ~3% power price rise
GAS_TO_POWER_PASS = 0.7      # gas usually sets the power price, so most of it passes through
MAX_DURATION_DAYS = 90       # cap on how long any single disruption is assumed to last
POWER_TO_GAS_ELASTICITY = 0.5 # 1% of GB power lost -> ~0.5% gas price rise (extra gas burn fills the gap)

# Oil: priced globally, so it uses the asset's share of WORLD oil supply
OIL_PRICE_ELASTICITY = 3.0   # 1% of world oil supply lost -> ~3% Brent rise (short run)
FUEL_PASS_THROUGH = 0.4      # UK pump prices move ~40% as much as Brent (fuel duty and VAT cushion it)

# Uncertainty band around every price impact
RANGE_LOW = {"lost": 0.6, "elasticity": 0.75}   # milder outage, calmer market
RANGE_HIGH = {"lost": 1.0, "elasticity": 1.5}   # full stated outage, panicky market

DISCLAIMER = "Indicative estimate. Stated assumptions, not a market forecast."

# Rough split of UK gas supply by source (%), used for the before/after chart
GAS_SOURCES = {"UK North Sea": 32, "Norway": 35, "LNG": 30, "Continent": 3}

# Fallback power mix (%) if the live API is unreachable
TYPICAL_MIX = {"gas": 30, "wind": 28, "nuclear": 14, "imports": 12,
               "biomass": 6, "solar": 4, "hydro": 2, "coal": 0, "other": 4}

CONF_WEIGHT = {"low": 0.5, "medium": 0.8, "high": 1.0}


def live_power_mix():
    """Current GB generation mix from the National Grid ESO Carbon Intensity API."""
    try:
        r = requests.get("https://api.carbonintensity.org.uk/generation", timeout=8)
        r.raise_for_status()
        mix = {m["fuel"]: m["perc"] for m in r.json()["data"]["generationmix"]}
        return mix, True
    except Exception:
        return dict(TYPICAL_MIX), False


def _duration_factor(days):
    # short outages are cushioned by storage/linepack; long ones bite harder
    if days < 2:
        return 0.5
    if days <= 14:
        return 1.0
    return 1.3


def _pct(v):
    return f"{v:+.1f}%"


def _asset_value(asset, col, default):
    if len(asset) and asset[col].notna().iloc[0]:
        return float(asset[col].iloc[0])
    return default


def _moves(fuel, asset, lost, dfac, gas_mix_pct, k):
    """Supply loss and price moves for one (lost fraction, elasticity multiplier k) case."""
    out = {"supply": 0.0, "gas": 0.0, "power": 0.0, "brent": 0.0, "uk_fuel": 0.0}
    if fuel == "gas":
        out["supply"] = _asset_value(asset, "uk_share_pct", 5.0) * lost
        out["gas"] = out["supply"] * GAS_PRICE_ELASTICITY * k * dfac
        out["power"] = out["gas"] * GAS_TO_POWER_PASS * gas_mix_pct / 30
    elif fuel == "power":
        out["supply"] = _asset_value(asset, "capacity_gw", 1.0) * lost / GB_DEMAND_GW * 100
        out["power"] = out["supply"] * POWER_PRICE_ELASTICITY * k * dfac
        out["gas"] = out["supply"] * POWER_TO_GAS_ELASTICITY * k * dfac
    else:  # oil: share of world supply drives Brent, Brent drives UK pump prices
        out["supply"] = _asset_value(asset, "global_share_pct", 1.0) * lost
        out["brent"] = out["supply"] * OIL_PRICE_ELASTICITY * k * dfac
        out["uk_fuel"] = out["brent"] * FUEL_PASS_THROUGH
    return out


def compute_impact(shock, assets_df, power_mix):
    """Return supply loss, price moves with low/central/high ranges, risk score,
    and the plugged-in formulas for the "How it's calculated" panel."""
    asset = assets_df[assets_df["id"] == shock.get("asset_id")]
    fuel = shock.get("fuel") if shock.get("fuel") in ("gas", "power", "oil") else "gas"
    lost = max(0.05, min(1.0, float(shock.get("lost_fraction") or 0.5)))  # relevant events always lose some flow
    days = max(1.0, min(MAX_DURATION_DAYS, float(shock.get("duration_days") or 7)))
    dfac = _duration_factor(days)
    gas_mix = float(power_mix.get("gas", 30))

    central = _moves(fuel, asset, lost, dfac, gas_mix, 1.0)
    low = _moves(fuel, asset, lost * RANGE_LOW["lost"], dfac, gas_mix, RANGE_LOW["elasticity"])
    high = _moves(fuel, asset, min(1.0, lost * RANGE_HIGH["lost"]), dfac, gas_mix, RANGE_HIGH["elasticity"])
    ranges = {k: {"low": round(low[k], 1), "central": round(central[k], 1), "high": round(high[k], 1)}
              for k in ("gas", "power", "brent", "uk_fuel")}

    conf_label = shock.get("confidence") if shock.get("confidence") in CONF_WEIGHT else "medium"
    conf = CONF_WEIGHT[conf_label]
    main = max(central["gas"], central["power"], central["brent"])
    score = min(100, round((main * 2 + min(days, 30)) * conf))
    level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"

    shown = ("brent", "uk_fuel") if fuel == "oil" else ("gas", "power")
    labels = {"gas": "Gas price", "power": "Power price", "brent": "Brent crude", "uk_fuel": "UK fuel"}
    supply_label = {"gas": "UK gas supply lost", "power": "GB power supply lost", "oil": "World oil supply lost"}[fuel]

    def rng(k):
        r = ranges[k]
        return f"{_pct(r['central'])} (range {_pct(r['low'])} to {_pct(r['high'])})"

    # ---- the formulas with this event's numbers plugged in ----
    name = asset["name"].iloc[0] if len(asset) else "an unmapped asset (default share)"
    calc = [{"by": "Gemma 4", "label": "Extraction",
             "formula": f"Event at {shock.get('event_location') or 'unknown location'} hits {name} · fuel {fuel} · "
                        f"{lost * 100:.0f}% of flow lost · {days:.0f} days · {conf_label} confidence"}]
    if fuel == "gas":
        share = _asset_value(asset, "uk_share_pct", 5.0)
        calc += [
            {"by": "Python", "label": "Supply lost", "formula": f"{share:g}% UK share × {lost * 100:.0f}% lost = {central['supply']:.1f}%"},
            {"by": "Python", "label": "Gas price", "formula": f"{central['supply']:.1f} × {GAS_PRICE_ELASTICITY} elasticity × {dfac} duration factor = {rng('gas')}"},
            {"by": "Python", "label": "Power price", "formula": f"{central['gas']:.1f} gas move × {GAS_TO_POWER_PASS} pass-through × {gas_mix:.0f}/30 gas share of GB mix = {rng('power')}"},
        ]
    elif fuel == "power":
        cap = _asset_value(asset, "capacity_gw", 1.0)
        calc += [
            {"by": "Python", "label": "Supply lost", "formula": f"{cap:g} GW × {lost * 100:.0f}% lost = {cap * lost:.2f} GW ÷ {GB_DEMAND_GW:g} GW demand = {central['supply']:.1f}%"},
            {"by": "Python", "label": "Power price", "formula": f"{central['supply']:.1f} × {POWER_PRICE_ELASTICITY} elasticity × {dfac} duration factor = {rng('power')}"},
            {"by": "Python", "label": "Gas price", "formula": f"{central['supply']:.1f} × {POWER_TO_GAS_ELASTICITY} extra gas burn × {dfac} duration factor = {rng('gas')}"},
        ]
    else:
        gshare = _asset_value(asset, "global_share_pct", 1.0)
        calc += [
            {"by": "Python", "label": "World supply lost", "formula": f"{gshare:g}% of world oil × {lost * 100:.0f}% lost = {central['supply']:.2f}%"},
            {"by": "Python", "label": "Brent", "formula": f"{central['supply']:.2f} × {OIL_PRICE_ELASTICITY} elasticity × {dfac} duration factor = {rng('brent')}"},
            {"by": "Python", "label": "UK fuel", "formula": f"{central['brent']:.1f} Brent move × {FUEL_PASS_THROUGH} pump pass-through = {rng('uk_fuel')}"},
        ]
    calc += [
        {"by": "Python", "label": "Duration factor", "formula": f"{days:.0f} days → {dfac} (under 2 days 0.5 · 2–14 days 1.0 · over 14 days 1.3)"},
        {"by": "Python", "label": "Range", "formula": f"low = lost × {RANGE_LOW['lost']} and elasticity × {RANGE_LOW['elasticity']} · "
                                                     f"high = lost × {RANGE_HIGH['lost']} (max 100%) and elasticity × {RANGE_HIGH['elasticity']}"},
        {"by": "Python", "label": "Risk score", "formula": f"({main:.1f} largest price move × 2 + {min(days, 30):.0f} days, max 30) × {conf} {conf_label}-confidence weight = {score} / 100"},
    ]

    return {
        "fuel": fuel,
        "supply_loss_pct": round(central["supply"], 2 if fuel == "oil" else 1),
        "supply_label": supply_label,
        "gas_price_change_pct": ranges["gas"]["central"],
        "power_price_change_pct": ranges["power"]["central"],
        "brent_price_change_pct": ranges["brent"]["central"],
        "uk_fuel_price_change_pct": ranges["uk_fuel"]["central"],
        "ranges": ranges,
        "prices": [{"key": k, "label": labels[k], **ranges[k]} for k in shown],
        "duration_days": days,
        "duration_factor": dfac,
        "confidence_weight": conf,
        "risk_score": score,
        "risk_level": level,
        "calc": calc,
        "disclaimer": DISCLAIMER,
    }
