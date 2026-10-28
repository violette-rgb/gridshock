"""GridShock: turns breaking news into energy risk for Great Britain.

Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import gemma
import model
import news

st.set_page_config(page_title="GridShock", page_icon="⚡", layout="wide")

ASSETS = pd.read_csv(Path(__file__).parent / "assets.csv")
RISK_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}


# ---------- cached data steps ----------
@st.cache_data(ttl=600, show_spinner=False)
def get_headlines(demo_mode):
    if demo_mode:
        return news.load_saved(), []
    return news.fetch_live()


@st.cache_data(ttl=600, show_spinner=False)
def get_mix():
    return model.live_power_mix()


@st.cache_data(ttl=600, show_spinner=False)
def screen(titles):
    heads = [{"title": t} for t in titles]
    return gemma.screen_headlines(heads, ASSETS)


@st.cache_data(ttl=3600, show_spinner=False)
def briefing(headline, shock_key, impact_key):
    import json
    return gemma.write_briefing(headline, json.loads(shock_key), json.loads(impact_key))


def build_events(headlines, mix):
    relevant = news.filter_energy(headlines)
    shocks = screen(tuple(h["title"] for h in relevant)) if relevant else []
    events = []
    for s in shocks:
        try:
            h = relevant[int(s["i"])]
        except (IndexError, ValueError, KeyError):
            continue
        impact = model.compute_impact(s, ASSETS, mix)
        events.append({"headline": h, "shock": s, "impact": impact})
    events.sort(key=lambda e: e["impact"]["risk_score"], reverse=True)
    return events, len(relevant)


def show_detail(title, shock, impact):
    import json
    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk", f"{RISK_ICON[impact['risk_level']]} {impact['risk_level']} · {impact['risk_score']}")
    c2.metric("Supply at risk", f"-{impact['supply_loss_pct']}%", help=f"Share of UK {impact['fuel']} supply")
    c3.metric("Gas price", f"+{impact['gas_price_change_pct']}%")
    c4.metric("Power price", f"+{impact['power_price_change_pct']}%")

    st.caption(
        f"Asset: **{shock.get('asset_id', 'other')}** · fuel: {shock.get('fuel')} · "
        f"capacity lost: {round(float(shock.get('lost_fraction', 0)) * 100)}% · "
        f"duration: ~{impact['duration_days']:g} days · Gemma confidence: {shock.get('confidence')}"
    )

    chart_key = "gas" if impact["fuel"] == "gas" else "power"
    data = impact["chart"][chart_key]
    df = pd.DataFrame({"Before": data["before"], "After shock": data["after"]})
    unit = "% of UK gas supply" if chart_key == "gas" else "GW of GB power"
    st.markdown(f"**{'UK gas supply by source' if chart_key == 'gas' else 'GB power mix'}** ({unit})")
    st.bar_chart(df, stack=False, height=260)

    with st.spinner("Gemma is writing the briefing…"):
        try:
            text = briefing(title, json.dumps(shock), json.dumps({k: v for k, v in impact.items() if k != "chart"}))
            st.info(f"**Gemma briefing**\n\n{text}")
        except Exception as e:
            st.warning(f"Briefing unavailable: {e}")


# ---------- layout ----------
st.title("⚡ GridShock")
st.caption("Live geopolitical early warning for Great Britain's energy supply · powered by Gemma 4")

with st.sidebar:
    st.header("Settings")
    demo_mode = st.toggle("Demo mode (use saved headlines)", value=False)
    if st.button("Refresh news"):
        st.cache_data.clear()
    st.divider()
    st.markdown(
        "**Model assumptions**\n"
        f"- GB demand ≈ {model.GB_DEMAND_GW:g} GW\n"
        f"- 1% gas supply lost → +{model.GAS_PRICE_ELASTICITY:g}% gas price\n"
        f"- 1% power supply lost → +{model.POWER_PRICE_ELASTICITY:g}% power price\n"
        f"- Gas → power pass-through {model.GAS_TO_POWER_PASS:g}\n\n"
        "Deliberately simple. Built to speed up analysts, not replace them."
    )

mix, live_mix = get_mix()
headlines, errors = get_headlines(demo_mode)
if errors and not demo_mode:
    st.sidebar.warning("Some feeds failed:\n" + "\n".join(errors))
if not demo_mode and headlines:
    if st.sidebar.button("Save these headlines for demo"):
        news.save_headlines(headlines)
        st.sidebar.success(f"Saved {len(headlines)} headlines")

st.caption(
    ("🟢 Live GB generation mix (Carbon Intensity API)" if live_mix else "⚪ Typical GB mix (live API unavailable)")
    + " · " + " · ".join(f"{k} {v:g}%" for k, v in sorted(mix.items(), key=lambda x: -x[1]) if v > 0)
)

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.subheader("Threat board")
    with st.spinner("Gemma is screening the news…"):
        try:
            events, n_checked = build_events(headlines, mix)
        except Exception as e:
            events, n_checked = [], 0
            st.error(f"Screening failed: {e}")
    st.caption(f"{len(headlines)} headlines read · {n_checked} matched energy keywords · {len(events)} flagged by Gemma")

    if not events:
        st.write("No supply threats flagged right now. Try a scenario on the right.")
    labels = [
        f"{RISK_ICON[e['impact']['risk_level']]} {e['impact']['risk_score']} · {e['headline']['title'][:90]}"
        for e in events
    ]
    choice = st.radio("Events", range(len(events)), format_func=lambda i: labels[i], label_visibility="collapsed") if events else None

with right:
    tab_event, tab_scenario = st.tabs(["Selected event", "What-if scenario"])

    with tab_event:
        if choice is not None:
            ev = events[choice]
            if ev["headline"].get("link"):
                st.markdown(f"[Read source]({ev['headline']['link']}) · {ev['headline'].get('source', '')}")
            show_detail(ev["headline"]["title"], ev["shock"], ev["impact"])
        else:
            st.write("Pick an event from the threat board.")

    with tab_scenario:
        text = st.text_input("Describe a scenario", placeholder="Drone strike shuts the Nyhamna gas plant in Norway for two weeks")
        if st.button("Run scenario", type="primary"):
            if not text.strip():
                st.error("Type a scenario first.")
            else:
                with st.spinner("Gemma is assessing the scenario…"):
                    try:
                        shock = gemma.extract_scenario(text, ASSETS)
                    except Exception as e:
                        shock = None
                        st.error(f"Gemma error: {e}")
                if shock:
                    show_detail(text, shock, model.compute_impact(shock, ASSETS, mix))
                elif shock is None:
                    st.warning("Gemma doesn't think this affects GB energy supply.")
