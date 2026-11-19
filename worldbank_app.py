"""
World Bank dashboard — Streamlit front end for worldbank_api.py.

Run:  py -m streamlit run app.py
Needs: streamlit, pandas, plotly, requests
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from worldbank_api import (INDICATORS, WorldBankAPIError, fmt, get_countries,
                           get_indicator, get_profile, latest_values)

st.set_page_config(page_title="World Bank Explorer", page_icon="🌍", layout="wide")
LAST_YEAR = 2025

try:
    with st.spinner("Loading country list from the World Bank…"):
        countries = get_countries()
except WorldBankAPIError as e:
    st.error(str(e))
    st.stop()

name_to_iso = dict(zip(countries["country"], countries["iso3"]))

# ---------- Sidebar ----------
st.sidebar.header("Settings")
ind_name = st.sidebar.selectbox("Indicator", list(INDICATORS))
ind_code, kind = INDICATORS[ind_name]

defaults = [c for c in ["United States", "China", "India", "Germany", "Brazil"] if c in name_to_iso]
picked = st.sidebar.multiselect("Countries", countries["country"], default=defaults, max_selections=10)
years = st.sidebar.slider("Years", 1960, LAST_YEAR, (2000, LAST_YEAR))
log_scale = st.sidebar.toggle("Log scale", value=kind in ("usd", "count"),
                              help="Useful when countries differ by orders of magnitude")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

st.title("🌍 World Bank Explorer")
st.caption(f"**{ind_name}** · indicator code `{ind_code}`")

tab_trend, tab_map, tab_profile, tab_data = st.tabs(
    ["📈 Compare over time", "🗺️ World map", "🏳️ Country profile", "🗂️ Data"]
)

# ---------- Load the selected series ----------
series = pd.DataFrame()
if picked:
    try:
        with st.spinner("Fetching time series…"):
            series = get_indicator(ind_code, tuple(name_to_iso[c] for c in picked), *years)
    except WorldBankAPIError as e:
        st.error(str(e))

# ---------- Tab 1: compare over time ----------
with tab_trend:
    if not picked:
        st.info("Pick one or more countries in the sidebar.")
    elif series.empty:
        st.warning("No data for that combination. Try a wider year range or another indicator.")
    else:
        # Latest value per country + change since the first year in range
        latest = series.groupby("country").tail(1).set_index("country")
        first = series.groupby("country").head(1).set_index("country")
        cols = st.columns(min(len(latest), 5), border=True)
        for col, (country, row) in zip(cols, latest.head(5).iterrows()):
            start_val = first.loc[country, "value"]
            if kind == "pct":
                delta = f"{row['value'] - start_val:+.1f} pts since {first.loc[country, 'year']}"
            elif start_val:
                delta = f"{(row['value'] / start_val - 1):+.0%} since {first.loc[country, 'year']}"
            else:
                delta = None
            col.metric(f"{country} ({row['year']})", fmt(row["value"], kind), delta)

        fig = px.line(series, x="year", y="value", color="country", markers=True,
                      log_y=log_scale, labels={"value": ind_name, "year": ""})
        fig.update_layout(height=480, legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2: world map ----------
with tab_map:
    try:
        with st.spinner("Fetching latest values for every country…"):
            latest_all = latest_values(ind_code)
    except WorldBankAPIError as e:
        st.error(str(e))
        latest_all = pd.DataFrame()

    if latest_all.empty:
        st.warning("No recent data for this indicator.")
    else:
        latest_all["display"] = latest_all["value"].apply(lambda v: fmt(v, kind))
        use_log = log_scale and (latest_all["value"] > 0).all()
        latest_all["color"] = np.log10(latest_all["value"]) if use_log else latest_all["value"]

        fig = px.choropleth(
            latest_all, locations="iso3", color="color", hover_name="country",
            hover_data={"display": True, "year": True, "color": False, "iso3": False},
            color_continuous_scale="Viridis", labels={"display": ind_name, "year": "Year"},
        )
        if use_log:
            lo, hi = np.floor(latest_all["color"].min()), np.ceil(latest_all["color"].max())
            ticks = np.arange(lo, hi + 1)
            fig.update_coloraxes(colorbar=dict(title="", tickvals=ticks,
                                               ticktext=[fmt(10 ** t, kind) for t in ticks]))
        else:
            fig.update_coloraxes(colorbar_title="")
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                          geo=dict(showframe=False, projection_type="natural earth"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each country shows its most recent available year (the newest year is often not published yet).")

        top, bottom = st.columns(2)
        show = latest_all[["country", "region", "year", "display", "value"]].rename(
            columns={"display": "Value", "country": "Country", "region": "Region", "year": "Year"})
        top.markdown("**Highest 10**")
        top.dataframe(show.nlargest(10, "value").drop(columns="value"), hide_index=True, use_container_width=True)
        bottom.markdown("**Lowest 10**")
        bottom.dataframe(show.nsmallest(10, "value").drop(columns="value"), hide_index=True, use_container_width=True)

# ---------- Tab 3: country profile ----------
with tab_profile:
    default_idx = int(countries.index[countries["country"] == (picked[0] if picked else "United States")][0]) \
        if (picked or "United States" in name_to_iso) else 0
    who = st.selectbox("Country", countries["country"], index=default_idx)
    info = countries[countries["country"] == who].iloc[0]
    st.markdown(f"**Region:** {info['region']} · **Income group:** {info['income']} · "
                f"**Capital:** {info['capital'] or '—'}")

    try:
        with st.spinner(f"Loading profile for {who}…"):
            profile = get_profile(info["iso3"], tuple(c for c, _ in INDICATORS.values()),
                                  LAST_YEAR - 15, LAST_YEAR)
    except WorldBankAPIError as e:
        st.error(str(e))
        profile = pd.DataFrame(columns=["indicator", "year", "value"])

    cards = st.columns(5, border=True)
    for i, (label, (code, k)) in enumerate(INDICATORS.items()):
        s = profile[profile["indicator"] == code].sort_values("year")
        if s.empty:
            cards[i % 5].metric(label, "—")
            continue
        last = s.iloc[-1]
        prev = s.iloc[-2] if len(s) > 1 else None
        delta = None
        if prev is not None:
            if k == "pct":
                delta = f"{last['value'] - prev['value']:+.1f} pts vs {prev['year']}"
            elif prev["value"]:
                delta = f"{last['value'] / prev['value'] - 1:+.1%} vs {prev['year']}"
        cards[i % 5].metric(f"{label} ({last['year']})", fmt(last["value"], k), delta)

# ---------- Tab 4: raw data ----------
with tab_data:
    if series.empty:
        st.info("Pick countries in the sidebar to see their data here.")
    else:
        wide = series.pivot(index="year", columns="country", values="value").sort_index(ascending=False)
        st.dataframe(wide, use_container_width=True)
        st.download_button("Download CSV", series.to_csv(index=False),
                           f"worldbank_{ind_code}.csv", "text/csv")

st.divider()
st.caption("Data: [World Bank Open Data](https://data.worldbank.org/) (CC BY 4.0)")
