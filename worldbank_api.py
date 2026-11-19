"""
worldbank_api.py — talks to the free World Bank Indicators API (v2) and
returns pandas DataFrames. No API key needed.

Data: World Bank Open Data (CC BY 4.0).
Test it on its own with:  py worldbank_api.py

The World Bank API can be slow, so this module:
  * retries failed/slow requests automatically (with increasing waits)
  * allows up to 60s per response
  * asks for small, targeted results (e.g. only the latest value per country)
"""

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://api.worldbank.org/v2"
TIMEOUT = (10, 60)  # (seconds to connect, seconds to wait for a response)

# A curated set of indicators. kind controls how numbers are formatted.
INDICATORS = {
    "GDP (current US$)":                   ("NY.GDP.MKTP.CD", "usd"),
    "GDP per capita (current US$)":        ("NY.GDP.PCAP.CD", "usd"),
    "GDP growth (annual %)":               ("NY.GDP.MKTP.KD.ZG", "pct"),
    "Inflation, consumer prices (%)":      ("FP.CPI.TOTL.ZG", "pct"),
    "Unemployment (% of labor force)":     ("SL.UEM.TOTL.ZS", "pct"),
    "Population":                          ("SP.POP.TOTL", "count"),
    "Life expectancy (years)":             ("SP.DYN.LE00.IN", "years"),
    "Internet users (% of population)":    ("IT.NET.USER.ZS", "pct"),
    "Exports (% of GDP)":                  ("NE.EXP.GNFS.ZS", "pct"),
    "Urban population (% of total)":       ("SP.URB.TOTL.IN.ZS", "pct"),
}


class WorldBankAPIError(Exception):
    """Raised when the API can't be reached or returns an error."""


@st.cache_resource  # one shared connection pool for the whole app
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=1.5,                       # waits ~1.5s, 3s, 6s between tries
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _get_all_pages(path: str, params: dict, per_page: int = 1000) -> list:
    """Call the API, following pagination, and return all data rows."""
    params = {"format": "json", "per_page": per_page, **params}
    rows, page = [], 1
    while True:
        try:
            r = _session().get(f"{BASE}/{path}", params={**params, "page": page}, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
        except (requests.RequestException, ValueError) as e:
            raise WorldBankAPIError(
                "The World Bank API is slow or unavailable right now. "
                "Wait a moment and press 🔄 Refresh data in the sidebar."
                f"\n\nDetails: {e}"
            ) from e

        # Errors come back as [{"message": [...]}] with no data list
        if not isinstance(payload, list) or len(payload) < 2:
            msg = payload[0].get("message", payload) if isinstance(payload, list) and payload else payload
            raise WorldBankAPIError(f"World Bank API error: {msg}")

        meta, data = payload[0], payload[1] or []
        rows.extend(data)
        if page >= int(meta.get("pages", 1) or 1):
            return rows
        page += 1


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([{
        "indicator": r["indicator"]["id"],
        "iso3": r.get("countryiso3code") or r["country"]["id"],
        "country": r["country"]["value"],
        "year": int(r["date"]),
        "value": r["value"],
    } for r in rows], columns=["indicator", "iso3", "country", "year", "value"])
    df = df.dropna(subset=["value"])
    df["value"] = df["value"].astype(float)
    return df.sort_values(["country", "year"]).reset_index(drop=True)


@st.cache_data(ttl=7 * 24 * 3600)  # the country list almost never changes
def get_countries() -> pd.DataFrame:
    """All real countries (no regional/income aggregates) with metadata."""
    rows = _get_all_pages("country", {}, per_page=400)
    df = pd.DataFrame([{
        "iso3": c["id"],
        "iso2": c["iso2Code"],
        "country": c["name"],
        "region": c["region"]["value"],
        "income": c["incomeLevel"]["value"],
        "capital": c.get("capitalCity", ""),
    } for c in rows])
    df = df[df["region"] != "Aggregates"]
    return df.sort_values("country").reset_index(drop=True)


@st.cache_data(ttl=24 * 3600)  # World Bank data updates a few times a year
def get_indicator(indicator: str, countries: tuple, start: int = 1960, end: int = 2025) -> pd.DataFrame:
    """Time series for one indicator and a few countries (tuple of ISO3 codes).
    Returns columns: iso3, country, year, value (nulls dropped)."""
    rows = _get_all_pages(f"country/{';'.join(countries)}/indicator/{indicator}",
                          {"date": f"{start}:{end}"})
    return _to_frame(rows).drop(columns="indicator")


@st.cache_data(ttl=24 * 3600)
def latest_values(indicator: str) -> pd.DataFrame:
    """Most recent non-empty value for every country, in one small request.
    (The newest year is often not published yet, so 'latest' differs by country.)"""
    rows = _get_all_pages(f"country/all/indicator/{indicator}", {"mrnev": 1})
    df = _to_frame(rows).drop(columns="indicator")
    countries = get_countries()
    df = df[df["iso3"].isin(countries["iso3"])]          # drop aggregates like "World"
    df = df.sort_values("year").groupby("iso3").tail(1)
    return df.merge(countries[["iso3", "region", "income"]], on="iso3", how="left")


@st.cache_data(ttl=24 * 3600)
def get_profile(iso3: str, codes: tuple, start: int, end: int) -> pd.DataFrame:
    """Several indicators for one country in a single request.
    Returns columns: indicator, year, value."""
    rows = _get_all_pages(f"country/{iso3}/indicator/{';'.join(codes)}",
                          {"date": f"{start}:{end}", "source": 2})  # source 2 = World Development Indicators
    return _to_frame(rows)[["indicator", "year", "value"]]


def fmt(value, kind: str) -> str:
    """Human-friendly formatting for metric boxes and tables."""
    if value is None or pd.isna(value):
        return "—"
    if kind in ("usd", "count"):
        prefix = "$" if kind == "usd" else ""
        for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(value) >= size:
                return f"{prefix}{value / size:,.2f}{suffix}"
        return f"{prefix}{value:,.0f}"
    if kind == "pct":
        return f"{value:.1f}%"
    if kind == "years":
        return f"{value:.1f} yrs"
    return f"{value:,.2f}"


if __name__ == "__main__":
    import time
    t = time.time()
    print(get_countries().head(), f"\n({time.time() - t:.1f}s)\n")
    code, _ = INDICATORS["GDP per capita (current US$)"]
    t = time.time()
    print(get_indicator(code, ("USA", "CHN", "IND"), 2015, 2025).tail(9), f"\n({time.time() - t:.1f}s)\n")
    t = time.time()
    print(latest_values(code).nlargest(5, "value")[["country", "year", "value"]], f"\n({time.time() - t:.1f}s)\n")
    t = time.time()
    codes = tuple(c for c, _ in INDICATORS.values())
    print(get_profile("USA", codes, 2015, 2025).groupby("indicator").tail(1), f"\n({time.time() - t:.1f}s)")
