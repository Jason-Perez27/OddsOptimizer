"""
src/data/weather.py — weather features for game day.

Part 1 of Spec ④ (2026-06-30): weather + Vegas context + boosted ensemble.

Pulls temperature, wind speed, humidity, and precipitation for a ballpark at
first-pitch time from the Open-Meteo free API (no API key required).  Two
endpoints are used:
  - Forecast (today / future): https://api.open-meteo.com/v1/forecast
  - Historical (past dates):   https://archive-api.open-meteo.com/v1/archive

Both return the same hourly JSON shape; the caller doesn't need to choose —
get_game_weather() picks the right endpoint by comparing game_date to today.

Dome/retractable-roof parks (is_dome=True in BALLPARK_TABLE) return a neutral
dict without hitting the API; weather is irrelevant in a controlled
environment.

Historical backtest note: Open-Meteo's archive endpoint provides hourly
historical data going back to 1940, so weather features CAN be used in
walk-forward backtests keyed to historical game dates.  This distinguishes
weather from sources that are forecast-only.

Feature columns (WEATHER_CANDIDATE_COLUMNS):
  temp_f   – temperature at first pitch in °F
  wind_mph – wind speed at first pitch in mph
  humidity – relative humidity at first pitch (0–100)
  is_dome  – 1.0 if park is domed/retractable, 0.0 otherwise

weather_was_imputed=True when:
  - the park is a dome (weather not applicable)
  - the API call fails or returns no data for the requested hour
  - the team is not in BALLPARK_TABLE

In all imputed cases the numeric weather columns are NaN; downstream
models treat them the same as any other imputed candidate feature.
"""

import json
from datetime import date
from urllib.request import urlopen

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public column names
# ---------------------------------------------------------------------------

WEATHER_CANDIDATE_COLUMNS = ["temp_f", "wind_mph", "humidity", "is_dome"]

# ---------------------------------------------------------------------------
# Ballpark reference table
# ---------------------------------------------------------------------------
# Keyed by home-team abbreviation (same 3-letter codes as park_factors.py).
# lat/lon: approximate stadium coordinates for the Open-Meteo lookup.
# is_dome: True for parks with a fixed or retractable roof that eliminates
#          meaningful weather effects (roof assumed closed for dome parks).
# Note: OAK → Sutter Health Park, Sacramento (A's relocated 2025+).

BALLPARK_TABLE = {
    "ARI": {"lat": 33.4453, "lon": -112.0667, "is_dome": True},   # Chase Field (retractable)
    "ATL": {"lat": 33.8908, "lon": -84.4678,  "is_dome": False},  # Truist Park
    "BAL": {"lat": 39.2839, "lon": -76.6217,  "is_dome": False},  # Oriole Park at Camden Yards
    "BOS": {"lat": 42.3467, "lon": -71.0972,  "is_dome": False},  # Fenway Park
    "CHC": {"lat": 41.9484, "lon": -87.6553,  "is_dome": False},  # Wrigley Field
    "CWS": {"lat": 41.8300, "lon": -87.6339,  "is_dome": False},  # Guaranteed Rate Field
    "CIN": {"lat": 39.0975, "lon": -84.5064,  "is_dome": False},  # Great American Ball Park
    "CLE": {"lat": 41.4962, "lon": -81.6853,  "is_dome": False},  # Progressive Field
    "COL": {"lat": 39.7559, "lon": -104.9942, "is_dome": False},  # Coors Field
    "DET": {"lat": 42.3390, "lon": -83.0485,  "is_dome": False},  # Comerica Park
    "HOU": {"lat": 29.7572, "lon": -95.3553,  "is_dome": True},   # Minute Maid Park (retractable)
    "KC":  {"lat": 39.0517, "lon": -94.4803,  "is_dome": False},  # Kauffman Stadium
    "LAA": {"lat": 33.8003, "lon": -117.8827, "is_dome": False},  # Angel Stadium
    "LAD": {"lat": 34.0739, "lon": -118.2400, "is_dome": False},  # Dodger Stadium
    "MIA": {"lat": 25.7781, "lon": -80.2197,  "is_dome": True},   # loanDepot Park (retractable)
    "MIL": {"lat": 43.0283, "lon": -87.9711,  "is_dome": True},   # American Family Field (retractable)
    "MIN": {"lat": 44.9817, "lon": -93.2783,  "is_dome": False},  # Target Field
    "NYM": {"lat": 40.7571, "lon": -73.8458,  "is_dome": False},  # Citi Field
    "NYY": {"lat": 40.8296, "lon": -73.9262,  "is_dome": False},  # Yankee Stadium
    "OAK": {"lat": 38.5803, "lon": -121.5011, "is_dome": False},  # Sutter Health Park, Sacramento
    "PHI": {"lat": 39.9057, "lon": -75.1665,  "is_dome": False},  # Citizens Bank Park
    "PIT": {"lat": 40.4469, "lon": -80.0058,  "is_dome": False},  # PNC Park
    "SD":  {"lat": 32.7073, "lon": -117.1567, "is_dome": False},  # Petco Park
    "SEA": {"lat": 47.5914, "lon": -122.3325, "is_dome": True},   # T-Mobile Park (retractable)
    "SF":  {"lat": 37.7786, "lon": -122.3893, "is_dome": False},  # Oracle Park
    "STL": {"lat": 38.6226, "lon": -90.1928,  "is_dome": False},  # Busch Stadium
    "TB":  {"lat": 27.7682, "lon": -82.6534,  "is_dome": True},   # Tropicana Field (fixed)
    "TEX": {"lat": 32.7473, "lon": -97.0842,  "is_dome": True},   # Globe Life Field (retractable)
    "TOR": {"lat": 43.6414, "lon": -79.3894,  "is_dome": True},   # Rogers Centre (retractable)
    "WSH": {"lat": 38.8731, "lon": -77.0074,  "is_dome": False},  # Nationals Park
}

DEFAULT_FIRST_PITCH_HOUR = 19  # 7 PM local — reasonable fallback when time unknown

# ---------------------------------------------------------------------------
# Open-Meteo API URLs
# ---------------------------------------------------------------------------

_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,windspeed_10m"
    "&temperature_unit=fahrenheit"
    "&windspeed_unit=mph"
    "&timezone=auto"
    "&start_date={date}&end_date={date}"
)

_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={date}&end_date={date}"
    "&hourly=temperature_2m,relative_humidity_2m,precipitation,windspeed_10m"
    "&temperature_unit=fahrenheit"
    "&windspeed_unit=mph"
    "&timezone=auto"
)

# ---------------------------------------------------------------------------
# Return shapes
# ---------------------------------------------------------------------------

def _dome_result() -> dict:
    """Neutral result for dome parks; weather is not a factor."""
    return {
        "temp_f": np.nan,
        "wind_mph": np.nan,
        "humidity": np.nan,
        "is_dome": 1.0,
        "weather_was_imputed": True,
    }


def _imputed_outdoor_result() -> dict:
    """Fallback when an outdoor park's weather fetch fails."""
    return {
        "temp_f": np.nan,
        "wind_mph": np.nan,
        "humidity": np.nan,
        "is_dome": 0.0,
        "weather_was_imputed": True,
    }


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def _default_fetcher(url: str) -> dict:
    with urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def fetch_weather(
    lat: float,
    lon: float,
    game_date: str,
    first_pitch_hour: int = DEFAULT_FIRST_PITCH_HOUR,
    *,
    fetcher=None,
) -> dict:
    """
    Fetch hourly weather from Open-Meteo for (lat, lon, game_date) and return
    conditions at first_pitch_hour local time.

    Parameters
    ----------
    lat, lon : float
        Ballpark coordinates.
    game_date : str
        ISO date string "YYYY-MM-DD".
    first_pitch_hour : int
        Local hour (0–23) of first pitch.  Used to select the matching hourly
        bucket.  Nearest available hour is chosen if an exact match isn't
        found.
    fetcher : callable, optional
        Injected for tests: ``fetcher(url) -> dict``.  Defaults to real
        urlopen.

    Returns
    -------
    dict
        Keys: temp_f, wind_mph, humidity, is_dome (0.0), weather_was_imputed.
        Returns imputed-outdoor dict on any failure.
    """
    _fetch = fetcher or _default_fetcher

    # Choose endpoint based on whether game_date is in the past or not.
    today = date.today().isoformat()
    if game_date < today:
        url = _ARCHIVE_URL.format(lat=lat, lon=lon, date=game_date)
    else:
        url = _FORECAST_URL.format(lat=lat, lon=lon, date=game_date)

    try:
        raw = _fetch(url)
    except Exception:
        return _imputed_outdoor_result()

    try:
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return _imputed_outdoor_result()

        # Find the index whose local-hour portion is closest to first_pitch_hour.
        # times are like "2025-06-15T19:00" (local, because timezone=auto).
        target_str = f"T{first_pitch_hour:02d}:00"
        idx = None
        for i, t in enumerate(times):
            if t.endswith(target_str):
                idx = i
                break
        if idx is None:
            # Fall back to the closest hour by integer distance
            def _hour(t):
                try:
                    return int(t[11:13])
                except (ValueError, IndexError):
                    return -99
            hours = [_hour(t) for t in times]
            dists = [abs(h - first_pitch_hour) for h in hours]
            idx = int(np.argmin(dists)) if dists else 0

        temp_f = hourly.get("temperature_2m", [None])[idx]
        humidity = hourly.get("relative_humidity_2m", [None])[idx]
        wind_mph = hourly.get("windspeed_10m", [None])[idx]

        return {
            "temp_f": float(temp_f) if temp_f is not None else np.nan,
            "wind_mph": float(wind_mph) if wind_mph is not None else np.nan,
            "humidity": float(humidity) if humidity is not None else np.nan,
            "is_dome": 0.0,
            "weather_was_imputed": False,
        }
    except Exception:
        return _imputed_outdoor_result()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_game_weather(
    home_team: str,
    game_date: str,
    first_pitch_hour: int = DEFAULT_FIRST_PITCH_HOUR,
    *,
    fetcher=None,
) -> dict:
    """
    Return weather features for a game keyed by home_team abbreviation.

    Dome parks return the neutral result without any network call.
    Unknown teams return the imputed-outdoor result (weather_was_imputed=True).

    Parameters
    ----------
    home_team : str
        Home-team abbreviation (e.g. "NYY", "BOS").  Must match a key in
        BALLPARK_TABLE.
    game_date : str
        ISO date string "YYYY-MM-DD".
    first_pitch_hour : int
        Local hour of first pitch (default 19, i.e. 7 PM).
    fetcher : callable, optional
        Injected for tests.

    Returns
    -------
    dict with keys: temp_f, wind_mph, humidity, is_dome, weather_was_imputed.
    """
    park = BALLPARK_TABLE.get(home_team)
    if park is None:
        return _imputed_outdoor_result()

    if park["is_dome"]:
        return _dome_result()

    return fetch_weather(
        lat=park["lat"],
        lon=park["lon"],
        game_date=game_date,
        first_pitch_hour=first_pitch_hour,
        fetcher=fetcher,
    )
