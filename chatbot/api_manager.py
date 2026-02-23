"""
api_manager.py
==============
Central manager for all external API integrations.

APIs managed:
  - Open-Meteo  (https://open-meteo.com)  — free, no key required
      • Geocoding : https://geocoding-api.open-meteo.com/v1/search
      • Forecast  : https://api.open-meteo.com/v1/forecast
  - Geek-Jokes  (https://geek-jokes.sameerkumar.website)  — free, no key required
      • Endpoint  : /api?format=json  →  { "joke": "..." }

Rate limits (sliding window, in-memory, per client IP):
  - Weather API : 10 calls / 60 seconds  per IP
  - Jokes API   : 15 calls / 60 seconds  per IP
  - Chat        : 30 messages / 60 seconds per IP

Add new API integrations as new classes below, then register them
in APIManager.__init__() and expose a clean public method.
"""

import time
import requests
from collections import defaultdict
from threading import Lock


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER  (sliding-window algorithm)
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Parameters
    ----------
    max_calls : int   — maximum number of calls allowed in `period` seconds
    period    : float — time window in seconds
    """

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period    = period
        self._calls: dict[str, list] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, identifier: str) -> bool:
        """Return True if the request is within the allowed limit."""
        with self._lock:
            now = time.monotonic()
            window = self._calls[identifier]
            # Drop timestamps outside the current window
            self._calls[identifier] = [t for t in window if now - t < self.period]
            if len(self._calls[identifier]) < self.max_calls:
                self._calls[identifier].append(now)
                return True
            return False

    def remaining(self, identifier: str) -> int:
        """Return how many calls are still allowed in the current window."""
        with self._lock:
            now = time.monotonic()
            recent = [t for t in self._calls[identifier] if now - t < self.period]
            return max(0, self.max_calls - len(recent))

    def retry_after(self, identifier: str) -> float:
        """Return seconds until the oldest call leaves the window."""
        with self._lock:
            now = time.monotonic()
            recent = sorted(t for t in self._calls[identifier] if now - t < self.period)
            if not recent or len(recent) < self.max_calls:
                return 0.0
            return round(self.period - (now - recent[0]), 1)


# ─────────────────────────────────────────────────────────────────────────────
# OPEN-METEO WEATHER API
# ─────────────────────────────────────────────────────────────────────────────

class WeatherAPI:
    """
    Wrapper around the Open-Meteo free weather API.
    No API key required.

    Usage
    -----
    api = WeatherAPI()
    ok, message = api.get_weather("London")
    """

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
    TIMEOUT       = 6  # seconds

    # WMO Weather Interpretation Codes → (description, emoji)
    WMO_CODES: dict[int, tuple[str, str]] = {
        0:  ("Clear sky",                   "☀️"),
        1:  ("Mainly clear",                "🌤️"),
        2:  ("Partly cloudy",               "⛅"),
        3:  ("Overcast",                    "☁️"),
        45: ("Fog",                         "🌫️"),
        48: ("Icy fog",                     "🌫️"),
        51: ("Light drizzle",               "🌦️"),
        53: ("Moderate drizzle",            "🌦️"),
        55: ("Dense drizzle",               "🌧️"),
        61: ("Slight rain",                 "🌧️"),
        63: ("Moderate rain",               "🌧️"),
        65: ("Heavy rain",                  "🌧️"),
        71: ("Slight snow",                 "🌨️"),
        73: ("Moderate snow",               "❄️"),
        75: ("Heavy snow",                  "❄️"),
        77: ("Snow grains",                 "🌨️"),
        80: ("Slight rain showers",         "🌦️"),
        81: ("Moderate rain showers",       "🌧️"),
        82: ("Violent rain showers",        "⛈️"),
        85: ("Slight snow showers",         "🌨️"),
        86: ("Heavy snow showers",          "❄️"),
        95: ("Thunderstorm",                "⛈️"),
        96: ("Thunderstorm with hail",      "⛈️"),
        99: ("Thunderstorm w/ heavy hail",  "⛈️"),
    }

    # ── private helpers ──────────────────────────────────────────────────────

    def _geocode(self, city: str) -> dict | None:
        """Resolve a city name to lat/lon via Open-Meteo geocoding."""
        resp = requests.get(
            self.GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=self.TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        r = results[0]
        return {
            "name":      r["name"],
            "country":   r.get("country", ""),
            "latitude":  r["latitude"],
            "longitude": r["longitude"],
        }

    def _fetch_forecast(self, lat: float, lon: float) -> dict:
        """Fetch current weather + first hourly data point from Open-Meteo."""
        resp = requests.get(
            self.FORECAST_URL,
            params={
                "latitude":         lat,
                "longitude":        lon,
                "current_weather":  True,
                "hourly":           "relative_humidity_2m,apparent_temperature",
                "forecast_days":    1,
                "timezone":         "auto",
                "wind_speed_unit":  "kmh",
            },
            timeout=self.TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    # ── public interface ─────────────────────────────────────────────────────

    def get_weather(self, city: str) -> tuple[bool, str]:
        """
        Return (success, message) where message is a formatted weather string
        or a user-friendly error string.
        """
        # 1. Geocode
        location = self._geocode(city)
        if not location:
            return False, (
                f"🗺️ I couldn't find a city called **{city}**. "
                "Could you double-check the spelling?"
            )

        # 2. Fetch weather
        data = self._fetch_forecast(location["latitude"], location["longitude"])
        cw   = data["current_weather"]

        temp   = cw["temperature"]
        wind   = cw["windspeed"]
        code   = int(cw["weathercode"])
        label, emoji = self.WMO_CODES.get(code, ("Unknown", "🌡️"))

        # First hourly snapshot for humidity / feels-like
        hourly     = data.get("hourly", {})
        humidity   = (hourly.get("relative_humidity_2m")   or [None])[0]
        feels_like = (hourly.get("apparent_temperature")   or [None])[0]

        lines = [
            f"{emoji} **Weather in {location['name']}, {location['country']}**",
            f"🌡️ Temperature : **{temp}°C**"
            + (f"  (feels like {feels_like}°C)" if feels_like is not None else ""),
            f"🌤️ Condition   : {label}",
            f"💨 Wind speed  : {wind} km/h",
        ]
        if humidity is not None:
            lines.append(f"💧 Humidity    : {humidity}%")

        lines.append("\n_Powered by [Open-Meteo](https://open-meteo.com) — free & open-source._")
        return True, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GEEK-JOKES API
# ─────────────────────────────────────────────────────────────────────────────

class JokesAPI:
    """
    Wrapper around the Geek-Jokes RESTful API.
    Endpoint: GET https://geek-jokes.sameerkumar.website/api?format=json
    Response : { "joke": "<joke text>" }
    No API key required.

    Usage
    -----
    api = JokesAPI()
    ok, joke = api.get_joke()
    """

    URL     = "https://geek-jokes.sameerkumar.website/api"
    TIMEOUT = 5  # seconds

    def get_joke(self) -> tuple[bool, str]:
        """
        Fetch a random geek joke.
        Returns (success: bool, message: str).
        """
        resp = requests.get(self.URL, params={"format": "json"}, timeout=self.TIMEOUT)
        resp.raise_for_status()
        joke = resp.json().get("joke", "").strip()
        if not joke:
            return False, "😅 Couldn't get a joke right now — the server returned an empty response."
        return True, f"😄 {joke}"


# ─────────────────────────────────────────────────────────────────────────────
# API MANAGER  (single entry-point used by app.py)
# ─────────────────────────────────────────────────────────────────────────────

class APIManager:
    """
    Central API manager.
    Instantiate once at app startup and pass the singleton around.

    Rate limit configuration
    ------------------------
    weather_limit : 10 requests / 60 s  per IP
    jokes_limit   : 15 requests / 60 s  per IP
    chat_limit    : 30 requests / 60 s  per IP

    To add a new API:
      1. Write a new class above (e.g. NewsAPI, TranslateAPI …)
      2. Instantiate it here in __init__
      3. Add a matching RateLimiter
      4. Expose a public method that checks the limiter then calls the API
    """

    def __init__(
        self,
        weather_max: int    = 10,
        weather_period: float = 60.0,
        jokes_max: int      = 15,
        jokes_period: float = 60.0,
        chat_max: int       = 30,
        chat_period: float  = 60.0,
    ):
        # ── rate limiters ────────────────────────────────────────────────────
        self.weather_limiter = RateLimiter(weather_max, weather_period)
        self.jokes_limiter   = RateLimiter(jokes_max,   jokes_period)
        self.chat_limiter    = RateLimiter(chat_max,    chat_period)

        # ── API clients ──────────────────────────────────────────────────────
        self.weather_api = WeatherAPI()
        self.jokes_api   = JokesAPI()

    # ── weather ──────────────────────────────────────────────────────────────

    def get_weather(self, city: str, client_ip: str = "global") -> str:
        """
        Fetch weather for `city`, respecting per-IP rate limits.
        Always returns a user-facing string (success or error).
        """
        if not self.weather_limiter.is_allowed(client_ip):
            wait = self.weather_limiter.retry_after(client_ip)
            return (
                f"⚠️ Weather API rate limit reached (10 requests/min). "
                f"Please try again in **{wait}s**."
            )
        try:
            _, message = self.weather_api.get_weather(city)
            return message
        except requests.exceptions.Timeout:
            return "⏱️ The weather service timed out. Please try again in a moment."
        except requests.exceptions.ConnectionError:
            return "🌐 Could not reach the weather service. Check your internet connection."
        except requests.exceptions.RequestException as exc:
            return f"❌ Weather API error: {exc}"

    # ── jokes ────────────────────────────────────────────────────────────────

    def get_joke(self, client_ip: str = "global") -> str:
        """
        Fetch a random geek joke, respecting per-IP rate limits.
        Always returns a user-facing string.
        """
        if not self.jokes_limiter.is_allowed(client_ip):
            wait = self.jokes_limiter.retry_after(client_ip)
            return (
                f"⚠️ Joke rate limit reached (15 jokes/min). "
                f"Please wait **{wait}s** before asking for another joke!"
            )
        try:
            _, message = self.jokes_api.get_joke()
            return message
        except requests.exceptions.Timeout:
            return "⏱️ The joke service timed out. Try again in a moment!"
        except requests.exceptions.ConnectionError:
            return "🌐 Can't reach the joke API right now. Check your connection."
        except requests.exceptions.RequestException as exc:
            return f"❌ Joke API error: {exc}"

    # ── chat rate check ───────────────────────────────────────────────────────

    def chat_allowed(self, client_ip: str = "global") -> bool:
        """Return True if this IP is within the chat rate limit."""
        return self.chat_limiter.is_allowed(client_ip)

    def chat_retry_after(self, client_ip: str = "global") -> float:
        """Seconds until the client may send another chat message."""
        return self.chat_limiter.retry_after(client_ip)


# ── module-level singleton (imported by app.py) ───────────────────────────────
api_manager = APIManager()
