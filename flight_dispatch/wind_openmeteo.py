"""Open-Meteo implementation of `WindSource`.

Kept separate from wind.py so the vector maths has no dependency on
`requests` or on any particular provider. Tests exercise the maths with
`ConstantWindSource` and never touch the network.

THE PERFORMANCE PROBLEM THIS SOLVES
-----------------------------------
Mesh construction asks for wind along every edge, and a transcontinental
graph has ~78,000 of them. One HTTP request per query would take hours.

Two things fix that:

  1. BATCHING. Open-Meteo accepts comma-separated coordinate lists, so
     hundreds of points come back in a single request.
  2. SNAPPING. Wind fields are smooth -- the difference between two points
     20 nm apart is negligible for planning. Coordinates are rounded to a
     grid before lookup, so nearby queries collapse onto the same cache
     entry. This turns tens of thousands of distinct requests into a few
     hundred.

The underlying model resolution is coarser than the snap grid anyway
(GFS is roughly 0.25 degrees), so snapping discards almost no real
information.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from .wind import KMH_PER_KNOT, Wind, nearest_pressure_level

API_URL = "https://api.open-meteo.com/v1/forecast"

# Coordinates are rounded to this many degrees before lookup. 0.5 degrees
# is ~30 nm of latitude -- well inside the distance over which winds aloft
# vary meaningfully, and coarser than the model's own grid.
DEFAULT_SNAP_DEG = 0.5

# Coordinates per request. 100 produced a ~2,000-character URL and drew
# 429s from Open-Meteo on larger meshes; 50 keeps URLs manageable and the
# request rate below the free tier's limit without doubling the useful
# work, since snapping already collapses most points.
MAX_POINTS_PER_REQUEST = 50

REQUEST_TIMEOUT_S = 30

# A burst of batched requests can trip the free tier's rate limit. Retry
# with backoff rather than failing the whole flight plan over it.
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0


class WindDataError(RuntimeError):
    """Raised when wind data cannot be fetched or parsed."""


class OpenMeteoWindSource:
    """Fetches winds aloft from Open-Meteo, with snapping and caching.

    Args:
        forecast_hour: Which hour of the forecast to use, counting from
            the start of the current model run. 0 is roughly now.
        snap_deg: Grid resolution for coordinate rounding.
        session: Optional `requests.Session` for connection reuse.
        offline_ok: If True, a network failure yields calm air instead of
            raising. Useful for demos without connectivity; the caller is
            responsible for saying so, since silently pretending the air
            is still would otherwise be a dangerous default.
    """

    def __init__(
        self,
        forecast_hour: int = 0,
        snap_deg: float = DEFAULT_SNAP_DEG,
        session: Optional[requests.Session] = None,
        offline_ok: bool = False,
    ):
        self.forecast_hour = forecast_hour
        self.snap_deg = snap_deg
        self.session = session or requests.Session()
        self.offline_ok = offline_ok

        # (snapped_lat, snapped_lon, pressure_level) -> Wind
        self._cache: Dict[Tuple[float, float, int], Wind] = {}

        # Counters, surfaced by the CLI so the batching is visible.
        self.requests_made = 0
        self.cache_hits = 0
        self.points_fetched = 0
        self.degraded = False  # True if any fetch failed and we fell back
        self.rate_limit_hits = 0

    # -- public interface ------------------------------------------------

    def wind_at(self, lat: float, lon: float, altitude_ft: float) -> Wind:
        return self.wind_at_many([(lat, lon)], altitude_ft)[0]

    def wind_at_many(
        self, points: Sequence[Tuple[float, float]], altitude_ft: float
    ) -> List[Wind]:
        """Wind at several points, fetching only what is not cached."""
        level = nearest_pressure_level(altitude_ft)
        snapped = [self._snap(lat, lon) for lat, lon in points]

        # Deduplicate before hitting the network. Many mesh edges snap to
        # the same grid cell, and a set collapses them.
        missing = sorted({p for p in snapped if (p[0], p[1], level) not in self._cache})
        self.cache_hits += len(snapped) - len(missing)

        for start in range(0, len(missing), MAX_POINTS_PER_REQUEST):
            self._fetch_batch(missing[start : start + MAX_POINTS_PER_REQUEST], level)

        return [
            self._cache.get(
                (lat, lon, level),
                Wind(direction_deg=0.0, speed_kt=0.0, altitude_ft=altitude_ft),
            )
            for lat, lon in snapped
        ]

    def prefetch(
        self, points: Sequence[Tuple[float, float]], altitude_ft: float
    ) -> None:
        """Warm the cache for a set of points.

        Call this once with every graph node before building edges, so the
        edge-cost function afterwards runs entirely from cache. Without
        it, the first mesh build would issue requests from inside a tight
        loop.
        """
        self.wind_at_many(points, altitude_ft)

    # -- internals -------------------------------------------------------

    def _snap(self, lat: float, lon: float) -> Tuple[float, float]:
        """Round a coordinate onto the lookup grid."""
        return (
            round(lat / self.snap_deg) * self.snap_deg,
            round(lon / self.snap_deg) * self.snap_deg,
        )

    def _fetch_batch(
        self, points: Sequence[Tuple[float, float]], level: int
    ) -> None:
        """Fetch one batch and populate the cache."""
        if not points:
            return

        params = {
            "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
            "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
            "hourly": ",".join(
                (
                    f"wind_speed_{level}hPa",
                    f"wind_direction_{level}hPa",
                    f"temperature_{level}hPa",
                )
            ),
            "forecast_days": 2,
        }

        payload = self._get_with_retry(params)
        if payload is None:
            return

        self.requests_made += 1

        # A single-coordinate request returns one object; a multi-
        # coordinate request returns a list. Normalise to a list.
        entries = payload if isinstance(payload, list) else [payload]

        for (lat, lon), entry in zip(points, entries):
            wind = self._parse_entry(entry, level)
            if wind is not None:
                self._cache[(lat, lon, level)] = wind
                self.points_fetched += 1

    def _get_with_retry(self, params: dict) -> Optional[dict]:
        """Fetch one batch, retrying rate limits with backoff.

        A mesh of a thousand nodes issues twenty-odd batched requests in
        quick succession, which the free tier answers with 429 Too Many
        Requests. That is a transient condition, not a failure -- waiting
        a moment and retrying gets the data, whereas propagating it loses
        an entire flight plan over weather that was available two seconds
        later.

        Returns the parsed payload, or None when the caller has opted
        into degraded mode and the fetch could not be completed.
        """
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(
                    API_URL, params=params, timeout=REQUEST_TIMEOUT_S
                )
                if response.status_code == 429:
                    last_error = requests.HTTPError("429 Too Many Requests")
                    self.rate_limit_hits += 1
                    if attempt < MAX_RETRIES - 1:
                        # Honour Retry-After when the server supplies it,
                        # otherwise back off exponentially.
                        wait = float(
                            response.headers.get(
                                "Retry-After", RETRY_BACKOFF_S * (2**attempt)
                            )
                        )
                        time.sleep(min(wait, 30.0))
                        continue
                    break

                response.raise_for_status()
                return response.json()

            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
                    continue
                break

        if not self.offline_ok:
            raise WindDataError(
                f"Could not fetch winds from Open-Meteo after "
                f"{MAX_RETRIES} attempts: {type(last_error).__name__}"
            ) from last_error

        # Degraded mode: leave the cache empty so lookups fall back to
        # calm air, and record that the data is not real.
        self.degraded = True
        return None

    def _parse_entry(self, entry: dict, level: int) -> Optional[Wind]:
        """Turn one Open-Meteo response object into a Wind."""
        hourly = entry.get("hourly")
        if not hourly:
            return None

        speeds = hourly.get(f"wind_speed_{level}hPa") or []
        directions = hourly.get(f"wind_direction_{level}hPa") or []
        temperatures = hourly.get(f"temperature_{level}hPa") or []

        index = min(self.forecast_hour, len(speeds) - 1)
        if index < 0 or index >= len(directions):
            return None

        speed_kmh = speeds[index]
        direction = directions[index]
        if speed_kmh is None or direction is None:
            return None

        temperature = (
            temperatures[index] if index < len(temperatures) else None
        )

        return Wind(
            direction_deg=float(direction),
            # Open-Meteo reports km/h; aviation works in knots.
            speed_kt=float(speed_kmh) / KMH_PER_KNOT,
            altitude_ft=0.0,  # filled by the caller's requested altitude
            temperature_c=temperature,
        )

    def stats(self) -> str:
        """One-line summary of fetch behaviour, for the CLI."""
        note = "  [DEGRADED: assuming calm air]" if self.degraded else ""
        if self.rate_limit_hits:
            note += f"  [{self.rate_limit_hits} rate-limit retries]"
        return (
            f"{self.points_fetched} grid points in {self.requests_made} "
            f"request(s), {self.cache_hits} cache hits{note}"
        )
