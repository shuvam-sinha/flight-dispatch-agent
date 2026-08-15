"""Climb, cruise and descent -- a flight is not one speed for its whole length.

THE PROBLEM
-----------
Up to here, flight time was `route_distance / cruise_speed`. Every mile
was flown at cruise TAS, which is the speed the aircraft reaches at
altitude and holds until the descent. That produced KORD to KMIA in a
777-300ER as 2h07m; airlines block it at about three hours.

Some of that gap is taxi and schedule padding, which are not flight time
and are not modelled here. But a real part of it is that the first
twenty minutes and the last twenty minutes are not flown at cruise speed.
The aircraft leaves the runway at a few hundred feet per minute, climbing
slowly through the densest air it will see, and comes down at reduced
power well before the field.

WHAT THIS MODELS
----------------
Three phases, each with its own speed and its own fuel flow:

    ft
    |          ______________________________
    |         /                              \\
    |        /            cruise              \\
    |       /                                  \\
    |      / climb                      descent \\
    +-----+------------------------------------- +-----> nm
       origin                                  destination

Climb and descent distances are subtracted from the route; whatever is
left is flown at cruise. Total time is the sum of the three, and fuel is
computed per phase, because a jet burns roughly 1.6x cruise flow on the
way up and a third of it coming down.

SHORT FLIGHTS
-------------
On a short enough leg the climb and descent do not fit -- a 777 cannot
reach FL370 in 150 nm and still come down. Rather than return a negative
cruise segment, the aircraft levels off lower, at the altitude where
climb distance plus descent distance exactly equals the route. That is
what actually happens, and it is why short hops cruise low.

WHERE THE NUMBERS COME FROM
---------------------------
Climb and descent rates are per category, not per type. Real per-type
climb schedules live in manufacturer performance manuals, which are not
public data, and inventing 47 sets of them would be dressing a guess up
as a specification. Category captures most of the variance -- a piston
single climbs at 700 fpm and a widebody at 1,800 -- and the assumption
is stated here rather than buried in a table.

Speeds are expressed as fractions of the aircraft's own cruise TAS,
which keeps a Cessna's climb speed sensible for a Cessna and a 787's
sensible for a 787.

WHAT IS STILL MISSING
---------------------
Winds are taken at cruise level for the cruise segment only. The climb
and descent pass through lower altitudes with different winds, which are
not sampled -- so climb and descent are computed in still air. Weight
also affects climb rate, and a heavy aircraft takes noticeably longer to
reach altitude than a light one. Neither is modelled.
"""

from dataclasses import dataclass
from typing import Optional

from .aircraft import AircraftProfile


@dataclass(frozen=True)
class PhaseModel:
    """Climb and descent assumptions for one category of aircraft.

    Attributes:
        climb_rate_fpm: Average rate of climb, feet per minute. Averaged
            over the whole climb, which is why it is well below the
            initial rate -- climb performance falls off with altitude as
            the air thins.
        descent_rate_fpm: Average rate of descent, feet per minute.
        climb_speed_factor: Climb TAS as a fraction of cruise TAS.
        descent_speed_factor: Descent TAS as a fraction of cruise TAS.
            Below cruise because descents are flown at reduced speed,
            and well below 250 kt under 10,000 ft.
        climb_burn_factor: Fuel flow in the climb, relative to cruise.
            Jets climb at high thrust and burn far more than they do in
            the cruise.
        descent_burn_factor: Fuel flow in the descent, relative to
            cruise. A jet descends near idle.
    """

    climb_rate_fpm: float
    descent_rate_fpm: float
    climb_speed_factor: float
    descent_speed_factor: float
    climb_burn_factor: float
    descent_burn_factor: float


# Piston singles climb slowly and descend gently, and their fuel flow
# barely changes -- a normally aspirated engine at cruise power in the
# climb is not doing anything dramatic. Turbines are the opposite: steep
# climbs at high thrust, and descents at flight idle.
PHASE_MODELS = {
    "ga": PhaseModel(700, 600, 0.75, 0.85, 1.15, 0.70),
    "business": PhaseModel(2200, 1900, 0.72, 0.75, 1.55, 0.35),
    "regional": PhaseModel(2000, 1900, 0.74, 0.72, 1.55, 0.35),
    "narrowbody": PhaseModel(2000, 1900, 0.75, 0.72, 1.60, 0.35),
    "widebody": PhaseModel(1800, 1900, 0.75, 0.72, 1.60, 0.35),
}

DEFAULT_PHASE_MODEL = PHASE_MODELS["narrowbody"]


def phase_model(aircraft: AircraftProfile) -> PhaseModel:
    """The climb and descent assumptions for an aircraft's category."""
    return PHASE_MODELS.get(aircraft.category, DEFAULT_PHASE_MODEL)


@dataclass(frozen=True)
class FlightPhases:
    """A flight split into climb, cruise and descent.

    Times are hours, distances nautical miles, fuel US gallons. Fuel
    here EXCLUDES the reserve -- `total_fuel_gal` is what the flight
    burns, and the reserve is added by the caller, which is where it was
    added before.
    """

    climb_time_hours: float
    climb_distance_nm: float
    climb_fuel_gal: float

    cruise_time_hours: float
    cruise_distance_nm: float
    cruise_fuel_gal: float

    descent_time_hours: float
    descent_distance_nm: float
    descent_fuel_gal: float

    cruise_altitude_ft: float
    reached_planned_altitude: bool

    @property
    def total_time_hours(self) -> float:
        return self.climb_time_hours + self.cruise_time_hours + self.descent_time_hours

    @property
    def total_fuel_gal(self) -> float:
        return self.climb_fuel_gal + self.cruise_fuel_gal + self.descent_fuel_gal


def _climb_descent_per_foot(aircraft: AircraftProfile, model: PhaseModel):
    """Time and distance cost of one foot of climb, and of descent.

    Factoring this out is what makes the short-flight case solvable:
    both phases are linear in height, so the altitude that exactly fills
    a short route can be found by division rather than by iterating.
    """
    climb_tas = aircraft.cruise_tas_kt * model.climb_speed_factor
    descent_tas = aircraft.cruise_tas_kt * model.descent_speed_factor

    # fpm -> hours per foot: 1 / (rate * 60)
    climb_hours_per_ft = 1.0 / (model.climb_rate_fpm * 60.0)
    descent_hours_per_ft = 1.0 / (model.descent_rate_fpm * 60.0)

    return (
        climb_hours_per_ft,
        climb_hours_per_ft * climb_tas,  # nm per foot climbed
        descent_hours_per_ft,
        descent_hours_per_ft * descent_tas,  # nm per foot descended
    )


def flight_phases(
    aircraft: AircraftProfile,
    route_distance_nm: float,
    cruise_ground_speed_kt: float,
    origin_elevation_ft: float = 0.0,
    dest_elevation_ft: float = 0.0,
    cruise_altitude_ft: Optional[float] = None,
) -> FlightPhases:
    """Split a route into climb, cruise and descent.

    Args:
        aircraft: The profile being flown.
        route_distance_nm: Total planned route distance.
        cruise_ground_speed_kt: Ground speed in the cruise -- the
            wind-affected figure the router already computed. Climb and
            descent use the aircraft's own TAS, because the winds at
            those altitudes are not sampled.
        origin_elevation_ft: Departure field elevation.
        dest_elevation_ft: Arrival field elevation.
        cruise_altitude_ft: Planned cruise level. Defaults to the
            aircraft's normal cruise altitude.

    Returns:
        A `FlightPhases` whose three segments sum to `route_distance_nm`.
    """
    model = phase_model(aircraft)
    planned_altitude = (
        aircraft.cruise_altitude_ft if cruise_altitude_ft is None else cruise_altitude_ft
    )

    climb_hpf, climb_nm_pf, descent_hpf, descent_nm_pf = _climb_descent_per_foot(
        aircraft, model
    )

    climb_height = max(0.0, planned_altitude - origin_elevation_ft)
    descent_height = max(0.0, planned_altitude - dest_elevation_ft)

    climb_distance = climb_height * climb_nm_pf
    descent_distance = descent_height * descent_nm_pf

    reached = True
    if climb_distance + descent_distance > route_distance_nm:
        # Not enough room. Level off wherever climb and descent meet.
        #
        # Both distances are linear in height above the respective field,
        # so with h measured above the higher of the two elevations the
        # total distance is a straight line in h and inverts directly. A
        # 200 nm hop in a 777 tops out around 20,000 ft, which is what
        # actually happens on short sectors.
        reached = False
        base = max(origin_elevation_ft, dest_elevation_ft)
        per_ft = climb_nm_pf + descent_nm_pf
        # Distance already used just getting to `base` from each field.
        fixed = (base - origin_elevation_ft) * climb_nm_pf + (
            base - dest_elevation_ft
        ) * descent_nm_pf
        height_above_base = max(0.0, (route_distance_nm - fixed) / per_ft)

        planned_altitude = base + height_above_base
        climb_height = max(0.0, planned_altitude - origin_elevation_ft)
        descent_height = max(0.0, planned_altitude - dest_elevation_ft)
        climb_distance = climb_height * climb_nm_pf
        descent_distance = descent_height * descent_nm_pf

    climb_time = climb_height * climb_hpf
    descent_time = descent_height * descent_hpf

    cruise_distance = max(0.0, route_distance_nm - climb_distance - descent_distance)
    speed = cruise_ground_speed_kt if cruise_ground_speed_kt > 0 else aircraft.cruise_tas_kt
    cruise_time = cruise_distance / speed

    burn = aircraft.fuel_burn_gph
    return FlightPhases(
        climb_time_hours=climb_time,
        climb_distance_nm=climb_distance,
        climb_fuel_gal=climb_time * burn * model.climb_burn_factor,
        cruise_time_hours=cruise_time,
        cruise_distance_nm=cruise_distance,
        cruise_fuel_gal=cruise_time * burn,
        descent_time_hours=descent_time,
        descent_distance_nm=descent_distance,
        descent_fuel_gal=descent_time * burn * model.descent_burn_factor,
        cruise_altitude_ft=planned_altitude,
        reached_planned_altitude=reached,
    )
