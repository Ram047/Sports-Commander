"""
event_engine/__init__.py
Exports a factory to get the right engine for a sport.
"""

from .basketball import BasketballEngine
from .volleyball import VolleyballEngine
from .football import FootballEngine


def get_engine(sport: str):
    sport = sport.lower()
    if sport == "basketball":
        return BasketballEngine()
    elif sport == "volleyball":
        return VolleyballEngine()
    elif sport in ("football", "soccer"):
        return FootballEngine()
    else:
        raise ValueError(f"Unknown sport: {sport}. Choose basketball/volleyball/football.")
