"""
Algorithms package for Emergency Routing GA.

Each algorithm module exposes a `run(problem, config)` function that returns a
metrics dict (e.g. {'objective': float, 'time_s': float, 'status': 'ok'}).
"""

from .standard_ga import run as run_standard
from .elitist_ga import run as run_elitist
from .adaptive_ga import run as run_adaptive
from .hybrid_ga import run as run_hybrid

__all__ = ["run_standard", "run_elitist", "run_adaptive", "run_hybrid"]
