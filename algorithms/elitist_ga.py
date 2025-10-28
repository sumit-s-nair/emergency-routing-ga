"""
Elitist GA variant for EVDRP.

This algorithm should preserve the best individuals each generation.
"""
import time


def run(problem=None, config=None):
    start = time.time()
    # TODO: implement elitism logic
    time.sleep(0.01)
    end = time.time()
    return {"algorithm": "elitist_ga", "objective": float('inf'), "time_s": end - start, "status": "not-implemented"}
