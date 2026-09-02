"""Wires jun_reliability's own acceptance scenarios into the pytest suite.

jun_reliability/synthetic_validation.py already encodes 16 scenario-based
acceptance checks (normal tracking, occlusion, stale-result rejection,
reacquisition ordering, ...) as assertions inside ``run_all()``, runnable
standalone via ``python -m jun_reliability.synthetic_validation``. Nothing
under tests/ (the only directory pytest.ini points ``testpaths`` at) called
into it, so a plain ``pytest`` run reported all-green while silently never
exercising the reliability layer at all. This module closes that gap without
duplicating the scenario logic itself.
"""

from __future__ import annotations

from jun_reliability.synthetic_validation import run_all


def test_all_synthetic_reliability_scenarios_pass() -> None:
    """run_all() raises AssertionError from inside a scenario on failure;
    reaching this point with 16 results back is the pass signal."""

    results = run_all()

    assert len(results) == 16
    assert all(result.passed for result in results), [
        result.name for result in results if not result.passed
    ]
