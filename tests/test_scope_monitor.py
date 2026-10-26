import numpy as np
from scripts.scope_monitor import StreamHealth, compose_view


def test_health_live_stale_and_recovery():
    health = StreamHealth()
    assert health.snapshot(0)[0] == 'CONNECTING'
    for t in (1.0, 1.1, 1.2):
        health.received(t)
    assert abs(health.snapshot(1.2)[2] - 10) < 1e-6
    assert health.snapshot(3.3)[0] == 'STALE'
    health.received(4)
    assert health.snapshot(4)[0] == 'LIVE'


def test_panel_preserves_source_and_marks_stale():
    source = np.full((640, 640, 3), 123, np.uint8)
    before = source.copy()
    health = StreamHealth()
    health.received(1)
    live = compose_view(source, health, 1)
    assert np.array_equal(live[:640, :640], source)
    stale = compose_view(source, health, 4)
    assert not np.array_equal(stale[:42, :640], source[:42])
    assert np.array_equal(source, before)
    assert stale.shape == (682, 930, 3)
