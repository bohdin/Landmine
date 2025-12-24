from core.mission_simulator import compute_adaptive_threshold


def test_adaptive_threshold_and_cooldown():
    window = [0.1 for _ in range(20)]
    thr = compute_adaptive_threshold(window)
    assert thr > 0.1

    metal_value = thr + 0.05
    cooldown_ms = 1000
    last_trigger = 0

    first_ts = 1500
    triggered_first = metal_value > thr and (first_ts - last_trigger) >= cooldown_ms
    assert triggered_first is True
    last_trigger = first_ts

    second_ts = 1800  # within cooldown
    triggered_second = metal_value > thr and (second_ts - last_trigger) >= cooldown_ms
    assert triggered_second is False

    third_ts = 2800
    triggered_third = metal_value > thr and (third_ts - last_trigger) >= cooldown_ms
    assert triggered_third is True
