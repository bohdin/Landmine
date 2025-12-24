from core.processing import metal_window, nearest_telemetry


def test_frame_sync_to_nearest_telemetry_and_metal():
    telemetry = [
        {"timestamp": 1000, "lat": 0, "lon": 0, "alt": 0, "battery": 1, "signal": 1, "yaw": 0},
        {"timestamp": 1200, "lat": 0, "lon": 0, "alt": 0, "battery": 1, "signal": 1, "yaw": 0},
        {"timestamp": 1600, "lat": 0, "lon": 0, "alt": 0, "battery": 1, "signal": 1, "yaw": 0},
    ]
    frame_ts = 1250
    nearest = nearest_telemetry(frame_ts, telemetry)
    assert nearest["timestamp"] == 1200

    metal = [
        {"timestamp": 900, "value": 0.1, "adaptive_threshold": 0.2, "lat": 0, "lon": 0, "alt": 0},
        {"timestamp": 1300, "value": 0.5, "adaptive_threshold": 0.3, "lat": 0, "lon": 0, "alt": 0},
        {"timestamp": 2000, "value": 0.4, "adaptive_threshold": 0.3, "lat": 0, "lon": 0, "alt": 0},
    ]
    window = metal_window(frame_ts, metal, window_ms=200)
    assert len(window) == 1
    assert window[0]["timestamp"] == 1300
