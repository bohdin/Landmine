from core.processing import cluster_detections, make_square_geometry


def test_risk_zone_clustering_and_geometry():
    detections = [
        {"lat": 48.5, "lon": 32.25, "risk_score": 0.9},
        {"lat": 48.50001, "lon": 32.25001, "risk_score": 0.8},
        {"lat": 48.5003, "lon": 32.2503, "risk_score": 0.7},
    ]
    clusters = cluster_detections(detections, eps_m=5.0, min_samples=1)
    # first two should cluster together (distance about 1.4 m), third is separate
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]

    geometry = make_square_geometry(48.5, 32.25, buffer_m=5.0)
    assert len(geometry) == 5
    # check geometry closes polygon
    assert geometry[0] == geometry[-1]
