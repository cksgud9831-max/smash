from benchmarks import integration_benchmark_v1 as benchmark


def test_bbox_iou_uses_xywh():
    assert benchmark.bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)) == 1 / 3
    assert benchmark.bbox_iou(None, (0, 0, 10, 10)) == 0.0


def test_tracking_metrics_match_v20_definitions():
    rows = [
        {"gt_present": True, "iou": 1.0},
        {"gt_present": True, "iou": 0.5},
        {"gt_present": True, "iou": 0.09},
        {"gt_present": False, "iou": None},
    ]
    result = benchmark.tracking_metrics(rows)
    assert result["evaluated_frames"] == 3
    assert result["mean_iou"] == (1.0 + 0.5 + 0.09) / 3
    assert result["tracking_precision"] == 2 / 3
    assert result["failure_count"] == 1
    assert 0.0 <= result["success_auc"] <= 1.0
    assert result["id_switch"] is None


def test_target_absent_frames_are_excluded_from_tracking_metrics():
    result = benchmark.tracking_metrics([{"gt_present": False, "iou": None}])
    assert result["evaluated_frames"] == 0
    assert result["mean_iou"] is None
    assert result["tracking_precision"] is None
    assert result["success_auc"] is None


def test_reliability_counts_and_transitions():
    rows = [
        {"frame_index": 0, "reliability_state": "LOST"},
        {"frame_index": 1, "reliability_state": "HOLD"},
        {"frame_index": 2, "reliability_state": "STABLE"},
        {"frame_index": 3, "reliability_state": "STABLE"},
    ]
    result = benchmark.reliability_statistics(rows)
    assert result["states"]["STABLE"]["frame_count"] == 2
    assert result["states"]["HOLD"]["entry_count"] == 1
    assert result["transitions"] == [
        {"frame_index": 1, "from": "LOST", "to": "HOLD"},
        {"frame_index": 2, "from": "HOLD", "to": "STABLE"},
    ]


def test_false_stable_statistics_measure_contiguous_runs():
    rows = [
        {"frame_index": 10, "gt_present": False, "reliability_state": "STABLE"},
        {"frame_index": 11, "gt_present": False, "reliability_state": "STABLE"},
        {"frame_index": 12, "gt_present": False, "reliability_state": "HOLD"},
        {"frame_index": 13, "gt_present": False, "reliability_state": "STABLE"},
    ]
    result = benchmark.false_stable_statistics(rows, fps=20.0)
    assert result["frame_count"] == 3
    assert result["longest_duration_frames"] == 2
    assert result["first_frame"] == 10
    assert result["state_recovery_frame"] == 12


def test_parse_tegrastats_line_extracts_jetson_resources():
    result = benchmark.parse_tegrastats_line(
        "RAM 2048/31919MB CPU [10%@729,20%@729,off] GR3D_FREQ 35% "
        "cpu@45.5C gpu@47C VDD_IN 7500mW/7000mW"
    )
    assert result == {
        "cpu_percent": 15.0,
        "gpu_percent": 35.0,
        "ram_mb": 2048.0,
        "power_w": 7.5,
        "temperature_c": 47.0,
    }


def test_decision_metrics_preserve_raw_and_use_simulated_ready():
    rows = [
        {"raw_aim_ready": True, "simulated_ready": True, "gate_reason": "passed"},
        {"raw_aim_ready": True, "simulated_ready": False, "gate_reason": "hold"},
        {"raw_aim_ready": False, "simulated_ready": False, "gate_reason": "raw"},
    ]
    result = benchmark._decision_metrics(rows)
    assert result["raw_aim_ready_count"] == 2
    assert result["simulated_ready_count"] == 1
    assert result["raw_aim_ready_ratio"] == 2 / 3
    assert result["simulated_ready_ratio"] == 1 / 3
