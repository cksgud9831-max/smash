from pathlib import Path

import yaml

from bridge.config import load_bridge_config


ROOT = Path(__file__).resolve().parent.parent


def test_default_bridge_config_selects_legacy_backend():
    config = load_bridge_config(ROOT / "config" / "bridge.yaml")
    assert config.tracker.backend == "legacy"


def test_final_v20_config_selects_engine_backend():
    config = load_bridge_config(ROOT / "config" / "bridge.final_v20.yaml")
    assert config.tracker.backend == "final_v20"
    assert config.tracker.model_path == "detector+tracker/final/models/ga_yolo11s_960_best.engine"


def test_config_without_backend_falls_back_to_legacy(tmp_path):
    source = ROOT / "config" / "bridge.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["tracker"].pop("backend")
    legacy_yaml = tmp_path / "bridge-without-backend.yaml"
    legacy_yaml.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_bridge_config(legacy_yaml).tracker.backend == "legacy"
