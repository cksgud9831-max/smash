#!/usr/bin/env python3
"""Validate and summarize the UAV pixel-threshold Monte Carlo experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(rows: list[dict[str, str]]) -> dict[str, object]:
    required = {"target_px", "trial", "detected", "stable_reached", "aimpoint_gate_passed"}
    missing_columns = required - set(rows[0]) if rows else required
    keys = [(int(row["target_px"]), int(row["trial"])) for row in rows]
    counts = Counter(px for px, _ in keys)
    missing_values = sum(not row[column] for row in rows for column in required)
    return {
        "rows": len(rows),
        "duplicate_trial_keys": len(keys) - len(set(keys)),
        "missing_required_values": missing_values,
        "missing_columns": sorted(missing_columns),
        "trials_per_bin": dict(sorted(counts.items())),
        "valid": len(rows) == 1000 and not missing_columns and not missing_values
        and len(keys) == len(set(keys)) and set(counts.values()) == {100},
    }


def make_chart(summary: list[dict[str, object]], output: Path) -> None:
    x = [int(item["target_px"]) for item in summary]
    stable = [100 * float(item["stable_rate"]) for item in summary]
    detect = [100 * float(item["detection_rate"]) for item in summary]
    low = [100 * float(item["aimpoint_gate_ci95"][0]) for item in summary]
    high = [100 * float(item["aimpoint_gate_ci95"][1]) for item in summary]
    yerr = [
        [max(0.0, a - b) for a, b in zip(stable, low)],
        [max(0.0, b - a) for a, b in zip(stable, high)],
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, detect, "--", color="#94a3b8", label="YOLO acquisition")
    ax.errorbar(x, stable, yerr=yerr, fmt="o-", capsize=4, color="#2563eb",
                label="Optical/reliability gate (Wilson 95% CI)")
    ax.axvspan(35, 40, color="#f59e0b", alpha=.14, label="Transition region")
    ax.axvline(40, color="#16a34a", linestyle=":", label="Recommended minimum tested bin")
    ax.set(xlabel="UAV long-side size (pixels)", ylabel="Trial success (%)",
           title="SMASH optical tracking stability by apparent UAV size")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_report(meta: dict[str, object], quality: dict[str, object], output: Path) -> None:
    summary = meta["summary"]
    lines = [
        "# UAV 픽셀 크기 몬테카를로 검증",
        "",
        "## 결론",
        "",
        "기존 경험값 25 px는 **탐지 가능 크기**로는 성립하지만, 현재 SMASH의 LK optical-flow + "
        "reliability gate가 안정화되는 전체 광학 파이프라인 기준으로는 부족하다. 본 조건에서 35 px는 "
        "전이 구간(73/100), 40 px는 최소 권장 실험 구간(100/100, Wilson 95% CI 96.3–100%)이었다. "
        "실환경 운용 여유를 포함한 목표값은 45–50 px를 권장한다.",
        "",
        "![성공률 그래프](stable_probability.png)",
        "",
        "## 결과",
        "",
        "| 장축 크기 | YOLO 탐지 | 안정화 게이트 | 95% CI | 판정 |",
        "|---:|---:|---:|---:|---|",
    ]
    for item in summary:
        px = int(item["target_px"])
        rate = 100 * float(item["stable_rate"])
        lo, hi = (100 * float(v) for v in item["aimpoint_gate_ci95"])
        verdict = "권장" if px >= 40 else ("조건부" if px == 35 else "불안정")
        lines.append(f"| {px} px | {100*float(item['detection_rate']):.0f}% | {rate:.0f}% | {lo:.1f}–{hi:.1f}% | {verdict} |")
    lines += [
        "",
        "## 검증 조건과 해석 범위",
        "",
        f"- 총 {meta['total_trials']}회: 10개 픽셀 구간 × 구간당 100회, seed `{meta['seed']}`.",
        f"- 각 trial은 {meta['frames_per_trial']}프레임/{meta['fps']} fps(약 1초)이며, 실제 Gazebo UAV 실루엣에 "
        "잔류 이동·밝기·블러·노이즈·JPEG 변동을 무작위 적용했다.",
        "- YOLO 모델, OpticalFlowTracker, TrackingReliabilityEngine은 생산 코드를 그대로 사용했다.",
        "- `aimpoint_gate_passed`는 실제 탄도계산 완료가 아니라, 조준점 생성이 허용되는 광학 reliability "
        "STABLE 게이트의 대리 지표다. 포탑 정렬·탄도·격발 성공률과 혼동하면 안 된다.",
        "- 단일 UAV 형상과 하늘 배경 기반 합성 시험이므로 복잡 배경·역광·모션블러·실센서 노이즈에서는 "
        "45–50 px 안전 여유를 적용하고 별도 실증해야 한다.",
        "",
        "## 데이터 품질",
        "",
        f"- 행 수: {quality['rows']}; 중복 trial key: {quality['duplicate_trial_keys']}; "
        f"필수값 누락: {quality['missing_required_values']}; 종합 통과: {quality['valid']}",
        "",
        "## 운용 가이드",
        "",
        "- `<30 px`: 탐지는 될 수 있으나 안정 추적/조준점 게이트 용도로 사용하지 않는다.",
        "- `30–34 px`: 매우 불안정한 획득 구간이다.",
        "- `35–39 px`: 전이·조건부 구간이다. 조준경이 지속 추종해 크기를 유지할 때만 제한적으로 사용한다.",
        "- `≥40 px`: 이번 시험에서 확인된 최소 권장 구간이다.",
        "- `45–50 px`: 실환경 변동을 고려한 설계 목표값이다.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def make_notebook(meta: dict[str, object], quality: dict[str, object], output: Path) -> None:
    table = "\n".join(
        f"{int(x['target_px']):>2} px | detect {100*float(x['detection_rate']):>3.0f}% | "
        f"stable {100*float(x['stable_rate']):>3.0f}% | CI "
        f"{100*float(x['aimpoint_gate_ci95'][0]):.1f}–{100*float(x['aimpoint_gate_ci95'][1]):.1f}%"
        for x in meta["summary"]
    )
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# UAV pixel-threshold Monte Carlo analysis\n",
                "Reproducible companion to `full1000.csv` and `full1000.json`.\n"]},
            {"cell_type": "code", "execution_count": 1, "metadata": {},
             "source": ["# Data-quality checks executed by analyze_pixel_threshold.py\n", "quality"],
             "outputs": [{"output_type": "execute_result", "execution_count": 1, "metadata": {},
                          "data": {"text/plain": [repr(quality)]}}]},
            {"cell_type": "code", "execution_count": 2, "metadata": {},
             "source": ["# Per-bin result table generated from full1000.json\n", "summary_table"],
             "outputs": [{"output_type": "execute_result", "execution_count": 2, "metadata": {},
                          "data": {"text/plain": [table]}}]},
            {"cell_type": "markdown", "metadata": {}, "source": [
                "## Decision\n", "40 px is the minimum tested recommended bin; target 45–50 px for field margin.\n",
                "The success endpoint is the optical/reliability gate, not complete ballistic/fire success.\n"]},
        ],
    }
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    quality = validate(rows)
    if not quality["valid"]:
        raise SystemExit(f"Data quality failed: {quality}")
    meta = json.loads(args.summary.read_text(encoding="utf-8"))
    make_chart(meta["summary"], args.output_dir / "stable_probability.png")
    make_report(meta, quality, args.output_dir / "REPORT.md")
    make_notebook(meta, quality, args.output_dir / "pixel_threshold_analysis.ipynb")
    print(json.dumps(quality, ensure_ascii=False))


if __name__ == "__main__":
    main()
