"""Property-based test for result numeric preservation (Property 6).

For each metrics CSV copied into ``clean_repo/results/metrics/``, this test
asserts cell-wise numeric equality against the corresponding Source_Repository
file, confirming that the reorganization did not alter any reported value.
"""

import csv
import math
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE_REPO_ROOT = os.path.dirname(_CLEAN_REPO_ROOT)

# Mapping of each copied metrics CSV (relative to clean_repo/results/metrics/)
# to its Source_Repository origin (relative to the source repo root). This must
# mirror the copies made in task 9.1.
_METRICS_CSV_MAP = {
    "summary_table.csv": "log/article_final_corr/results_mm/summary_table.csv",
    "m1_pure_pcn_results.csv": "log/article_final_corr/results_mm/m1_pure_pcn_results.csv",
    "m3_lrpcn_label_guided_results.csv": "log/article_final_corr/results_mm/m3_lrpcn_label_guided_results.csv",
    "m4_blind_displacement_results.csv": "log/article_final_corr/results_mm/m4_blind_displacement_results.csv",
    "m5_flow_displacement_results.csv": "log/article_final_corr/results_mm/m5_flow_displacement_results.csv",
    "m7_seed_displacement_results.csv": "log/article_final_corr/results_mm/m7_seed_displacement_results.csv",
    "m7_ablation_no_local_results.csv": "log/article_final_corr/results_mm/m7_ablation_no_local_results.csv",
    "per_surface_metrics.csv": "log/article_final_corr/per_surface_eval/per_surface_metrics.csv",
    "damage_sensitivity.csv": "log/article_final_corr/damage_sensitivity/damage_sensitivity.csv",
}


def _read_cells(path):
    """Read a CSV into a list of rows, each a list of raw string cells."""
    with open(path, newline="") as handle:
        return [row for row in csv.reader(handle)]


def _as_number(cell):
    """Return the float value of a cell, or None if it is not numeric."""
    try:
        value = float(cell)
    except (TypeError, ValueError):
        return None
    return value


@settings(max_examples=100, deadline=None)
@given(clean_name=st.sampled_from(sorted(_METRICS_CSV_MAP.keys())))
def test_result_numeric_preservation(clean_name):
    """**Feature: repo-reorganization, Property 6: For any metrics result file
    copied into the clean repo results directory, every numeric value equals the
    corresponding value in the Source_Repository result file.**

    Validates: Requirements 8.3, 11.1
    """
    clean_path = os.path.join(
        _CLEAN_REPO_ROOT, "results", "metrics", clean_name
    )
    source_path = os.path.join(_SOURCE_REPO_ROOT, _METRICS_CSV_MAP[clean_name])

    assert os.path.exists(clean_path), f"missing copied file: {clean_path}"
    assert os.path.exists(source_path), f"missing source file: {source_path}"

    clean_rows = _read_cells(clean_path)
    source_rows = _read_cells(source_path)

    assert len(clean_rows) == len(source_rows), (
        f"row count differs for {clean_name}: "
        f"clean={len(clean_rows)} source={len(source_rows)}"
    )

    for r, (clean_row, source_row) in enumerate(zip(clean_rows, source_rows)):
        assert len(clean_row) == len(source_row), (
            f"column count differs in {clean_name} row {r}: "
            f"clean={len(clean_row)} source={len(source_row)}"
        )
        for c, (clean_cell, source_cell) in enumerate(zip(clean_row, source_row)):
            clean_num = _as_number(clean_cell)
            source_num = _as_number(source_cell)
            if clean_num is None or source_num is None:
                # Non-numeric cells (headers, labels, "N/A") must match exactly.
                assert clean_cell == source_cell, (
                    f"non-numeric cell mismatch in {clean_name} at "
                    f"({r},{c}): clean={clean_cell!r} source={source_cell!r}"
                )
            elif math.isnan(source_num):
                assert math.isnan(clean_num), (
                    f"NaN mismatch in {clean_name} at ({r},{c})"
                )
            else:
                assert clean_num == pytest.approx(source_num, abs=0.0, rel=0.0), (
                    f"numeric cell mismatch in {clean_name} at ({r},{c}): "
                    f"clean={clean_num} source={source_num}"
                )
