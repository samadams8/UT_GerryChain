import os
import sys
import csv
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv/bin/python"
SCRIPT = PROJECT_ROOT / "03_run_ensemble.py"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "ensemble_summary.csv"


def run_cmd(args):
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(PROJECT_ROOT))
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}")


def read_csv_header(path: Path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
    return headers


def assert_contains(headers, cols):
    missing = [c for c in cols if c not in headers]
    if missing:
        raise AssertionError(f"Missing expected columns: {missing}")


def assert_not_contains(headers, cols):
    present = [c for c in cols if c in headers]
    if present:
        raise AssertionError(f"Unexpected columns present: {present}")


def test_default_aggregation():
    # Defaults: years=2016,2020,2024; offices=PRE,GOV,ATG,AUD,TRE; vote-share-agg=median
    run_cmd([str(VENV_PYTHON), str(SCRIPT), "--steps", "3", "--viz-every", "1"])  # skip frequent visuals
    headers = read_csv_header(SUMMARY_CSV)
    # Aggregated metrics should be present
    assert_contains(headers, [
        "Republican_agg_seats",
        "agg_mean_median",
        "agg_partisan_bias",
        "agg_efficiency_gap",
    ])
    # District share columns appended and sorted
    district_cols = [h for h in headers if h.startswith("Republican_agg_share_d")]
    if len(district_cols) == 0:
        raise AssertionError("Expected aggregated district share columns")
    # Per-election outputs should be absent in aggregation mode
    assert_not_contains(headers, [
        "2016_PRE_Republican_total",
        "2016_PRE_Republican_wins",
    ])


def test_no_aggregation_subset():
    # No aggregation; per-election outputs should appear for selected elections only
    run_cmd([
        str(VENV_PYTHON), str(SCRIPT),
        "--years", "2016,2020",
        "--offices", "PRE,USS",
        "--vote-share-agg", "none",
        "--steps", "3",
        "--viz-every", "1",
    ])
    headers = read_csv_header(SUMMARY_CSV)
    # Aggregated metrics should be absent
    assert_not_contains(headers, [
        "Republican_agg_seats",
        "agg_mean_median",
        "agg_partisan_bias",
        "agg_efficiency_gap",
    ])
    # Per-election outputs present for selected filters
    assert_contains(headers, [
        "2016_PRE_Republican_total",
        "2016_PRE_Republican_wins",
        "2020_PRE_Republican_total",
        "2020_PRE_Republican_wins",
    ])
    # And not present for unselected office
    assert_not_contains(headers, [
        "2016_GOV_Republican_total",
        "2016_GOV_Republican_wins",
    ])


def test_mean_aggregation_office_filter():
    run_cmd([
        str(VENV_PYTHON), str(SCRIPT),
        "--years", "2016,2024",
        "--offices", "GOV,ATG",
        "--vote-share-agg", "mean",
        "--steps", "3",
        "--viz-every", "1",
    ])
    headers = read_csv_header(SUMMARY_CSV)
    # Aggregated metrics present
    assert_contains(headers, [
        "Republican_agg_seats",
        "agg_mean_median",
        "agg_partisan_bias",
        "agg_efficiency_gap",
    ])
    # Per-election outputs absent
    assert_not_contains(headers, [
        "2016_GOV_Republican_total",
        "2016_GOV_Republican_wins",
        "2024_ATG_Republican_total",
        "2024_ATG_Republican_wins",
    ])


if __name__ == "__main__":
    try:
        test_default_aggregation()
        test_no_aggregation_subset()
        test_mean_aggregation_office_filter()
        print("All tests passed.")
        sys.exit(0)
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)


