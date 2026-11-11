"""Evidence conversion tests; no MLflow dependency in the training environment."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("mlflow_sync", Path(__file__).resolve().parents[1] / "scripts/mlflow_sync.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def test_scalars_preserve_values_and_ignore_nonfinite_and_booleans():
    assert sync.scalar_metrics({"accuracy": .8, "nested": {"mmd": -.01}, "PASS": True,
                                "bad": float("nan"), "curve": [1, 2]}) == {"accuracy": .8, "nested.mmd": -.01}


def test_curve_selection_preserves_actual_endpoints_and_steps():
    rows = [{"global_step": i * 2, "loss": i / 17} for i in range(57)]
    chosen = sync.sample_rows(rows, 7)
    assert len(chosen) == 7 and chosen[0] == rows[0] and chosen[-1] == rows[-1]
    assert all(r in rows for r in chosen)
    assert sync.sample_rows(rows, 100) == rows


def test_unknown_and_interrupted_are_never_success():
    assert sync.state_status("COMPLETED_EVALUATION") == "FINISHED"
    assert sync.state_status("FAILED") == "FAILED"
    assert sync.state_status("INTERRUPTED_AT_EPOCH_BOUNDARY") == "KILLED"
    assert sync.state_status("UNKNOWN") == "KILLED"


def test_original_start_time_has_timezone():
    assert sync.timestamp("2026-11-02T16:11:39.035834+00:00") == 1793635899035
    assert sync.timestamp(None) is None


def test_flatten_arrays_are_not_invented_as_scalar_results():
    assert dict(sync.flatten({"model": {"name": "cnn"}, "deltas": [1, 2, 3]})) == {"model.name": "cnn"}
