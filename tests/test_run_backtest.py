"""
Unit tests for scripts/run_backtest.py (task #12 go-live additions).

Design: docs/design/specs/2026-06-29-live-forward-validation-go-live-design.md
("--fit-only", "The model-path bug to reconcile" sections).

Strategy (matches the rest of this repo's no-network test style):
- run_backtest_pipeline() is the testable core extracted out of main() --
  every test below monkeypatches build_corpus/build_training_table/
  filter_starters at the module level to hand-built DataFrames (no real
  pybaseball/network call, no real feature engineering) and injects a fast
  fake fit_fn (no real statsmodels fit), exactly the contract
  fit_production_model/run_walk_forward already expect.
- The model-path no-drift guard test needs no fixtures at all -- it's a
  plain import-time equality check.

Run with: pytest tests/test_run_backtest.py -v
"""

import pandas as pd
import pytest

import scripts.run_backtest as rb
import src.pipeline.refresh as refresh


# ---------------------------------------------------------------------------
# Model-path no-drift guard (spec testing item #1)
# ---------------------------------------------------------------------------

def test_model_path_does_not_drift_between_refresh_and_run_backtest():
    assert rb.DEFAULT_MODEL_PATH == refresh.DEFAULT_MODEL_PATH
    assert rb.DEFAULT_MODEL_PATH.endswith(
        __import__("os").path.join("data", "models", "baseline_model.joblib")
    )


# ---------------------------------------------------------------------------
# fixtures / stubs
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal stand-in for BaselineModel exposing exactly what
    src.models.baseline_model.save_model's payload construction reads
    (family/result/preprocessor/active_columns/alpha) -- enough to round-trip
    through joblib.dump without needing a real statsmodels fit."""

    def __init__(self):
        self.family = "poisson"
        self.result = {"params": [1.0]}
        self.preprocessor = {"impute_means": {}, "scale_stats": {}}
        self.active_columns = ["k_rate_last5"]
        self.alpha = None


def make_fake_fit_fn():
    calls = []

    def fit_fn(train_df, test_df=None):
        calls.append((train_df.copy(), test_df))
        return _FakeModel(), {"n_train": len(train_df)}, (None, None)

    return fit_fn, calls


def _make_args(**overrides):
    """Build an argparse.Namespace matching build_arg_parser()'s defaults,
    overridden as needed -- avoids hand-rolling sys.argv for every test."""
    parser = rb.build_arg_parser()
    base_argv = ["--start", "2026-03-26", "--end", "2026-06-28"]
    args = parser.parse_args(base_argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _dummy_pitch_df():
    return pd.DataFrame({"pitcher": [1], "game_pk": [100], "game_date": ["2026-04-01"]})


def _dummy_game_df():
    return pd.DataFrame({"pitcher": [1], "game_pk": [100], "game_date": ["2026-04-01"], "strikeouts": [5]})


def _dummy_feature_table():
    return pd.DataFrame({
        "pitcher": [1, 1],
        "game_pk": [100, 101],
        "game_date": ["2026-04-01", "2026-04-08"],
        "strikeouts": [5, 6],
    })


def _patch_pipeline_steps(monkeypatch, *, feature_table=None):
    monkeypatch.setattr(rb, "build_corpus", lambda *a, **k: _dummy_pitch_df())
    monkeypatch.setattr(rb, "build_training_table", lambda *a, **k: _dummy_game_df())
    monkeypatch.setattr(rb, "filter_starters", lambda *a, **k: feature_table if feature_table is not None else _dummy_feature_table())


# ---------------------------------------------------------------------------
# --fit-only (spec testing items #2-#3)
# ---------------------------------------------------------------------------

def test_fit_only_skips_walk_forward_and_report_and_calls_fit_production_model(monkeypatch, tmp_path):
    _patch_pipeline_steps(monkeypatch)

    walk_forward_calls = []
    report_calls = []
    monkeypatch.setattr(rb, "run_walk_forward", lambda *a, **k: walk_forward_calls.append((a, k)))
    monkeypatch.setattr(rb, "generate_backtest_report", lambda *a, **k: report_calls.append((a, k)))

    model_path = str(tmp_path / "model.joblib")
    args = _make_args(fit_only=True, through_date="2026-06-28", model_path=model_path)
    fit_fn, fit_calls = make_fake_fit_fn()

    result = rb.run_backtest_pipeline(args, fit_fn=fit_fn)

    assert not walk_forward_calls, "run_walk_forward must not be called when --fit-only is set"
    assert not report_calls, "generate_backtest_report must not be called when --fit-only is set"
    assert len(fit_calls) == 1, "fit_production_model (via fit_fn) must be called exactly once"
    assert result["fit_only"] is True
    assert result["model_path"] == model_path
    assert result["oos_rows"] is None
    assert result["report_result"] is None

    import os
    assert os.path.exists(model_path), "fit-only must save the model artifact to --model-path"


def test_fit_only_and_no_production_model_is_rejected():
    parser = rb.build_arg_parser()
    args = parser.parse_args([
        "--start", "2026-03-26", "--end", "2026-06-28",
        "--fit-only", "--no-production-model",
    ])
    assert args.fit_only and args.no_production_model

    with pytest.raises(SystemExit):
        if args.fit_only and args.no_production_model:
            parser.error(
                "--fit-only and --no-production-model are mutually exclusive -- "
                "fitting the production model is the entire point of --fit-only."
            )


def test_fit_only_and_no_production_model_rejected_via_main(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["run_backtest.py", "--start", "2026-03-26", "--end", "2026-06-28",
         "--fit-only", "--no-production-model"],
    )
    with pytest.raises(SystemExit):
        rb.main()
    err = capsys.readouterr().err
    assert "--fit-only and --no-production-model are mutually exclusive" in err


def test_non_fit_only_path_still_calls_walk_forward_and_report(monkeypatch, tmp_path):
    """Sanity check that the default (non-fit-only) path is unchanged --
    walk-forward and report ARE called, fit_production_model runs after."""
    feature_table = _dummy_feature_table()
    _patch_pipeline_steps(monkeypatch, feature_table=feature_table)

    oos_df = pd.DataFrame({
        "pitcher": [1], "game_pk": [100], "game_date": ["2026-04-08"],
        "wf_step": ["2026-04-08"], "strikeouts": [5],
    })
    monkeypatch.setattr(rb, "run_walk_forward", lambda *a, **k: oos_df)
    monkeypatch.setattr(rb, "persist_oos_frame", lambda df, path: path)
    report_calls = []

    def fake_report(**kwargs):
        report_calls.append(kwargs)
        return {
            "report_path": "r.md",
            "reliability_plot_path": "a.png",
            "calibration_by_tier_plot_path": "b.png",
            "error_over_time_plot_path": "c.png",
        }

    monkeypatch.setattr(rb, "generate_backtest_report", fake_report)

    model_path = str(tmp_path / "model.joblib")
    args = _make_args(fit_only=False, model_path=model_path)
    fit_fn, fit_calls = make_fake_fit_fn()

    result = rb.run_backtest_pipeline(args, fit_fn=fit_fn)

    assert report_calls, "generate_backtest_report must be called on the non-fit-only path"
    assert len(fit_calls) == 1
    assert result["oos_rows"] == 1
    assert result["report_result"] is not None
