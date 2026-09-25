"""Whole-trajectory summaries; GT is read only after stitching succeeds."""
import json
from pathlib import Path
import numpy as np


COMMON_PAIRS = (("000059", "000060"), ("000089", "000090"))


def _rmse(rows, key):
    return float(np.sqrt(np.mean(np.square([row[key] for row in rows])))) if rows else None


def summarize(output):
    output = Path(output)
    rows = json.loads((output / "adjacent_pose_errors.json").read_text())
    if not rows:
        raise ValueError("trajectory needs at least two frames")
    boundary = [row for row in rows if row["boundary"]]
    internal = [row for row in rows if not row["boundary"]]
    if len(boundary) + len(internal) != len(rows):
        raise ValueError("incomplete adjacent frame classification")
    def metrics(data):
        return dict(count=len(data),
                    translation_rmse_m=_rmse(data, "translation_error"),
                    rotation_rmse_deg=_rmse(data, "rotation_error_deg"))
    common = []
    for before, after in COMMON_PAIRS:
        match = [row for row in rows if str(row["before"]) == before and str(row["after"]) == after]
        if match:
            common.append(dict(pair=f"{before}->{after}", is_ownership_boundary=bool(match[0]["boundary"]),
                               translation_error_m=match[0]["translation_error"],
                               rotation_error_deg=match[0]["rotation_error_deg"]))
    ate = json.loads((output / "trajectory_metrics.json").read_text())["ate_rmse_m"]
    result = dict(ate_rmse_m=ate, adjacent=metrics(rows), ownership_boundaries=metrics(boundary),
                  within_window=metrics(internal), common_pairs=common,
                  gt_alignment="one whole-trajectory proper Sim(3); never per window")
    (output / "evaluation_summary.json").write_text(json.dumps(result, indent=2))
    return result
