from pathlib import Path
import numpy as np
import pytest


def test_measurement_excludes_load_and_snapshot_time_and_covers_native_pipeline(monkeypatch):
    from kitti_eval.resources import measure_inference
    from kitti_eval.backends.common import BackendPrediction
    events, time = [], [0.]
    class Backend:
        def load(self):
            events.append("load")
            time[0] += 100
        def infer(self, frame_ids, image_paths):
            assert frame_ids == ("000007",) and image_paths == (Path("000007.png"),)
            for stage in ("preprocess", "forward", "reconstruct", "stitch", "loop"):
                events.append(stage)
                time[0] += .5
            return BackendPrediction(frame_ids, np.eye(4)[None], None,
                {"pose_convention": "c2w", "pose_scale": "rigid"})
    class Cuda:
        def reset_peak_memory_stats(self): events.append("reset")
        def synchronize(self): events.append("sync")
        def max_memory_allocated(self): return 128 * 2**20
        def max_memory_reserved(self): return 192 * 2**20
    def clock():
        events.append("clock")
        return time[0]
    def smi(label):
        events.append(label)
        time[0] += 7
        return {"label": label}
    monkeypatch.setattr("kitti_eval.resources._smi", smi)
    prediction, usage = measure_inference(Backend(), ("000007",), (Path("000007.png"),), Cuda(), clock)
    assert events == ["load", "reset", "before", "sync", "clock", "preprocess", "forward",
                      "reconstruct", "stitch", "loop", "sync", "clock", "after"]
    assert usage.inference_seconds == pytest.approx(2.5)
    assert usage.peak_allocated_mib == 128
    assert usage.peak_reserved_mib == 192
    assert usage.nvidia_smi_before == {"label": "before"}
    assert usage.nvidia_smi_after == {"label": "after"}
    assert prediction.frame_ids == ("000007",)
    time[0] += 500  # Downstream metric work is not part of the captured interval.
    assert usage.inference_seconds == 2.5


@pytest.mark.parametrize("field,value", [
    ("inference_seconds", -1), ("peak_allocated_mib", float("nan")),
    ("peak_reserved_mib", float("inf")), ("peak_allocated_mib", True)])
def test_resource_numbers_are_finite_nonnegative(field, value):
    from kitti_eval.resources import ResourceUsage
    values = dict(inference_seconds=1., peak_allocated_mib=2., peak_reserved_mib=3.,
                  nvidia_smi_before={}, nvidia_smi_after={})
    values[field] = value
    with pytest.raises(ValueError):
        ResourceUsage(**values)
