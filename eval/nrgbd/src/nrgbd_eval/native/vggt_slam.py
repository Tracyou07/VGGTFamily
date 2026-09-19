import sys
import time
from pathlib import Path
import numpy as np


def infer(req):
    import os

    os.environ["TORCH_HOME"] = req["settings"]["torch_home"]
    import torch

    source_root = Path(req["source_root"]).resolve()
    sys.path[:0] = [
        str(source_root),
        str(source_root / "third_party" / "vggt"),
        str(source_root / "third_party" / "salad"),
    ]
    from . import _slam_support as helper

    max_loops = int(req["settings"].get("max_loops", 1))
    if max_loops == 0:
        helper.install_noop_salad_import()
    import vggt_slam.solver as solver_module

    solver_module.Viewer = helper.NullViewer
    if max_loops == 0:
        solver_module.ImageRetrieval = helper.NoOpImageRetrieval
    else:
        with helper.offline_dinov2_hub():
            shared_retrieval = solver_module.ImageRetrieval()
        solver_module.ImageRetrieval = lambda: shared_retrieval
    from vggt_slam.solver import Solver

    model = helper.load_model(req["checkpoint"], req["device"])
    solver = Solver(
        init_conf_threshold=float(req["settings"].get("conf_threshold", 1.5)),
        lc_thres=float(req["settings"].get("lc_thres", 0.95)),
    )
    submap_size = int(req["settings"].get("submap_size", 16))
    torch.cuda.reset_peak_memory_stats()
    forward_start = model.total_forward_seconds
    started = time.perf_counter()
    with torch.no_grad():
        for window in helper.iter_submap_windows(
            [Path(x) for x in req["rgb_paths"]], submap_size
        ):
            predictions = solver.run_predictions(
                [str(x) for x in window], model, max_loops, None, None
            )
            solver.add_points(predictions)
            solver.graph.optimize()

    by_id = {}
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        points, frame_ids, masks = submap.get_points_list_in_world_frame(solver.graph)
        for point_map, frame_id, mask in zip(points, frame_ids, masks):
            by_id.setdefault(str(int(round(frame_id))), (point_map, mask))
    missing = [x for x in req["frame_ids"] if x not in by_id]
    if missing:
        raise RuntimeError(
            f"VGGT-SLAM did not export requested frame IDs: {missing[:10]}"
        )
    dense = np.stack([by_id[x][0] for x in req["frame_ids"]]).astype(np.float32)
    valid = np.stack([by_id[x][1] for x in req["frame_ids"]]).astype(bool)
    return {
        "world_points": dense,
        "valid_masks": valid,
        "inference_seconds": model.total_forward_seconds - forward_start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "pipeline_seconds": time.perf_counter() - started,
        "loop_closures": solver.graph.get_num_loops(),
    }
