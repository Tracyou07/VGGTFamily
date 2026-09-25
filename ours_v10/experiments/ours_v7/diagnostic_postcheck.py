"""CPU downstream regression after exact full-window output checks."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.ours_v7.diagnostics import difference,write_json,csv_rows
from vggt.v5.alignment import Stitcher,LongAlignmentConfig
from experiments.compare_long.evaluate import evaluate_poses
def run(out):
    import csv
    checks=list(csv.DictReader(open(out/'patch_cached_keys_regression.csv')))
    assert len(checks)>=42 and all(r['exact']=='True' and r['finite']=='True' for r in checks)
    # Every freshly optimized prediction equals this frozen source elementwise.
    # Reuse it to avoid storing another 600 MB of identical raw predictions.
    source=Path('/data/yjh/output/vggt/ours_v7/20260922T023028Z_v7_f100_camera_patch_exchange')
    stitch=Stitcher(out/'cached_keys_alignment_replay',LongAlignmentConfig())
    for w in range(3):
        with np.load(source/'camera_patch_exchange/windows'/f'{w:04d}'/'local.npz') as z:
            prediction={k:z[k] for k in z.files}
        stitch.add(prediction,w)
    ids=[f'{i:06d}' for i in range(100)]
    result=stitch.finish(ids)
    with np.load(source/'camera_patch_exchange/global_trajectory.npz') as z:
        original=z['c2w']
    delta=difference(original,result['c2w']);assert delta['exact']
    scene=json.load(open(source/'config.json'))['base']['scene_root']
    before,_=evaluate_poses(ids,original,scene);after,_=evaluate_poses(ids,result['c2w'],scene)
    assert before==after
    write_json(out/'optimized_downstream_regression.json',dict(raw_comparison_rows=len(checks),
        raw_all_elementwise_exact=True,stitching=delta,evaluation_identical=True,
        evaluation=after,
        prediction_reuse='all seven fresh optimized raw fields exactly match frozen source; reuse avoids duplicate large artifacts',
        formal_gt_scope='one whole trajectory Sim3; GT never participates in prediction or stitching'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
