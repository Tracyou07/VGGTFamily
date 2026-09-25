"""Read-only analysis of completed campaign outputs; no model execution."""
import json,sys
from pathlib import Path
import numpy as np

r=Path(sys.argv[1]); summary={}
lines=['# ours_v4 validation report','',f'Campaign: `{r}`','',
       'Implementation tested: `ccebb0b61ee8d886e836e8ea45499b92ec7a45cb`. CPU tests: 16/16 passed.',
       'Original VGGT*: `cc1d8ac15861aea54d14961653cd340e7d984f29`. Separate original reference package, unmodified model attention.',
       'Stitching source: ours_v3 `d54c6e2c6e8e0491d04bd7d4105e455217aeb884`.',
       'Checkpoint SHA256: `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.',
       'Same fixed scene0150_00 frame IDs and preprocessed tensors; no GT or alignment in equivalence comparisons.','',
       '## Equivalence gates','',
       '55 frames: [0,30), [20,50), [40,55); packed groups [[0,1],[2]]. Full encoder and head-cache tensors were compared, not samples.','']
for precision in ('fp32','bf16'):
 for stage in ('repeat','equivalence'):
    p=r/f'{precision}_{stage}_report.json'
    if not p.exists():
        lines.append(f'- {precision} {stage}: not reached.'); continue
    d=json.loads(p.read_text()); summary[p.stem]=d
    lines.append(f'- {precision} {stage}: **{"PASS" if d["passed"] else "FAIL"}**; fixed tolerances `{d["tolerance"]}`.')
    for win,rows in d['windows'].items():
        failed={k:v for k,v in rows.items() if v.get('failed_elements',0)}
        pose=rows['pose_geometry']
        lines.append(f'  - {win}: center {pose["max_center_m"]:.8g} m; rotation {pose["max_rotation_deg"]:.8g} deg; failed tensor fields: {list(failed)}.')
        for k,v in failed.items(): lines.append(f'    - {k}: max abs {v["max_abs"]:.8g}, mean abs {v["mean_abs"]:.8g}, out-of-tolerance elements {v["failed_elements"]}.')
lines+=['','## Resources','', '| Run | Forward seconds | Allocated GiB | Reserved GiB | CPU peak GiB |','|---|---:|---:|---:|---:|']
for folder in sorted(r.iterdir()):
 p=folder/'manifest.json'
 if not p.exists(): continue
 d=json.loads(p.read_text())
 if 'inference_seconds' not in d: continue
 summary[folder.name]=d
 lines.append(f'| {folder.name} | {d["inference_seconds"]:.3f} | {d["peak_allocated_bytes"]/2**30:.3f} | {d["peak_reserved_bytes"]/2**30:.3f} | {d["cpu_peak_rss_bytes"]/2**30:.3f} |')
lines+=['','Gate resource measurements include feature-capture hooks. They are not a substitute for the 100-frame timing comparison. Peak reserved memory can increase with batching.','', '## 100-frame comparison','']
for name in ('reference100','packed100'):
 p=r/name/'trajectory_metrics.json'
 if p.exists():
    metrics=json.loads(p.read_text()); lines.append(f'- {name}: ATE {metrics["ate_rmse_m"]:.8f} m (one global Sim3 for evaluation only).'); summary[name+'_metrics']=metrics
 else: lines.append(f'- {name}: no valid completed trajectory metrics.')
if all((r/n/'COMPLETE.json').exists() for n in ('reference100','packed100')):
 diffs={}
 for p in sorted((r/'reference100/windows').glob('*/local.npz')):
    a=np.load(p); b=np.load(r/'packed100/windows'/p.parent.name/'local.npz')
    rows={}
    for k in ('pose_encoding','c2w','intrinsics','depth','confidence'):
        delta=np.abs(a[k]-b[k]); rows[k]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()))
    diffs[p.parent.name]=rows
 summary['100_frame_raw_window_differences']=diffs
 (r/'100_frame_raw_window_differences.json').write_text(json.dumps(diffs,indent=2))
 lines.append('- Raw per-window numerical differences: `100_frame_raw_window_differences.json`; no Sim3 applied to this comparison.')
for name in ('reference100','packed100'):
 folder=r/name
 if folder.exists():
    summary[name+'_edges']=[json.loads(p.read_text()) for p in sorted((folder/'alignment').glob('edge_*.json'))]
    p=folder/'boundary_diagnostics.json'
    if p.exists(): summary[name+'_boundaries']=json.loads(p.read_text())
failure=r/'FAILED.json'
if failure.exists():
 summary['failure']=json.loads(failure.read_text()); lines+=['',f'**Stopped:** {summary["failure"]["reason"]}. No thresholds relaxed, no automatic retry, no downstream experiment started after failure.']
lines+=['','## Historical v3 context','',
 'Existing v3 100-frame results (20260920T112529Z): local_shared_ref ATE 0.10254751 m; camera_global 0.10510145 m. v3 uses a different shared-state backbone, so comparisons cannot be interpreted as a batching-only effect.',
 '', '## Limitations','',
 'Only the fixed 30/10 window campaign is validated here. No 1000-frame run, no feature cache, no training. Full FP32/BF16 validity requires both gates to pass. Failed tensor equivalence cannot be overridden by a visually plausible point cloud or a lower ATE.',
 '', 'Commands and architecture: `README_OURS_V4.md`. Gate configuration: `configs/v4_validation.json`. Existing versions and outputs were left untouched.']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
(r/'summary.json').write_text(json.dumps(summary,indent=2))
print('\n'.join(lines))
