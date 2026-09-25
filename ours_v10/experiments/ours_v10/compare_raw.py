"""Compare frozen, unaligned local predictions for the four v10 modes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.ours_v6.runtime import sha256,write_json

FIELDS=("c2w","intrinsics","depth","world_points","world_points_conf")
PAIRS=(("independent","camera_only"),
       ("overlap_correspondence","camera_global_overlap"),
       ("independent","overlap_correspondence"))


def field_stats(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:
        raise ValueError("raw output shape differs")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite raw output")
    delta=a.astype(np.float64)-b.astype(np.float64)
    magnitude=np.abs(delta)
    denominator=max(float(np.linalg.norm(a.astype(np.float64).ravel())),1e-30)
    return dict(shape=list(a.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype),
        exact=bool(np.array_equal(a,b)),finite=True,
        within_2e5_information_only=bool(np.allclose(a,b,atol=2e-5,rtol=2e-5)),
        max_abs=float(magnitude.max()),mean_abs=float(magnitude.mean()),
        relative_l2=float(np.linalg.norm(delta.ravel())/denominator))


def execute(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError(output)
    manifests={mode:json.loads((root/mode/"run_manifest.json").read_text())
               for pair in PAIRS for mode in pair}
    first=manifests["independent"]
    for mode,m in manifests.items():
        if not (root/mode/"COMPLETE.json").is_file():
            raise ValueError(f"incomplete prediction: {mode}")
        for key in ("dataset","scene","condition","frame_ids","windows",
                    "input_sha256","image_tensor_sha256","checkpoint_sha256","precision"):
            if m[key]!=first[key]:raise ValueError(f"prediction mismatch {mode}: {key}")
        if len(m["prediction_files"])!=len(first["windows"]):
            raise ValueError(f"missing windows: {mode}")
    rows=[]
    for left,right in PAIRS:
        for window in range(len(first["windows"])):
            paths=[]
            for mode in (left,right):
                path=root/mode/"windows"/f"{window:04d}"/"local.npz"
                if sha256(path)!=manifests[mode]["prediction_files"][window]["sha256"]:
                    raise ValueError(f"prediction file hash changed: {path}")
                paths.append(path)
            with np.load(paths[0],allow_pickle=False) as a, np.load(paths[1],allow_pickle=False) as b:
                if list(a["frame_ids"])!=list(b["frame_ids"]):
                    raise ValueError("window frame IDs differ")
                for field in FIELDS:
                    rows.append(dict(left=left,right=right,window=window,field=field,
                                     **field_stats(a[field],b[field])))
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,dict(status="success",prediction_root=str(root),
                           pairs=[list(pair) for pair in PAIRS],rows=rows))
    csv_path=output.with_suffix(".csv")
    with csv_path.open("x",newline="") as stream:
        columns=("left","right","window","field","shape","dtype_a","dtype_b",
                 "exact","finite","within_2e5_information_only","max_abs",
                 "mean_abs","relative_l2")
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow(dict(row,shape=json.dumps(row["shape"])))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    execute(args.prediction_root,args.output)


if __name__=="__main__":main()
