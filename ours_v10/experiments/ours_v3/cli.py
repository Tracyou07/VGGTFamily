"""Explicit configuration, immutable run directories, no automatic experiments."""
import argparse
import json
from pathlib import Path
from .runner import Config, run, resolve_frames
from .geometry import AlignmentConfig


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--frames',type=int)
    parser.add_argument('--frame-manifest')
    parser.add_argument('--device')
    parser.add_argument('--single-window-gate',action='store_true')
    parser.add_argument('--check-inputs',action='store_true',help='validate paths and exact frames without loading checkpoint/model')
    args=parser.parse_args(argv)
    values=json.loads(args.config.read_text())
    values['alignment']=AlignmentConfig(**values.get('alignment',{}))
    for key in ('frames','frame_manifest','device'):
        if getattr(args,key) is not None: values[key]=getattr(args,key)
    if args.single_window_gate: values['single_window_gate']=True
    config=Config(**values); config.validate()
    if args.check_inputs:
        ids,paths,source=resolve_frames(config)
        print(json.dumps(dict(frame_count=len(ids),first=str(paths[0]),last=str(paths[-1]),frame_manifest=source),indent=2))
        return 0
    if args.output is None: parser.error('--output is required unless --check-inputs')
    run(config,args.output)
    return 0
