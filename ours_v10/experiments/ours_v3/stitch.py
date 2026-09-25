"""Sequential prediction-only stitcher. No GT imports or inputs."""
import json
from pathlib import Path
import numpy as np
from .geometry import Sim3, overlap_correspondences, robust_sim3, transform_predictions, append_unique


def json_write(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


class Stitcher:
    def __init__(self, directory, config):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config
        self.previous=None
        self.global_transform=Sim3(1.,np.eye(3),np.zeros(3))
        self.seen=set()
        self.frames=[]
        self.poses=[]
        self.intrinsics=[]
        self.sources=[]

    def add(self, prediction, window_id):
        local=prediction
        if self.previous is not None:
            edge=self.directory/f'edge_{window_id-1:04d}_{window_id:04d}'
            try:
                source,target,labels=overlap_correspondences(self.previous,local,self.config)
                adjacent,stats=robust_sim3(source,target,self.config)
                self.global_transform=self.global_transform.compose(adjacent)
                residual=np.linalg.norm(adjacent.apply(source)-target,axis=1)
                np.savez_compressed(str(edge)+'.npz',source_B=source,target_A=target,
                                    aligned_B_in_A=adjacent.apply(source),residual=residual,
                                    inliers=residual<=self.config.threshold,
                                    frame_ids=np.asarray([row[0] for row in labels]),
                                    pixels_rc=np.asarray([row[1:] for row in labels]))
                json_write(str(edge)+'.json',dict(status='success',direction='B_local -> A_local',
                           composition='S_B_global = S_A_global compose S_B_to_A',
                           adjacent=adjacent.record(),global_transform=self.global_transform.record(),**stats))
            except (ValueError,np.linalg.LinAlgError) as error:
                json_write(str(edge)+'.json',dict(status='failed',reason=str(error),direction='B_local -> A_local'))
                raise ValueError(f'window {window_id} alignment failed: {error}') from error
        pose,depth=transform_predictions(local['c2w'],local['depth'],self.global_transform)
        fresh=append_unique(self.seen,list(local['frame_ids']))
        for i in fresh:
            self.frames.append(local['frame_ids'][i]); self.poses.append(pose[i])
            self.intrinsics.append(local['intrinsics'][i]); self.sources.append(window_id)
        json_write(self.directory/f'window_{window_id:04d}_transform.json',
                   dict(direction='window_local -> global',**self.global_transform.record(),
                        appended_frame_ids=[local['frame_ids'][i] for i in fresh]))
        self.previous=local
        return fresh,pose,depth

    def finish(self, expected_ids):
        if self.frames!=list(expected_ids): raise ValueError('final frame mapping is not exact')
        return dict(frame_ids=np.asarray(self.frames),c2w=np.asarray(self.poses),
                    intrinsics=np.asarray(self.intrinsics),source_window=np.asarray(self.sources))
