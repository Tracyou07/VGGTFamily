"""Prediction-only Sim(3) fitting, composition and correspondence selection."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("non-positive/nonfinite scale")
        if not np.isfinite(self.rotation).all() or not np.isfinite(self.translation).all():
            raise ValueError("nonfinite transform")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(self.rotation),1,atol=1e-6):
            raise ValueError("rotation must be proper orthogonal")

    def apply(self, points):
        return self.scale * np.asarray(points) @ self.rotation.T + self.translation

    def compose(self, inner):
        """self: A->global; inner: B->A; result: B->global."""
        return Sim3(self.scale*inner.scale, self.rotation@inner.rotation,
                    self.scale*self.rotation@inner.translation+self.translation)

    def record(self):
        return dict(scale=float(self.scale), rotation=self.rotation.tolist(), translation=self.translation.tolist())


@dataclass(frozen=True)
class AlignmentConfig:
    threshold: float = 0.05  # destination-window units, never tuned using GT
    iterations: int = 256
    min_inliers: int = 100
    min_ratio: float = 0.25
    max_rmse: float = 0.05
    degeneracy_ratio: float = 1e-6
    seed: int = 17
    pixel_stride: int = 8
    max_points: int = 20000
    confidence_quantile: float = 0.5
    min_confidence: float = 0.0

    def validate(self):
        if not (np.isfinite(self.threshold) and self.threshold>0 and np.isfinite(self.max_rmse) and self.max_rmse>0):
            raise ValueError("residual thresholds must be finite and positive")
        if self.iterations<1 or self.min_inliers<3 or self.pixel_stride<1 or self.max_points<self.min_inliers:
            raise ValueError("invalid sampling/inlier limits")
        if not 0<self.min_ratio<=1 or not 0<=self.confidence_quantile<=1 or not 0<self.degeneracy_ratio<1:
            raise ValueError("invalid ratio/quantile")


def fit_sim3(source, target, degeneracy_ratio=1e-6):
    source=np.asarray(source,dtype=np.float64); target=np.asarray(target,dtype=np.float64)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<3:
        raise ValueError("need matching Nx3 correspondences with N>=3")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite correspondences")
    x=source-source.mean(0); y=target-target.mean(0)
    for values in (x,y):
        singular=np.linalg.svd(values,compute_uv=False)
        # Planar points are admissible; collinear or collapsed points are not.
        if singular[0]<1e-12 or singular[1]/singular[0]<degeneracy_ratio:
            raise ValueError("degenerate correspondences")
    u,s,vt=np.linalg.svd(y.T@x/len(x))
    correction=np.diag([1.,1.,1. if np.linalg.det(u@vt)>=0 else -1.])
    rotation=u@correction@vt
    scale=float(np.sum(s*np.diag(correction))/np.mean(np.sum(x*x,axis=1)))
    return Sim3(scale,rotation,target.mean(0)-scale*rotation@source.mean(0))


def robust_sim3(source, target, config):
    config.validate()
    source=np.asarray(source,dtype=np.float64); target=np.asarray(target,dtype=np.float64)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3:
        raise ValueError("invalid correspondence shapes")
    if len(source)<config.min_inliers or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("insufficient valid correspondences")
    rng=np.random.default_rng(config.seed); best=None; score=(-1,-np.inf)
    for _ in range(config.iterations):
        sample=rng.choice(len(source),3,replace=False)
        try: transform=fit_sim3(source[sample],target[sample],config.degeneracy_ratio)
        except ValueError: continue
        residual=np.linalg.norm(transform.apply(source)-target,axis=1)
        mask=residual<=config.threshold
        candidate=(int(mask.sum()),-float(residual[mask].mean()) if mask.any() else -np.inf)
        if candidate>score: best=mask; score=candidate
    if best is None or best.sum()<config.min_inliers:
        raise ValueError("RANSAC failed: insufficient inliers or degenerate geometry")
    for _ in range(3):
        transform=fit_sim3(source[best],target[best],config.degeneracy_ratio)
        residual=np.linalg.norm(transform.apply(source)-target,axis=1)
        updated=residual<=config.threshold
        if updated.sum()<config.min_inliers: raise ValueError("refit lost inliers")
        if np.array_equal(updated,best): break
        best=updated
    transform=fit_sim3(source[best],target[best],config.degeneracy_ratio)
    residual=np.linalg.norm(transform.apply(source)-target,axis=1)
    best=residual<=config.threshold
    count=int(best.sum()); ratio=count/len(source)
    rmse=float(np.sqrt(np.mean(residual[best]**2))) if count else float('inf')
    if count<config.min_inliers or ratio<config.min_ratio or rmse>config.max_rmse:
        raise ValueError(f"alignment rejected: inliers={count}, ratio={ratio}, rmse={rmse}")
    return transform,dict(inlier_count=count,inlier_ratio=ratio,inlier_rmse=rmse,
                          all_residual_median=float(np.median(residual)),
                          correspondence_count=len(source),config=asdict(config))


def transform_predictions(c2w, depth, transform):
    result=np.asarray(c2w).copy()
    result[:,:3,:3]=transform.rotation@result[:,:3,:3]
    result[:,:3,3]=transform.apply(result[:,:3,3])
    return result,np.asarray(depth)*transform.scale


def append_unique(seen, frame_ids):
    if len(set(frame_ids))!=len(frame_ids): raise ValueError("duplicate local frame IDs")
    indices=[]
    for i, frame in enumerate(frame_ids):
        if frame not in seen: indices.append(i); seen.add(frame)
    return indices


def unproject_pixels(depth, intrinsic, c2w, rows, cols):
    pixels=np.stack([cols,rows,np.ones_like(cols)],axis=-1)
    rays=pixels@np.linalg.inv(intrinsic).T
    camera=rays*depth[rows,cols,None]
    return camera@c2w[:3,:3].T+c2w[:3,3]


def overlap_correspondences(a, b, config):
    """Return B->A pairs at exactly the same frame IDs and pixel coordinates."""
    config.validate()
    a_ids=list(a['frame_ids']); b_ids=list(b['frame_ids'])
    if len(set(a_ids))!=len(a_ids) or len(set(b_ids))!=len(b_ids): raise ValueError("duplicate IDs")
    common=[frame for frame in a_ids if frame in set(b_ids)]
    if not common: raise ValueError("no common frame IDs")
    source=[]; target=[]; labels=[]
    for frame in common:
        ai=a_ids.index(frame); bi=b_ids.index(frame)
        da=np.asarray(a['depth'][ai]).squeeze(-1) if a['depth'][ai].ndim==3 else a['depth'][ai]
        db=np.asarray(b['depth'][bi]).squeeze(-1) if b['depth'][bi].ndim==3 else b['depth'][bi]
        ca=np.asarray(a['confidence'][ai]).reshape(da.shape); cb=np.asarray(b['confidence'][bi]).reshape(db.shape)
        if da.shape!=db.shape: raise ValueError("overlap image dimensions differ")
        rows,cols=np.mgrid[0:da.shape[0]:config.pixel_stride,0:da.shape[1]:config.pixel_stride]
        rows=rows.ravel(); cols=cols.ravel()
        valid=(np.isfinite(da[rows,cols]) & (da[rows,cols]>0) & np.isfinite(db[rows,cols]) & (db[rows,cols]>0)
               & np.isfinite(ca[rows,cols]) & np.isfinite(cb[rows,cols]))
        if not valid.any(): continue
        ta=max(config.min_confidence,float(np.quantile(ca[rows[valid],cols[valid]],config.confidence_quantile)))
        tb=max(config.min_confidence,float(np.quantile(cb[rows[valid],cols[valid]],config.confidence_quantile)))
        valid &= (ca[rows,cols]>=ta)&(cb[rows,cols]>=tb)
        rows=rows[valid]; cols=cols[valid]
        pa=unproject_pixels(da,a['intrinsics'][ai],a['c2w'][ai],rows,cols)
        pb=unproject_pixels(db,b['intrinsics'][bi],b['c2w'][bi],rows,cols)
        finite=np.isfinite(pa).all(1)&np.isfinite(pb).all(1)
        source.append(pb[finite]); target.append(pa[finite])
        labels.extend((frame,int(r),int(c)) for r,c in zip(rows[finite],cols[finite]))
    if not source: raise ValueError("no valid overlap correspondences")
    source=np.concatenate(source); target=np.concatenate(target)
    choose=np.arange(len(source))
    if len(choose)>config.max_points:
        choose=np.sort(np.random.default_rng(config.seed).choice(len(choose),config.max_points,replace=False))
    return source[choose],target[choose],[labels[i] for i in choose]
