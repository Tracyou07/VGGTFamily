import numpy as np
import pytest
from virtual_kitti_eval.metrics import ate_rmse_m, umeyama_sim3

def poses(points):
    result = np.repeat(np.eye(4)[None], len(points), axis=0)
    result[:, :3, 3] = points
    return result

XYZ = np.array([[0.,0,0], [1,0,0], [0,2,0], [0,0,3]])
IDS = ("00000", "00001", "00002", "00003")

def test_sim3_removes_known_scale_rotation_translation_and_matches_ids():
    # GT = 2 * rotation90(pred) + (4,-3,2), manually derived.
    gt = np.array([[4.,-3,2], [4,-1,2], [0,-3,2], [4,-3,8]])
    order = [3,1,0,2]
    score = ate_rmse_m(tuple(IDS[i] for i in order), poses(XYZ[order]), IDS, poses(gt))
    assert score.rmse_m < 1e-12
    assert score.matched_frames == 4
    assert score.alignment.scale == pytest.approx(2.)
    np.testing.assert_allclose(score.alignment.translation, [4,-3,2], atol=1e-12)

def test_nonzero_ate_has_hand_calculated_residual():
    # Centered orthogonal axes. Anisotropic lengths 2,3,4 have optimal scale 3.
    pred = np.array([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    gt = pred * [2,3,4]
    ids = tuple(map(str, range(6)))
    score = ate_rmse_m(ids, poses(pred), ids, poses(gt))
    assert score.rmse_m == pytest.approx(np.sqrt(2/3))
    assert score.alignment.scale == pytest.approx(3.)

@pytest.mark.parametrize("points", [
    [[0.,0,0],[1,0,0]], [[0.,0,0],[1,0,0],[2,0,0]], [[0.,0,0]] * 4])
def test_degenerate_sim3_is_rejected(points):
    with pytest.raises(ValueError, match="DEGENERATE"):
        umeyama_sim3(points, points)

def test_reflection_and_nonfinite_rejected():
    with pytest.raises(ValueError, match="REFLECTION"):
        umeyama_sim3(XYZ, XYZ * [-1,1,1])
    bad = XYZ.copy(); bad[0,0] = np.nan
    with pytest.raises(ValueError): umeyama_sim3(bad, XYZ)

@pytest.mark.parametrize("change", ["ids", "duplicate", "scale_rotation", "complex"])
def test_exact_frame_and_rigid_real_pose_contract(change):
    pred = poses(XYZ); ids = IDS
    if change == "ids": ids = ("other", *IDS[1:])
    if change == "duplicate": ids = (IDS[0], *IDS[:3])
    if change == "scale_rotation": pred[:,0,0] = 2
    if change == "complex": pred = pred.astype(complex) + 1j
    with pytest.raises(ValueError): ate_rmse_m(ids, pred, IDS, poses(XYZ))
