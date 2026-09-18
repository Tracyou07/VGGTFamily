import numpy as np
import pytest


def api():
    from kitti_eval import metrics
    return metrics


def poses(xyz):
    out = np.repeat(np.eye(4)[None], len(xyz), axis=0)
    out[:, :3, 3] = xyz
    return out


@pytest.fixture
def xyz():
    return np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [2, 1, 1]])


def test_known_sim3_and_exact_id_reordering(xyz):
    m = api()
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    gt = 2.5 * xyz @ rotation.T + [3, -4, 7]
    ids = tuple(map(str, range(len(xyz))))
    result = m.ate_rmse_m(ids[::-1], poses(xyz[::-1]), ids, poses(gt))
    assert result.rmse_m == pytest.approx(0, abs=1e-12)
    assert result.matched_frames == 5
    assert result.alignment.scale == pytest.approx(2.5)
    np.testing.assert_allclose(result.alignment.rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(result.alignment.translation, [3, -4, 7], atol=1e-12)


def test_noise_has_hand_calculated_rmse():
    # Paired points along each axis; anisotropic scale (2,1,1).
    # Best uniform scale is 4/3; mean squared residual = 2/9.
    xyz = np.vstack([np.eye(3), -np.eye(3)])
    gt = xyz * [2, 1, 1]
    result = api().ate_rmse_m(tuple("abcdef"), poses(xyz), tuple("abcdef"), poses(gt))
    assert result.rmse_m == pytest.approx(np.sqrt(2 / 9), abs=1e-12)


@pytest.mark.parametrize("xyz", [
    np.zeros((2, 3)), np.zeros((3, 3)), np.array([[0.,0,0],[1,0,0],[2,0,0]])
])
def test_degenerate_trajectories_rejected(xyz):
    with pytest.raises(ValueError, match="DEGENERATE"):
        api().umeyama_sim3(xyz, xyz)


def test_planar_triangle_is_valid():
    xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    assert api().umeyama_sim3(xyz, xyz).scale == pytest.approx(1)


def test_reflection_rejected(xyz):
    with pytest.raises(ValueError, match="REFLECTION"):
        api().umeyama_sim3(xyz, xyz * [-1, 1, 1])


@pytest.mark.parametrize("ids", [("a", "a", "c"), ("a", "b", "x")])
def test_duplicate_or_different_ids_rejected(ids):
    xyz = poses(np.eye(3))
    with pytest.raises(ValueError, match="FRAME_ID"):
        api().match_poses_by_frame_id(ids, xyz, ("a", "b", "c"), xyz)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_poses_and_centers_rejected(xyz, bad):
    xyz[0, 0] = bad
    with pytest.raises(ValueError, match="NONFINITE"):
        api().umeyama_sim3(xyz, xyz)
    with pytest.raises(ValueError, match="NONFINITE"):
        api().match_poses_by_frame_id(tuple("abcde"), poses(xyz), tuple("abcde"), poses(xyz))


def test_pose_rotation_and_shape_are_validated(xyz):
    malformed = poses(xyz)
    malformed[0, 0, 0] = -1
    with pytest.raises(ValueError, match="POSE"):
        api().match_poses_by_frame_id(tuple("abcde"), malformed, tuple("abcde"), poses(xyz))
