from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import scannet_eval.fastvggt_eval as adapter
from scannet_eval.data import SceneData
from scannet_eval.fastvggt_eval import (
    FastVGGTEvaluationError,
    evaluate_prediction,
    validate_scene_inputs,
)
from scannet_eval.vendor import fastvggt_eval_utils as reused


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVO132_ORACLE_ROOT = REPOSITORY_ROOT / "reference" / "evo-1.32.0"
PRISTINE_ROOT = Path(
    os.environ.get(
        "FASTVGGT_PRISTINE_ROOT",
        "/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes/reference/FastVGGT-main",
    )
)


def _pose(yaw_degrees: float, translation: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", yaw_degrees, degrees=True).as_matrix()
    pose[:3, 3] = translation
    return pose


def _write_ply(path: Path, points: np.ndarray) -> None:
    rows = "\n".join(" ".join(str(float(value)) for value in row) for row in points)
    path.write_text(
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
        f"{rows}\n",
        encoding="utf-8",
    )


def _make_scene(root: Path) -> SceneData:
    frame_ids = (0, 2, 5, 9)
    poses = np.stack(
        [
            _pose(15.0, (1.2, -0.4, 0.3)),
            _pose(22.0, (1.8, -0.1, 0.5)),
            _pose(33.0, (2.1, 0.6, 0.8)),
            _pose(47.0, (2.9, 0.9, 1.4)),
        ]
    )
    gt_points = np.array(
        [
            [-1.3, -0.7, -0.2], [1.4, -0.5, 0.1], [-0.9, 1.2, 0.4],
            [0.8, 0.9, 1.5], [0.2, -0.1, 0.6], [1.0, 0.3, -0.4],
        ],
        dtype=np.float64,
    )
    gt_ply = root / "ground-truth-at-an-arbitrary-path.ply"
    _write_ply(gt_ply, gt_points)
    return SceneData(
        scene_id="scene0123_00",
        frame_ids=frame_ids,
        image_paths=tuple(root / f"{frame_id}.jpg" for frame_id in frame_ids),
        poses_c2w=poses,
        intrinsics_color=np.eye(3),
        gt_ply=gt_ply,
        manifest_sha256="synthetic-manifest",
    )


def _make_prediction(scene: SceneData) -> SimpleNamespace:
    first_gt = scene.poses_c2w[0]
    normalized_gt = np.linalg.inv(first_gt) @ scene.poses_c2w
    gt_w2c = np.linalg.inv(normalized_gt)
    global_rotation = Rotation.from_euler("xyz", [7.0, -13.0, 28.0], degrees=True).as_matrix()
    estimated_w2c = np.repeat(np.eye(4)[None], len(scene.frame_ids), axis=0)
    for index, pose in enumerate(gt_w2c):
        estimated_w2c[index, :3, :3] = global_rotation @ pose[:3, :3]
        estimated_w2c[index, :3, 3] = 1.7 * (global_rotation @ pose[:3, 3]) + np.array(
            [0.4, -0.8, 0.2]
        )
    estimated_w2c[2, :3, :3] = (
        Rotation.from_euler("y", 4.0, degrees=True).as_matrix()
        @ estimated_w2c[2, :3, :3]
    )
    estimated_w2c[0, :3, 3] += np.array([0.13, -0.07, 0.11])
    base = np.array(
        [
            [-1.0, -0.8, -0.3], [1.2, -0.7, 0.2], [-0.6, 1.0, 0.5],
            [0.9, 0.7, 1.2], [0.1, 0.2, 0.8], [0.7, -0.1, -0.5],
        ],
        dtype=np.float64,
    )
    point_rotation = Rotation.from_euler("z", 39.0, degrees=True).as_matrix()
    points = 0.42 * (base @ point_rotation.T) + np.array([0.3, -0.2, 0.1])
    return SimpleNamespace(
        points=points,
        poses_c2w=np.linalg.inv(estimated_w2c),
        frame_ids=scene.frame_ids,
        inference_seconds=0.125,
    )


def _run_pristine_reference(
    root: Path, scene: SceneData, prediction: SimpleNamespace, max_dist: float
) -> dict[str, float]:
    if not PRISTINE_ROOT.is_dir():
        raise AssertionError(f"missing pristine FastVGGT checkout: {PRISTINE_ROOT}")
    fixture = root / "pristine-input.npz"
    first_gt = scene.poses_c2w[0].copy()
    normalized_gt = np.linalg.inv(first_gt) @ scene.poses_c2w
    np.savez(
        fixture,
        c2ws=normalized_gt,
        first_gt=first_gt,
        frame_ids=np.asarray(scene.frame_ids),
        predicted_w2c=np.linalg.inv(prediction.poses_c2w),
        points=prediction.points,
    )
    gt_root = root / "pristine-gt"
    gt_scene = gt_root / scene.scene_id
    gt_scene.mkdir(parents=True)
    shutil.copy2(scene.gt_ply, gt_scene / f"{scene.scene_id}_vh_clean_2.ply")
    output = root / "pristine-output"
    script = """
import sys
from pathlib import Path
import numpy as np
source_root, oracle_root, fixture_path, gt_root, output_path, scene_id, max_dist = sys.argv[1:]
sys.path.insert(0, source_root)
import ast
import typing
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe

def load_official_function(path, function_name, installed_module):
    source = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    definition = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    namespace = dict(vars(installed_module))
    namespace['typing'] = typing
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function_name]

main_ape.ape = load_official_function(Path(oracle_root) / 'main_ape.py', 'ape', main_ape)
main_rpe.rpe = load_official_function(Path(oracle_root) / 'main_rpe.py', 'rpe', main_rpe)
from vggt.utils.eval_utils import evaluate_scene_and_save
data = np.load(fixture_path)
evaluate_scene_and_save(
    scene_id, data['c2ws'], data['first_gt'], data['frame_ids'].tolist(),
    list(data['predicted_w2c']), [data['points']], Path(output_path), Path(gt_root),
    float(max_dist), 125.0, False,
)
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, "-c", script, str(PRISTINE_ROOT), str(EVO132_ORACLE_ROOT),
         str(fixture), str(gt_root), str(output), scene.scene_id, str(max_dist)],
        check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))


class FastVGGTEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scene = _make_scene(self.root)
        self.prediction = _make_prediction(self.scene)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_validate_scene_inputs_decodes_finite_nonempty_gt_before_evaluation(self) -> None:
        validate_scene_inputs(self.scene)
        with self.assertRaisesRegex(FastVGGTEvaluationError, "GT PLY"):
            validate_scene_inputs(replace(self.scene, gt_ply=self.root / "missing.ply"))
        empty_ply = self.root / "empty.ply"
        _write_ply(empty_ply, np.empty((0, 3)))
        with self.assertRaisesRegex(FastVGGTEvaluationError, "empty"):
            validate_scene_inputs(replace(self.scene, gt_ply=empty_ply))
        nonfinite_ply = self.root / "nonfinite.ply"
        _write_ply(nonfinite_ply, np.array([[0.0, np.nan, 1.0]]))
        with self.assertRaisesRegex(FastVGGTEvaluationError, "finite"):
            validate_scene_inputs(replace(self.scene, gt_ply=nonfinite_ply))

    def test_validate_scene_inputs_rejects_bad_ids_transforms_and_degenerate_trajectory(self) -> None:
        short = replace(
            self.scene, frame_ids=self.scene.frame_ids[:2], image_paths=self.scene.image_paths[:2],
            poses_c2w=self.scene.poses_c2w[:2],
        )
        with self.assertRaisesRegex(FastVGGTEvaluationError, "at least 3"):
            validate_scene_inputs(short)
        with self.assertRaisesRegex(FastVGGTEvaluationError, "frame_ids"):
            validate_scene_inputs(replace(self.scene, frame_ids=(0, 2, 2, 9)))
        invalid_pose = self.scene.poses_c2w.copy()
        invalid_pose[1, 3, 0] = 1.0
        with self.assertRaisesRegex(FastVGGTEvaluationError, "rigid"):
            validate_scene_inputs(replace(self.scene, poses_c2w=invalid_pose))
        degenerate = np.repeat(self.scene.poses_c2w[:1], len(self.scene.frame_ids), axis=0)
        with self.assertRaisesRegex(FastVGGTEvaluationError, "degenerate"):
            validate_scene_inputs(replace(self.scene, poses_c2w=degenerate))

    def test_preflight_rejects_rank_one_evo_trajectories(self) -> None:
        collinear_scene_poses = np.repeat(np.eye(4)[None], len(self.scene.frame_ids), axis=0)
        collinear_scene_poses[:, 0, 3] = np.arange(len(self.scene.frame_ids))
        with self.assertRaisesRegex(FastVGGTEvaluationError, "rank"):
            validate_scene_inputs(replace(self.scene, poses_c2w=collinear_scene_poses))

        collinear_prediction_w2c = np.repeat(
            np.eye(4)[None], len(self.scene.frame_ids), axis=0
        )
        collinear_prediction_w2c[:, 1, 3] = np.arange(len(self.scene.frame_ids))
        collinear_prediction = SimpleNamespace(
            **{
                **vars(self.prediction),
                "poses_c2w": np.linalg.inv(collinear_prediction_w2c),
            }
        )
        with self.assertRaisesRegex(FastVGGTEvaluationError, "rank"):
            evaluate_prediction(self.scene, collinear_prediction, self.root / "rank-one")

    def test_reused_txt_loader_and_frame_selection_preserve_upstream_behavior(self) -> None:
        pose_dir = self.root / "pose"
        pose_dir.mkdir()
        original = [_pose(10.0, (1.0, 2.0, 3.0)), _pose(20.0, (2.0, 3.0, 4.0))]
        np.savetxt(pose_dir / "000002.txt", original[0])
        np.savetxt(pose_dir / "000009.txt", original[1])
        (pose_dir / "000010.txt").write_text("nan " * 16, encoding="utf-8")
        normalized, first, available = reused.load_poses(pose_dir)
        np.testing.assert_allclose(first, original[0], atol=1e-12)
        np.testing.assert_allclose(normalized[0], np.eye(4), atol=1e-12)
        np.testing.assert_allclose(normalized[1], np.linalg.inv(original[0]) @ original[1])
        np.testing.assert_array_equal(available, np.array([2, 9]))
        images = [self.root / f"{value}.jpg" for value in (0, 1, 2, 3, 4, 5, 6)]
        selected_ids, selected_paths, pose_indices = reused.build_frame_selection(
            images, np.array([0, 2, 3, 4, 5, 6, 8]), 4
        )
        self.assertEqual(selected_ids, [0, 2, 3, 4])
        self.assertEqual(selected_paths, [images[index] for index in (0, 2, 3, 4)])
        self.assertEqual(pose_indices, [0, 1, 2, 3])

    def test_evaluate_prediction_rejects_invalid_prediction_contract(self) -> None:
        cases = [
            (SimpleNamespace(**{**vars(self.prediction), "frame_ids": (0, 2, 5, 10)}), "frame_ids"),
            (SimpleNamespace(**{**vars(self.prediction), "points": np.empty((0, 3))}), "points"),
            (SimpleNamespace(**{**vars(self.prediction), "points": np.array([[0.0, np.inf, 1.0]])}), "finite"),
            (SimpleNamespace(**{**vars(self.prediction), "poses_c2w": np.repeat(np.eye(4)[None], 4, axis=0)}), "degenerate"),
            (SimpleNamespace(**{**vars(self.prediction), "inference_seconds": np.nan}), "inference"),
        ]
        for prediction, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(FastVGGTEvaluationError, message):
                    evaluate_prediction(self.scene, prediction, self.root / f"bad-{message}")

    def test_evaluate_prediction_rejects_missing_or_nonfinite_upstream_metrics(self) -> None:
        valid = {
            "ate": 0.1, "are": 0.2, "rpe_rot": 0.3, "rpe_trans": 0.4,
            "chamfer_distance": 0.5, "scale_factor": 1.2,
            "aligned_ate": 0.1, "aligned_are": 0.2, "aligned_rpe_rot": 0.3,
            "aligned_rpe_trans": 0.4, "aligned_chamfer_distance": 0.5,
            "aligned_scale_factor": 1.2, "inference_time_ms": 125.0,
        }
        with patch(
            "scannet_eval.fastvggt_eval.evaluate_scene_and_save",
            return_value={key: value for key, value in valid.items() if key != "chamfer_distance"},
        ):
            with self.assertRaisesRegex(FastVGGTEvaluationError, "missing"):
                evaluate_prediction(self.scene, self.prediction, self.root / "missing-metric")
        nonfinite = dict(valid, ate=np.nan, aligned_ate=np.nan)
        with patch("scannet_eval.fastvggt_eval.evaluate_scene_and_save", return_value=nonfinite):
            with self.assertRaisesRegex(FastVGGTEvaluationError, "finite"):
                evaluate_prediction(self.scene, self.prediction, self.root / "nonfinite-metric")

    def test_real_metrics_match_pristine_reference_with_scale_rotation_and_clipping(self) -> None:
        actual = evaluate_prediction(
            self.scene, self.prediction, self.root / "adapter-output", chamfer_max_dist=0.08, plot=False,
        )
        loose_clip = evaluate_prediction(
            self.scene, self.prediction, self.root / "loose-clip-output", chamfer_max_dist=0.5, plot=False,
        )
        expected = _run_pristine_reference(self.root, self.scene, self.prediction, 0.08)
        expected_keys = {
            "chamfer_distance", "ate", "are", "rpe_rot", "rpe_trans", "inference_time_ms",
            "scale_factor", "aligned_chamfer_distance", "aligned_ate", "aligned_are",
            "aligned_rpe_rot", "aligned_rpe_trans", "aligned_scale_factor",
        }
        self.assertEqual(set(actual), expected_keys)
        self.assertEqual(set(expected), expected_keys)
        for key in sorted(expected_keys):
            self.assertAlmostEqual(actual[key], expected[key], places=9, msg=key)
        self.assertNotAlmostEqual(actual["scale_factor"], 1.0, places=3)
        self.assertGreater(actual["ate"], 1e-4)
        self.assertGreater(actual["chamfer_distance"], 0.0)
        self.assertLessEqual(actual["chamfer_distance"], 0.16)
        self.assertLess(actual["chamfer_distance"], loose_clip["chamfer_distance"])
        self.assertEqual(actual["inference_time_ms"], 125.0)
        on_disk = json.loads((self.root / "adapter-output" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, actual)

    def test_importing_evaluator_does_not_load_model_specific_vggt_modules(self) -> None:
        script = """
import sys
import scannet_eval.fastvggt_eval
unexpected = {'vggt.utils.geometry', 'vggt.utils.pose_enc'} & set(sys.modules)
if unexpected:
    raise SystemExit(f'unexpected model imports: {sorted(unexpected)}')
"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_source_manifest_hashes_every_pristine_snapshot_file(self) -> None:
        snapshot = REPOSITORY_ROOT / "reference" / "FastVGGT"
        manifest = json.loads((snapshot / "SOURCE.json").read_text(encoding="utf-8"))
        expected_paths = {
            "eval/eval_scannet.py", "vggt/utils/eval_utils.py",
            "eval/scannet_50.yaml", "LICENSE.txt",
        }
        self.assertEqual(set(manifest["files"]), expected_paths)
        for relative, recorded in manifest["files"].items():
            digest = hashlib.sha256((snapshot / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, recorded["sha256"], relative)
        self.assertIn("w2c", manifest["protocol_notes"])
        self.assertIn("bbox", manifest["protocol_notes"])

    def test_protocol_is_pinned_to_independent_official_evo132_oracle(self) -> None:
        self.assertEqual(getattr(adapter, "PROTOCOL_ID", None), "fastvggt_scannet_evo132")
        manifest = json.loads(
            (REPOSITORY_ROOT / "reference" / "FastVGGT" / "SOURCE.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest.get("protocol_id"), "fastvggt_scannet_evo132")
        oracle_manifest_path = EVO132_ORACLE_ROOT / "SOURCE.json"
        self.assertTrue(oracle_manifest_path.is_file())
        oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(oracle_manifest["version"], "1.32.0")
        for relative, metadata in oracle_manifest["files"].items():
            digest = hashlib.sha256((EVO132_ORACLE_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, metadata["sha256"], relative)


if __name__ == "__main__":
    unittest.main()
