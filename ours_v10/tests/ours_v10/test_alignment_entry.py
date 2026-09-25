"""v10 CPU alignment accepts new prediction sets with frozen v9 math."""
import unittest

from experiments.ours_v10.compare_frozen_sparse import parse_args


class V10AlignmentEntryTest(unittest.TestCase):
    def test_all_four_prediction_sets_can_use_sparse_joint(self):
        for mode in ("independent","camera_only","overlap_correspondence",
                     "camera_global_overlap"):
            args=parse_args(["--predictions",f"/tmp/{mode}/windows",
                "--source-manifest",f"/tmp/{mode}/run_manifest.json",
                "--prediction-set",mode,"--mode","sparse_point_camera_joint",
                "--output",f"/tmp/new_{mode}",
                "--vkitti-raw-root","/tmp/raw","--vkitti-condition","rain"])
            self.assertEqual(args.prediction_set,mode)
            self.assertEqual(args.mode,"sparse_point_camera_joint")


if __name__=="__main__":
    unittest.main()
