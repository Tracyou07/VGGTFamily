# Third-party notices

The evaluation infrastructure contains reference code from FastVGGT and evo. Preserve the source notices and license files when redistributing or modifying these files.

- FastVGGT: `eval/7scenes/reference/FastVGGT-main/LICENSE.txt` and `eval/scannet/reference/FastVGGT/LICENSE.txt`. ScanNet source provenance is recorded in `eval/scannet/reference/FastVGGT/SOURCE.json`.
- evo 1.32.0: `eval/scannet/reference/evo-1.32.0/LICENSE` and `SOURCE.json`.
- ScanNet vendored scoring utilities: consult `eval/scannet/scannet_eval/vendor/fastvggt_eval_utils.py` and the FastVGGT reference notices.

The source-only `ours_v10/` model snapshot includes VGGT-derived code and a copy of its license at `ours_v10/LICENSE.txt`, plus vendored and reference code retained in that tree. Preserve their original notices and check the applicable terms before redistribution. Datasets and checkpoints are not bundled.