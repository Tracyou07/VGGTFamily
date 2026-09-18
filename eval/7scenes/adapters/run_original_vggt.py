"""Run the original upstream-style VGGT against the shared 7-Scenes protocol."""

from pathlib import Path
import importlib.util
import runpy
import sys
import torch

ADAPTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPTER_DIR.parents[2]
ORIGINAL_VGGT_ROOT = PROJECT_ROOT / "vggtlong" / "base_models"
FASTVGGT_ROOT = PROJECT_ROOT / "eval" / "7scenes" / "reference" / "FastVGGT-main"
FASTVGGT_EVAL = FASTVGGT_ROOT / "eval"
sys.path[:0] = [str(ORIGINAL_VGGT_ROOT), str(FASTVGGT_EVAL)]

import vggt.models.vggt as original_vggt_module

eval_utils_path = FASTVGGT_ROOT / "vggt" / "utils" / "eval_utils.py"
eval_utils_spec = importlib.util.spec_from_file_location("vggt.utils.eval_utils", eval_utils_path)
eval_utils_module = importlib.util.module_from_spec(eval_utils_spec)
sys.modules["vggt.utils.eval_utils"] = eval_utils_module
eval_utils_spec.loader.exec_module(eval_utils_module)
sys.path.append(str(FASTVGGT_ROOT))

_torch_load = torch.load

def _load_checkpoint(path, *args, **kwargs):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return _torch_load(path, *args, **kwargs)

torch.load = _load_checkpoint

class ProtocolCompatibleVGGT(original_vggt_module.VGGT):
    """Ignore FastVGGT-only constructor switches for the original VGGT."""
    def __init__(self, *args, merging=None, merge_ratio=None, enable_point=True,
                 enable_track=False, **kwargs):
        del merging, merge_ratio, enable_point, enable_track
        super().__init__(*args, **kwargs)

    def to(self, *args, **kwargs):
        # The original heads disable autocast and require their FP32 weights.
        # Keep the checkpoint in FP32 while the evaluator supplies BF16 input.
        if args == (torch.bfloat16,) and not kwargs:
            return self
        if not args and kwargs.get("dtype") is torch.bfloat16 and "device" not in kwargs:
            return self
        return super().to(*args, **kwargs)

original_vggt_module.VGGT = ProtocolCompatibleVGGT

if "--verify-import" in sys.argv:
    print(original_vggt_module.__file__)
    print(f"eval_utils={eval_utils_module.__file__}")
    raise SystemExit(0)

if __name__ == "__main__":
    print(f"Original VGGT module: {original_vggt_module.__file__}", flush=True)
    runpy.run_path(str(FASTVGGT_EVAL / "eval_7andN.py"), run_name="__main__")
