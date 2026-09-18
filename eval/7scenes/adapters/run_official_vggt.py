"""Run the FastVGGT 7-Scenes protocol with the canonical VGGT package."""

from pathlib import Path
import importlib.util
import runpy
import sys
import torch


ADAPTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPTER_DIR.parents[2]
OFFICIAL_VGGT_ROOT = PROJECT_ROOT / "vggt"
FASTVGGT_ROOT = PROJECT_ROOT / "eval" / "7scenes" / "reference" / "FastVGGT-main"
FASTVGGT_EVAL = FASTVGGT_ROOT / "eval"

sys.path[:0] = [
    str(OFFICIAL_VGGT_ROOT),
    str(FASTVGGT_EVAL),
]

import vggt.models.vggt as canonical_vggt_module

eval_utils_path = FASTVGGT_ROOT / "vggt" / "utils" / "eval_utils.py"
eval_utils_spec = importlib.util.spec_from_file_location(
    "vggt.utils.eval_utils", eval_utils_path
)
eval_utils_module = importlib.util.module_from_spec(eval_utils_spec)
sys.modules["vggt.utils.eval_utils"] = eval_utils_module
eval_utils_spec.loader.exec_module(eval_utils_module)

# Add the reference root only after the namespace package has resolved to the
# canonical repository. Its presence prevents eval_7andN.py from promoting the
# bundled FastVGGT model to sys.path[0].
sys.path.append(str(FASTVGGT_ROOT))


if "--verify-import" in sys.argv:
    print(canonical_vggt_module.__file__)
    print(f"eval_utils={eval_utils_module.__file__}")
    raise SystemExit(0)


CanonicalVGGT = canonical_vggt_module.VGGT


_torch_load = torch.load


def _load_checkpoint(path, *args, **kwargs):
    """Load the shared VGGT safetensors without creating a local .pt copy."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return _torch_load(path, *args, **kwargs)


torch.load = _load_checkpoint


class ProtocolCompatibleVGGT(CanonicalVGGT):
    """Ignore FastVGGT-only constructor switches without changing VGGT."""

    def __init__(self, *args, merging=None, merge_ratio=None, **kwargs):
        del merging, merge_ratio
        super().__init__(*args, **kwargs)

    def to(self, *args, **kwargs):
        # This canonical code snapshot disables autocast around its heads and
        # feeds them float32 aggregator outputs. FastVGGT's evaluator casts the
        # whole model to bf16, which makes the head weights incompatible.
        if args == (torch.bfloat16,) and not kwargs:
            return self
        if not args and kwargs.get("dtype") is torch.bfloat16 and "device" not in kwargs:
            return self
        return super().to(*args, **kwargs)


canonical_vggt_module.VGGT = ProtocolCompatibleVGGT


if __name__ == "__main__":
    print(f"Canonical VGGT module: {canonical_vggt_module.__file__}", flush=True)
    runpy.run_path(str(FASTVGGT_EVAL / "eval_7andN.py"), run_name="__main__")
