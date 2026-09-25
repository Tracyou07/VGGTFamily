"""Opt-in process-local backend policies, selected before CUDA initialization."""
import argparse
import os

PROFILES=("legacy","native_vggt")
LEGACY_CUBLAS=":4096:8"

def prepare_environment(profile,environ=None,explicit=True):
    if profile not in PROFILES:
        raise ValueError(f"unknown backend profile: {profile}")
    if environ is None:environ=os.environ
    current=environ.get("CUBLAS_WORKSPACE_CONFIG")
    if profile=="legacy":
        if explicit and current not in (None,LEGACY_CUBLAS):
            raise ValueError("legacy requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        environ.setdefault("CUBLAS_WORKSPACE_CONFIG",LEGACY_CUBLAS)
    elif current is not None:
        raise ValueError("native_vggt requires CUBLAS_WORKSPACE_CONFIG unset; remove it in the launcher")
    return profile

def early_prepare(argv,environ=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend-profile",choices=PROFILES)
    args,_=parser.parse_known_args(argv)
    profile=args.backend_profile or "legacy"
    prepare_environment(profile,environ,explicit=args.backend_profile is not None)
    return profile

def snapshot(torch):
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        sdpa=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
            memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math=torch.backends.cuda.math_sdp_enabled()),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        torch=str(torch.__version__),cuda=str(torch.version.cuda),
        cudnn_version=torch.backends.cudnn.version())

def apply_backend_profile(profile,torch):
    if profile not in PROFILES:raise ValueError(f"unknown backend profile: {profile}")
    if torch.cuda.is_initialized():
        raise RuntimeError("backend profile must be applied before CUDA initialization")
    if profile=="legacy":
        # Preserve the original v7 assignments, including its untouched flags.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False, warn_only=False)
        torch.backends.cudnn.deterministic=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=True
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    return profile

def restore_snapshot(torch,state):
    torch.use_deterministic_algorithms(state["deterministic_algorithms"],
        warn_only=state["warn_only"])
    torch.backends.cudnn.deterministic=state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark=state["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32=state["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(state["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32=state["matmul_allow_tf32"]
    torch.backends.cuda.enable_flash_sdp(state["sdpa"]["flash"])
    torch.backends.cuda.enable_mem_efficient_sdp(state["sdpa"]["memory_efficient"])
    torch.backends.cuda.enable_math_sdp(state["sdpa"]["math"])
