from .runtime import RuntimeBackend, doctor_backend as doctor_backend

NAMES = ("vggt", "vggt_long", "streamvggt", "vggt_slam", "vggt_omega")


def create_backend(name, config, device):
    if name not in NAMES:
        raise ValueError(f"unknown backend {name}; choose {NAMES}")
    return RuntimeBackend(name, config, device)
