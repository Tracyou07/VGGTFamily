"""Native bridge for the original VGGT backend."""

from ._vggt_core import infer_full


def infer(request):
    return infer_full(request)
