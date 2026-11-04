"""ONNX export of the three models, with a parity check against PyTorch.

Each export takes RGB values in [0, 1]; the detector accepts any height and width that are multiples of 32, and the
two face models accept any batch size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from facekit.detect import OUTPUT_NAMES

OPSET = 17


def _export(model: nn.Module, example: torch.Tensor, path: Path, output_names: list[str], dynamic_axes: dict) -> Path:
    model = model.eval().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["image"], output_names=output_names, dynamic_axes=dynamic_axes, opset_version=OPSET)
    try:
        torch.onnx.export(model, (example,), str(path), dynamo=False, **kwargs)
    except TypeError:   # older torch without the dynamo switch
        torch.onnx.export(model, (example,), str(path), **kwargs)
    return path


def export_detector(model: nn.Module, path: str | Path, size: int = 640) -> Path:
    """Exports FaceDetNet with dynamic batch, height and width."""
    axes = {"image": {0: "batch", 2: "height", 3: "width"}}
    axes.update({name: {0: "batch", 2: "map_height", 3: "map_width"} for name in OUTPUT_NAMES})

    class _Tuple(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            return tuple(out[name] for name in OUTPUT_NAMES)

    return _export(_Tuple(model), torch.rand(1, 3, size, size), Path(path), list(OUTPUT_NAMES), axes)


def export_classifier(model: nn.Module, path: str | Path, size: int, output_name: str) -> Path:
    """Exports a face model (expression or embedding) with a dynamic batch dimension."""
    axes = {"image": {0: "batch"}, output_name: {0: "batch"}}
    return _export(model, torch.rand(2, 3, size, size), Path(path), [output_name], axes)


def parity(model: nn.Module, path: str | Path, example: torch.Tensor) -> float:
    """Returns the largest difference between PyTorch and ONNX Runtime outputs on `example`, relative to the largest
    PyTorch output value (so an untrained network with very large activations is judged fairly)."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = model.eval().cpu()(example)
    if isinstance(reference, dict):
        reference = [reference[name] for name in OUTPUT_NAMES]
    elif torch.is_tensor(reference):
        reference = [reference]
    onnx_out = session.run(None, {session.get_inputs()[0].name: example.numpy()})
    return float(max(np.abs(r.numpy() - o).max() / max(1e-6, float(np.abs(r.numpy()).max()))
                     for r, o in zip(reference, onnx_out)))
