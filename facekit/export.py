"""Exports the three models to ONNX and checks that the exported files agree with PyTorch.

ONNX is a file format for sharing trained models, which lets the pipeline run without PyTorch. Each export takes RGB
values between 0 and 1. The detector accepts any height and width that are multiples of 32, and the two face models
accept any batch size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from facekit.detect import OUTPUT_NAMES

# The version of the ONNX operator set that the exported files use.
OPSET = 17


def _export(model: nn.Module, example: torch.Tensor, path: Path, output_names: list[str], dynamic_axes: dict) -> Path:
    """Exports `model` to an ONNX file at `path` and returns the path.

    `example` is a sample input that the exporter runs through the model to trace it. `dynamic_axes` names the input
    and output dimensions that are allowed to change size, such as the batch size.
    """
    model = model.eval().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["image"], output_names=output_names, dynamic_axes=dynamic_axes, opset_version=OPSET)

    # Newer versions of torch have a switch for the new exporter. This code asks for the older exporter, and an older
    # torch that does not know the switch is called without it.
    try:
        torch.onnx.export(model, (example,), str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, (example,), str(path), **kwargs)
    return path


def export_detector(model: nn.Module, path: str | Path, size: int = 640) -> Path:
    """Exports FaceDetNet to `path` with dynamic batch, height and width, and returns the path.

    `size` is the side of the example image used for tracing and does not limit the sizes that the file accepts.
    """
    axes = {"image": {0: "batch", 2: "height", 3: "width"}}
    axes.update({name: {0: "batch", 2: "map_height", 3: "map_width"} for name in OUTPUT_NAMES})

    class _Tuple(nn.Module):
        """Wraps the detector so that it returns its output maps as a tuple, which is what the exporter needs."""

        def __init__(self, inner):
            """Stores the detector that is wrapped."""
            super().__init__()
            self.inner = inner

        def forward(self, x):
            """Returns the detector's output maps in the order of OUTPUT_NAMES."""
            out = self.inner(x)
            return tuple(out[name] for name in OUTPUT_NAMES)

    return _export(_Tuple(model), torch.rand(1, 3, size, size), Path(path), list(OUTPUT_NAMES), axes)


def export_classifier(model: nn.Module, path: str | Path, size: int, output_name: str) -> Path:
    """Exports a face model (expression or embedding) to `path` with a dynamic batch dimension, and returns the path."""
    axes = {"image": {0: "batch"}, output_name: {0: "batch"}}
    return _export(model, torch.rand(2, 3, size, size), Path(path), [output_name], axes)


def parity(model: nn.Module, path: str | Path, example: torch.Tensor) -> float:
    """Returns the largest difference between the PyTorch and ONNX Runtime outputs on `example`.

    The difference is divided by the largest absolute PyTorch output value, so that an untrained network with very
    large activations is judged fairly. The result is the worst case over all outputs of the model.
    """
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    # Run the original PyTorch model and collect its outputs in a list, in the order of the exported outputs.
    with torch.no_grad():
        reference = model.eval().cpu()(example)
    if isinstance(reference, dict):
        reference = [reference[name] for name in OUTPUT_NAMES]
    elif torch.is_tensor(reference):
        reference = [reference]

    # Run the exported file and compare each output with the matching PyTorch output.
    onnx_out = session.run(None, {session.get_inputs()[0].name: example.numpy()})
    return float(max(np.abs(r.numpy() - o).max() / max(1e-6, float(np.abs(r.numpy()).max()))
                     for r, o in zip(reference, onnx_out)))
