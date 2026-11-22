"""Check block argument forwarding in the pinned NVIDIA model source.

Pass the unmodified modeling_nemotron_h.py from the pinned revision.  This
executes its actual NemotronHBlock.forward with lightweight stand-ins for
the tensors, normalization, and mixer; neither CUDA nor weights are needed.
"""

import argparse
import ast
import contextlib
import hashlib
from pathlib import Path
from types import SimpleNamespace


MODEL_HASH = "52273a23a2c768ffe5938e96f6832afd762029293a8be9bfed886757ba6a39b5"


def load_forward(path):
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != MODEL_HASH:
        raise ValueError(f"unexpected NVIDIA model source SHA-256: {digest}")
    module = ast.parse(source.decode("utf-8"))
    block = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "NemotronHBlock"
    )
    forward = next(
        node
        for node in block.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    for argument in (*forward.args.posonlyargs, *forward.args.args, *forward.args.kwonlyargs):
        argument.annotation = None
    forward.returns = None
    isolated = ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[]))
    cuda = SimpleNamespace(
        default_stream=lambda _device: None,
        stream=lambda _stream: contextlib.nullcontext(),
    )
    namespace = {"torch": SimpleNamespace(cuda=cuda, float32="float32")}
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["forward"]


class Tensor:
    device = "cpu"

    def to(self, *, dtype):
        return self

    def __add__(self, other):
        return self


class Norm:
    weight = SimpleNamespace(dtype="float32")

    def __call__(self, tensor):
        return tensor


class Mixer:
    def __init__(self, attention):
        self.attention = attention
        self.calls = []

    def __call__(self, tensor, **kwargs):
        self.calls.append(kwargs)
        return (tensor, None) if self.attention else tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_source", type=Path)
    args = parser.parse_args()
    forward = load_forward(args.model_source)
    cache, position, mask = object(), object(), object()

    attention = Mixer(attention=True)
    block = SimpleNamespace(
        norm=Norm(), mixer=attention, block_type="attention", residual_in_fp32=False
    )
    forward(block, Tensor(), cache_params=cache, cache_position=position, attention_mask=mask)
    if attention.calls != [{"cache_position": position}]:
        raise AssertionError(f"unexpected attention forwarding: {attention.calls!r}")

    mamba = Mixer(attention=False)
    block.mixer, block.block_type = mamba, "mamba"
    forward(block, Tensor(), cache_params=cache, cache_position=position, attention_mask=mask)
    if mamba.calls != [{"cache_params": cache, "cache_position": position}]:
        raise AssertionError(f"unexpected Mamba forwarding: {mamba.calls!r}")
    print(
        "Pinned block: attention receives only cache_position; Mamba receives "
        "cache_params and cache_position; neither receives the mask."
    )


if __name__ == "__main__":
    main()
