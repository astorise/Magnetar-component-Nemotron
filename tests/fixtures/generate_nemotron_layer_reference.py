"""Generate synthetic Nemotron-H mixer cases from pinned upstream CPU code.

Usage: python generate_nemotron_layer_reference.py MODELING.py LAYERNORM.py [--biased]
Requires PyTorch 2.15 and einops. Both source files are SHA-256 verified before
their relevant definitions are executed. The NVIDIA checkpoint's CUDA-only
gated norm is replaced by mamba-ssm's own portable PyTorch reference.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MODEL_HASH = "52273a23a2c768ffe5938e96f6832afd762029293a8be9bfed886757ba6a39b5"
NORM_HASH = "eb6252e247b90f1c8a75946efbc1a221e0c4da701b6757ddae49f3495cf7a42f"


def load_definitions(path, expected, names, namespace):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError(f"unexpected source digest: {path}")
    tree = ast.parse(source, filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("missing upstream definitions")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class Cache:
    """Expose the upstream cache interface with separately shaped state tensors.

    The pinned HybridMambaAttentionDynamicCache implementation stores lists but
    its mixer accesses `.device` on them and expects a `conv_kernel_size` field.
    This shim provides those operations without changing the mixer equations.
    """
    def __init__(self, conv_dim, kernel, heads, head_dim, state_dim):
        self.conv_kernel_size = kernel
        self.conv_states = torch.zeros(1, 1, conv_dim, kernel)
        self.ssm_states = torch.zeros(1, 1, heads, head_dim, state_dim)

    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(-1, -1)
            self.conv_states[layer_idx, :, :, -1] = new_conv_state[:, 0, :]

    def update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx] = new_ssm_state


def main():
    biased = len(sys.argv) == 4 and sys.argv[3] == "--biased"
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and not biased):
        raise ValueError("expected MODELING.py LAYERNORM.py [--biased]")
    namespace = {"torch": torch, "nn": nn, "F": F, "rearrange": rearrange}
    load_definitions(sys.argv[2], NORM_HASH, ["rms_norm_ref"], namespace)
    namespace.update({
        "rmsnorm_fn": namespace["rms_norm_ref"],
        "ACT2FN": {"silu": nn.SiLU()},
        "is_fast_path_available": False,
        "logger": SimpleNamespace(warning_once=lambda *_: None),
    })
    load_definitions(sys.argv[1], MODEL_HASH,
                     ["pad_tensor_by_size", "reshape_into_chunks", "segment_sum",
                      "apply_mask_to_padding_states", "MambaRMSNormGated", "NemotronHMamba2Mixer"], namespace)
    cfg = SimpleNamespace(mamba_num_heads=4, hidden_size=5, ssm_state_size=3,
                          conv_kernel=3, mamba_head_dim=2, n_groups=2, chunk_size=4,
                          use_conv_bias=True, use_bias=biased, mamba_hidden_act="silu",
                          layer_norm_epsilon=1e-5, time_step_limit=(0.05, 1.25),
                          time_step_min=0.05, time_step_max=1.25)
    torch.manual_seed(91273)
    model = namespace["NemotronHMamba2Mixer"](cfg, 0).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.25)
            if name == "norm.weight":
                parameter.add_(1.0)
        inputs = torch.randn(1, 8, cfg.hidden_size) * 0.65
        def make_cache():
            return Cache(model.conv_dim, cfg.conv_kernel, cfg.mamba_num_heads,
                         cfg.mamba_head_dim, cfg.ssm_state_size)
        full_cache = make_cache()
        full = model.torch_forward(inputs, full_cache, torch.arange(8))
        cache = make_cache()
        prefix = model.torch_forward(inputs[:, :4], cache, torch.arange(4))
        prefix_conv = cache.conv_states.reshape(-1).tolist()
        prefix_ssm = cache.ssm_states.reshape(-1).tolist()
        steps = []
        for i in range(4, 8):
            out = model.torch_forward(inputs[:, i:i+1], cache, torch.tensor([i]))
            steps.append({"output": out.reshape(-1).tolist(),
                          "conv_state": cache.conv_states.reshape(-1).tolist(),
                          "ssm_state": cache.ssm_states.reshape(-1).tolist()})
        if not torch.allclose(full[:, :4], prefix, atol=2e-6, rtol=1e-6):
            raise AssertionError("upstream prefill disagreement")
        if not torch.allclose(full[:, 4:], torch.tensor([s["output"] for s in steps]).unsqueeze(0), atol=2e-6, rtol=1e-6):
            raise AssertionError("upstream prefill and decode disagree")
        if not torch.allclose(full_cache.ssm_states, cache.ssm_states, atol=2e-6, rtol=1e-6):
            raise AssertionError("upstream final states disagree")
    tensor = lambda value: None if value is None else value.detach().reshape(-1).tolist()
    result = {"config": {"hidden_size": cfg.hidden_size, "heads": cfg.mamba_num_heads,
                         "head_dim": cfg.mamba_head_dim, "groups": cfg.n_groups,
                         "state_dim": cfg.ssm_state_size, "kernel": cfg.conv_kernel,
                         "eps": cfg.layer_norm_epsilon, "dt_min": cfg.time_step_limit[0],
                         "dt_max": cfg.time_step_limit[1]},
              "weights": {"in_proj": tensor(model.in_proj.weight), "in_bias": tensor(model.in_proj.bias),
                          "conv": tensor(model.conv1d.weight), "conv_bias": tensor(model.conv1d.bias),
                          "dt_bias": tensor(model.dt_bias), "a_log": tensor(model.A_log),
                          "d": tensor(model.D), "norm": tensor(model.norm.weight),
                          "out_proj": tensor(model.out_proj.weight), "out_bias": tensor(model.out_proj.bias)},
              "inputs": tensor(inputs), "full_output": tensor(full), "prefix_output": tensor(prefix),
              "prefix_conv_state": prefix_conv, "prefix_ssm_state": prefix_ssm, "steps": steps}
    destination = Path(__file__).with_name("nemotron-layer-biased-reference.json" if biased else "nemotron-layer-reference.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
