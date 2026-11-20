"""Generate the recurrence reference using the upstream PyTorch decode branch.

Usage: python generate_mamba2_reference.py /path/to/modeling_mamba2.py
Requires PyTorch 2.15.0. Source must be Transformers v4.55.4 and its hash is
verified. Only synthetic input, state and results are saved; no weights.
"""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

SOURCE_HASH = "52fbd9d3e9f7cee9347a36577d761e84127f276eb9b4044d8b84d1ba7bb1a9dc"


def load_decode_branch(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("Transformers Mamba-2 source digest differs from the pinned version")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Mamba2Mixer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "torch_forward")
    branch = next(node for node in ast.walk(method) if isinstance(node, ast.If)
                  and "cache_device = cache_params.ssm_states.device" in ast.unparse(node))
    return compile(ast.fix_missing_locations(ast.Module(body=branch.body, type_ignores=[])), str(path), "exec")


class Cache:
    def __init__(self, values, heads, width, state_dim):
        self.ssm_states = torch.tensor(values, dtype=torch.float32).reshape(1, 1, heads, width, state_dim)

    def update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx] = new_ssm_state


def main():
    code = load_decode_branch(sys.argv[1])
    heads, width, groups, state_dim = 4, 2, 2, 3
    initial = [((i * 7) % 19 - 9) / 17.0 for i in range(heads * width * state_dim)]
    dt_bias = [0.1, -0.2, 0.3, 0.0]
    a_log = [-0.7, 0.0, 0.4, 1.0]
    d = [0.5, -0.1, 0.8, 1.2]
    cases = [
        {"x": [0.3, -0.4, 0.7, 0.1, -0.2, 0.9, 0.5, -0.6],
         "dt": [-0.4, 1.5, -1.2, 0.3], "b": [0.2, -0.3, 0.4, -0.1, 0.5, 0.8],
         "c": [-0.5, 0.3, 0.7, 0.6, -0.2, 0.1]},
        {"x": [-0.6, 0.5, 0.8, -0.1, 0.4, -0.3, 0.2, 0.7],
         "dt": [25.0, -30.0, 0.0, 0.8], "b": [0.3, 0.2, -0.4, 0.6, -0.8, 0.1],
         "c": [0.4, -0.2, 0.1, -0.7, 0.5, 0.3]},
        {"x": [0.0, -0.2, 0.6, 0.5, -0.7, 0.1, 0.8, 0.3],
         "dt": [0.2, -0.1, 0.7, -0.5], "b": [0.1, -0.2, 0.6, 0.2, 0.4, -0.3],
         "c": [0.8, 0.2, -0.1, 0.3, 0.5, -0.4]},
    ]
    cache = Cache(initial, heads, width, state_dim)
    self = SimpleNamespace(num_heads=heads, head_dim=width, n_groups=groups,
                           ssm_state_size=state_dim, dt_bias=torch.tensor(dt_bias),
                           A_log=torch.tensor(a_log), D=torch.tensor(d),
                           time_step_limit=(0.05, 1.25), layer_idx=0)
    for case in cases:
        namespace = {"torch": torch, "self": self, "cache_params": cache, "batch_size": 1,
                     "dt": torch.tensor(case["dt"]).reshape(1, 1, heads),
                     "A": -torch.exp(self.A_log.float()),
                     "B": torch.tensor(case["b"]).reshape(1, 1, -1),
                     "C": torch.tensor(case["c"]).reshape(1, 1, -1),
                     "hidden_states": torch.tensor(case["x"]).reshape(1, 1, -1)}
        with torch.no_grad():
            exec(code, namespace)
        case["output"] = namespace["y"].reshape(-1).tolist()
        case["state"] = cache.ssm_states.reshape(-1).tolist()
    reference = {"heads": heads, "head_dim": width, "groups": groups, "state_dim": state_dim,
                 "initial": initial, "dt_bias": dt_bias, "a_log": a_log, "d": d,
                 "dt_min": 0.05, "dt_max": 1.25, "steps": cases}
    destination = Path(__file__).with_name("mamba2-decode-reference.json")
    destination.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
