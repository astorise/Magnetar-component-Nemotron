# magnetar-component-nemotron

## Purpose

A WebAssembly Component implementing the NVIDIA Nemotron model families for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It will
supply prefill/decode graph semantics for Nemotron-compatible Model Artifacts
through the Runtime's Model Component contract, without ever selecting a
Provider, Device, or receiving native Runtime handles -- per Magnetar's
architectural invariants (Components request Capabilities, not Providers
or Devices).

## Status

The Rust library validates a Nemotron-H hybrid layer pattern and implements a
standalone float32 reference for the cached Mamba-2 selective-state recurrence.
The recurrence is tested across three consecutive steps against the upstream
PyTorch decode branch, including grouped heads, nonzero initial state, time
step clamping and the learned D skip. Its input is already convolved and
projected: causal convolution, gated normalization, model graph production,
weights, WIT exports and complete inference are not implemented. No Nemotron
checkpoint is yet runnable in Magnetar through this component.

The parser recognizes `M` (Mamba-2), `*` (attention), `-` (dense MLP) and
`E` (MoE), and tests the factual 52-layer layout of NVIDIA's Nemotron-3 Nano
checkpoint. At the cited revision, that checkpoint's pattern contains `E`,
although its bundled `configuration_nemotron_h.py` appears to exclude `E` in
its pattern validation. NVIDIA's model code has an MoE branch and a newer
NVIDIA configuration explicitly maps `E` to MoE. Parsing the published layout
here does not claim that this upstream configuration loads successfully.

## What this family needs

What sets this family apart from the generic decoder block
`components/qwen` and `components/llama` already implement, each to be
checked against the exact pinned checkpoint's `config.json` before it is
implemented:

- Nemotron-4 / Minitron checkpoints use a non-gated MLP with a squared-ReLU activation and LayerNorm (with a `1 + weight` scale) rather than RMSNorm, and rotate only part of each head (partial rotary embedding): an activation, a normalization and a RoPE coverage the current graph contract does not express.
- Llama-Nemotron checkpoints are Llama-derived but architecture-searched: blocks can differ per layer (some attention-free or with a narrower MLP), so the graph cannot assume one block shape repeated `num_hidden_layers` times.
- Nemotron-H checkpoints are hybrid: most layers are Mamba-2 state-space layers, interleaved with a few attention layers, which needs a recurrent state (not a KV history) carried from step to step.

## Governing contract

The shared state, graph and Reference CPU work required for Nemotron-H Mamba-2
is tracked in [Magnetar #223](https://github.com/astorise/Magnetar/issues/223).
Its grouped and biased expert routing also depends on
[Magnetar #198](https://github.com/astorise/Magnetar/issues/198) and
[#216](https://github.com/astorise/Magnetar/issues/216). A real component will target the same
`model-component-graph-contract` WIT interface (`magnetar:model-component-graph`)
`components/qwen` and `components/llama` implement, extended additively
where this family needs Operators or state the contract does not model yet.
Family-specific requirements will be scoped in the main repository's
`openspec/` (the `qualify-additional-model-and-state-families` change tracks
qualifying new families).

## Relationship to magnetar-runtime

This Component will be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/nemotron`.
