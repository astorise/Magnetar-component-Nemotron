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

**Empty template.** This crate is a scaffold: no Nemotron graph logic, no WIT
bindings and no Component Model exports exist here yet, and the Runtime has
no Nemotron implementation of its own either. It is pinned into Magnetar as a
placeholder so the work has a home; it is not yet scheduled against an
OpenSpec change of its own.

## What this family needs

What sets this family apart from the generic decoder block
`components/qwen` and `components/llama` already implement, each to be
checked against the exact pinned checkpoint's `config.json` before it is
implemented:

- Nemotron-4 / Minitron checkpoints use a non-gated MLP with a squared-ReLU activation and LayerNorm (with a `1 + weight` scale) rather than RMSNorm, and rotate only part of each head (partial rotary embedding): an activation, a normalization and a RoPE coverage the current graph contract does not express.
- Llama-Nemotron checkpoints are Llama-derived but architecture-searched: blocks can differ per layer (some attention-free or with a narrower MLP), so the graph cannot assume one block shape repeated `num_hidden_layers` times.
- Nemotron-H checkpoints are hybrid: most layers are Mamba-2 state-space layers, interleaved with a few attention layers, which needs a recurrent state (not a KV history) carried from step to step.

## Governing contract

None dedicated yet. A real implementation will target the same
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
