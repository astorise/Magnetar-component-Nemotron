# Mamba-2 cached decode reference

`mamba2-decode-reference.json` is produced by executing the actual cached
decode recurrence branch of Hugging Face Transformers'
[`modeling_mamba2.py` at v4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/mamba2/modeling_mamba2.py),
the implementation identified as the source of the Nemotron-H mixer. The
source is SHA-256 checked (`52fbd9d3e9f7cee9347a36577d761e84127f276eb9b4044d8b84d1ba7bb1a9dc`)
before its relevant Python AST statements run. Inputs, state, grouped heads
and projection values are synthetic; no NVIDIA model weights are included.
Three sequential decode steps exercise updated state, grouping, learned D
skip, and both time-step clamping boundaries.

To reproduce with PyTorch 2.15.0:

```sh
python tests/fixtures/generate_mamba2_reference.py /path/to/modeling_mamba2.py
```

The published 52-layer NVIDIA Nemotron-3 Nano layout in `layout_integration.rs`
is a factual subset of its
[configuration at revision fe4143e](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/fe4143e388795a58a92974891b2e45092a53dc1d/config.json).
The fixture does not redistribute that checkpoint's weights.
