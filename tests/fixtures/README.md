# Mamba-2 cached decode reference

`nemotron-layer-reference.json` and `nemotron-layer-biased-reference.json`
compare an entire synthetic mixer, including projections and separate
convolution/SSM states, to
[NVIDIA's pinned Nemotron-H model code](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/fe4143e388795a58a92974891b2e45092a53dc1d/modeling_nemotron_h.py)
(SHA-256 `52273a23a2c768ffe5938e96f6832afd762029293a8be9bfed886757ba6a39b5`).
Its gated RMS normalization depends on a CUDA-only kernel; the fixture uses
the `rms_norm_ref` PyTorch reference from
[mamba-ssm v2.2.4](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/layernorm_gated.py)
(SHA-256 `eb6252e247b90f1c8a75946efbc1a221e0c4da701b6757ddae49f3495cf7a42f`).
Both sources are checked before extracting their actual definitions. A small
cache shim fixes the pinned upstream cache's inconsistent list/`.device`
interface without changing the mixer. The first fixture uses the published
checkpoint's bias flags (`use_bias=false`, `use_conv_bias=true`); the second
also exercises linear biases. No checkpoint weights are included.

Run with PyTorch 2.15 and einops, supplying those two verified source files:

```sh
python tests/fixtures/generate_nemotron_layer_reference.py MODELING.py LAYERNORM.py
python tests/fixtures/generate_nemotron_layer_reference.py MODELING.py LAYERNORM.py --biased
```

The pinned upstream CPU path handles cached decode one token at a time. The
reference checks those steps against full prefill; Rust additionally checks
multi-token chunks, reset, session isolation and transactional failure.

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
