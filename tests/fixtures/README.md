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

`nano-config.json` is the published configuration at that revision (SHA-256
`dd9fa380ac107b0477db5a26108db9febe6378e7bb3966a107944853ec4f76f8`).
`nano-mamba2-header.json` records only the names, shapes and dtypes of the
eight layer-0 mixer tensors from the first Safetensors shard; it contains no
weight data or offsets. The source header was read with HTTP byte ranges; its
55,432 JSON bytes have SHA-256
`77a5f9ac2281aacc2209e622c4dbabb1d1cb78d76796055a765729e868165579`.
The index, model code above and
[`configuration_nemotron_h.py`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/fe4143e388795a58a92974891b2e45092a53dc1d/configuration_nemotron_h.py)
(SHA-256 `09a3179517667d302068c2f8c96c9f059297418a829bf3cb96273040563b955c`)
are the sources for these shapes and the default inference time-step limits.
The config's `time_step_min` and `time_step_max` govern initialization;
the inference scan defaults to `(0, infinity)`. BF16 execution, other shards,
and loading or running the published checkpoint are not qualified by this test.
