//! Shape and weight mapping for the pinned Nemotron-3 Nano Mamba-2 mixer.
//!
//! This describes the portable f32 operator, not a checkpoint loader, graph
//! export, runtime session or a qualification of the BF16 model.

use crate::layout::{LayerKind, LayerLayout};
use serde::Deserialize;
use std::{error::Error, fmt};

const MAX_ELEMENTS: usize = 67_108_864;
const NANO_PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";

/// The exact published mixer geometry and layer locations for the pinned Nano.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2Profile {
    pub layers: Vec<usize>,
    pub hidden_size: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state_dim: usize,
    pub conv_kernel: usize,
    pub epsilon: f64,
    /// Initializer bounds; these are not inference-time clamping bounds.
    pub initialization_dt_min: f64,
    pub initialization_dt_max: f64,
    pub max_tokens: usize,
}

/// How a checkpoint tensor must be adapted to the portable operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightTransform {
    Identity,
    Transpose,
    FlattenDepthwiseConvolution,
}

/// A required checkpoint tensor, in operator input order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSpec {
    pub checkpoint_name: String,
    pub checkpoint_shape: Vec<usize>,
    pub checkpoint_dtype: &'static str,
    pub operator_shape: Vec<usize>,
    pub transform: WeightTransform,
}

/// One single-sequence f32 reference invocation with explicit caller state.
#[derive(Clone, Debug, PartialEq)]
pub struct InvocationPlan {
    pub operator: &'static str,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub weights: Vec<WeightSpec>,
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state_dim: usize,
    pub conv_kernel: usize,
    pub epsilon: f64,
    pub dt_min: f64,
    /// `None` means unbounded, as in the pinned configuration's default.
    pub dt_max: Option<f64>,
    pub conv_bias: bool,
    pub linear_bias: bool,
}

#[derive(Deserialize)]
struct RawConfig {
    hybrid_override_pattern: String,
    hidden_size: usize,
    mamba_num_heads: usize,
    mamba_head_dim: usize,
    n_groups: usize,
    ssm_state_size: usize,
    conv_kernel: usize,
    layer_norm_epsilon: f64,
    mamba_hidden_act: String,
    mamba_ssm_cache_dtype: String,
    torch_dtype: String,
    use_bias: bool,
    use_conv_bias: bool,
    mamba_proj_bias: bool,
    time_step_min: f64,
    time_step_max: f64,
}

impl Mamba2Profile {
    /// Accept the pinned Nano mixer geometry and numeric settings only.
    /// The upstream config's default `mamba_dt_limit` is (0, infinity):
    /// `time_step_min` and `time_step_max` initialize weights, not the scan.
    /// Alternate alias fields are rejected instead of silently ignored.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, ProfileError> {
        let layout = LayerLayout::from_huggingface_config(bytes)
            .map_err(|error| ProfileError(error.to_string()))?;
        let raw: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| ProfileError(format!("invalid mixer configuration: {error}")))?;
        let document: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| ProfileError(format!("invalid mixer configuration: {error}")))?;
        if [
            "time_step_limit",
            "mamba_dt_limit",
            "mamba_n_groups",
            "mamba_d_conv",
            "mamba_conv_bias",
        ]
        .iter()
        .any(|key| document.get(*key).is_some())
        {
            return Err(ProfileError(
                "unsupported alternate mixer configuration fields".into(),
            ));
        }
        if raw.hybrid_override_pattern != NANO_PATTERN
            || raw.hidden_size != 2688
            || raw.mamba_num_heads != 64
            || raw.mamba_head_dim != 64
            || raw.n_groups != 8
            || raw.ssm_state_size != 128
            || raw.conv_kernel != 4
            || raw.layer_norm_epsilon != 1e-5
            || raw.mamba_hidden_act != "silu"
            || raw.mamba_ssm_cache_dtype != "float32"
            || raw.torch_dtype != "bfloat16"
            || raw.use_bias
            || !raw.use_conv_bias
            || raw.mamba_proj_bias
            || raw.time_step_min != 0.001
            || raw.time_step_max != 0.1
        {
            return Err(ProfileError(
                "unsupported Nemotron-3 Nano mixer profile".into(),
            ));
        }
        let inner = raw.mamba_num_heads * raw.mamba_head_dim;
        let conv = inner + 2 * raw.n_groups * raw.ssm_state_size;
        let projection = inner + conv + raw.mamba_num_heads;
        Ok(Self {
            layers: layout
                .layers()
                .iter()
                .enumerate()
                .filter_map(|(i, layer)| (*layer == LayerKind::Mamba2).then_some(i))
                .collect(),
            hidden_size: raw.hidden_size,
            heads: raw.mamba_num_heads,
            head_dim: raw.mamba_head_dim,
            groups: raw.n_groups,
            state_dim: raw.ssm_state_size,
            conv_kernel: raw.conv_kernel,
            epsilon: raw.layer_norm_epsilon,
            initialization_dt_min: raw.time_step_min,
            initialization_dt_max: raw.time_step_max,
            max_tokens: MAX_ELEMENTS / projection.max(conv).max(raw.hidden_size),
        })
    }

    /// Plan the operator shapes. Actual loading, conversions, state ownership
    /// and dispatch remain the caller's responsibility.
    pub fn invocation(&self, layer: usize, tokens: usize) -> Result<InvocationPlan, ProfileError> {
        if NANO_PATTERN.as_bytes().get(layer) != Some(&b'M')
            || !self.layers.contains(&layer)
            || tokens == 0
            || tokens > self.max_tokens
        {
            return Err(ProfileError(
                "invalid mixer layer or sequence length".into(),
            ));
        }
        // Profiles are publicly readable; reject fabricated or modified values.
        if self.hidden_size != 2688
            || self.heads != 64
            || self.head_dim != 64
            || self.groups != 8
            || self.state_dim != 128
            || self.conv_kernel != 4
            || self.epsilon != 1e-5
            || self.initialization_dt_min != 0.001
            || self.initialization_dt_max != 0.1
            || self.max_tokens != MAX_ELEMENTS / 10304
            || self.layers
                != NANO_PATTERN
                    .bytes()
                    .enumerate()
                    .filter_map(|(index, kind)| (kind == b'M').then_some(index))
                    .collect::<Vec<_>>()
        {
            return Err(ProfileError("modified mixer profile".into()));
        }
        let inner = self.heads * self.head_dim;
        let conv = inner + 2 * self.groups * self.state_dim;
        let projection = inner + conv + self.heads;
        let prefix = format!("backbone.layers.{layer}.mixer");
        let weight = |name: &str, dtype, checkpoint_shape, operator_shape, transform| WeightSpec {
            checkpoint_name: format!("{prefix}.{name}"),
            checkpoint_dtype: dtype,
            checkpoint_shape,
            operator_shape,
            transform,
        };
        let weights = vec![
            weight(
                "in_proj.weight",
                "BF16",
                vec![projection, self.hidden_size],
                vec![self.hidden_size, projection],
                WeightTransform::Transpose,
            ),
            weight(
                "conv1d.weight",
                "BF16",
                vec![conv, 1, self.conv_kernel],
                vec![conv, self.conv_kernel],
                WeightTransform::FlattenDepthwiseConvolution,
            ),
            weight(
                "dt_bias",
                "BF16",
                vec![self.heads],
                vec![self.heads],
                WeightTransform::Identity,
            ),
            weight(
                "A_log",
                "F32",
                vec![self.heads],
                vec![self.heads],
                WeightTransform::Identity,
            ),
            weight(
                "D",
                "F32",
                vec![self.heads],
                vec![self.heads],
                WeightTransform::Identity,
            ),
            weight(
                "norm.weight",
                "BF16",
                vec![inner],
                vec![inner],
                WeightTransform::Identity,
            ),
            weight(
                "out_proj.weight",
                "BF16",
                vec![self.hidden_size, inner],
                vec![inner, self.hidden_size],
                WeightTransform::Transpose,
            ),
            weight(
                "conv1d.bias",
                "BF16",
                vec![conv],
                vec![conv],
                WeightTransform::Identity,
            ),
        ];
        let mut inputs = vec![
            vec![tokens, self.hidden_size],
            vec![conv, self.conv_kernel],
            vec![self.heads, self.head_dim, self.state_dim],
        ];
        inputs.extend(weights.iter().map(|spec| spec.operator_shape.clone()));
        Ok(InvocationPlan {
            operator: "mamba2-mixer@1",
            input_shapes: inputs,
            output_shapes: vec![
                vec![tokens, self.hidden_size],
                vec![conv, self.conv_kernel],
                vec![self.heads, self.head_dim, self.state_dim],
            ],
            weights,
            heads: self.heads,
            head_dim: self.head_dim,
            groups: self.groups,
            state_dim: self.state_dim,
            conv_kernel: self.conv_kernel,
            epsilon: self.epsilon,
            dt_min: 0.0,
            dt_max: None,
            conv_bias: true,
            linear_bias: false,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileError(String);

impl fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl Error for ProfileError {}
