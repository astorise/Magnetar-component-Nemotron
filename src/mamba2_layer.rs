//! Standalone float32 reference for one Nemotron-H Mamba-2 mixer.
//!
//! This is not yet a graph operator or a runnable model component. State is
//! explicit and belongs to one layer of one sequence, not to an attention KV
//! cache. The implementation processes chunks as consecutive token steps.

use crate::mamba2::{Mamba2State, StateError, StepInput};
use serde::Deserialize;

// The published 30B-A3B in-projection has 27,697,152 elements.
const MAX_ELEMENTS: usize = 67_108_864;

/// Mixer geometry and numerical parameters for a single sequence.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state_dim: usize,
    pub kernel: usize,
    pub eps: f32,
    pub dt_min: f32,
    pub dt_max: f32,
}

struct Geometry {
    inner: usize,
    grouped: usize,
    conv: usize,
    projection: usize,
}

impl Config {
    fn geometry(&self) -> Result<Geometry, StateError> {
        if self.hidden_size == 0
            || self.heads == 0
            || self.head_dim == 0
            || self.groups == 0
            || self.state_dim == 0
            || self.kernel == 0
            || !self.heads.is_multiple_of(self.groups)
            || !self.eps.is_finite()
            || self.eps <= 0.0
        {
            return Err(StateError::InvalidDimensions);
        }
        if !self.dt_min.is_finite()
            || self.dt_min < 0.0
            || self.dt_max.is_nan()
            || self.dt_max < self.dt_min
        {
            return Err(StateError::InvalidTimeLimit);
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .filter(|&n| n <= MAX_ELEMENTS)
                .ok_or(StateError::InvalidDimensions)
        };
        let sum = |a: usize, b: usize| {
            a.checked_add(b)
                .filter(|&n| n <= MAX_ELEMENTS)
                .ok_or(StateError::InvalidDimensions)
        };
        let inner = product(self.heads, self.head_dim)?;
        let grouped = product(self.groups, self.state_dim)?;
        let conv = sum(inner, product(2, grouped)?)?;
        let projection = sum(sum(inner, conv)?, self.heads)?;
        // Reject impossible or excessive tensors before any allocation.
        product(conv, self.kernel)?;
        product(projection, self.hidden_size)?;
        product(self.hidden_size, inner)?;
        product(inner, self.state_dim)?;
        Ok(Geometry {
            inner,
            grouped,
            conv,
            projection,
        })
    }
}

/// Checkpoint weights, flattened in PyTorch's row-major linear/conv order.
/// Optional biases must be absent when the corresponding checkpoint omits them.
pub struct Weights<'a> {
    pub in_proj: &'a [f32],
    pub in_bias: Option<&'a [f32]>,
    pub conv: &'a [f32],
    pub conv_bias: Option<&'a [f32]>,
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub d: &'a [f32],
    pub norm: &'a [f32],
    pub out_proj: &'a [f32],
    pub out_bias: Option<&'a [f32]>,
}

impl Weights<'_> {
    fn validate(&self, config: &Config, geometry: &Geometry) -> Result<(), StateError> {
        if self.in_proj.len() != geometry.projection * config.hidden_size
            || self
                .in_bias
                .is_some_and(|bias| bias.len() != geometry.projection)
            || self.conv.len() != geometry.conv * config.kernel
            || self
                .conv_bias
                .is_some_and(|bias| bias.len() != geometry.conv)
            || self.dt_bias.len() != config.heads
            || self.a_log.len() != config.heads
            || self.d.len() != config.heads
            || self.norm.len() != geometry.inner
            || self.out_proj.len() != config.hidden_size * geometry.inner
            || self
                .out_bias
                .is_some_and(|bias| bias.len() != config.hidden_size)
        {
            return Err(StateError::InvalidDimensions);
        }
        if [
            self.in_proj,
            self.in_bias.unwrap_or_default(),
            self.conv,
            self.conv_bias.unwrap_or_default(),
            self.dt_bias,
            self.a_log,
            self.d,
            self.norm,
            self.out_proj,
            self.out_bias.unwrap_or_default(),
        ]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite())
        {
            return Err(StateError::NonFinite);
        }
        Ok(())
    }
}

/// Fixed-size convolution and SSM state for one layer in one sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerState {
    config: Config,
    conv_history: Vec<f32>,
    ssm: Mamba2State,
}

impl LayerState {
    pub fn new(config: Config) -> Result<Self, StateError> {
        let geometry = config.geometry()?;
        let ssm = Mamba2State::new(
            config.heads,
            config.head_dim,
            config.groups,
            config.state_dim,
        )?;
        Ok(Self {
            conv_history: vec![0.0; geometry.conv * config.kernel],
            config,
            ssm,
        })
    }

    /// History is `[conv_channel, kernel]`, oldest token first.
    pub fn conv_history(&self) -> &[f32] {
        &self.conv_history
    }

    /// SSM state is `[head, head_dim, state_dim]`.
    pub fn ssm_state(&self) -> &[f32] {
        self.ssm.values()
    }

    /// Clear both state tensors at a sequence/document boundary.
    pub fn reset(&mut self) {
        self.conv_history.fill(0.0);
        self.ssm.reset();
    }

    /// Execute one or more tokens in `[sequence, hidden_size]` order.
    ///
    /// Failure at any token leaves both original state tensors unchanged.
    pub fn run(&mut self, weights: &Weights<'_>, input: &[f32]) -> Result<Vec<f32>, StateError> {
        let geometry = self.config.geometry()?;
        weights.validate(&self.config, &geometry)?;
        if input.is_empty() || !input.len().is_multiple_of(self.config.hidden_size) {
            return Err(StateError::InvalidDimensions);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(StateError::NonFinite);
        }
        let mut next = self.clone();
        let mut output = Vec::with_capacity(input.len());
        for token in input.chunks_exact(self.config.hidden_size) {
            output.extend(next.step(weights, &geometry, token)?);
        }
        *self = next;
        Ok(output)
    }

    fn step(
        &mut self,
        weights: &Weights<'_>,
        geometry: &Geometry,
        input: &[f32],
    ) -> Result<Vec<f32>, StateError> {
        let config = &self.config;
        let projected = linear(input, weights.in_proj, weights.in_bias, geometry.projection);
        if projected.iter().any(|value| !value.is_finite()) {
            return Err(StateError::NonFinite);
        }
        let gate = &projected[..geometry.inner];
        let raw = &projected[geometry.inner..geometry.inner + geometry.conv];
        let dt = &projected[geometry.inner + geometry.conv..];
        let mut convolved = vec![0.0; geometry.conv];
        for channel in 0..geometry.conv {
            let start = channel * config.kernel;
            let history = &mut self.conv_history[start..start + config.kernel];
            history.rotate_left(1);
            history[config.kernel - 1] = raw[channel];
            let mut value = weights.conv_bias.map_or(0.0, |bias| bias[channel]);
            for (&x, &w) in history
                .iter()
                .zip(&weights.conv[start..start + config.kernel])
            {
                value += x * w;
            }
            convolved[channel] = silu(value);
        }
        let b_start = geometry.inner;
        let c_start = b_start + geometry.grouped;
        let scan = self.ssm.step(&StepInput {
            x: &convolved[..b_start],
            dt,
            dt_bias: weights.dt_bias,
            a_log: weights.a_log,
            b: &convolved[b_start..c_start],
            c: &convolved[c_start..],
            d: weights.d,
            dt_min: config.dt_min,
            dt_max: config.dt_max,
        })?;
        let mut norm: Vec<f32> = scan.iter().zip(gate).map(|(&x, &z)| x * silu(z)).collect();
        let group_size = geometry.inner / config.groups;
        for (group, scale) in norm
            .chunks_exact_mut(group_size)
            .zip(weights.norm.chunks_exact(group_size))
        {
            let variance = group.iter().map(|x| x * x).sum::<f32>() / group_size as f32;
            if !variance.is_finite() {
                return Err(StateError::NonFinite);
            }
            let inverse_rms = (variance + config.eps).sqrt().recip();
            for (value, &weight) in group.iter_mut().zip(scale) {
                *value *= inverse_rms * weight;
            }
        }
        let result = linear(
            &norm,
            weights.out_proj,
            weights.out_bias,
            config.hidden_size,
        );
        if result
            .iter()
            .chain(norm.iter())
            .any(|value| !value.is_finite())
        {
            return Err(StateError::NonFinite);
        }
        Ok(result)
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn linear(input: &[f32], weights: &[f32], bias: Option<&[f32]>, rows: usize) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            let offset = row * input.len();
            let mut sum = bias.map_or(0.0, |values| values[row]);
            for (&value, &weight) in input.iter().zip(&weights[offset..offset + input.len()]) {
                sum += value * weight;
            }
            sum
        })
        .collect()
}
