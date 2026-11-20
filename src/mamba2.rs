//! A float32 CPU reference for the cached Mamba-2 selective-state step.
//!
//! This covers only the recurrence after the upstream causal convolution and
//! before gated normalization and output projection. It is not a complete
//! Mamba layer, a Magnetar Provider operator, or a runnable Model Component.

use std::{error::Error, fmt};

const MAX_STATE_ELEMENTS: usize = 16_777_216;

/// Parameters and already-projected inputs for one token and one batch item.
///
/// `x` is `[head, head_dim]`, `dt` and `dt_bias` are `[head]`, `a_log`
/// and `d` are `[head]`, and `b` and `c` are `[group, state_dim]`.
pub struct StepInput<'a> {
    pub x: &'a [f32],
    pub dt: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub b: &'a [f32],
    pub c: &'a [f32],
    pub d: &'a [f32],
    pub dt_min: f32,
    pub dt_max: f32,
}

/// One layer and one sequence's `[head, head_dim, state_dim]` recurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct Mamba2State {
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_dim: usize,
    values: Vec<f32>,
}

impl Mamba2State {
    /// Construct a zero state with valid grouped-head geometry.
    pub fn new(
        heads: usize,
        head_dim: usize,
        groups: usize,
        state_dim: usize,
    ) -> Result<Self, StateError> {
        let count = heads
            .checked_mul(head_dim)
            .and_then(|n| n.checked_mul(state_dim));
        if heads == 0
            || head_dim == 0
            || groups == 0
            || state_dim == 0
            || !heads.is_multiple_of(groups)
            || count.is_none_or(|n| n > MAX_STATE_ELEMENTS)
        {
            return Err(StateError::InvalidDimensions);
        }
        Ok(Self {
            heads,
            head_dim,
            groups,
            state_dim,
            values: vec![0.0; count.unwrap()],
        })
    }

    /// Construct from a state snapshot in `[head, head_dim, state_dim]` order.
    pub fn from_values(
        heads: usize,
        head_dim: usize,
        groups: usize,
        state_dim: usize,
        values: Vec<f32>,
    ) -> Result<Self, StateError> {
        let mut state = Self::new(heads, head_dim, groups, state_dim)?;
        if values.len() != state.values.len() {
            return Err(StateError::InvalidDimensions);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(StateError::NonFinite);
        }
        state.values = values;
        Ok(state)
    }

    /// Current recurrence state, flattened in `[head, head_dim, state_dim]` order.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Reset the sequence without changing its geometry.
    pub fn reset(&mut self) {
        self.values.fill(0.0);
    }

    /// Advance one token and return the ungated `[head, head_dim]` SSM output.
    ///
    /// Errors leave the previous state unchanged. This intentionally excludes
    /// convolution, input/output projections and the gated RMS normalization.
    pub fn step(&mut self, input: &StepInput<'_>) -> Result<Vec<f32>, StateError> {
        let width = self.heads * self.head_dim;
        let grouped = self.groups * self.state_dim;
        if input.x.len() != width
            || input.dt.len() != self.heads
            || input.dt_bias.len() != self.heads
            || input.a_log.len() != self.heads
            || input.d.len() != self.heads
            || input.b.len() != grouped
            || input.c.len() != grouped
        {
            return Err(StateError::InvalidDimensions);
        }
        if !input.dt_min.is_finite()
            || input.dt_min < 0.0
            || input.dt_max.is_nan()
            || input.dt_max < input.dt_min
        {
            return Err(StateError::InvalidTimeLimit);
        }
        if [
            input.x,
            input.dt,
            input.dt_bias,
            input.a_log,
            input.b,
            input.c,
            input.d,
        ]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite())
        {
            return Err(StateError::NonFinite);
        }
        let mut next = self.values.clone();
        let mut output = vec![0.0; width];
        let heads_per_group = self.heads / self.groups;
        for head in 0..self.heads {
            let raw_dt = input.dt[head] + input.dt_bias[head];
            let a = -input.a_log[head].exp();
            if !raw_dt.is_finite() || !a.is_finite() {
                return Err(StateError::NonFinite);
            }
            // PyTorch softplus uses the linear branch above its threshold 20.
            let dt = if raw_dt > 20.0 {
                raw_dt
            } else {
                raw_dt.exp().ln_1p()
            }
            .clamp(input.dt_min, input.dt_max);
            let decay = (dt * a).exp();
            let group = head / heads_per_group;
            for dimension in 0..self.head_dim {
                let x_index = head * self.head_dim + dimension;
                let mut sum = 0.0_f32;
                for index in 0..self.state_dim {
                    let state_index = x_index * self.state_dim + index;
                    let bc_index = group * self.state_dim + index;
                    let current = self.values[state_index] * decay
                        + (dt * input.b[bc_index]) * input.x[x_index];
                    if !current.is_finite() {
                        return Err(StateError::NonFinite);
                    }
                    next[state_index] = current;
                    sum += current * input.c[bc_index];
                }
                let value = sum + input.x[x_index] * input.d[head];
                if !value.is_finite() {
                    return Err(StateError::NonFinite);
                }
                output[x_index] = value;
            }
        }
        self.values = next;
        Ok(output)
    }
}

/// Invalid shapes, time-step limits, or numerical values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    InvalidDimensions,
    InvalidTimeLimit,
    NonFinite,
}

impl fmt::Display for StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDimensions => "invalid Mamba-2 state or tensor dimensions",
            Self::InvalidTimeLimit => "invalid nonnegative time-step limits",
            Self::NonFinite => "non-finite Mamba-2 input or intermediate",
        })
    }
}

impl Error for StateError {}
