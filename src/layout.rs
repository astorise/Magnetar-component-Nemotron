//! Parse the ordered Nemotron-H hybrid layer pattern.

use serde::Deserialize;
use std::{error::Error, fmt};

const MAX_LAYERS: usize = 4096;

/// One block in the ordered checkpoint layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Mamba2,
    Attention,
    Mlp,
    Moe,
}

/// A validated checkpoint layer ordering. It is not an inference capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerLayout {
    layers: Vec<LayerKind>,
}

impl LayerLayout {
    /// Read a Nemotron-H Hugging Face `config.json` hybrid pattern.
    ///
    /// Only `model_type`, `num_hidden_layers` and `hybrid_override_pattern`
    /// are inspected. Successful parsing does not mean the remaining config
    /// or the model is supported. Unknown patterns are rejected.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, LayoutError> {
        let config: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| LayoutError(format!("invalid Nemotron-H config: {error}")))?;
        if config.model_type != "nemotron_h" {
            return Err(LayoutError(format!(
                "expected model_type nemotron_h, got {:?}",
                config.model_type
            )));
        }
        if !(1..=MAX_LAYERS).contains(&config.num_hidden_layers) {
            return Err(LayoutError(format!(
                "num_hidden_layers must be between 1 and {MAX_LAYERS}"
            )));
        }
        let layers = config
            .hybrid_override_pattern
            .chars()
            .enumerate()
            .map(|(index, kind)| match kind {
                'M' => Ok(LayerKind::Mamba2),
                '*' => Ok(LayerKind::Attention),
                '-' => Ok(LayerKind::Mlp),
                'E' => Ok(LayerKind::Moe),
                _ => Err(LayoutError(format!(
                    "unsupported layer pattern {kind:?} at index {index}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layers.len() != config.num_hidden_layers {
            return Err(LayoutError(format!(
                "hybrid pattern has {} entries, expected {}",
                layers.len(),
                config.num_hidden_layers
            )));
        }
        Ok(Self { layers })
    }

    /// Layers in checkpoint order.
    pub fn layers(&self) -> &[LayerKind] {
        &self.layers
    }
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    num_hidden_layers: usize,
    hybrid_override_pattern: String,
}

/// A malformed or unsupported layer layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutError(String);

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for LayoutError {}
