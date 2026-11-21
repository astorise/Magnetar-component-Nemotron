//! Initial Nemotron-H model component support.
//!
//! The layout parser and standalone Mamba-2 mixer reference are not yet
//! connected to graph exports or a runnable Magnetar inference pipeline.

pub mod layout;
pub mod mamba2;
pub mod mamba2_layer;
pub mod mamba2_profile;

/// Project-level family label; Hugging Face Nemotron-H uses `nemotron_h`.
pub const ARCHITECTURE_FAMILY: &str = "nemotron";
