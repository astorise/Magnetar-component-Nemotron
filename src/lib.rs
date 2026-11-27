//! Initial Nemotron-H model component support.
//!
//! Includes an experimental mixed-graph producer and standalone numerical
//! references. Full checkpoint ingestion and end-to-end qualification remain
//! separate work.

mod graph;
pub mod layout;
pub mod mamba2;
pub mod mamba2_layer;
pub mod mamba2_profile;

/// Project-level family label; Hugging Face Nemotron-H uses `nemotron_h`.
pub const ARCHITECTURE_FAMILY: &str = "nemotron";
