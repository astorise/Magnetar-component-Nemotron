use magnetar_component_nemotron::{
    mamba2::StateError,
    mamba2_layer::{Config, LayerState, Weights},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    config: Config,
    weights: OwnedWeights,
    inputs: Vec<f32>,
    full_output: Vec<f32>,
    prefix_output: Vec<f32>,
    prefix_conv_state: Vec<f32>,
    prefix_ssm_state: Vec<f32>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct OwnedWeights {
    in_proj: Vec<f32>,
    in_bias: Option<Vec<f32>>,
    conv: Vec<f32>,
    conv_bias: Option<Vec<f32>>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    d: Vec<f32>,
    norm: Vec<f32>,
    out_proj: Vec<f32>,
    out_bias: Option<Vec<f32>>,
}

impl OwnedWeights {
    fn borrowed(&self) -> Weights<'_> {
        Weights {
            in_proj: &self.in_proj,
            in_bias: self.in_bias.as_deref(),
            conv: &self.conv,
            conv_bias: self.conv_bias.as_deref(),
            dt_bias: &self.dt_bias,
            a_log: &self.a_log,
            d: &self.d,
            norm: &self.norm,
            out_proj: &self.out_proj,
            out_bias: self.out_bias.as_deref(),
        }
    }
}

#[derive(Deserialize)]
struct Step {
    output: Vec<f32>,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

fn fixture() -> Fixture {
    serde_json::from_slice(include_bytes!("fixtures/nemotron-layer-reference.json")).unwrap()
}

fn close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2e-6,
            "{what}[{index}]: {actual} != {expected}"
        );
    }
}

#[test]
fn full_prefill_and_cached_decode_match_pinned_nvidia_mixer() {
    let fixtures = [
        fixture(),
        serde_json::from_slice(include_bytes!(
            "fixtures/nemotron-layer-biased-reference.json"
        ))
        .unwrap(),
    ];
    for fixture in fixtures {
        let weights = fixture.weights.borrowed();
        let width = fixture.config.hidden_size;
        let mut full = LayerState::new(fixture.config.clone()).unwrap();
        let output = full.run(&weights, &fixture.inputs).unwrap();
        close(&output, &fixture.full_output, "full prefill");

        let mut incremental = LayerState::new(fixture.config.clone()).unwrap();
        let prefix = incremental
            .run(&weights, &fixture.inputs[..4 * width])
            .unwrap();
        close(&prefix, &fixture.prefix_output, "prefix");
        close(
            incremental.conv_history(),
            &fixture.prefix_conv_state,
            "prefix convolution",
        );
        close(
            incremental.ssm_state(),
            &fixture.prefix_ssm_state,
            "prefix recurrence",
        );
        for (i, reference) in fixture.steps.iter().enumerate() {
            let begin = (4 + i) * width;
            let output = incremental
                .run(&weights, &fixture.inputs[begin..begin + width])
                .unwrap();
            close(&output, &reference.output, "decode output");
            close(
                incremental.conv_history(),
                &reference.conv_state,
                "decode convolution",
            );
            close(
                incremental.ssm_state(),
                &reference.ssm_state,
                "decode recurrence",
            );
        }
        close(
            incremental.conv_history(),
            full.conv_history(),
            "final convolution",
        );
        close(
            incremental.ssm_state(),
            full.ssm_state(),
            "final recurrence",
        );
    }
}

#[test]
fn chunks_reset_and_sequences_preserve_independent_states() {
    let fixture = fixture();
    let weights = fixture.weights.borrowed();
    let width = fixture.config.hidden_size;
    let mut first = LayerState::new(fixture.config.clone()).unwrap();
    let mut second = LayerState::new(fixture.config.clone()).unwrap();
    let mut outputs = first.run(&weights, &fixture.inputs[..2 * width]).unwrap();
    let second_prefix = second.run(&weights, &fixture.inputs[..4 * width]).unwrap();
    outputs.extend(
        first
            .run(&weights, &fixture.inputs[2 * width..5 * width])
            .unwrap(),
    );
    outputs.extend(first.run(&weights, &fixture.inputs[5 * width..]).unwrap());
    close(&outputs, &fixture.full_output, "chunked output");
    close(&second_prefix, &fixture.prefix_output, "other session");
    close(
        second.conv_history(),
        &fixture.prefix_conv_state,
        "other convolution",
    );
    close(
        second.ssm_state(),
        &fixture.prefix_ssm_state,
        "other recurrence",
    );
    first.reset();
    assert!(first.conv_history().iter().all(|value| *value == 0.0));
    assert!(first.ssm_state().iter().all(|value| *value == 0.0));
    let restarted = first.run(&weights, &fixture.inputs[..4 * width]).unwrap();
    close(&restarted, &fixture.prefix_output, "reset output");
    close(
        first.conv_history(),
        second.conv_history(),
        "reset convolution",
    );
    close(first.ssm_state(), second.ssm_state(), "reset recurrence");
}

#[test]
fn invalid_weights_inputs_and_overflow_leave_both_states_unchanged() {
    let fixture = fixture();
    let mut invalid_config = fixture.config.clone();
    invalid_config.groups = 3;
    assert_eq!(
        LayerState::new(invalid_config),
        Err(StateError::InvalidDimensions)
    );
    let mut invalid_config = fixture.config.clone();
    invalid_config.kernel = usize::MAX;
    assert_eq!(
        LayerState::new(invalid_config),
        Err(StateError::InvalidDimensions)
    );
    let mut state = LayerState::new(fixture.config.clone()).unwrap();
    let weights = fixture.weights.borrowed();
    state
        .run(&weights, &fixture.inputs[..fixture.config.hidden_size])
        .unwrap();
    let original = state.clone();
    let mut malformed = fixture.weights.borrowed();
    malformed.conv = &[];
    assert_eq!(
        state.run(&malformed, &fixture.inputs),
        Err(StateError::InvalidDimensions)
    );
    assert_eq!(state, original);
    assert_eq!(state.run(&weights, &[]), Err(StateError::InvalidDimensions));
    assert_eq!(
        state.run(&weights, &[f32::NAN; 5]),
        Err(StateError::NonFinite)
    );
    let mut later_overflow = fixture.inputs[..2 * fixture.config.hidden_size].to_vec();
    later_overflow[fixture.config.hidden_size..].fill(f32::MAX);
    assert_eq!(
        state.run(&weights, &later_overflow),
        Err(StateError::NonFinite)
    );
    assert_eq!(state, original);
}

#[test]
fn published_nemotron_nano_mixer_dimensions_are_admitted() {
    let config = Config {
        hidden_size: 2688,
        heads: 64,
        head_dim: 64,
        groups: 8,
        state_dim: 128,
        kernel: 4,
        eps: 1e-5,
        dt_min: 0.0,
        dt_max: f32::INFINITY,
    };
    let state = LayerState::new(config).unwrap();
    assert_eq!(state.conv_history().len(), (4096 + 2 * 8 * 128) * 4);
    assert_eq!(state.ssm_state().len(), 64 * 64 * 128);
}
