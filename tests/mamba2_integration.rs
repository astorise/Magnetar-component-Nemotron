use magnetar_component_nemotron::mamba2::{Mamba2State, StateError, StepInput};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    heads: usize,
    head_dim: usize,
    groups: usize,
    state_dim: usize,
    initial: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    d: Vec<f32>,
    dt_min: f32,
    dt_max: f32,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    x: Vec<f32>,
    dt: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    output: Vec<f32>,
    state: Vec<f32>,
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2e-6,
            "{label}[{index}]: {actual} != {expected}"
        );
    }
}

#[test]
fn successive_cached_steps_match_the_upstream_pytorch_decode_branch() {
    let reference: Reference =
        serde_json::from_slice(include_bytes!("fixtures/mamba2-decode-reference.json")).unwrap();
    let mut state = Mamba2State::from_values(
        reference.heads,
        reference.head_dim,
        reference.groups,
        reference.state_dim,
        reference.initial.clone(),
    )
    .unwrap();
    assert_eq!(reference.steps.len(), 3);
    for (index, step) in reference.steps.iter().enumerate() {
        let output = state
            .step(&StepInput {
                x: &step.x,
                dt: &step.dt,
                dt_bias: &reference.dt_bias,
                a_log: &reference.a_log,
                b: &step.b,
                c: &step.c,
                d: &reference.d,
                dt_min: reference.dt_min,
                dt_max: reference.dt_max,
            })
            .unwrap();
        close(&output, &step.output, &format!("step {index} output"));
        close(state.values(), &step.state, &format!("step {index} state"));
    }
    state.reset();
    assert!(state.values().iter().all(|value| *value == 0.0));
}

#[test]
fn invalid_steps_fail_without_changing_state() {
    assert_eq!(
        Mamba2State::new(3, 2, 2, 3),
        Err(StateError::InvalidDimensions)
    );
    assert_eq!(
        Mamba2State::new(usize::MAX, 2, 1, 3),
        Err(StateError::InvalidDimensions)
    );
    assert_eq!(
        Mamba2State::from_values(1, 1, 1, 1, vec![f32::NAN]),
        Err(StateError::NonFinite)
    );
    let mut state = Mamba2State::from_values(1, 1, 1, 1, vec![0.5]).unwrap();
    let original = state.clone();
    let mut input = StepInput {
        x: &[1.0],
        dt: &[0.0],
        dt_bias: &[0.0],
        a_log: &[0.0],
        b: &[1.0],
        c: &[1.0],
        d: &[0.0],
        dt_min: 0.0,
        dt_max: f32::INFINITY,
    };
    input.b = &[];
    assert_eq!(state.step(&input), Err(StateError::InvalidDimensions));
    assert_eq!(state, original);
    input.b = &[1.0];
    input.dt_min = 2.0;
    input.dt_max = 1.0;
    assert_eq!(state.step(&input), Err(StateError::InvalidTimeLimit));
    assert_eq!(state, original);
    input.dt_min = 0.0;
    input.dt_max = f32::INFINITY;
    input.x = &[f32::NAN];
    assert_eq!(state.step(&input), Err(StateError::NonFinite));
    assert_eq!(state, original);
    input.x = &[f32::MAX];
    input.b = &[f32::MAX];
    assert_eq!(state.step(&input), Err(StateError::NonFinite));
    assert_eq!(state, original);
}
