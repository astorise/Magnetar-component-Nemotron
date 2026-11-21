use magnetar_component_nemotron::mamba2_profile::{Mamba2Profile, WeightTransform};
use serde_json::{Value, json};

// Exact pinned NVIDIA Nemotron-3 Nano BF16 config; see fixtures/README.md.
fn config() -> Value {
    serde_json::from_str(include_str!("fixtures/nano-config.json")).unwrap()
}

fn profile(config: &Value) -> Result<Mamba2Profile, Box<dyn std::error::Error>> {
    Ok(Mamba2Profile::from_huggingface_config(
        &serde_json::to_vec(config)?,
    )?)
}

#[test]
fn pinned_nano_mixer_shapes_match_real_safetensors_header() {
    let profile = profile(&config()).unwrap();
    assert_eq!(profile.layers.len(), 23);
    assert_eq!(profile.initialization_dt_min, 0.001);
    assert_eq!(profile.initialization_dt_max, 0.1);
    let plan = profile.invocation(0, 7).unwrap();
    assert_eq!(plan.operator, "mamba2-mixer@1");
    assert_eq!(plan.dt_min, 0.0);
    assert_eq!(plan.dt_max, None);
    assert!(plan.conv_bias);
    assert!(!plan.linear_bias);
    assert_eq!(plan.input_shapes.len(), 11);
    assert_eq!(plan.input_shapes[0], [7, 2688]);
    assert_eq!(plan.input_shapes[1], [6144, 4]);
    assert_eq!(plan.input_shapes[2], [64, 64, 128]);
    assert_eq!(
        plan.output_shapes,
        vec![vec![7, 2688], vec![6144, 4], vec![64, 64, 128]]
    );
    let header: Value =
        serde_json::from_str(include_str!("fixtures/nano-mamba2-header.json")).unwrap();
    assert_eq!(plan.weights.len(), header.as_object().unwrap().len());
    for spec in &plan.weights {
        let actual = &header[&spec.checkpoint_name];
        assert_eq!(actual["dtype"].as_str().unwrap(), spec.checkpoint_dtype);
        let shape: Vec<usize> = actual["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dimension| dimension.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(shape, spec.checkpoint_shape, "{}", spec.checkpoint_name);
    }
    assert_eq!(plan.weights[0].transform, WeightTransform::Transpose);
    assert_eq!(plan.weights[0].operator_shape, [2688, 10304]);
    assert_eq!(
        plan.weights[1].transform,
        WeightTransform::FlattenDepthwiseConvolution
    );
    assert_eq!(plan.weights[6].operator_shape, [4096, 2688]);
    assert_eq!(
        profile.invocation(2, 1).unwrap().output_shapes[0],
        [1, 2688]
    );
}

#[test]
fn unsupported_configs_and_invalid_invocations_fail_closed() {
    for (key, value) in [
        ("n_groups", json!(1)),
        ("mamba_hidden_act", json!("gelu")),
        ("use_bias", json!(true)),
        ("use_conv_bias", json!(false)),
        ("torch_dtype", json!("float32")),
        ("mamba_ssm_cache_dtype", json!("bfloat16")),
        ("time_step_limit", json!([0.001, 0.1])),
        ("mamba_dt_limit", json!([0.001, 0.1])),
        ("mamba_n_groups", json!(1)),
        ("mamba_d_conv", json!(3)),
    ] {
        let mut altered = config();
        altered[key] = value;
        assert!(profile(&altered).is_err(), "accepted {key}");
    }
    let p = profile(&config()).unwrap();
    assert!(p.invocation(1, 1).is_err());
    assert!(p.invocation(0, 0).is_err());
    assert!(p.invocation(0, p.max_tokens + 1).is_err());
    assert!(p.invocation(0, p.max_tokens).is_ok());
    let mut forged = p.clone();
    forged.layers.push(1);
    assert!(forged.invocation(1, 1).is_err());
    assert!(forged.invocation(0, 1).is_err());
    forged = p;
    forged.heads = usize::MAX;
    assert!(forged.invocation(0, 1).is_err());
}
