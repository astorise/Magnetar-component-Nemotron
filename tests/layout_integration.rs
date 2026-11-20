use magnetar_component_nemotron::layout::{LayerKind, LayerLayout};

#[test]
fn published_nemotron_3_nano_pattern_preserves_order_and_kinds() {
    // Factual layout from NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, revision
    // fe4143e388795a58a92974891b2e45092a53dc1d; no model weights are used.
    let config = br#"{
        "model_type":"nemotron_h", "num_hidden_layers":52,
        "hybrid_override_pattern":"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    }"#;
    let layout = LayerLayout::from_huggingface_config(config).unwrap();
    assert_eq!(layout.layers().len(), 52);
    assert_eq!(
        layout
            .layers()
            .iter()
            .filter(|layer| **layer == LayerKind::Mamba2)
            .count(),
        23
    );
    assert_eq!(
        layout
            .layers()
            .iter()
            .filter(|layer| **layer == LayerKind::Moe)
            .count(),
        23
    );
    assert_eq!(
        layout
            .layers()
            .iter()
            .filter(|layer| **layer == LayerKind::Attention)
            .count(),
        6
    );
    assert_eq!(
        &layout.layers()[0..6],
        &[
            LayerKind::Mamba2,
            LayerKind::Moe,
            LayerKind::Mamba2,
            LayerKind::Moe,
            LayerKind::Mamba2,
            LayerKind::Attention
        ]
    );
}

#[test]
fn accepts_dense_mlp_blocks_and_rejects_malformed_or_unknown_layouts() {
    let config =
        br#"{"model_type":"nemotron_h","num_hidden_layers":4,"hybrid_override_pattern":"M*-E"}"#;
    assert_eq!(
        LayerLayout::from_huggingface_config(config)
            .unwrap()
            .layers(),
        &[
            LayerKind::Mamba2,
            LayerKind::Attention,
            LayerKind::Mlp,
            LayerKind::Moe
        ]
    );
    let cases: &[&[u8]] = &[
        br#"{"model_type":"llama","num_hidden_layers":1,"hybrid_override_pattern":"M"}"#,
        br#"{"model_type":"nemotron_h","num_hidden_layers":0,"hybrid_override_pattern":""}"#,
        br#"{"model_type":"nemotron_h","num_hidden_layers":4097,"hybrid_override_pattern":"M"}"#,
        br#"{"model_type":"nemotron_h","num_hidden_layers":2,"hybrid_override_pattern":"M"}"#,
        br#"{"model_type":"nemotron_h","num_hidden_layers":2,"hybrid_override_pattern":"MX"}"#,
        br#"{"model_type":"nemotron_h","num_hidden_layers":1}"#,
        b"not-json",
    ];
    for input in cases {
        assert!(
            LayerLayout::from_huggingface_config(input).is_err(),
            "{}",
            String::from_utf8_lossy(input)
        );
    }
}
