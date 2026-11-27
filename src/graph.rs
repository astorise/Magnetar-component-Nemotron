//! Experimental Nano-shaped graph producer. Checkpoint ingestion, a complete
//! mixed-model numerical reference and BF16 execution are not yet qualified.
//!
//! The pinned NVIDIA block does not forward its attention cache. This graph
//! deliberately declares explicit causal KV history, as required by the
//! Magnetar state contract; it is not presented as equivalent to that block's
//! cached execution until a reference profile has been qualified.

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "model-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    Attribute, AttributeValue, GraphPhase, KvCacheRole, KvInvolvement, TensorShape, add_node,
    add_node_outputs, begin_graph, declare_input, finish_graph, kv_resource, recurrent_input,
    recurrent_output, recurrent_resource, weight_edge,
};
use magnetar::model_component_graph::model_config::{
    self, ArchitectureConfig, ExpertGroupScore, ExpertNormalizationMode, ExpertScoring,
    HybridDecoderBlock, Mamba2HybridConfig, MoeConfig,
};

fn shape(dimensions: Vec<u64>) -> TensorShape {
    TensorShape { dimensions }
}

fn attr(name: &str, value: AttributeValue) -> Attribute {
    Attribute {
        name: name.into(),
        value,
    }
}

fn integer(name: &str, value: u64) -> Attribute {
    attr(
        name,
        AttributeValue::Integer(i64::try_from(value).expect("dimension fits i64")),
    )
}

fn weight(layer: u64, name: &str) -> String {
    weight_edge(&format!("layers.{layer}.{name}"))
}

/// Read all settings needed by this narrow profile. Other declared forms are
/// left unread, so the Runtime rejects the resulting graph rather than
/// silently substituting a different decoder.
fn read_profile(config: &ArchitectureConfig) -> (Mamba2HybridConfig, Option<MoeConfig>) {
    let hybrid = model_config::mamba2_hybrid().expect("Mamba-2 hybrid configuration required");
    assert_eq!(hybrid.layers.len(), config.num_hidden_layers as usize);
    assert!(
        !config.attention_bias,
        "attention projection bias is unsupported"
    );
    assert!(
        !config.tie_word_embeddings,
        "tied embeddings are unsupported"
    );
    assert!(config.rope_scaling_factor.is_none() && model_config::rope_scaling().is_none());
    assert_eq!(
        model_config::hidden_activation().as_deref(),
        Some("relu_squared")
    );
    let positions = model_config::rope_layers();
    if hybrid
        .layers
        .iter()
        .any(|layer| matches!(layer, HybridDecoderBlock::Attention))
    {
        assert!(
            positions.is_some(),
            "attention position mode must be explicit"
        );
    }
    if let Some(positions) = positions {
        assert_eq!(positions.len(), hybrid.layers.len());
        assert!(
            positions.iter().all(|enabled| !enabled),
            "RoPE is unsupported"
        );
    }
    assert!(
        hybrid.conv_bias && !hybrid.linear_bias,
        "unsupported Mamba bias profile"
    );
    assert_eq!(hybrid.dt_min, 0.0, "unsupported scan lower limit");
    assert!(hybrid.dt_max.is_none(), "unsupported scan upper limit");

    let moe = model_config::mixture_of_experts();
    let moe_layers = model_config::mixture_of_experts_layers();
    let routing = model_config::expert_routing();
    let normalization = model_config::expert_normalization();
    let expected: Vec<bool> = hybrid
        .layers
        .iter()
        .map(|layer| matches!(layer, HybridDecoderBlock::RoutedMoe))
        .collect();
    if expected.iter().any(|enabled| *enabled) {
        let config = moe.as_ref().expect("MoE configuration required");
        assert!(config.experts > 1 && config.experts_per_token > 0);
        assert!(config.normalize_top_k && config.shared_expert_intermediate_size.is_some());
        assert_eq!(moe_layers.as_ref(), Some(&expected));
        let routing = routing.expect("expert routing required");
        assert!(matches!(routing.scoring, ExpertScoring::Sigmoid));
        assert!(routing.selection_bias && !routing.shared_expert_gated);
        assert!(routing.routed_scaling_factor.is_finite() && routing.routed_scaling_factor > 0.0);
        let groups = routing.groups.expect("expert groups required");
        // NVIDIA's published Nano selects its only group. General group
        // masking is not equivalent when corrected scores are negative.
        assert!(groups.count == 1 && groups.per_token == 1);
        assert!(matches!(groups.score, ExpertGroupScore::TopTwoSum));
        assert!(matches!(
            normalization,
            Some(ExpertNormalizationMode::Always)
        ));
    } else {
        assert!(
            moe.is_none() && moe_layers.is_none() && routing.is_none() && normalization.is_none()
        );
    }
    (hybrid, moe)
}

fn mamba(
    config: &ArchitectureConfig,
    hybrid: &Mamba2HybridConfig,
    layer: u64,
    input: String,
    tokens: u64,
) -> String {
    let inner = u64::from(hybrid.heads) * hybrid.head_dimension;
    let channels = inner + 2 * u64::from(hybrid.groups) * hybrid.state_dimension;
    let conv_shape = shape(vec![channels, u64::from(hybrid.conv_kernel)]);
    let ssm_shape = shape(vec![
        u64::from(hybrid.heads),
        hybrid.head_dimension,
        hybrid.state_dimension,
    ]);
    let conv = recurrent_resource(layer, "conv", &conv_shape);
    let ssm = recurrent_resource(layer, "ssm", &ssm_shape);
    let mut inputs = vec![input, recurrent_input(&conv), recurrent_input(&ssm)];
    for name in [
        "mamba2.in_proj",
        "mamba2.conv",
        "mamba2.dt_bias",
        "mamba2.a_log",
        "mamba2.d",
        "mamba2.norm",
        "mamba2.out_proj",
        "mamba2.conv_bias",
    ] {
        inputs.push(weight(layer, name));
    }
    let attributes = [
        integer("heads", u64::from(hybrid.heads)),
        integer("head_dim", hybrid.head_dimension),
        integer("groups", u64::from(hybrid.groups)),
        integer("state_dim", hybrid.state_dimension),
        integer("conv_kernel", u64::from(hybrid.conv_kernel)),
        attr("epsilon", AttributeValue::FloatValue(hybrid.epsilon)),
        attr("dt_min", AttributeValue::FloatValue(hybrid.dt_min)),
        attr("conv_bias", AttributeValue::Boolean(true)),
        attr("linear_bias", AttributeValue::Boolean(false)),
    ];
    let outputs = add_node_outputs(
        &format!("layer{layer}.mamba2"),
        "mamba2-mixer",
        "LinearAlgebra",
        &attributes,
        &inputs,
        &[
            shape(vec![tokens, config.hidden_size]),
            conv_shape,
            ssm_shape,
        ],
    );
    recurrent_output(&conv, &outputs[1]);
    recurrent_output(&ssm, &outputs[2]);
    outputs[0].clone()
}

fn attention(
    config: &ArchitectureConfig,
    layer: u64,
    input: String,
    tokens: u64,
    append: bool,
) -> String {
    let query = u64::from(config.num_attention_heads) * config.head_dimension;
    let kv = u64::from(config.num_key_value_heads) * config.head_dimension;
    let project = |name: &str, width: u64, role: Option<KvCacheRole>| {
        let involvement = role.map(|role| KvInvolvement {
            resource_id: kv_resource(layer, role),
            role,
            append,
        });
        add_node(
            &format!("layer{layer}.{name}"),
            "matmul",
            "LinearAlgebra",
            &[],
            &[input.clone(), weight(layer, &format!("attention.{name}"))],
            &shape(vec![tokens, width]),
            involvement.as_ref(),
        )
    };
    let q = project("q_proj", query, None);
    let k = project("k_proj", kv, Some(KvCacheRole::K));
    let v = project("v_proj", kv, Some(KvCacheRole::V));
    let result = add_node(
        &format!("layer{layer}.attention"),
        "attention",
        "Attention",
        &[
            attr("causal", AttributeValue::Boolean(true)),
            integer("head_count", u64::from(config.num_attention_heads)),
            integer("kv_head_count", u64::from(config.num_key_value_heads)),
            integer("head_dimension", config.head_dimension),
            attr("attention_mask_kind", AttributeValue::Text("causal".into())),
        ],
        &[q, k, v],
        &shape(vec![tokens, query]),
        None,
    );
    add_node(
        &format!("layer{layer}.o_proj"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[result, weight(layer, "attention.o_proj")],
        &shape(vec![tokens, config.hidden_size]),
        None,
    )
}

fn dense(
    layer: u64,
    prefix: &str,
    input: String,
    tokens: u64,
    hidden: u64,
    intermediate: u64,
) -> String {
    let projected = add_node(
        &format!("layer{layer}.{prefix}.in"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[input, weight(layer, &format!("{prefix}.in"))],
        &shape(vec![tokens, intermediate]),
        None,
    );
    let activated = add_node(
        &format!("layer{layer}.{prefix}.activation"),
        "activation",
        "Activation",
        &[attr("kind", AttributeValue::Text("relu_squared".into()))],
        &[projected],
        &shape(vec![tokens, intermediate]),
        None,
    );
    add_node(
        &format!("layer{layer}.{prefix}.out"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[activated, weight(layer, &format!("{prefix}.out"))],
        &shape(vec![tokens, hidden]),
        None,
    )
}

fn experts(
    config: &ArchitectureConfig,
    moe: &MoeConfig,
    layer: u64,
    input: String,
    tokens: u64,
) -> String {
    let routing = model_config::expert_routing().expect("expert routing required");
    let logits = add_node(
        &format!("layer{layer}.router"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[input.clone(), weight(layer, "moe.router")],
        &shape(vec![tokens, u64::from(moe.experts)]),
        None,
    );
    let routed = add_node(
        &format!("layer{layer}.experts"),
        "moe-experts",
        "LinearAlgebra",
        &[
            integer("experts", u64::from(moe.experts)),
            integer("top_k", u64::from(moe.experts_per_token)),
            attr("activation", AttributeValue::Text("relu_squared".into())),
            attr("routing_scoring", AttributeValue::Text("sigmoid".into())),
            attr(
                "routing_normalization",
                AttributeValue::Text("always".into()),
            ),
            attr(
                "routing_scale",
                AttributeValue::FloatValue(routing.routed_scaling_factor),
            ),
            integer("routing_group_count", 1),
            integer("routing_group_selected", 1),
        ],
        &[
            input.clone(),
            logits,
            weight(layer, "moe.experts_in"),
            weight(layer, "moe.experts_out"),
            weight(layer, "moe.selection_bias"),
        ],
        &shape(vec![tokens, config.hidden_size]),
        None,
    );
    let shared = dense(
        layer,
        "moe.shared",
        input,
        tokens,
        config.hidden_size,
        moe.shared_expert_intermediate_size
            .expect("shared expert required"),
    );
    add_node(
        &format!("layer{layer}.moe"),
        "add",
        "Tensor",
        &[],
        &[routed, shared],
        &shape(vec![tokens, config.hidden_size]),
        None,
    )
}

fn graph(
    phase: GraphPhase,
    tokens: u64,
    position: u64,
    append: bool,
    start: u32,
    end: u32,
) -> String {
    let config = model_config::model_config();
    let (hybrid, moe) = read_profile(&config);
    assert!(tokens > 0 && start <= end && end <= config.num_hidden_layers);
    begin_graph(phase, tokens, position);
    let mut hidden = if start == 0 {
        let ids = declare_input("token_ids", &shape(vec![tokens]));
        add_node(
            "embedding",
            "embedding",
            "Tensor",
            &[],
            &[ids, weight_edge("token_embedding")],
            &shape(vec![tokens, config.hidden_size]),
            None,
        )
    } else {
        declare_input("hidden_states_in", &shape(vec![tokens, config.hidden_size]))
    };
    for layer in u64::from(start)..u64::from(end) {
        let normalized = add_node(
            &format!("layer{layer}.norm"),
            "rmsnorm",
            "Normalization",
            &[attr(
                "epsilon",
                AttributeValue::FloatValue(config.rms_norm_eps),
            )],
            &[hidden.clone(), weight(layer, "norm")],
            &shape(vec![tokens, config.hidden_size]),
            None,
        );
        let mixed = match hybrid.layers[layer as usize] {
            HybridDecoderBlock::Mamba2 => mamba(&config, &hybrid, layer, normalized, tokens),
            HybridDecoderBlock::Attention => attention(&config, layer, normalized, tokens, append),
            HybridDecoderBlock::DenseMlp => dense(
                layer,
                "mlp",
                normalized,
                tokens,
                config.hidden_size,
                config.intermediate_size,
            ),
            HybridDecoderBlock::RoutedMoe => experts(
                &config,
                moe.as_ref().expect("MoE required"),
                layer,
                normalized,
                tokens,
            ),
        };
        hidden = add_node(
            &format!("layer{layer}.residual"),
            "residual-add",
            "Tensor",
            &[],
            &[hidden, mixed],
            &shape(vec![tokens, config.hidden_size]),
            None,
        );
    }
    if end == config.num_hidden_layers {
        let normalized = add_node(
            "final_norm",
            "rmsnorm",
            "Normalization",
            &[attr(
                "epsilon",
                AttributeValue::FloatValue(config.rms_norm_eps),
            )],
            &[hidden, weight_edge("final_norm")],
            &shape(vec![tokens, config.hidden_size]),
            None,
        );
        let logits = add_node(
            "lm_head",
            "matmul",
            "LinearAlgebra",
            &[],
            &[normalized, weight_edge("lm_head")],
            &shape(vec![tokens, config.vocabulary_size]),
            None,
        );
        finish_graph(&logits)
    } else {
        finish_graph(&hidden)
    }
}

struct Component;
impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        let count = model_config::model_config().num_hidden_layers;
        graph(GraphPhase::Prefill, sequence_length, 0, false, 0, count)
    }
    fn build_decode_graph(cached_token_count: u64) -> String {
        let count = model_config::model_config().num_hidden_layers;
        graph(GraphPhase::Decode, 1, cached_token_count, true, 0, count)
    }
    fn build_prefill_graph_segment(
        sequence_length: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        graph(
            GraphPhase::Prefill,
            sequence_length,
            0,
            false,
            start_layer,
            end_layer,
        )
    }
    fn build_decode_graph_segment(
        cached_token_count: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        graph(
            GraphPhase::Decode,
            1,
            cached_token_count,
            true,
            start_layer,
            end_layer,
        )
    }
}
export!(Component);
