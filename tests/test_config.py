from manas.config import ManasConfig


def test_defaults_describe_the_64m_model():
    config = ManasConfig()
    assert (config.hidden_size, config.num_hidden_layers) == (768, 8)
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (8, 4, 96)
    assert config.intermediate_size == 2432
    assert config.vocab_size == 6400
    assert (config.bos_token_id, config.eos_token_id) == (1, 2)
    assert config.rope_theta == 1e6 and config.max_position_embeddings == 32768
    assert config.tie_word_embeddings is True


def test_intermediate_size_is_a_multiple_of_64():
    for hidden in (512, 640, 768, 1024):
        size = ManasConfig(hidden_size=hidden).intermediate_size
        assert size % 64 == 0
        assert 3.0 < size / hidden < 3.3


def test_moe_knobs_default_to_four_top1_experts():
    config = ManasConfig(use_moe=True)
    assert config.use_moe is True
    assert (config.num_experts, config.num_experts_per_tok) == (4, 1)
    assert config.moe_intermediate_size == config.intermediate_size
    assert config.norm_topk_prob is True and config.router_aux_loss_coef == 5e-4
    assert ManasConfig().use_moe is False


def test_overrides_and_round_trip():
    config = ManasConfig(hidden_size=512, num_key_value_heads=2, dropout=0.1)
    assert config.head_dim == 64 and config.dropout == 0.1
    restored = ManasConfig(**config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert config.model_type == "manas"
