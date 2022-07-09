import numpy as np


def take_out(flat_params, shape):
    return flat_params[:np.prod(shape)], flat_params[np.prod(shape):]


# noinspection PyUnresolvedReferences
def f(opt_model, flat_params, config):
    out, flat_params = take_out(flat_params, (config.vocab_size // 2, config.hidden_size))
    assert (out == opt_model.model.decoder.embed_tokens.weight[:config.vocab_size // 2].flatten()).all()
    out, flat_params = take_out(flat_params, (config.max_position_embeddings, config.hidden_size))
    assert (out == opt_model.model.decoder.embed_positions.weight.flatten()).all()
    for i in range(config.num_hidden_layers):
        out, flat_params = take_out(flat_params, (config.num_attention_heads // 2, config.head_dim, config.hidden_size))
        assert (out == opt_model.model.decoder.layers[i].self_attn.k_proj.weight.reshape(
            config.num_attention_heads, config.head_dim, config.hidden_size,
        )[:config.num_attention_heads//2, :, :].flatten()).all()
        out, flat_params = take_out(flat_params, (config.num_attention_heads // 2, config.head_dim, config.hidden_size))
        assert (out == opt_model.model.decoder.layers[i].self_attn.v_proj.weight.reshape(
            config.num_attention_heads, config.head_dim, config.hidden_size,
        )[:config.num_attention_heads//2, :, :].flatten()).all()
        out, flat_params = take_out(flat_params, (config.num_attention_heads // 2, config.head_dim, config.hidden_size))
        assert (out == opt_model.model.decoder.layers[i].self_attn.q_proj.weight.reshape(
            config.num_attention_heads, config.head_dim, config.hidden_size,
        )[:config.num_attention_heads//2, :, :].flatten()).all()
        out, flat_params = take_out(flat_params, (config.hidden_size // 2,))
        assert (out == opt_model.model.decoder.layers[i].self_attn.k_proj.bias[:config.hidden_size // 2]).all()
        out, flat_params = take_out(flat_params, (config.hidden_size // 2,))
        assert (out == opt_model.model.decoder.layers[i].self_attn.v_proj.bias[:config.hidden_size // 2]).all()
        out, flat_params = take_out(flat_params, (config.hidden_size // 2,))
        assert (out == opt_model.model.decoder.layers[i].self_attn.q_proj.bias[:config.hidden_size // 2]).all()
        out, flat_params = take_out(flat_params, (config.hidden_size, config.num_attention_heads // 2, config.head_dim))
        assert (out == opt_model.model.decoder.layers[i].self_attn.out_proj.weight.reshape(
            config.hidden_size, config.num_attention_heads, config.head_dim
        )[:, :config.num_attention_heads//2, :].flatten()).all()
        out, flat_params = take_out(flat_params, (config.hidden_size,))
        assert (out == opt_model.model.decoder.layers[i].self_attn.out_proj.bias).all()
        out, flat_params = take_out(flat_params, (config.hidden_size,))
        assert (out == opt_model.model.decoder.layers[i].self_attn_layer_norm.weight).all()
        out, flat_params = take_out(flat_params, (config.hidden_size,))
        assert (out == opt_model.model.decoder.layers[i].self_attn_layer_norm.bias).all()
        out, flat_params = take_out(flat_params, (config.ffn_dim // 2, config.hidden_size))
        assert (out == opt_model.model.decoder.layers[i].fc1.weight[:config.ffn_dim // 2].flatten()).all()
        out, flat_params = take_out(flat_params, (config.ffn_dim // 2,))
        assert (out == opt_model.model.decoder.layers[i].fc1.bias[:config.ffn_dim // 2]).all()
        out, flat_params = take_out(flat_params, (config.hidden_size, config.ffn_dim // 2))
        assert (out == opt_model.model.decoder.layers[i].fc2.weight[:, :config.ffn_dim // 2].flatten()).all()
        out, flat_params = take_out(flat_params, (config.hidden_size,))
        assert (out == opt_model.model.decoder.layers[i].fc2.bias).all()
        out, flat_params = take_out(flat_params, (config.hidden_size,))
        assert (out == opt_model.model.decoder.layers[i].final_layer_norm.weight).all()
        out, flat_params = take_out(flat_params, (config.hidden_size,))
        assert (out == opt_model.model.decoder.layers[i].final_layer_norm.bias).all()
    out, flat_params = take_out(flat_params, (config.hidden_size,))
    assert (out == opt_model.model.decoder.final_layer_norm.weight).all()
    out, flat_params = take_out(flat_params, (config.hidden_size,))
    assert (out == opt_model.model.decoder.final_layer_norm.bias).all()
