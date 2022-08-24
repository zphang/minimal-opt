import numpy as np
import torch


def take_out(flat_params, shape):
    return flat_params[:np.prod(shape)].view(*shape), flat_params[np.prod(shape):]


def get_slice(shard_size, shard_i):
    return slice(shard_size * shard_i, shard_size * (shard_i + 1))


def load_sharded_weights(opt_model, sharded_checkpoint_list):
    config = opt_model.config
    model = opt_model.model.decoder
    num_shards = len(sharded_checkpoint_list)
    for shard_i in range(num_shards):
        loaded = torch.load(sharded_checkpoint_list[shard_i], map_location="cpu")
        if len(loaded["model"]) == 2:
            # small_model
            flat_params = loaded["model"]["flat_param_0"]
            load_final_layer_norm_first = False
        else:
            # big model
            load_final_layer_norm_first = True
            flat_params = torch.cat([
                v.flatten()
                for k, v in loaded["model"].items()
                if k != "decoder.version"
            ])

        # noinspection PyUnresolvedReferences
        vocab_size_per_shard = config.vocab_size // num_shards
        heads_per_shard = config.num_attention_heads // num_shards
        hidden_size_per_shard = config.hidden_size // num_shards
        ffn_dim_per_shard = config.ffn_dim // num_shards
        dims_per_head = config.hidden_size // config.num_attention_heads

        # Vocab
        out, flat_params = take_out(flat_params, (vocab_size_per_shard, config.hidden_size))
        model.embed_tokens.weight.data[get_slice(vocab_size_per_shard, shard_i)] = out
        opt_model.lm_head.weight = model.embed_tokens.weight

        # Pos encoding (fixed offset=2)
        out, flat_params = take_out(flat_params, (config.max_position_embeddings + 2, config.hidden_size))
        model.embed_positions.weight.data[:] = out

        if load_final_layer_norm_first:
            # Post-attention LayerNorm
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.final_layer_norm.weight.data[:] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.final_layer_norm.bias.data[:] = out
            # If code fails here, you need to update transformers.
            # An earlier version was missing the final_layer_norm

        for layer_i in range(config.num_hidden_layers):

            # K/V/Q weights
            out, flat_params = take_out(flat_params, (heads_per_shard, dims_per_head, config.hidden_size))
            model.layers[layer_i].self_attn.k_proj.weight.data.reshape(
                config.num_attention_heads, dims_per_head, config.hidden_size,
            )[get_slice(heads_per_shard, shard_i), :, :] = out
            out, flat_params = take_out(flat_params, (heads_per_shard, dims_per_head, config.hidden_size))
            model.layers[layer_i].self_attn.v_proj.weight.data.reshape(
                config.num_attention_heads, dims_per_head, config.hidden_size,
            )[get_slice(heads_per_shard, shard_i), :, :] = out
            out, flat_params = take_out(flat_params, (heads_per_shard, dims_per_head, config.hidden_size))
            model.layers[layer_i].self_attn.q_proj.weight.data.reshape(
                config.num_attention_heads, dims_per_head, config.hidden_size,
            )[get_slice(heads_per_shard, shard_i), :, :] = out

            # K/V/Q bias
            out, flat_params = take_out(flat_params, (hidden_size_per_shard,))
            model.layers[layer_i].self_attn.k_proj.bias.data[
                get_slice(hidden_size_per_shard, shard_i)] = out
            out, flat_params = take_out(flat_params, (hidden_size_per_shard,))
            model.layers[layer_i].self_attn.v_proj.bias.data[
                get_slice(hidden_size_per_shard, shard_i)] = out
            out, flat_params = take_out(flat_params, (hidden_size_per_shard,))
            model.layers[layer_i].self_attn.q_proj.bias.data[
                get_slice(hidden_size_per_shard, shard_i)] = out

            # O weight, O bias
            out, flat_params = take_out(flat_params, (config.hidden_size, heads_per_shard, dims_per_head))
            model.layers[layer_i].self_attn.out_proj.weight.data.reshape(
                config.hidden_size, config.num_attention_heads, dims_per_head,
            )[:, get_slice(heads_per_shard, shard_i), :] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.layers[layer_i].self_attn.out_proj.bias.data[:] = out

            # Input LayerNorm
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.layers[layer_i].self_attn_layer_norm.weight.data[:] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.layers[layer_i].self_attn_layer_norm.bias.data[:] = out

            # MLP dense_h_to_4h
            out, flat_params = take_out(flat_params, (ffn_dim_per_shard, config.hidden_size))
            model.layers[layer_i].fc1.weight.data[
                get_slice(ffn_dim_per_shard, shard_i), :] = out
            out, flat_params = take_out(flat_params, (ffn_dim_per_shard,))
            model.layers[layer_i].fc1.bias.data[
                get_slice(ffn_dim_per_shard, shard_i)] = out

            # MLP dense_4h_to_h
            out, flat_params = take_out(flat_params, (config.hidden_size, ffn_dim_per_shard))
            model.layers[layer_i].fc2.weight.data[
                :, get_slice(ffn_dim_per_shard, shard_i)] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.layers[layer_i].fc2.bias.data[:] = out

            # Post-attention LayerNorm
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.layers[layer_i].final_layer_norm.weight.data[:] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.layers[layer_i].final_layer_norm.bias.data[:] = out

        if not load_final_layer_norm_first:
            # Post-attention LayerNorm
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.final_layernorm.weight.data[:] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            model.final_layernorm.bias.data[:] = out
            # If code fails here, you need to update transformers.
            # An earlier version was missing the final_layer_norm

        assert flat_params.numel() == 0
