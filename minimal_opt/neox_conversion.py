import torch
from minimal_opt.model import OPTConfig
import os
import numpy as np
import glob
from tqdm import auto as tqdm_lib


def create_embedding_state(config: OPTConfig, num_shards=1, dtype=torch.float16):
    return {
        "word_embeddings.weight": torch.empty([config.vocab_size // num_shards, config.hidden_size], dtype=dtype),
        "position_embeddings.weight": torch.empty([config.max_position_embeddings, config.hidden_size], dtype=dtype),
    }


def create_transformer_layer_state(config: OPTConfig, num_shards=1, dtype=torch.float16):
    return {
        "input_layernorm.weight": torch.empty([config.hidden_size], dtype=dtype),
        "input_layernorm.bias": torch.empty([config.hidden_size], dtype=dtype),
        "post_attention_layernorm.weight": torch.empty([config.hidden_size], dtype=dtype),
        "post_attention_layernorm.bias": torch.empty([config.hidden_size], dtype=dtype),
        "attention.query_key_value.weight": torch.empty([
            3 * config.hidden_size // num_shards,
            config.hidden_size,
        ], dtype=dtype),
        "attention.query_key_value.bias": torch.empty([3 * config.hidden_size // num_shards], dtype=dtype),
        "attention.dense.weight": torch.empty([config.hidden_size, config.hidden_size // num_shards], dtype=dtype),
        "attention.dense.bias": torch.empty([config.hidden_size], dtype=dtype),
        "mlp.dense_h_to_4h.weight": torch.empty([config.ffn_dim // num_shards, config.hidden_size], dtype=dtype),
        "mlp.dense_h_to_4h.bias": torch.empty([config.ffn_dim // num_shards], dtype=dtype),
        "mlp.dense_4h_to_h.weight": torch.empty([config.hidden_size, config.ffn_dim // num_shards], dtype=dtype),
        "mlp.dense_4h_to_h.bias": torch.empty([config.hidden_size], dtype=dtype),
    }


def create_final_layer_norm_state(config: OPTConfig, dtype=torch.float16):
    return {
        "norm.weight": torch.empty([config.hidden_size], dtype=dtype),
        "norm.bias": torch.empty([config.hidden_size], dtype=dtype),
    }


def convert_opt_to_neox_weights(config: OPTConfig,
                                opt_sharded_checkpoint_list,
                                num_neox_shards,
                                neox_output_path,
                                dtype=torch.float16,
                                delete_existing=False):

    num_opt_shards = len(opt_sharded_checkpoint_list)
    assert num_neox_shards <= num_opt_shards
    opt_shards_per_neox_shard = num_opt_shards // num_neox_shards
    os.makedirs(neox_output_path, exist_ok=True)

    tqdm_total = num_opt_shards * (config.num_hidden_layers + 3) + int(delete_existing)
    pbar = tqdm_lib.tqdm(total=tqdm_total)

    if delete_existing:
        pbar.set_description("Deleting existing model states")
        path_ls = glob.glob(os.path.join(neox_output_path, "layer_*-model_*-model_states.pt"))
        for path in path_ls:
            os.remove(path)
        pbar.update()

    # OPT
    vocab_size_per_opt_shard = config.vocab_size // num_opt_shards
    heads_per_opt_shard = config.num_attention_heads // num_opt_shards
    hidden_size_per_opt_shard = config.hidden_size // num_opt_shards
    ffn_dim_per_opt_shard = config.ffn_dim // num_opt_shards

    # NeoX
    vocab_size_per_neox_shard = config.vocab_size // num_neox_shards
    heads_per_neox_shard = config.num_attention_heads // num_neox_shards
    hidden_size_per_neox_shard = config.hidden_size // num_neox_shards
    ffn_dim_per_neox_shard = config.ffn_dim // num_neox_shards

    for opt_shard_i, shard_path in enumerate(opt_sharded_checkpoint_list):
        pbar.set_description(f"OPT Shard {opt_shard_i}/{num_opt_shards}: Loading shard")
        # Mapping from OPT shard to NeoX shard
        corresponding_neox_shard_i = opt_shard_i // opt_shards_per_neox_shard
        index_within_neox_shard = opt_shard_i % opt_shards_per_neox_shard

        loaded = torch.load(shard_path, map_location="cpu")
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
        pbar.update()

        # === Embeddings === #
        pbar.set_description(f"OPT Shard {opt_shard_i}/{num_opt_shards}: Vocab")
        neox_embeddings_path = os.path.join(
            neox_output_path, f"layer_00-model_{corresponding_neox_shard_i:02d}-model_states.pt")
        if os.path.exists(neox_embeddings_path):
            neox_vocab_state = torch.load(neox_embeddings_path)
        else:
            neox_vocab_state = create_embedding_state(config=config, num_shards=num_neox_shards, dtype=dtype)

        out, flat_params = take_out(flat_params, (vocab_size_per_opt_shard, config.hidden_size))
        neox_vocab_state["word_embeddings.weight"][
            get_slice(vocab_size_per_opt_shard, index_within_neox_shard), :
        ] = out
        out, flat_params = take_out(flat_params, (config.max_position_embeddings, config.hidden_size))
        neox_vocab_state["position_embeddings.weight"][:, :] = out
        torch.save(neox_vocab_state, neox_embeddings_path)
        pbar.update()

        # === Post-attention LayerNorm (Before) === #
        if load_final_layer_norm_first:
            pbar.set_description(f"OPT Shard {opt_shard_i}/{num_opt_shards}: Final Layer Norm")
            neox_layer_i = config.num_hidden_layers + 3
            neox_final_layer_norm_path = os.path.join(
                neox_output_path,
                f"layer_{neox_layer_i:02d}-model_{corresponding_neox_shard_i:02d}-model_states.pt")
            if os.path.exists(neox_final_layer_norm_path):
                neox_final_layer_norm_state = torch.load(neox_final_layer_norm_path)
            else:
                neox_final_layer_norm_state = create_final_layer_norm_state(config=config, dtype=dtype)
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_final_layer_norm_state["norm.weight"] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_final_layer_norm_state["norm.bias"] = out
            torch.save(neox_final_layer_norm_state, neox_final_layer_norm_path)
            pbar.update()

        # === Transformer Layer === #
        for layer_i in range(config.num_hidden_layers):
            pbar.set_description(
                f"OPT Shard {opt_shard_i}/{num_opt_shards}: Transformer Layer {layer_i}/{config.num_hidden_layers}"
            )
            neox_layer_i = layer_i + 2
            neox_layer_path = os.path.join(
                neox_output_path, f"layer_{neox_layer_i:02d}-model_{corresponding_neox_shard_i:02d}-model_states.pt")
            if os.path.exists(neox_layer_path):
                neox_layer_state = torch.load(neox_layer_path)
            else:
                neox_layer_state = create_transformer_layer_state(config=config, num_shards=num_neox_shards, dtype=dtype)

            k_weight, flat_params = take_out(flat_params, (heads_per_opt_shard, config.head_dim, config.hidden_size))
            v_weight, flat_params = take_out(flat_params, (heads_per_opt_shard, config.head_dim, config.hidden_size))
            q_weight, flat_params = take_out(flat_params, (heads_per_opt_shard, config.head_dim, config.hidden_size))
            reshaped_neox_kvq_weight = neox_layer_state["attention.query_key_value.weight"].view(
                3 * heads_per_neox_shard,
                config.head_dim,
                config.hidden_size,
            )
            reshaped_neox_kvq_weight[
                :heads_per_opt_shard, :, :
            ] = k_weight
            reshaped_neox_kvq_weight[
                heads_per_neox_shard: heads_per_neox_shard + heads_per_opt_shard, :, :
            ] = v_weight
            reshaped_neox_kvq_weight[
                2 * heads_per_neox_shard: 2 * heads_per_neox_shard + heads_per_opt_shard, :, :
            ] = q_weight
            neox_layer_state["attention.query_key_value.weight"] = reshaped_neox_kvq_weight.view(
                3 * config.hidden_size // num_neox_shards,
                config.hidden_size,
            )

            # K/V/Q bias
            k_bias, flat_params = take_out(flat_params, (heads_per_opt_shard, config.head_dim,))
            v_bias, flat_params = take_out(flat_params, (heads_per_opt_shard, config.head_dim,))
            q_bias, flat_params = take_out(flat_params, (heads_per_opt_shard, config.head_dim,))
            reshaped_neox_kvq_bias = neox_layer_state["attention.query_key_value.bias"].view(
                3 * heads_per_neox_shard,
                config.head_dim,
            )
            reshaped_neox_kvq_bias[
                :heads_per_opt_shard, :
            ] = k_bias
            reshaped_neox_kvq_bias[
                heads_per_neox_shard: heads_per_neox_shard + heads_per_opt_shard, :
            ] = v_bias
            reshaped_neox_kvq_bias[
                2 * heads_per_neox_shard: 2 * heads_per_neox_shard + heads_per_opt_shard, :
            ] = q_bias
            neox_layer_state["attention.query_key_value.bias"] = reshaped_neox_kvq_bias.view(
                3 * config.hidden_size // num_neox_shards,
            )

            # O weight, O bias
            out, flat_params = take_out(flat_params, (config.hidden_size, heads_per_opt_shard, config.head_dim))
            neox_layer_state["attention.dense.weight"][
                :, get_slice(hidden_size_per_opt_shard, index_within_neox_shard)
            ] = out.view(config.hidden_size, config.hidden_size // num_opt_shards)
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_layer_state["attention.dense.bias"] = out

            # Input LayerNorm
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_layer_state["input_layernorm.weight"] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_layer_state["input_layernorm.bias"] = out

            # MLP dense_h_to_4h
            out, flat_params = take_out(flat_params, (ffn_dim_per_opt_shard, config.hidden_size))
            neox_layer_state["mlp.dense_h_to_4h.weight"][
                get_slice(ffn_dim_per_opt_shard, index_within_neox_shard), :
            ] = out
            out, flat_params = take_out(flat_params, (ffn_dim_per_opt_shard,))
            neox_layer_state["mlp.dense_h_to_4h.bias"][
                get_slice(ffn_dim_per_opt_shard, index_within_neox_shard)
            ] = out

            # MLP dense_4h_to_h
            out, flat_params = take_out(flat_params, (config.hidden_size, ffn_dim_per_opt_shard))
            neox_layer_state["mlp.dense_4h_to_h.weight"][
                :, get_slice(ffn_dim_per_opt_shard, index_within_neox_shard)
            ] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_layer_state["mlp.dense_4h_to_h.bias"] = out

            # Post-attention LayerNorm
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_layer_state["post_attention_layernorm.weight"] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_layer_state["post_attention_layernorm.bias"] = out

            torch.save(neox_layer_state, neox_layer_path)
            pbar.update()

        # === Post-attention LayerNorm (After) === #
        if not load_final_layer_norm_first:
            pbar.set_description(f"OPT Shard {opt_shard_i}/{num_opt_shards}: Final Layer Norm")
            neox_layer_i = config.num_hidden_layers + 3
            neox_final_layer_norm_path = os.path.join(
                neox_output_path,
                f"layer_{neox_layer_i:02d}-model_{corresponding_neox_shard_i:02d}-model_states.pt")
            if os.path.exists(neox_final_layer_norm_path):
                neox_final_layer_norm_state = torch.load(neox_final_layer_norm_path)
            else:
                neox_final_layer_norm_state = create_final_layer_norm_state(config=config, dtype=dtype)
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_final_layer_norm_state["norm.weight"] = out
            out, flat_params = take_out(flat_params, (config.hidden_size,))
            neox_final_layer_norm_state["norm.bias"] = out
            torch.save(neox_final_layer_norm_state, neox_final_layer_norm_path)
            pbar.update()

    pbar.set_description("Done.")


def take_out(flat_params, shape):
    return flat_params[:np.prod(shape)].view(*shape), flat_params[np.prod(shape):]


def get_slice(shard_size, shard_i):
    return slice(shard_size * shard_i, shard_size * (shard_i + 1))
