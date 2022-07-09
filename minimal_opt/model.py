from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class OPTConfig:
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    num_attention_heads: int
    head_dim: int
    ffn_dim: int
    num_hidden_layers: int


OPT_125M_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=768,
    max_position_embeddings=2050,
    num_attention_heads=12,
    head_dim=64,
    ffn_dim=3072,
    num_hidden_layers=12,
)

OPT_1_3B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=2048,
    max_position_embeddings=2050,
    num_attention_heads=32,
    head_dim=64,
    ffn_dim=8192,
    num_hidden_layers=24,
)

OPT_2_7B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=2560,
    max_position_embeddings=2050,
    num_attention_heads=32,
    head_dim=80,
    ffn_dim=10240,
    num_hidden_layers=32,
)

OPT_6_7B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=4096,
    max_position_embeddings=2050,
    num_attention_heads=32,
    head_dim=128,
    ffn_dim=16384,
    num_hidden_layers=32,
)

OPT_13B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=5120,
    max_position_embeddings=2050,
    num_attention_heads=40,
    head_dim=128,
    ffn_dim=20480,
    num_hidden_layers=40,
)

OPT_30B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=7168,
    max_position_embeddings=2050,
    num_attention_heads=56,
    head_dim=128,
    ffn_dim=28672,
    num_hidden_layers=48,
)


class OPTModel(nn.Module):
    def __init__(self, config: OPTConfig, use_cache=False, device=None):
        super().__init__()
        self.config = config
        self.use_cache = use_cache
        self.embed_tokens = EmbeddingNoInit(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            device=device, dtype=torch.float16,
        )
        self.embed_positions = LearnedPositionalEmbedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            device=device, dtype=torch.float16,
        )
        self.layer_list = nn.ModuleList([])
        for layer_i in range(config.num_hidden_layers):
            self.layer_list.append(TransformerLayer(config, use_cache, device=device))
        self.final_layernorm = LayerNormNoInit(
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.logits_out = LinearNoInit(
            config.hidden_size, config.vocab_size, bias=False,
            device=device,
            dtype=torch.float16,
        )

    @classmethod
    def pre_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    @classmethod
    def post_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    def forward(self, input_ids, attention_mask=None, layer_past=None):

        if attention_mask is None:
            attention_mask = generate_mask(input_ids.shape[1]).to(input_ids.device)
        if self.use_cache:
            if layer_past is None:
                kv_length = input_ids.shape[1]
            else:
                kv_length = layer_past[0].shape[1] + 1
            attention_mask = attention_mask[..., :input_ids.shape[1], :kv_length]

        if layer_past is None:
            layer_past = [None] * len(self.layer_list)
        kv_cache_list = []
        token_embeddings = self.embed_tokens(input_ids)
        pos_embeddings = self.embed_positions(token_embeddings, kv_cache=layer_past)
        hidden_states = token_embeddings + pos_embeddings
        hidden_states = self.pre_transformer_transpose(hidden_states)
        for layer_i, layer in enumerate(self.layer_list):
            hidden_states, kv_cache = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past[layer_i],
            )
            kv_cache_list.append(kv_cache)
        hidden_states = self.post_transformer_transpose(hidden_states)
        hidden_states = self.final_layernorm(hidden_states)

        logits = self.logits_out(hidden_states)
        if self.use_cache:
            return logits, kv_cache_list
        else:
            return logits


class SelfAttention(nn.Module):
    def __init__(self, config: OPTConfig, use_cache=False, device=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_cache = use_cache
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.q_proj = LinearNoInit(
            config.hidden_size,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.k_proj = LinearNoInit(
            config.hidden_size,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.v_proj = LinearNoInit(
            config.hidden_size,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.out_proj = LinearNoInit(
            config.hidden_size,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )

    def forward(self, hidden_states, attention_mask, layer_past=None):
        has_layer_past = layer_past is not None and layer_past.numel() > 0
        q_seq_len, batch_size, hidden_dim = hidden_states.shape

        # [sq, b, np, hn]
        query_layer = self.q_proj(hidden_states).reshape(
            q_seq_len, batch_size, self.num_attention_heads, self.hidden_size_per_attention_head
        )
        query_layer /= self.norm_factor
        key_layer = self.k_proj(hidden_states).reshape(
            q_seq_len, batch_size, self.num_attention_heads, self.hidden_size_per_attention_head
        )
        value_layer = self.v_proj(hidden_states).reshape(
            q_seq_len, batch_size, self.num_attention_heads, self.hidden_size_per_attention_head
        )

        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        if self.use_cache:
            kv_cache = torch.stack((key_layer, value_layer))
        else:
            kv_cache = None

        # Compute attention
        # noinspection PyTypeChecker
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.out_proj(context_layer)

        return output, kv_cache

    # noinspection PyMethodMayBeStatic
    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2],
            output_size[0] * output_size[1],
            -1
        )
        key_layer = key_layer.view(
            output_size[3],
            output_size[0] * output_size[1],
            -1,
        )

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        masked_scores = attention_mask_func(attention_scores, attention_mask) \
            if attention_mask is not None else attention_scores
        # noinspection PyTypeChecker
        attention_probs = nn.functional.softmax(
            masked_scores, dim=-1, dtype=torch.float32).to(masked_scores.dtype)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer


class MLP(nn.Module):
    def __init__(self, config: OPTConfig, device=None):
        super().__init__()
        self.dense_h_to_4h = LinearNoInit(config.hidden_size, config.ffn_dim, device=device, dtype=torch.float16)
        self.dense_4h_to_h = LinearNoInit(config.ffn_dim, config.hidden_size, device=device, dtype=torch.float16)

    def forward(self, hidden_states):
        hidden_states_shape = hidden_states.shape
        intermediate_parallel = self.dense_h_to_4h(
            hidden_states.view(-1, hidden_states_shape[-1])
        )
        intermediate_parallel = F.relu(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output.view(*hidden_states_shape)


class TransformerLayer(nn.Module):
    def __init__(self, config: OPTConfig, use_cache, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = LayerNormNoInit(
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.post_attention_layernorm = LayerNormNoInit(
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.attention = SelfAttention(config, self.use_cache, device=device)
        self.mlp = MLP(config, device=device)

    def forward(self, hidden_states, attention_mask, layer_past=None):
        residual = hidden_states
        ln_output = self.input_layernorm(hidden_states)
        hidden_states, kv_cache = self.attention(
            ln_output,
            attention_mask,
            layer_past=layer_past,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, kv_cache


class EmbeddingNoInit(nn.Embedding):
    def reset_parameters(self):
        pass


class LayerNormNoInit(nn.LayerNorm):
    def reset_parameters(self):
        pass


class LinearNoInit(nn.Linear):
    def reset_parameters(self):
        pass


class LearnedPositionalEmbedding(EmbeddingNoInit):
    def __init__(self, num_embeddings: int, embedding_dim: int, magic_offset=2,
                 device=None, dtype=None):
        self.magic_offset = magic_offset
        super().__init__(num_embeddings, embedding_dim, device=device, dtype=dtype)

    # noinspection PyMethodOverriding
    def forward(self, token_embeddings, kv_cache):
        """`attention_mask` is expected to be [bsz x seqlen]."""
        # positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        batch_size, seq_len, _ = token_embeddings.shape
        if kv_cache is None or kv_cache[0] is None:
            positions = torch.arange(seq_len, device=token_embeddings.device)[None].expand(
                batch_size, -1,
            )
        else:
            kv_cache_len = kv_cache[0].shape[1]
            positions = torch.arange(kv_cache_len + seq_len, device=token_embeddings.device)[None].expand(
                batch_size, -1,
            )
            positions = positions[:, kv_cache_len:]

        return super().forward(positions + self.magic_offset)


def generate_mask(seq_len):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))


def attention_mask_func(attention_scores, ltor_mask):
    """Assign dtype minimum to False cells in ltor_mask"""
    attention_scores.masked_fill_(~ltor_mask, torch.tensor(torch.finfo(torch.float16).min))
    return attention_scores
