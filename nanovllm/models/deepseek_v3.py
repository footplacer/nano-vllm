import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


def _get_qk_head_dim(config: Any) -> int:
    qk_head_dim = getattr(config, "qk_head_dim", None)
    if qk_head_dim is not None:
        return qk_head_dim
    return config.qk_nope_head_dim + config.qk_rope_head_dim


def _get_rope_parameters(config: Any) -> dict:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        return rope_parameters
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        return rope_scaling
    return {}


def _get_rope_theta(config: Any) -> float:
    rope_parameters = _get_rope_parameters(config)
    return rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_attention_scaling(config: Any, qk_head_dim: int) -> float:
    scaling = qk_head_dim ** -0.5
    rope_parameters = _get_rope_parameters(config)
    rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
    if rope_type != "default":
        mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
        factor = rope_parameters.get("factor", 1.0)
        if mscale_all_dim:
            mscale = _yarn_get_mscale(factor, mscale_all_dim)
            scaling = scaling * mscale * mscale
    return scaling


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _apply_rotary_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x = x.float()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y = torch.empty_like(x)
    y[..., 0::2] = x_even * cos - x_odd * sin
    y[..., 1::2] = x_odd * cos + x_even * sin
    return y


class DeepseekV3RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        max_position: int,
        base: float,
        interleaved: bool = True,
    ) -> None:
        super().__init__()
        assert head_size % 2 == 0
        self.interleaved = interleaved
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.interleaved:
            query = _apply_rotary_emb_interleaved(query, cos, sin).to(query.dtype)
            key = _apply_rotary_emb_interleaved(key, cos, sin).to(key.dtype)
        else:
            query = _apply_rotary_emb(query, cos, sin)
            key = _apply_rotary_emb(key, cos, sin)
        return query, key


class DeepseekV3MLP(nn.Module):

    def __init__(
        self,
        config: Any,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
        )
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class DeepseekV3TopkRouter(nn.Module):

    def __init__(
        self,
        config: Any,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = getattr(config, "num_local_experts", getattr(config, "n_routed_experts"))
        self.top_k = config.num_experts_per_tok
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias

        if self.num_group is not None and self.topk_group is not None:
            group_scores = (
                scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.num_group, self.num_experts // self.num_group)
                .reshape(-1, self.num_experts)
            )
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_indices


class DeepseekV3Experts(nn.Module):

    def __init__(
        self,
        config: Any,
    ) -> None:
        super().__init__()
        self.num_experts = getattr(config, "num_local_experts", getattr(config, "n_routed_experts"))
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            topk_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            current_hidden_states = self.act_fn(gate_up)
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * topk_weights[token_idx, topk_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class DeepseekV3MoE(nn.Module):

    def __init__(
        self,
        config: Any,
    ) -> None:
        super().__init__()
        self.experts = DeepseekV3Experts(config)
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        orig_shape = hidden_states.shape
        topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        return hidden_states + self.shared_experts(residual)


class DeepseekV3Attention(nn.Module):

    def __init__(
        self,
        config: Any,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = _get_qk_head_dim(config)
        self.v_head_dim = getattr(config, "v_head_dim", self.qk_head_dim)
        attention_bias = getattr(config, "attention_bias", False)
        assert self.v_head_dim <= self.qk_head_dim
        self.scaling = _get_attention_scaling(config, self.qk_head_dim)

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = ReplicatedLinear(
                config.hidden_size,
                self.q_lora_rank,
                bias=attention_bias,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            config.hidden_size,
            bias=attention_bias,
        )
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            self.qk_rope_head_dim,
            config.max_position_embeddings,
            _get_rope_theta(config),
            interleaved=getattr(config, "rope_interleave", True),
        )
        self.attn = Attention(
            self.num_heads,
            self.qk_head_dim,
            self.scaling,
            self.num_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_lora, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_lora))
        kv = kv.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rope = k_rope.view(-1, 1, self.qk_rope_head_dim)

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
        k_rope = k_rope.expand(-1, self.num_heads, -1)
        q = torch.cat((q_nope, q_rope), dim=-1)
        k = torch.cat((k_nope, k_rope), dim=-1)
        if self.v_head_dim != self.qk_head_dim:
            v = F.pad(v, [0, self.qk_head_dim - self.v_head_dim])
        o = self.attn(q, k, v)
        if self.v_head_dim != self.qk_head_dim:
            o = o[..., : self.v_head_dim]
        output = self.o_proj(o.flatten(1, -1))
        return output


class DeepseekV3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Any,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = DeepseekV3Attention(config)
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV3Model(nn.Module):

    def __init__(
        self,
        config: Any,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    ignored_weights = ()
    enforce_eager = True

    def __init__(
        self,
        config: Any,
    ) -> None:
        super().__init__()
        self.model = DeepseekV3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.ignored_weights = (f"model.layers.{config.num_hidden_layers}.",)
        self.kvcache_num_heads = config.num_attention_heads
        self.kvcache_head_dim = _get_qk_head_dim(config)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
