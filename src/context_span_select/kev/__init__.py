"""Kev decision-model loaders and attention capture (MLX)."""

from context_span_select.kev.attention import (
    ATTN_LAYERS,
    N_HEADS,
    N_TOTAL,
    branch_attention,
    build_labeled_state,
    chunk_scores_from_attn,
    head_qr_scores,
    install_attention_capture,
    layer_module_ids,
    state_token_chunk_ids,
)
from context_span_select.kev.load import load_kev4b_mlx8

__all__ = [
    "ATTN_LAYERS",
    "N_HEADS",
    "N_TOTAL",
    "branch_attention",
    "build_labeled_state",
    "chunk_scores_from_attn",
    "head_qr_scores",
    "install_attention_capture",
    "layer_module_ids",
    "load_kev4b_mlx8",
    "state_token_chunk_ids",
]
