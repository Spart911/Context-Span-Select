"""Load Kev-4B MLX-8bit + official pointer head.

Requires the `kev` package and mlx-lm (run via the kev uv env):
  cd ~/.cache/kev && uv run --extra serve python ...
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from kev.checkpoint import Checkpoint
from kev.mlx_model import CACHE_LIMIT, MLXDecisionModel
from kev.model import PointerHead, load_tokenizer, pad_id
from mlx_lm.utils import load_model

KEV4B_REV = "jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101"
WEIGHTS_REPO = "RoderickQiu/kev-4b-mlx-8bit"


def embed_dim(text) -> int:
    emb = text.embed_tokens
    if hasattr(emb, "dims"):
        return int(emb.dims)
    w = emb.weight
    if hasattr(w, "shape") and len(w.shape) >= 2:
        return int(w.shape[-1])
    raise RuntimeError("cannot infer embedding width for quantized model")


def load_kev4b_mlx8():
    """-> (checkpoint, tokenizer, MLXDecisionModel)."""
    mx.set_cache_limit(CACHE_LIMIT)
    weights = Path(
        snapshot_download(
            WEIGHTS_REPO,
            allow_patterns=["config.json", "model.safetensors"],
        )
    )
    ck = Checkpoint(KEV4B_REV)
    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
    lm = load_model(weights)[0]
    lm.eval()
    model = MLXDecisionModel(lm, pad_id(tok), head_dim=ck.meta.head_dim)
    d = embed_dim(model.text)
    model.head = PointerHead(d, dp=ck.meta.head_dim).eval()
    model.head.load_state_dict(ck.meta.head)
    model.head.temperature = ck.meta.temperature
    ck.requested = f"{WEIGHTS_REPO}+{KEV4B_REV}"
    return ck, tok, model
