"""Attention capture inside Kev (Qwen3.5 full-attention layers) for evidence ranking.

Uses query rows (instruction + <decide>) → state tokens. Heads are scored with QR-style
gold mass / passage mass, then top-K heads re-rank pack chunks.
"""
from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import numpy as np
import torch.nn.functional as F
from kev.model import OPT_DECIDE, OPT_NONE, encode, rows_of

ATTN_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
N_HEADS = 16
N_TOTAL = len(ATTN_LAYERS) * N_HEADS
GATE_INSTR = (
    "Does the selected passages contain enough factual evidence to answer the question correctly? "
    "Answer yes only if the answer can be read from the passages, not guessed."
)
GATE_OPTS = ["no", "yes"]

_CAPTURING = False
_CAPTURE: dict[int, mx.array] = {}
_ORIG_CALL = None


def install_attention_capture() -> None:
    """Monkey-patch Qwen3NextAttention to also materialize softmax weights."""
    global _ORIG_CALL
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.qwen3_next import Qwen3NextAttention

    if _ORIG_CALL is not None:
        return
    _ORIG_CALL = Qwen3NextAttention.__call__

    def wrapped(self, x, mask=None, cache=None):
        B, L, D = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
        keys, values = self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if _CAPTURING:
            rep = self.num_attention_heads // self.num_key_value_heads
            k = mx.repeat(keys, rep, axis=1)
            scores = (queries.astype(mx.float32) * self.scale) @ mx.swapaxes(
                k.astype(mx.float32), -1, -2
            )
            attn = mx.softmax(scores, axis=-1)
            _CAPTURE[id(self)] = attn
            mx.eval(attn)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    Qwen3NextAttention.__call__ = wrapped


def layer_module_ids(text_model) -> dict[int, int]:
    out = {}
    for li in ATTN_LAYERS:
        out[id(text_model.layers[li].self_attn)] = ATTN_LAYERS.index(li)
    return out


def build_labeled_state(row: dict[str, Any]) -> tuple[str, list[int], list[tuple[int, int, int]]]:
    header = f"Question: {row['question']}\n\n"
    parts = [header]
    ranges: list[tuple[int, int, int]] = []
    pos = len(header)
    ordered = list(row["pack"])
    for n, i in enumerate(ordered):
        label = f"[P{n + 1:02d}] "
        body = row["chunks"][str(i)]
        line = label + body + "\n"
        body_start = pos + len(label)
        body_end = body_start + len(body)
        ranges.append((i, body_start, body_end))
        parts.append(line)
        pos += len(line)
    state = "".join(parts)
    if len(state) > 20000:
        state = state[:20000]
        ranges = [(i, a, min(b, 20000)) for i, a, b in ranges if a < 20000]
    return state, ordered, ranges


def token_char_spans(tok, text: str) -> list[tuple[int, int]]:
    try:
        out = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = out["offset_mapping"]
        if offs and offs[0] is not None:
            return [(int(a), int(b)) for a, b in offs]
    except (TypeError, KeyError, ValueError):
        pass
    ids = tok(text, add_special_tokens=False).input_ids
    spans = []
    cursor = 0
    for tid in ids:
        piece = tok.decode([tid], skip_special_tokens=False)
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = cursor
            end = min(len(text), cursor + max(1, len(piece)))
        else:
            end = idx + len(piece)
        spans.append((idx, end))
        cursor = end
    return spans


def state_token_chunk_ids(tok, state: str, ranges) -> np.ndarray:
    spans = token_char_spans(tok, state)
    out = np.full(len(spans), -1, dtype=np.int32)
    for ti, (a, b) in enumerate(spans):
        if a >= b:
            continue
        mid = (a + b) // 2
        for cid, ca, cb in ranges:
            if ca <= mid < cb:
                out[ti] = cid
                break
    return out


def branch_attention(model, tok, state: str, layer_ids: dict[int, int], instr: str = GATE_INSTR):
    """-> attn [n_layers, n_heads, n_query_rows, n_state], p_true, wall_ms, Ls"""
    global _CAPTURING, _CAPTURE
    rec = {
        "state": state,
        "questions": [{"instr": instr, "options": GATE_OPTS, "label": 1}],
    }
    enc = encode(tok, rec, max_state=65536, max_branch=65536 + 8192, strict=False)
    Ls = enc["seg"].count(0)
    _, _, rows = rows_of(enc)
    row = rows[0]
    opt_branch = enc["opt"][Ls:]
    qrows = [i for i, o in enumerate(opt_branch) if o in (OPT_NONE, OPT_DECIDE)]
    if not qrows:
        qrows = [row["decide"]]

    t0 = time.perf_counter()
    Ls_p, cache = model.prefix(enc)
    assert Ls_p == Ls

    _CAPTURE.clear()
    _CAPTURING = True
    try:
        batch = [type(c).merge([c]) for c in cache]
        h = model._hidden([row["ids"]], batch)
        mx.eval(h)
        logits = model._logits(h[0], row["decide"], row["opts"])
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        p_true = float(probs[1])
    finally:
        _CAPTURING = False
    wall = 1000 * (time.perf_counter() - t0)

    attn = np.zeros((len(ATTN_LAYERS), N_HEADS, len(qrows), Ls), dtype=np.float32)
    for mid, arr in _CAPTURE.items():
        if mid not in layer_ids:
            continue
        li = layer_ids[mid]
        a = np.asarray(arr.astype(mx.float32))
        Lq = a.shape[2]
        valid = [r for r in qrows if r < Lq]
        if not valid:
            continue
        piece = a[0, :, :, :Ls][:, valid, :]
        mass = piece.sum(axis=-1, keepdims=True)
        piece = piece / np.clip(mass, 1e-12, None)
        attn[li, :, : len(valid), :] = piece
    _CAPTURE.clear()
    mx.clear_cache()
    return attn, p_true, wall, Ls


def head_qr_scores(attn: np.ndarray, tok_chunk: np.ndarray, gold: set[int]) -> np.ndarray:
    scores = np.zeros(N_TOTAL, dtype=np.float32)
    if not gold or tok_chunk.size == 0:
        return scores
    Ls = attn.shape[-1]
    gold_mask = np.zeros(Ls, dtype=np.float32)
    pass_mask = np.zeros(Ls, dtype=np.float32)
    for si in range(1, Ls):
        ci = si - 1
        if ci >= len(tok_chunk):
            break
        cid = int(tok_chunk[ci])
        if cid < 0:
            continue
        pass_mask[si] = 1.0
        if cid in gold:
            gold_mask[si] = 1.0
    if gold_mask.sum() < 1 or pass_mask.sum() < 1:
        return scores
    g = (attn * gold_mask[None, None, None, :]).sum(-1).mean(-1)
    p = (attn * pass_mask[None, None, None, :]).sum(-1).mean(-1)
    return (g / np.clip(p, 1e-12, None)).reshape(-1)


def chunk_scores_from_attn(
    attn: np.ndarray, tok_chunk: np.ndarray, head_idx: np.ndarray
) -> dict[int, float]:
    Ls = attn.shape[-1]
    flat = attn.reshape(N_TOTAL, attn.shape[2], Ls)
    sel = flat[head_idx].mean(0).mean(0)
    acc: dict[int, float] = {}
    cnt: dict[int, int] = {}
    for si in range(1, Ls):
        ci = si - 1
        if ci >= len(tok_chunk):
            break
        cid = int(tok_chunk[ci])
        if cid < 0:
            continue
        acc[cid] = acc.get(cid, 0.0) + float(sel[si])
        cnt[cid] = cnt.get(cid, 0) + 1
    return {c: acc[c] / max(1, cnt[c]) for c in acc}


def flat_to_layer_head(flat: int) -> tuple[int, int]:
    li, hi = divmod(int(flat), N_HEADS)
    return ATTN_LAYERS[li], hi
