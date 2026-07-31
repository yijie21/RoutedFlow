"""L3/L5 flow-module SKELETON (interfaces + injection blocks; training NOT wired).

Scope agreed 2026-07-30 (grill session): L1+L2 are fully implemented; L3/L5 get
interface-level skeletons only, because their training is gated on D3 final
choice, the stage-0 retrain schedule, and flow-GT generation. What IS final here:

    * the module interface (what L3/L5 consume/produce) per plan §2.7/§2.8
    * D8 = option (a): 2D tracks + per-query-point initial depth channel
    * the two candidate conditioning blocks (D3): CondCrossAttn / CondAdaLN
    * the text asymmetry: L3 takes cond (C-latent) and NO text; L5 takes text
      (ATM's native task-emb conditioning) and NO C-latent

Integration plan (when L3 training starts): ATM's TrackTransformer forward is
wrapped, the injection block operating on its token stream after the encoder's
k-th layer (cross-attn) or on its AdaLN params (adaln). The wrapper below holds
the injection modules and validates shapes; the ATM surgery lands with the L3
training PR, mirroring how stage-0's BCViLTPolicyGated overrode track_encode.

Shape contract (B=batch, N=query points, T=horizon):
    query_points (B, N, 2)  pixel xy in ATM's track convention (dim0=x/W)
    query_depth  (B, N, 1)  metric depth at the query pixel (D8a channel)
    cond         (B, Dc)    C-latent from L1 (L3 only)
    tracks out   (B, T, N, 2)
"""
import torch
import torch.nn as nn

COND_MODES = ("cross_attn", "adaln")


class CondCrossAttn(nn.Module):
    """One cross-attention block: token stream attends to the conditioning vector.
    D3 candidate #1. Zero-init output projection => identity at start of training
    (safe to insert into a pretrained ATM without wrecking it)."""

    def __init__(self, d_model, cond_dim, n_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.kv = nn.Linear(cond_dim, 2 * d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, tokens, cond):
        k, v = self.kv(cond)[:, None].chunk(2, dim=-1)
        h, _ = self.attn(self.norm(tokens), k, v)
        return tokens + self.out(h)


class CondAdaLN(nn.Module):
    """Adaptive LayerNorm modulation from the conditioning vector.
    D3 candidate #2. Zero-init modulation => identity at start of training."""

    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.mod.weight), nn.init.zeros_(self.mod.bias)

    def forward(self, tokens, cond):
        scale, shift = self.mod(cond)[:, None].chunk(2, dim=-1)
        return self.norm(tokens) * (1 + scale) + shift


class QueryDepthEmbed(nn.Module):
    """D8 option (a): lift each query point's initial depth into the query
    embedding space and ADD it to ATM's query-point embedding."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.proj[-1].weight), nn.init.zeros_(self.proj[-1].bias)

    def forward(self, query_emb, query_depth):
        return query_emb + self.proj(query_depth)


class ConditionedTrackTransformer(nn.Module):
    """Wrapper interface for ATM's TrackTransformer with routed conditioning.

    role="robot"  (L3): forward(vid, query_points, query_depth, cond) — no text.
    role="object" (L5): forward(vid, query_points, query_depth, text_emb) — no cond.

    `inner` is the (pretrained) ATM track transformer; None is allowed for
    interface/shape work before the surgery lands (forward then raises).
    """

    def __init__(self, inner=None, role="robot", d_model=384, cond_dim=384,
                 cond_mode="cross_attn"):
        super().__init__()
        assert role in ("robot", "object") and cond_mode in COND_MODES
        self.inner, self.role, self.cond_mode = inner, role, cond_mode
        self.depth_embed = QueryDepthEmbed(d_model)
        if role == "robot":
            self.cond_block = (CondCrossAttn(d_model, cond_dim) if cond_mode == "cross_attn"
                               else CondAdaLN(d_model, cond_dim))

    def forward(self, vid, query_points, query_depth, cond=None, text_emb=None):
        if self.role == "robot":
            assert cond is not None and text_emb is None, \
                "L3 consumes the C-latent and MUST NOT see text (plan §2.7 fix #3)"
        else:
            assert text_emb is not None and cond is None, \
                "L5 consumes text and has no C-latent input"
        if self.inner is None:
            raise NotImplementedError(
                "ATM TrackTransformer surgery lands with the L3 training PR; "
                "this skeleton fixes the interface + injection blocks only.")
        raise NotImplementedError("wire injection into inner.forward here")
