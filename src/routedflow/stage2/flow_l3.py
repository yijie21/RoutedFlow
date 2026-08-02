"""L3 complete: ATM TrackTransformer with z-conditioning — the real surgery.

Design (D3 v1, grill session 2026-07-31): **z takes the text token's slot**.
The pretrained transformer expects [track tokens, image patches, 1 language
token]; we keep the token layout identical but the last token is now
`cond_proj(z)` instead of `language_encoder(task_emb)` — task information can
ONLY enter through z (plan §2.7 fix #3). An optional zero-init cross-attention
block (`use_cross_attn`) is kept as the D3 alternative for a later comparison.

Also adds the D8a query-depth channel: each query point's camera-frame depth is
embedded (zero-init MLP) and added to its track tokens.

The whole inner TrackTransformer stays TRAINABLE (warm-started from the ATM
checkpoint); forward mirrors TrackTransformer.forward/reconstruct line by line.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat

# Import ORDER matters (circular-import trap): atm.policy.vilt must be imported
# BEFORE atm.model.track_transformer. Importing atm.model first leaves vilt's
# `from atm.model import *` executed against a half-initialized package, caching
# a vilt module that lacks TrackTransformer (its _setup_track eval then fails).
from atm.policy.vilt import BCViLTPolicy  # noqa: F401  (order guard, see above)

from routedflow.flow_models import CondCrossAttn, QueryDepthEmbed


def load_atm_track_transformer(track_fn):
    """Instantiate the pretrained ATM TrackTransformer (trainable copy)."""
    from omegaconf import OmegaConf

    from atm.model.track_transformer import TrackTransformer
    cfg = OmegaConf.load(f"{track_fn}/config.yaml")
    cfg.model_cfg.load_path = f"{track_fn}/model_best.ckpt"
    return TrackTransformer(**cfg.model_cfg)


class L3RobotFlow(nn.Module):
    def __init__(self, track_fn, cond_dim=384, use_cross_attn=False):
        super().__init__()
        self.inner = load_atm_track_transformer(track_fn)
        d = self.inner.dim
        self.cond_proj = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, d))
        self.depth_embed = QueryDepthEmbed(d)  # zero-init => no-op at start
        self.cross = CondCrossAttn(d, cond_dim) if use_cross_attn else None

    @property
    def num_track_ts(self):
        return self.inner.num_track_ts

    def forward(self, vid, query_uv, z, query_depth, p_img=0.0):
        """vid (B, fs, c, h, w) raw 0-255 · query_uv (B, n, 2) normalized xy ·
        z (B, cond_dim) · query_depth (B, n) meters -> tracks (B, tl, n, 2)."""
        inner = self.inner
        B, n = query_uv.shape[0], query_uv.shape[1]
        tl = inner.num_track_ts

        track = repeat(query_uv, "b n d -> b tl n d", tl=tl)
        track = inner._preprocess_track(track)
        vid = inner._preprocess_vid(vid)

        patches = inner._encode_video(vid, p_img)                       # (b, n_img, d)

        # _encode_track mirrored, with the depth embedding inserted (zero-init MLP
        # => exact pretrained behavior at step 0)
        tr = inner._mask_track_as_first(track)
        tr = inner.track_proj_encoder(tr)                               # (b, t', n, d)
        dep = self.depth_embed.proj(query_depth[..., None])             # (b, n, d)
        tr = tr + dep[:, None]                                          # broadcast over t'
        tr = tr + inner.track_embed
        tr = rearrange(tr, "b t n d -> b (t n) d")

        z_token = self.cond_proj(z)[:, None]                            # (b, 1, d) — text slot
        x = torch.cat([tr, patches, z_token], dim=1)
        if self.cross is not None:
            x = self.cross(x, z)
        x = inner.transformer(x)

        rec_track = x[:, : inner.num_track_patches]
        rec_track = inner.track_decoder(rec_track)                      # (b, (t n), 2*p)
        num_track_h = inner.num_track_ts // inner.track_patch_size
        rec_track = rearrange(rec_track, "b (t n) (p c) -> b (t p) n c",
                              p=inner.track_patch_size, t=num_track_h)
        return rec_track
