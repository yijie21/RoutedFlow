"""L1 front end (option a / D1=C') + L2 C-head.

L1FrontEnd:
    inputs  dino (B,1369,768) frozen features, prior (B,37,37) AFUN coverage,
            text (B,768) BERT task emb
    fusion  patch tokens = Linear([dino || prior_channel]) + pos emb;
            sequence [patches, text_token, C_token] through a 3-layer
            pre-norm transformer encoder (full self-attention)
    heads   heatmap logits (B,128,128) via conv upsampler over the 37x37 patch
            token map; z = C-token output (B,d)
    prior channel dropout (train-time, per-sample) makes the model functional
    without AFUN and yields the with/without-prior ablation for free (plan §2.2).

CHead: MLP([z, feat(p_hat)]) -> yaw logits (36) + pitch logits (12) + w.
feat(p_hat) is gathered bilinearly from the decoder feature map at the GT
contact point during training (teacher-forced; predicted peak at inference).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from routedflow.stage1.dataset import GRID, HEATMAP_RES, N_PITCH, N_YAW


class L1FrontEnd(nn.Module):
    def __init__(self, d=384, n_layers=3, n_heads=6, dino_dim=768, text_dim=768,
                 feat_ch=96, prior_dropout=0.3):
        super().__init__()
        self.d, self.prior_dropout = d, prior_dropout
        self.patch_proj = nn.Linear(dino_dim + 1, d)
        self.text_proj = nn.Linear(text_dim, d)
        self.c_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, GRID * GRID, d) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(1, 2, d) * 0.02)  # [text, C]
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d,
                                           activation="gelu", batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.dec1 = nn.Sequential(nn.Conv2d(d, 192, 3, padding=1), nn.GroupNorm(8, 192), nn.GELU())
        self.dec2 = nn.Sequential(nn.Conv2d(192, feat_ch, 3, padding=1), nn.GroupNorm(8, feat_ch), nn.GELU())
        self.out = nn.Conv2d(feat_ch, 1, 1)
        self.feat_ch = feat_ch

    def forward(self, dino, prior, text):
        """dino (B,1369,768) · prior (B,37,37) · text (B,768) ->
        (heatmap_logits (B,128,128), z (B,d), feat_map (B,feat_ch,128,128))"""
        B = dino.shape[0]
        if self.training and self.prior_dropout > 0:
            keep = (torch.rand(B, 1, 1, device=prior.device) > self.prior_dropout).float()
            prior = prior * keep
        x = torch.cat([dino, prior.reshape(B, -1, 1)], dim=-1)
        tokens = self.patch_proj(x) + self.pos
        text_t = self.text_proj(text)[:, None] + self.type_emb[:, :1]
        c_t = self.c_token.expand(B, 1, -1) + self.type_emb[:, 1:]
        h = self.encoder(torch.cat([tokens, text_t, c_t], dim=1))
        patch_h, z = h[:, : GRID * GRID], h[:, -1]
        fm = patch_h.transpose(1, 2).reshape(B, self.d, GRID, GRID)
        fm = self.dec1(fm)
        fm = F.interpolate(fm, scale_factor=2, mode="bilinear", align_corners=False)
        fm = self.dec2(fm)
        fm = F.interpolate(fm, size=HEATMAP_RES, mode="bilinear", align_corners=False)
        return self.out(fm)[:, 0], z, fm


class CHead(nn.Module):
    def __init__(self, d=384, feat_ch=96, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d + feat_ch, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.yaw = nn.Linear(hidden, N_YAW)
        self.pitch = nn.Linear(hidden, N_PITCH)
        self.w = nn.Linear(hidden, 1)

    @staticmethod
    def gather_feat(feat_map, rowcol512):
        """Bilinear sample feat_map (B,C,128,128) at contact points given @512 px."""
        # grid_sample wants xy in [-1,1]
        xy = torch.stack([rowcol512[:, 1], rowcol512[:, 0]], dim=-1) / 511.0 * 2 - 1
        g = xy.view(-1, 1, 1, 2)
        return F.grid_sample(feat_map, g, mode="bilinear", align_corners=True)[:, :, 0, 0]

    def forward(self, z, feat_map, rowcol512):
        f = self.gather_feat(feat_map, rowcol512)
        h = self.mlp(torch.cat([z, f], dim=-1))
        return self.yaw(h), self.pitch(h), self.w(h)[:, 0]


def stage1_loss(hm_logits, z, yaw_logits, pitch_logits, w_pred, batch,
                lam=(1.0, 0.5, 0.5)):
    """L_C = lam0*KL(heatmap) + lam1*(CE_yaw+CE_pitch) + lam2*L1(w). Returns (loss, parts)."""
    B = hm_logits.shape[0]
    logp = F.log_softmax(hm_logits.reshape(B, -1), dim=-1)
    l_hm = F.kl_div(logp, batch["heatmap"].reshape(B, -1), reduction="batchmean")
    l_ori = F.cross_entropy(yaw_logits, batch["yaw_bin"]) + \
        F.cross_entropy(pitch_logits, batch["pitch_bin"])
    l_w = F.l1_loss(w_pred, batch["w"])
    loss = lam[0] * l_hm + lam[1] * l_ori + lam[2] * l_w
    return loss, {"hm": l_hm.item(), "ori": l_ori.item(), "w": l_w.item()}
