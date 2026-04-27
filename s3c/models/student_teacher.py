import torch
import torch.nn as nn

from copy import deepcopy


class ViTBlockWithYFull(nn.Module):
    def __init__(self, blk):
        super().__init__()

        # --- Image branch (identique au ViT original)
        self.norm1_x = blk.norm1
        self.norm2_x = blk.norm2
        self.attn_x = blk.attn
        self.mlp_x = blk.mlp

        # --- Y branch (copie explicite)
        self.norm1_y = deepcopy(blk.norm1)
        self.norm2_y = deepcopy(blk.norm2)
        self.mlp_y   = deepcopy(blk.mlp)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=blk.attn.num_heads * blk.attn.head_dim, #blk.attn.embed_dim,
            num_heads=blk.attn.num_heads,
            batch_first=True
        )

    def forward(self, x, y):
        """
        x : [B, N, D]  image tokens
        y : [B, 1, D]  y token
        """

        # ======================================================
        # Image stream (strict ViT)
        # ======================================================

        x = x + self.attn_x(self.norm1_x(x))

        x = x + self.mlp_x(self.norm2_x(x))

        # ======================================================
        # Y stream (cross-attention + MLP)
        # ======================================================
        y_res = y
        y = self.norm1_y(y)

        y_attn, _ = self.cross_attn(
            query=y,
            key=x, # !!!.detach(),
            value=x # !!!.detach()
        )
        y = y_res + y_attn

        y = y + self.mlp_y(self.norm2_y(y))

        return x, y


class StudentWithYPredictor(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.norm = vit.norm

        """num_y_blocks = 6
        blocks = vit.blocks
        self.blocks = nn.ModuleList([
            ViTBlockWithYShared(blk, enable_y=(i >= len(blocks) - num_y_blocks))
            for i, blk in enumerate(blocks)
        ])"""

        self.blocks = nn.ModuleList([
            ViTBlockWithYFull(blk) for blk in vit.blocks
        ])

        self.y_proj = nn.Linear(2, vit.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.y_proj.weight)  # Initialisation soignée

        # ✅ norme spécifique pour le token Y
        self.norm_y = nn.LayerNorm(vit.embed_dim)

    def forward(self, x, y, layernorm=True):
        B = x.shape[0]

        # ---- tokens image ----
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        # ---- token Y ----
        y = self.y_proj(y).unsqueeze(1)  # [B,1,D]

        for blk in self.blocks:
            x, y = blk(x, y)

        if layernorm:
            x = self.norm(x)
            y = self.norm_y(y)

        return x[:, 0], y.squeeze(1)


@torch.no_grad()
def update_ema_student_teacher(student, teacher, momentum=0.999):

    # --- patch embedding ---
    for p_s, p_t in zip(student.patch_embed.parameters(),
                        teacher.patch_embed.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

    # --- cls token ---
    teacher.cls_token.data.mul_(momentum).add_(
        student.cls_token.data, alpha=1.0 - momentum
    )

    # --- pos embed ---
    teacher.pos_embed.data.mul_(momentum).add_(
        student.pos_embed.data, alpha=1.0 - momentum
    )

    # --- transformer blocks ---
    for blk_s, blk_t in zip(student.blocks, teacher.blocks):

        # attention
        for p_s, p_t in zip(blk_s.attn_x.parameters(),
                            blk_t.attn.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

        # norm1
        for p_s, p_t in zip(blk_s.norm1_x.parameters(),
                            blk_t.norm1.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

        # norm2
        for p_s, p_t in zip(blk_s.norm2_x.parameters(),
                            blk_t.norm2.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

        # mlp
        for p_s, p_t in zip(blk_s.mlp_x.parameters(),
                            blk_t.mlp.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

    # --- final norm ---
    for p_s, p_t in zip(student.norm.parameters(),
                        teacher.norm.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)