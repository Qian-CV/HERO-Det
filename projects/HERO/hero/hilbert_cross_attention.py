import torch
import torch.nn as nn
from mmengine.model import (BaseModule)


# Reuse the previous CrossAttentionBlock
class CrossAttention(nn.Module):
    """
    Cross-attention module for interaction between two sequences.
    Args:
        dim (int): Input channel dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in Q, K, V linear layers
        qk_scale (float): Scaling factor, defaults to head_dim^-0.5 when None
        attn_drop (float): Dropout rate for attention
        proj_drop (float): Dropout rate for output projection
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, C]
        Returns:
            cls_out: Tensor of shape [B, 1, C]
        """
        B, N, C = x.shape
        # Q uses only the 0-th token (CLS)
        q = self.wq(x[:, :1]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # K, V use all tokens
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B,H,1,N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B,H,1,C/H
        x = x.transpose(1, 2).reshape(B, 1, C)  # B,1,C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EnhancedCrossAttention(nn.Module):
    """
    Enhanced cross-attention module with bidirectional information flow and global context modeling.
    Args:
        dim (int): Input channel dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in Q, K, V linear layers
        qk_scale (float): Scaling factor, defaults to head_dim^-0.5 when None
        attn_drop (float): Dropout rate for attention
        proj_drop (float): Dropout rate for output projection
        global_token (bool): Whether to use a global context token
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., global_token=True):
        super().__init__()
        self.num_heads = num_heads
        self.global_token = global_token
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Projections for query, key, value
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Global context token
        if global_token:
            self.global_ctx = nn.Parameter(torch.zeros(1, 1, dim))
            self.global_proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Feature enhancement
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, C]
        Returns:
            cls_out: Tensor of shape [B, 1, C]
        """
        B, N, C = x.shape
        
        # Add global context token
        if self.global_token:
            global_ctx = self.global_ctx.expand(B, -1, -1)
            x_with_global = torch.cat([global_ctx, x], dim=1)
            
            # Project query, key, value
            q = self.wq(x_with_global).reshape(B, N+1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.wk(x_with_global).reshape(B, N+1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.wv(x_with_global).reshape(B, N+1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # Apply attention
            out = (attn @ v).transpose(1, 2).reshape(B, N+1, C)
            
            # Use global token information to update CLS only
            global_info = out[:, 0:1]
            global_info = self.global_proj(global_info)
            
            # Adaptive fusion
            gate = self.gate(x[:, 0:1])
            cls_out = gate * global_info + (1 - gate) * x[:, 0:1]
        else:
            # Use only the CLS token as query
            q = self.wq(x[:, 0:1]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            cls_out = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        
        cls_out = self.proj(cls_out)
        cls_out = self.proj_drop(cls_out)
        return cls_out


class CrossAttentionBlock(nn.Module):
    """Cross-attention block handling interactions between CLS token and other tokens; input and output are [B, C, N]
    Args:
        dim (int): Input channel dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to input dim, default 4
        qkv_bias (bool): Whether to use bias in Q, K, V projections
        qk_scale (float): Scaling factor, defaults to head_dim^-0.5 when None
        drop (float): Dropout rate for output projection
        attn_drop (float): Dropout rate for attention
        drop_path (float): Dropout rate for residual connection
        act_layer (nn.Module): Activation function, default GELU
        norm_layer (nn.Module): Normalization layer, default LayerNorm
        has_mlp (bool): Whether to include an MLP layer
        use_enhanced_attn (bool): Whether to use enhanced cross-attention
    """

    def __init__(self, dim, num_heads, mlp_ratio=5., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False,
                 use_enhanced_attn=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # Choose basic or enhanced cross-attention
        if use_enhanced_attn:
            self.attn = EnhancedCrossAttention(dim, num_heads, qkv_bias,
                                         qk_scale, attn_drop, drop)
        else:
            self.attn = CrossAttention(dim, num_heads, qkv_bias,
                                   qk_scale, attn_drop, drop)
                                   
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                act_layer(),
                nn.Dropout(drop),  # Add dropout after activation
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(drop),
            )

    def forward(self, x):
        """Perform CLS‑token level cross‑layer cross‑attention on sequences after multi‑level Hilbert flattening.
            Args:
                x: Tensor of shape [B, N, C]
            Returns:
                Tensor of shape [B, C, N]  (only the first token is updated)
        """
        
        # 1) Cross-attention after normalization
        cls_updated = self.drop_path(self.attn(self.norm1(x)))  # [B,1,C]

        # Extract cls and remaining tokens
        cls, rest = x[:, :1], x[:, 1:]
        cls = cls + cls_updated  # Residual connection
        
        # 3) Optional MLP residual connection
        if self.has_mlp:
            cls = cls + self.drop_path(self.mlp(self.norm2(cls)))
            
        # 4) Concatenate and convert back to original shape
        out_seq = torch.cat([cls, rest], dim=1)  # [B,N,C]
        return out_seq.transpose(1, 2)  # Back to [B,C,N]


class RelativePositionEncoding(nn.Module):
    """Relative position encoding module that adds positional information to Hilbert-flattened sequences.
    
    Args:
        dim (int): Feature dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout rate
    """
    def __init__(self, dim, max_len=1024, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        # Learnable relative positional encoding
        self.rel_pos_bias = nn.Parameter(torch.zeros(max_len))
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
    
    def forward(self, x):
        """Add relative positional encoding.
        
        Args:
            x: Tensor of shape [B, C, L]
        Returns:
            Features with positional information [B, C, L]
        """
        B, C, L = x.shape
        
        # Handle sequence length dynamically
        if L <= self.max_len:
            # Within preset length, use directly
            pos_emb = self.rel_pos_bias[:L].unsqueeze(0).unsqueeze(1).expand(B, C, -1)
        else:
            # Exceeds preset length, use cyclic mode
            # Method 1: reuse positional encoding
            indices = torch.fmod(torch.arange(L, device=x.device), self.max_len)
            pos_emb = self.rel_pos_bias[indices].unsqueeze(0).unsqueeze(1).expand(B, C, -1)
            
        x = x + pos_emb
        return self.dropout(x)


class HilbertCrossScaleAttention(BaseModule):
    """Multi-scale cross-layer attention module based on Hilbert curve flattening.
    
    Args:
        channels (int): Number of input feature channels
        num_scales (int): Number of multi-scale features, default 4
        num_heads (int): Number of attention heads, default 8
        mlp_ratio (float): Ratio of MLP hidden dim to input dim, default 4.0
        qkv_bias (bool): Whether to use bias in QKV projections, default True
        qk_scale (float): Scaling factor for QK dot product, default None
        drop (float): Dropout rate for linear layers, default 0.0
        attn_drop (float): Dropout rate for attention, default 0.0
        drop_path (float): Dropout rate for residual connections, default 0.0
        init_cfg (dict): Initialization config, default None
        use_pos_embed (bool): Whether to use positional encoding, default True
        bidirectional (bool): Whether to use bidirectional fusion, default True
        use_enhanced_attn (bool): Whether to use enhanced cross-attention, default True
    """

    def __init__(self,
                 channels: int,
                 num_scales: int = 5,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 init_cfg=None,
                 use_pos_embed: bool = True,
                 bidirectional: bool = True,
                 use_enhanced_attn: bool = True):
        super().__init__(init_cfg=init_cfg)
        self.num_scales = num_scales
        self.channels = channels
        self.bidirectional = bidirectional
        self.use_pos_embed = use_pos_embed

        # One learnable CLS token per scale
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, channels))
            for _ in range(num_scales)
        ])
        
        # Initialize CLS tokens
        for cls_token in self.cls_tokens:
            nn.init.trunc_normal_(cls_token, std=0.02)

        # Relative positional encoding
        if use_pos_embed:
            self.pos_embeds = nn.ModuleList([
                RelativePositionEncoding(channels, dropout=drop)
                for _ in range(num_scales)
            ])

        # Feature projection layers (source to target)
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.Dropout(drop),
                nn.GELU()
            )
            for _ in range(num_scales)
        ])
        
        # Feature re-projection layers (target back to source)
        self.reprojs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.Dropout(drop),
                nn.GELU()
            )
            for _ in range(num_scales)
        ])

        # Cross-attention fusion blocks
        self.fusions = nn.ModuleList([
            CrossAttentionBlock(
                dim=channels, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                has_mlp=True, use_enhanced_attn=use_enhanced_attn
            )
            for _ in range(num_scales)
        ])
        
        # If bidirectional fusion is enabled, add reverse fusion blocks (high-to-low)
        if bidirectional:
            self.reverse_fusions = nn.ModuleList([
                CrossAttentionBlock(
                    dim=channels, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop,
                    attn_drop=attn_drop, drop_path=drop_path,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                    has_mlp=True, use_enhanced_attn=use_enhanced_attn
                )
                for _ in range(num_scales)
            ])
        
        # Feature enhancement layers
        self.feature_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.Dropout(drop)
            )
            for _ in range(num_scales)
        ])

    def forward(self, seqs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            seqs: list with length = num_scales,
                  each item has shape [B, C, L_i] (after Hilbert flattening)
        Returns:
            new_seqs: list of the same length, each with shape [B, C, L_i]
        """
        B = seqs[0].shape[0]
        
        # Add positional encoding (if enabled)
        if self.use_pos_embed:
            seqs = [self.pos_embeds[i](x) for i, x in enumerate(seqs)]

        # 1) Build token sequences with CLS [B, L_i+1, C]
        tokens = []
        for i, x in enumerate(seqs):
            # [B, C, L] -> [B, L, C]
            t = x.permute(0, 2, 1)
            # Prepend CLS
            cls = self.cls_tokens[i].expand(B, -1, -1)
            tokens.append(torch.cat([cls, t], dim=1))

        # 2) Forward cross-layer fusion: low-to-high
        fused_tokens = []
        for i in range(self.num_scales):
            j = (i + 1) % self.num_scales
            # Project CLS from level i to level j
            proj_cls = self.projs[i](tokens[i][:, :1, :])  # [B,1,C]
            # Build cross-layer input
            inp = torch.cat([proj_cls, tokens[j][:, 1:, :]], dim=1)  # [B, L_j+1, C]
            # Cross-attention, update CLS
            out = self.fusions[i](inp)  # [B, C, L_j+1]
            # Re-project updated CLS back to level i
            out_cls = out[:, :, 0:1].transpose(1, 2)  # [B,1,C]
            new_cls = self.reprojs[i](out_cls)  # [B,1,C]
            # Update CLS at level i
            fused_tokens.append(torch.cat([new_cls, tokens[i][:, 1:, :]], dim=1))
            
        # 3) If bidirectional, perform reverse fusion: high-to-low
        if self.bidirectional:
            reverse_fused = []
            for i in range(self.num_scales):
                j = (i - 1) % self.num_scales  # Reverse index
                # Project
                proj_cls = self.projs[i](fused_tokens[i][:, :1, :])  # [B,1,C]
                # Build input
                inp = torch.cat([proj_cls, fused_tokens[j][:, 1:, :]], dim=1)
                # Cross-attention
                out = self.reverse_fusions[i](inp)  # [B, C, L_j+1]
                # Re-project
                out_cls = out[:, :, 0:1].transpose(1, 2)  # [B,1,C]
                new_cls = self.reprojs[i](out_cls)  # [B,1,C]
                # Update
                reverse_fused.append(torch.cat([new_cls, fused_tokens[i][:, 1:, :]], dim=1))
            fused_tokens = reverse_fused
            
        # 4) Feature enhancement
        enhanced_tokens = []
        for i, t in enumerate(fused_tokens):
            cls = t[:, :1, :]
            # Enhance CLS features
            enhanced_cls = cls + self.feature_enhancers[i](cls)
            # Concatenate back to original sequence
            enhanced_tokens.append(torch.cat([enhanced_cls, t[:, 1:, :]], dim=1))
            
        # 5) Drop CLS and restore shape [B, C, L_i]
        new_seqs = []
        for i, t in enumerate(enhanced_tokens):
            t = t[:, 1:, :].permute(0, 2, 1)  # -> [B, C, L_i]
            new_seqs.append(t)

        return new_seqs
