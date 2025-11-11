import torch
import torch.nn as nn
import torch.nn.functional as F


class RotationAttentionFusion(nn.Module):
    def __init__(self, channels, seq_len, num_rotations=4):
        super().__init__()
        self.num_rotations = num_rotations
        
        # Dimensionality reduction projection to reduce computation
        self.query = nn.Conv1d(channels*num_rotations, channels, kernel_size=1)
        self.key = nn.Conv1d(channels*num_rotations, channels, kernel_size=1) 
        self.value = nn.Conv1d(channels*num_rotations, channels*num_rotations, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv1d(channels*num_rotations, channels, kernel_size=1)
        
        # Positional encodings to help model understand rotation order - fix channel count!
        self.pos_emb = nn.Parameter(torch.randn(1, channels*num_rotations, seq_len))
        
        # Scaling factor
        self.scale = channels ** -0.5
        
    def forward(self, x_rot):
        """
        Input: x_rot [B, C, R, L] - features from R rotation directions
        Output: fused features [B, C, L]
        """
        B, C, R, L = x_rot.shape
        # Ensure rotation count matches
        assert R == self.num_rotations, f"Input rotation count {R} does not match initialized {self.num_rotations}"
        
        # Reshape for processing
        x = x_rot.reshape(B, C*R, L)
        
        # Add positional encoding - ensure channel dimension matches
        v = self.value(x) + self.pos_emb
        
        # Compute attention
        q = self.query(x) * self.scale  # [B, C, L]
        k = self.key(x)                # [B, C, L]
        
        # Attention scores
        attn = torch.einsum('bcl,bcm->blm', q, k)  # [B, L, L]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.einsum('blm,bcm->bcl', attn, v)  # [B, C*R, L]
        
        # Project back to original channel count
        out = self.out_proj(out)  # [B, C, L]
        
        return out


class RotationChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8, num_rotations=4):
        super().__init__()
        self.num_rotations = num_rotations
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Angle channel attention - adjust input size to match rotation count
        self.angle_fc = nn.Sequential(
            nn.Linear(num_rotations, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_rotations),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        # Final fusion layer - adjust channels to fit rotation count
        self.fusion = nn.Conv1d(channels*num_rotations, channels, kernel_size=3, padding=1)
        
    def forward(self, x_rot):
        """
        Input: x_rot [B, C, R, L] - features from R rotation directions
        Output: fused features [B, C, L]
        """
        B, C, R, L = x_rot.shape
        # Ensure rotation count matches
        assert R == self.num_rotations, f"Input rotation count {R} does not match initialized {self.num_rotations}"
        
        # 1. Compute importance for each rotation
        # First compute global statistics for each rotation
        angle_weights = []
        for i in range(R):
            feat = x_rot[:, :, i, :]  # [B, C, L]
            pool = self.avg_pool(feat).squeeze(-1)  # [B, C]
            angle_weights.append(pool.mean(dim=1, keepdim=True))  # [B, 1]
            
        angle_weights = torch.cat(angle_weights, dim=1)  # [B, R]
        angle_weights = self.angle_fc(angle_weights).unsqueeze(-1)  # [B, R, 1]
        
        # 2. Assign weights to each rotation
        weighted_features = []
        for i in range(R):
            # Channel attention
            feat = x_rot[:, :, i, :]  # [B, C, L]
            pool = self.avg_pool(feat).squeeze(-1)  # [B, C]
            channel_weight = self.channel_fc(pool).unsqueeze(-1)  # [B, C, 1]
            
            # Apply channel weights and rotation weights
            angle_w = angle_weights[:, i, :].unsqueeze(1)  # [B, 1, 1]
            weighted = feat * channel_weight * angle_w
            weighted_features.append(weighted)
        
        # 3. Concatenate and fuse
        concat = torch.cat(weighted_features, dim=1)  # [B, C*R, L]
        output = self.fusion(concat)  # [B, C, L]
        
        return output


class AngularCircularFusion(nn.Module):
    def __init__(self, channels, seq_len=64, num_rotations=4):
        super().__init__()
        self.num_rotations = num_rotations
        
        # Angle circular convolution
        self.angular_conv = nn.Conv2d(
            channels, channels, kernel_size=(3, 3),
            padding=(1, 1), groups=channels
        )
        
        # Second angular conv to enhance interaction across rotations
        self.angular_conv2 = nn.Conv2d(
            channels, channels, kernel_size=(3, 1),
            padding=(1, 0), groups=channels
        )
        
        # Channel interaction - update input channels to fit rotation count
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(channels*num_rotations, channels*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels*2, channels, kernel_size=3, padding=1)
        )
        
        # Weighting across rotations - updated for dynamic rotation count
        self.angle_weights = nn.Parameter(torch.ones(1, 1, num_rotations, 1) / num_rotations)
        
    def forward(self, x_rot):
        """
        Input: x_rot [B, C, R, L] - features from R rotation directions
        Output: fused features [B, C, L]
        """
        B, C, R, L = x_rot.shape
        # Ensure rotation count matches
        assert R == self.num_rotations, f"Input rotation count {R} does not match initialized {self.num_rotations}"
        
        # 1. Reshape 1D sequence to 2D for angular circular convolution
        # Assume L = H*W is a perfect square
        H = W = int(L**0.5)
        x_reshaped = x_rot.view(B, C, R, H, W)  # [B, C, R, H, W]
        
        # 2. Permute dims, treat rotation as a spatial dimension
        x_angular = x_reshaped.permute(0, 1, 3, 2, 4).contiguous()  # [B, C, H, R, W]
        x_angular = x_angular.view(B, C, H, R*W)  # [B, C, H, R*W]
        
        # 3. Apply angular circular convolutions
        x_conv = self.angular_conv(x_angular)  # [B, C, H, R*W]
        x_conv = self.angular_conv2(x_conv)
        
        # 4. Reshape back to original form
        x_conv = x_conv.view(B, C, H, R, W)  # [B, C, H, R, W]
        x_conv = x_conv.permute(0, 1, 3, 2, 4).contiguous()  # [B, C, R, H, W]
        x_conv = x_conv.view(B, C, R, L)  # [B, C, R, L]
        
        # 5. Learn weights for each rotation and apply
        weighted = x_conv * (self.angle_weights + 1.0)  # Add 1.0 as a residual connection
        
        # 6. Concatenate and fuse channels
        concat = weighted.reshape(B, C*R, L)  # [B, C*R, L]
        output = self.channel_mixer(concat)  # [B, C, L]
        
        return output


class DynamicRotationFusion(nn.Module):
    def __init__(self, channels, seq_len, num_rotations=4):
        super().__init__()
        # Reduce dimensionality to lower computational complexity
        reduced_dim = channels // 2
        self.num_rotations = num_rotations

        # Feature extractors per rotation
        self.angle_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, reduced_dim, kernel_size=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_rotations)
        ])

        # Cross-rotation attention
        self.cross_angle_attn = nn.MultiheadAttention(
            embed_dim=reduced_dim,
            num_heads=4,
            batch_first=True
        )

        # Rotation importance network
        self.angle_importance = nn.Sequential(
            nn.Linear(reduced_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Feature fusion network
        self.fusion = nn.Sequential(
            nn.Conv1d(reduced_dim * num_rotations + channels * num_rotations, channels * 2, kernel_size=1),  # Include residual features
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 2, channels, kernel_size=3, padding=1)
        )

        # Residual connection parameter
        self.res_param = nn.Parameter(torch.ones(1))

    def forward(self, x_rot):
        """
        Input: x_rot [B, C, 4, L] - features from 4 rotation directions
        Output: fused features [B, C, L]
        """
        B, C, R, L = x_rot.shape

        # 1. Extract features for each rotation
        angle_feats = []
        for i in range(R):
            feat = self.angle_extractors[i](x_rot[:, :, i, :])  # [B, C/2, L]
            angle_feats.append(feat)

        # 2. Rearrange sequence features to shapes required by attention
        query = torch.stack(angle_feats, dim=1)  # [B, 4, C/2, L]
        query = query.permute(0, 3, 1, 2).contiguous()  # [B, L, 4, C/2]
        query = query.view(B * L, R, -1)  # [B*L, 4, C/2]

        # 3. Apply cross-rotation attention
        attn_out, _ = self.cross_angle_attn(query, query, query)  # [B*L, 4, C/2]

        # 4. Compute importance weights for each rotation
        avg_feat = attn_out.mean(dim=1)  # [B*L, C/2]
        angle_weights = self.angle_importance(avg_feat)  # [B*L, 1]
        angle_weights = angle_weights.view(B, L, 1, 1)  # [B, L, 1, 1]

        # 5. Reshape attention output
        attn_out = attn_out.view(B, L, R, -1)  # [B, L, 4, C/2]
        attn_out = attn_out.permute(0, 3, 2, 1)  # [B, C/2, 4, L]

        # 6. Combine with original features via residual connection
        res_feats = []
        for i in range(R):
            orig = x_rot[:, :, i, :]  # [B, C, L]
            attn = attn_out[:, :, i, :]  # [B, C/2, L]

            # Expand reduced features back to original channels (simple duplication)
            attn_expanded = torch.cat([attn, attn], dim=1)  # [B, C, L]

            # Residual connection
            res = orig + self.res_param * attn_expanded
            res_feats.append(res)

        # 7. Weighted fusion
        weighted_feats = []
        for i in range(R):
            # Use attention to assign position-dependent importance across rotations
            weighted = angle_feats[i] * angle_weights[:, :, :, 0].permute(0, 2, 1)
            weighted_feats.append(weighted)

        # 8. Concatenate and fuse; include residual features
        concat_angle_feats = torch.cat(weighted_feats, dim=1)  # [B, C/2*4, L]
        concat_res_feats = torch.cat(res_feats, dim=1)  # [B, C*4, L]

        # Concatenate both feature types
        all_feats = torch.cat([concat_angle_feats, concat_res_feats], dim=1)
        output = self.fusion(all_feats)  # [B, C, L]

        return output


# Helper function for testing
def test_fusion_modules():
    # Create test inputs
    B, C = 2, 256
    L = 64
    
    # Test different numbers of rotation angles
    for num_rotations in [2, 3, 4]:  # test 2 (0,90), 3 (0,90,180), 4 (0,90,180,270) rotations
        print(f"\nTesting with {num_rotations} rotation angles:")
        x = torch.randn(B, C, num_rotations, L)
        
        # Test RotationAttentionFusion
        fusion1 = RotationAttentionFusion(C, L, num_rotations)
        out1 = fusion1(x)
        print(f"RotationAttentionFusion output shape: {out1.shape}")
        
        # Test RotationChannelAttention
        fusion2 = RotationChannelAttention(C, 8, num_rotations)
        out2 = fusion2(x)
        print(f"RotationChannelAttention output shape: {out2.shape}")
        
        # Test AngularCircularFusion
        fusion3 = AngularCircularFusion(C, L, num_rotations)
        out3 = fusion3(x)
        print(f"AngularCircularFusion output shape: {out3.shape}")
        
        # All fusion modules should output [B, C, L]
        assert out1.shape == (B, C, L)
        assert out2.shape == (B, C, L)
        assert out3.shape == (B, C, L)
    print("All tests passed!")


if __name__ == "__main__":
    test_fusion_modules()


