import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionMask(nn.Module):
    def __init__(self, emb_dim, context_size=3,channels= 16):
        super(ConditionMask, self).__init__()
        device = torch.device("cuda:0")
        self.context_size = context_size
        self.channels = channels
        self.emb_dim = emb_dim
        self.stat_proj = nn.Linear(3, emb_dim).to(device)        # Local statistics projection
        self.time_proj = nn.Linear(128, emb_dim).to(device)      # Time embedding projection
        self.feature_proj = nn.Linear(16, emb_dim).to(device)    # Feature embedding projection
        self.final_proj = nn.Linear(emb_dim * 3, 1).to(device)    # Final projection

    def forward(self, observed_data, x_mask, cond_mask, target_mask, time_embed, feature_embed):
        B, K, L = x_mask.shape  # (B, K, L)

        device = torch.device("cuda:0")

        # 移动模型到设备


        # 确保所有输入数据和中间数据也在同一个设备上
        observed_data = observed_data.to(device)
        x_mask = x_mask.to(device)
        cond_mask = cond_mask.to(device)
        target_mask = target_mask.to(device)
        time_embed = time_embed.to(device)
        feature_embed = feature_embed.to(device)


        # Create valid mask
        valid_mask = cond_mask.float()

        # 1. Compute local statistics
        mean_local = (observed_data * target_mask).sum(dim=1, keepdim=True) / \
                     (target_mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, 1, L)
        std_local = torch.sqrt(((observed_data * target_mask - mean_local) ** 2).sum(dim=1, keepdim=True) / \
                               (target_mask.sum(dim=1, keepdim=True) + 1e-8))  # (B, 1, L)

        # Compute residual
        window_size = 5
        padding = window_size // 2
        observed_padded = F.pad(observed_data, (padding, padding), mode='replicate')
        rolling_mean = observed_padded.unfold(2, window_size, 1).mean(dim=-1)  # (B, K, L)
        residual = observed_data - rolling_mean  # (B, K, L)
        residual_mean = residual.mean(dim=1, keepdim=True)  # (B, 1, L)

        # 2. Project local statistics
        local_stats = torch.cat([mean_local, std_local, residual_mean], dim=1)  # (B, 3, L)
        local_stats = local_stats.to(device)
        local_stats_proj = self.stat_proj(local_stats.permute(0, 2, 1))  # (B, L, emb_dim)

        # 3. Project time embeddings
        xt = self.time_proj(time_embed)  # (B, L, emb_dim)

        # 4. Project feature embeddings
        xf = self.feature_proj(feature_embed)  # (B, K, emb_dim)

        # 5. Adjust dimensions for concatenation
        xt_repeated = xt.unsqueeze(1).repeat(1, K, 1, 1)              # (B, K, L, emb_dim)
        xf_repeated = xf.unsqueeze(2).repeat(1, 1, L, 1)              # (B, K, L, emb_dim)
        local_stats_proj = local_stats_proj.unsqueeze(1).repeat(1, K, 1, 1)  # (B, K, L, emb_dim)

        # 6. Concatenate embeddings
        combined_info = torch.cat([xt_repeated, xf_repeated, local_stats_proj], dim=-1)  # (B, K, L, emb_dim * 3)

        # 7. Final projection
        mask_embed = self.final_proj(combined_info).squeeze(-1)  # (B, K, L)

        # 8. Expand dimensions to match expected output
        mask_embed = mask_embed.unsqueeze(1).repeat(1, 2*self.channels, 1, 1)  # (B, B, K, L)

        return mask_embed


if __name__ == '__main__':


    # 使用示例
    condition_mask_model = ConditionMask(128)
    observed_data = torch.randn(32, 55, 100)  # 原始数据
    x_mask = torch.randn(32, 55, 100)  # 被 mask 的数据
    cond_mask = torch.ones(32, 55, 100)  # 未被 mask 部分的位置
    target_mask = torch.ones(32, 55, 100)  # 被 mask 部分的位置
    time_embed = torch.randn(32, 100, 128)
    feature_embed = torch.randn(32, 55, 16)  # MTS 特征维度为 25, 长度为 100

    mask_embed = condition_mask_model(observed_data, x_mask, cond_mask, target_mask, time_embed, feature_embed)
    print(mask_embed.shape)  # 输出应为 (32, 32, 25, 100)
