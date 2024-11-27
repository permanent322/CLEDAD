import torch
import torch.nn as nn


class ConditionMask(nn.Module):
    def __init__(self, time_emb_dim,feature_emb_dim,num_feature, context_size=3,channels=16):
        super(ConditionMask, self).__init__()
        self.context_size = context_size
        self.channels = channels
        self.stat_proj = nn.Linear(3, time_emb_dim)  # 局部统计信息投影到 embedding 空间
        self.time_proj = nn.Linear(128, time_emb_dim)  # 时间嵌入投影到 embedding 空间
        self.feature_proj = nn.Linear(num_feature, feature_emb_dim)  # 特征维度从 25 嵌入到 16
        self.final_proj = nn.Linear(272, 1)  # 最终融合的投影层，输入为 272，输出为 1

    def forward(self, observed_data, x_mask, cond_mask, target_mask, time_embed, feature_embed):
        B, K, L = x_mask.shape  # x_mask 的形状是 (B, K, L)

        # 构建掩码，标记未被 mask 的部分为 cond_mask（1 表示未被 mask，0 表示被 mask）
        valid_mask = cond_mask.float()

        # 1. 计算局部统计信息（沿特征维度 K 计算）
        mean_local = (observed_data * target_mask).sum(dim=1, keepdim=True) / (
                    target_mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, 1, L)
        std_local = torch.sqrt(((observed_data * target_mask - mean_local) ** 2).sum(dim=1, keepdim=True) / (
                    target_mask.sum(dim=1, keepdim=True) + 1e-8))  # (B, 1, L)

        # 计算局部残差（使用滑动窗口）
        window_size = 5  # 滑动窗口大小
        rolling_mean = observed_data.unfold(2, window_size, 1).mean(dim=-1)  # (B, K, L - window_size + 1)
        residual = observed_data[:, :, (window_size // 2):-(window_size // 2)] - rolling_mean  # 计算残差
        residual = torch.cat([torch.zeros(B, K, window_size // 2).to(observed_data.device), residual,
                              torch.zeros(B, K, window_size // 2).to(observed_data.device)], dim=-1)  # 对齐原始序列 (B, K, L)

        # 计算残差的均值
        residual_mean = residual.mean(dim=1, keepdim=True)  # (B, 1, L)

        # 2. 提取上下文信息（仅考虑未被 mask 的部分）
        context_info = []
        for i in range(L):
            start_idx = max(0, i - self.context_size)
            end_idx = min(L, i + self.context_size + 1)
            context_patch = observed_data[:, :, start_idx:end_idx] * valid_mask[:, :, start_idx:end_idx]

            # 补齐上下文信息到固定大小，前后补0以使所有patch的大小一致
            if context_patch.shape[2] < 2 * self.context_size + 1:
                padding_size = 2 * self.context_size + 1 - context_patch.shape[2]
                padding = torch.zeros(B, K, padding_size).to(observed_data.device)
                context_patch = torch.cat([context_patch, padding], dim=2)

            context_info.append(context_patch.mean(dim=1))  # 沿特征维度 K 计算均值

        context_info = torch.stack(context_info, dim=1)  # (B, L, 1)

        # 3. 投影时间嵌入和特征嵌入
        xt = self.time_proj(time_embed)  # 投影时间嵌入 (B, L, emb_dim)

        # 调整 feature_embed 形状，使其变为 (B, L, K)
        feature_embed = feature_embed.permute(0, 2, 1)  # (B, K, L) -> (B, L, K)
        xf = self.feature_proj(feature_embed)  # 投影特征嵌入 (B, L, 25) -> (B, L, 16)

        # 4. 调整 xt 和 xf 的维度以进行拼接
        xt_repeated = xt.unsqueeze(2).repeat(1, 1, K, 1)  # (B, L, K, emb_dim)
        xf_repeated = xf.unsqueeze(2).repeat(1, 1, K, 1)  # (B, L, K, 16)

        # 5. 将局部统计信息与时间和特征嵌入进行拼接
        local_stats = torch.cat([mean_local, std_local, residual_mean], dim=1)  # (B, 3, L)
        local_stats_proj = self.stat_proj(local_stats.permute(0, 2, 1))  # (B, L, emb_dim)
        local_stats_proj = local_stats_proj.unsqueeze(2).repeat(1, 1, K, 1)  # (B, L, K, emb_dim)

        # 6. 融合时间、特征和局部统计信息
        combined_info = torch.cat([
            xt_repeated,  # (B, L, K, emb_dim)
            xf_repeated,  # (B, L, K, 16)
            local_stats_proj  # (B, L, K, emb_dim)
        ], dim=-1)  # (B, L, K, emb_dim + 16 + emb_dim)

        combined_info = combined_info.permute(0, 2, 1, 3)  # (B, K, L, emb_dim * 3)

        # 7. 最终投影得到 mask_embed
        mask_embed = self.final_proj(combined_info).squeeze(-1)  # 去掉最后的 embedding 维度 (B, K, L)
        mask_embed = mask_embed.unsqueeze(1).repeat(1, 2 * self.channels, 1, 1)  # 扩展为 (B, 2*channels, K, L)

        return mask_embed


if __name__ == '__main__':


    # 使用示例
    condition_mask_model = ConditionMask(128,16,55)
    observed_data = torch.randn(12, 55, 100)  # 原始数据
    x_mask = torch.randn(12, 55, 100)  # 被 mask 的数据
    cond_mask = torch.ones(12, 55, 100)  # 未被 mask 部分的位置
    target_mask = torch.ones(12, 55, 100)  # 被 mask 部分的位置
    time_embed = torch.randn(12, 100, 128)
    feature_embed = torch.randn(12, 55, 100)  # MTS 特征维度为 25, 长度为 100

    mask_embed = condition_mask_model(observed_data, x_mask, cond_mask, target_mask, time_embed, feature_embed)
    print(mask_embed.shape)  # 输出应为 (32, 32, 25, 100)









