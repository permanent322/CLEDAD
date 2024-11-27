import torch
import numpy as np
import random
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw


def imputation_mask_batch(observed_mask):
    """
    Generates a batch of masks for imputation task where certain observations are randomly masked based on a 
    sample-specific ratio.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values (1 for observed, 0 for missing).
    
    Returns:
    - Tensor: A mask tensor for imputation with the same shape as `observed_mask`.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    min_value, max_value = 0.1, 0.9
    for i in range(len(observed_mask)):
        sample_ratio = min_value + (max_value - min_value)*np.random.rand()  # missing ratio ## at random
        num_observed = observed_mask[i].sum().item()
        num_masked = round(num_observed * sample_ratio)
        rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask

def interpolation_mask_batch(observed_mask):
    """
    Generates a batch of masks for interpolation task by randomly selecting timestamps to mask across all features.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values.
    
    Returns:
    - Tensor: A mask tensor for interpolation tasks.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[2]
    timestamps = np.arange(total_timestamps)
    for i in range(len(observed_mask)):
        mask_timestamp = np.random.choice(
            timestamps
        )
        rand_for_mask[i][:,mask_timestamp] = -1
    cond_mask = (rand_for_mask > 0).float()
    return cond_mask
    

def forecasting_mask_batch(observed_mask):
    """
    Generates a batch of masks for forecasting task by masking out all future values beyond a randomly selected start
    point in the sequence (30% timestamps at most).
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values.
    
    Returns:
    - Tensor: A mask tensor for forecasting tasks.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[2]
    start_pred_timestamps = round(total_timestamps/3)
    timestamps = np.arange(total_timestamps)[start_pred_timestamps:]
    for i in range(len(observed_mask)):
        start_forecast_mask = np.random.choice(
            timestamps
        )
        rand_for_mask[i][:,-start_forecast_mask:] = -1
    cond_mask = (rand_for_mask > 0).float()
    return cond_mask
    

def forecasting_imputation_mask_batch(observed_mask):
    """
    Generates a batch of masks for forecasting/imputation task by masking out all 
    future values for a random subset of features beyond a randomly selected start
    point in the sequence (30% timestamps at most).
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values.
    
    Returns:
    - Tensor: A mask tensor for forecasting tasks.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[2]
    start_pred_timestamps = round(total_timestamps/3)
    timestamps = np.arange(total_timestamps)[start_pred_timestamps:]

    for i in range(len(observed_mask)):
        batch_indices = list(np.arange(0, len(rand_for_mask[i])))
        n_keep_dims = random.choice([1, 2, 3]) # pick how many dims to keep unmasked
        keep_dims_idx = random.sample(batch_indices, n_keep_dims) # choose the dims to keep
        mask_dims_idx = [i for i in batch_indices if i not in keep_dims_idx] # choose the dims to mask
        start_forecast_mask = np.random.choice(
            timestamps
        )
        rand_for_mask[i][mask_dims_idx, -start_forecast_mask:] = -1
    cond_mask = (rand_for_mask > 0).float()
    return cond_mask


def imputation_mask_sample(observed_mask):
    """
    Generates a mask for imputation for a single sample, similar to `imputation_mask_batch` but for an individual sample.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a single sample.
    
    Returns:
    - Tensor: A mask tensor for imputation for the sample.
    """
    ## Observed mask of shape KxL
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values   
    
    rand_for_mask = rand_for_mask.reshape(-1)
    min_value, max_value = 0.1, 0.9
    sample_ratio = min_value + (max_value - min_value)*np.random.rand()
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask

def interpolation_mask_sample(observed_mask):
    """
    Generates a mask for interpolation for a single sample by randomly selecting a timestamp to mask.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a single sample.
    
    Returns:
    - Tensor: A mask tensor for interpolation for the sample.
    """
    ## Observed mask of shape KxL
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[1]
    timestamps = np.arange(total_timestamps)
    
    mask_timestamp = np.random.choice(
        timestamps
    )
    rand_for_mask[:,mask_timestamp] = -1
    cond_mask = (rand_for_mask > 0).float()
    return cond_mask
    

def forecasting_mask_sample(observed_mask):
    """
    Generates a mask for forecasting for a single sample by masking out all future values beyond a selected timestamp.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a single sample.
    
    Returns:
    - Tensor: A mask tensor for forecasting for the sample.
    """
    ## Observed mask of shape KxL
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[1]
    
    start_pred_timestamps = round(total_timestamps/3)
    timestamps = np.arange(total_timestamps)[-start_pred_timestamps:]
    
    start_forecast_mask = np.random.choice(
        timestamps
    )
    rand_for_mask[:,start_forecast_mask:] = -1
    cond_mask = (rand_for_mask > 0).float()
    
    return cond_mask



def get_mask_equal_p_sample(observed_mask):
    """
    IIF mix masking strategy.
    Generates masks for a batch of samples where each sample has an equal probability of being assigned one of the
    three mask types: imputation, interpolation, or forecasting.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a batch of samples.
    
    Returns:
    - Tensor: A batch of masks with a mix of the three types.
    """
    B, K, L = observed_mask.shape
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    for i in range(B):
        
        threshold = 1/3

        imp_mask = imputation_mask_sample(observed_mask[i])
        p = np.random.rand()  # missing probability at random

        if p<threshold: 

            cond_mask = imp_mask

        elif p<2*threshold:

            cond_mask = interpolation_mask_sample(imp_mask)

        else:

            cond_mask = forecasting_mask_sample(imp_mask)

        rand_for_mask[i]=cond_mask
    
    return rand_for_mask
                            
def get_mask_probabilistic_layering(observed_mask):
    """
    Mix masking strategy.
    Generates masks for a batch of samples using a probabilistic layering approach where masks are applied in a 
    random order and with a random chance, potentially layering multiple types of masks.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a batch of samples.
    
    Returns:
    - Tensor: A batch of masks generated by probabilistic layering of mask types.
    """

    B, K, L = observed_mask.shape
    types = ['imputation', 'forecasting','interpolation']
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    for i in range(B):
        random.shuffle(types)
        mask = observed_mask[i]
        # missing rate
        m_initial = torch.sum(torch.eq(mask, 0))/(K*L)
        m = m_initial
        for mask_type in types:
            p = np.random.rand()
            if mask_type == types[-1] and m==m_initial:
                p = 1
            if p>0.5:
                if mask_type == 'imputation':
                    mask = imputation_mask_sample(mask)
                elif mask_type == 'interpolation':
                    mask = interpolation_mask_sample(mask)
                else:
                    mask = forecasting_mask_sample(mask)
                    
                m = torch.sum(torch.eq(mask, 0))/(K*L)
    
        rand_for_mask[i]=mask
    
    return rand_for_mask





def pattern_mask_batch(observed_mask):
    """
    Generates a batch of masks based on a predetermined pattern or a random choice between imputation mask and a
    previously used mask pattern. Used for finetuning CLEDAD on PM25 dataset.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a batch of samples.
    
    Returns:
    - Tensor: A batch of masks where each mask is either an imputation mask or follows a specific pattern.
    """
    pattern_mask = observed_mask
    rand_mask = imputation_mask_batch(observed_mask)

    cond_mask = observed_mask.clone()  ### Gradients can flow back to observed_mask
    for i in range(len(cond_mask)):
        mask_choice = np.random.rand()
        if mask_choice > 0.5:
            cond_mask[i] = rand_mask[i]
        else:  # draw another sample for histmask (i-1 corresponds to another sample) ###### Not randomly sampled?
            cond_mask[i] = cond_mask[i] * pattern_mask[i - 1] 
    return cond_mask


def patch_mask(observed_data):
    data = observed_data.clone()  # 避免修改原始tensor

    data = data.permute(0, 2, 1)  # 将tensor的维度换位
    B,L,M = data.shape

    # 生成
    patch_for_mask = torch.ones_like(data)


    for i in range(B):
        size = 10
        start_index = 0
        end_index = size
        # 计算tensor每行的方差
        mean = torch.mean(data[i], dim=1)   # 每个时间步，所有特征的均值
        # 生成一个tensor_pre_label，有两列，value和pre_label，都是0，行数和tensor的行数一样
        tensor_pre_label = torch.zeros((L,2))
        tensor_pre_label[:, 0] = mean  # 第一列放均值

        all_var = torch.var(data.flatten())  # 计算整体数据的方差
        last_size_var = 0

        for x in range(int(L / size)):
            # 计算[start_index: end_index]这个patch的方差，和上一个patch的方差比较
            last_size_var = get_mutation_point(tensor_pre_label, start_index, end_index, last_size_var)

            start_index += size
            end_index += size

        # 计算最后一个patch的方差，并标记异常点的pre_label
        get_mutation_point(tensor_pre_label, start_index,data.size(0) - 1, last_size_var)
        index = torch.where(tensor_pre_label[:, 1] == 1)  # 异常点的index
        patch_for_mask[i][index] = 0   # 异常点的mask为0
        # 返回的是一个tensor，有两列，value和pre_label
    return patch_for_mask.permute(0,2,1)

def get_mutation_point(tensor_pre_label, start_index, end_index, last_size_var):
    patch = tensor_pre_label[start_index:end_index, 0]  # 取出当前patch的数据
    # print("torch.isnan(patch).any() : " , torch.isnan(patch).any())
    # print("torch.unique(patch).numel() > 1:", torch.unique(patch).numel() > 1)
    #size_var = torch.var(patch)  # 计算该个patch的方差

    if patch.numel() > 1 and torch.unique(patch).numel() > 1:
        size_var = torch.var(patch)
    else:
        size_var = 0

        # 计算当前patch的方差和上一个patch的方差的比值
    if last_size_var == 0:
        times = float('nan')
    else:
        times = size_var / last_size_var

    if times != float('nan') and times >= 5:  # 如果times大于等于5，就认为是异常点
        tensor_pre_label[start_index:end_index, 1] = 1  # pre_label赋值为1
    else:
        tensor_pre_label[start_index:end_index, 1] = 0  # 否则保持0

    return size_var  # 返回当前patch的方差


def cluster_blocks_dtw(patchs, n_clusters=3, random_state=42):
    """使用DTW对块进行聚类"""
    patchs_np = patchs.cpu().numpy()
    model = TimeSeriesKMeans(n_clusters=n_clusters,
                             metric="dtw",
                             random_state=random_state)
    cluster_labels = model.fit_predict(patchs_np)
    cluster_centers = model.cluster_centers_
    return torch.tensor(cluster_labels), torch.tensor(cluster_centers)

def calculate_anomaly_scores_dtw(patch, cluster_centers):
    """计算每个块的异常分数，使用DTW距离"""
    anomaly_scores = []
    patch_np = patch.cpu().numpy()
    centers_np = cluster_centers.cpu().numpy()
    for patch in patch_np:
        distances = [dtw(patch, center) for center in centers_np]
        anomaly_scores.append(min(distances))
    return torch.tensor(anomaly_scores)

def cluster_patch_mask(observed_data, patch_size=10):
    """将批次数据分割成不重叠的块"""
    threshold = 0.95
    B, M, L = observed_data.shape
    data = observed_data.clone()
    data = data.permute(0, 2, 1)
    num_patchs = L // patch_size
    patchs = data.unfold(1, patch_size, patch_size).transpose(2,3)
    patch_for_mask = torch.ones_like(data)
    for i in range(B):
        cluster_labels, cluster_centers = cluster_blocks_dtw(patchs[i], n_clusters=3)
        anomaly_scores = calculate_anomaly_scores_dtw(patchs[i], cluster_centers)
        threshold_value = torch.quantile(anomaly_scores, threshold)
        anomalous_patchs = torch.where(anomaly_scores > threshold_value)[0]
        patch_mask = torch.ones_like(patchs[i])
        for block in anomalous_patchs:
            patch_mask[block] = 0
        patch_mask = patch_mask.reshape(L,-1)
        index = torch.where(patch_mask == 0)
        patch_for_mask[i][index] = 0
    return patch_for_mask.permute(0,2,1)


def random_mask(observed_mask, p=0.7):
    """
    Generates random masks for a batch of samples, setting p% of (K, L) elements to 1 (visible) and the rest to 0 (invisible).

    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a batch of samples, shape (B, K, L).
    - p (float): The percentage of elements to set as visible (between 0 and 1).

    Returns:
    - Tensor: A batch of masks with p% of elements randomly set to 1 (visible) and the rest to 0 (invisible).
    """
    B, K, L = observed_mask.shape  # Batch size, number of features, length of sequence
    rand_for_mask = torch.zeros_like(observed_mask)  # Initialize all elements to 0 (invisible)

    for i in range(B):
        total_elements = K * L
        num_visible_elements = int(total_elements * p)  # Number of elements to make visible

        # Generate random indices to set as visible
        visible_indices = torch.randperm(total_elements)[:num_visible_elements]

        # Flatten the mask and set the chosen indices to 1 (visible)
        mask = rand_for_mask[i].reshape(-1)  # Flatten the (K, L) tensor
        mask[visible_indices] = 1  # Set the randomly selected elements to 1
        rand_for_mask[i] = mask.view(K, L)  # Reshape back to (K, L)

    return rand_for_mask


def hist_mask(observed_mask):
    B, K, L = observed_mask.shape  # 获取输入的维度
    patch_size = L // 10  # 将L均分成10个patch

    # 创建一个新的 mask，初始化为全0
    mask = torch.zeros_like(observed_mask)

    # 为每一个奇数编号的patch置1，偶数编号的patch置0
    for i in range(0, 10, 2):
        mask[:, :, i * patch_size:(i + 1) * patch_size] = 1

    return mask


if __name__ == '__main__':
    # import pandas as pd
    # df = pd.read_excel('../test.xlsx', header=None, index_col=0)
    # data = df.values
    # observed_data = torch.tensor(data)
    # observed_data = observed_data.permute(1,0)
    # L, M = observed_data.shape
    # observed_data = observed_data.reshape(1,L,M)
    # mask = patch_mask(observed_data)
    # print(mask)
    # for i in range(len(mask)):
    #     print(mask[i])
    #
    # mask_np = mask.numpy()
    # print(mask_np)


    # cluster_patch_mask(observed_data, patch_size=10)
    observed_mask = torch.zeros(2, 3, 20)  # 假设输入维度为 (2, 3, 20)
    result = hist_mask(observed_mask)
    print(result.shape)
    print(result)
