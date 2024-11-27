import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .mtsEmbedding import embedding_MTS
from .TSLANet_with_residual import TSLANet_with_residual
from utils.masking_strategies import get_mask_probabilistic_layering,cluster_patch_mask, hist_mask,random_mask, patch_mask, get_mask_equal_p_sample, imputation_mask_batch, pattern_mask_batch, interpolation_mask_batch
from .Condition_Embedding import ConditionMask
from .maskmodel import ConditionMask as ConditionMask_old
from .CL import  ConditionMask as CL_model

class CLEDAD_base(nn.Module):
    """
    Base class for CLEDAD model.
    
    Attributes:
        - device: The device on which the model will run (CPU or CUDA).
        - target_dim: number of features in the MTS.
        - sample_feat: Whether to sample subset of features during training.
        - mix_masking_strategy: Strategy for mixing masks during pretraining.
        - time_strategy: Strategy for embedding time points.
        - emb_time_dim: Dimension of time embeddings.
        - emb_cat_feature_dim: Dimension of categorical feature embeddings.
        - mts_emb_dim: Dimension of the MTS embeddings.
        - embed_layer: Embedding layer for feature embeddings.
        - diffmodel: Model block for diffusion.
        - embdmodel: Model for embedding MTS.
        - mlp: Multi-layer perceptron for classification tasks.
        - conv: Convolutional layer for anomaly detection.
        
    Methods:
        - time_embedding: Generates sinusoidal embeddings for time points.
        - get_mts_emb: Generates embeddings for MTS.
        - calc_loss: Calculates training loss for a given batch of data.
        - calc_loss_valid: Calculates validation loss for a given batch of data.
        - impute: Imputes missing values in the time series.
        - forward: Forward pass for pretraining, and fine-tuning for imputation, interpolation, and forecasting.
        - forward_finetuning: Forward pass for fine-tuning on specific tasks (classification or anomaly detection).
        - evaluate_finetuned_model: Evaluates the fine-tuned model for classification and anomaly detection.
        - evaluate: Evaluates the model on imputation, interpolation and forecasting.
    """
    def __init__(self, target_dim, config, device, sample_feat=20):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.sample_feat = sample_feat
        self.seq_len = config["data"]["seq_len"]
        self.mix_masking_strategy = config["model"]["mix_masking_strategy"]   
        self.time_strategy = config["model"]["time_strategy"]
        
        self.emb_time_dim = config["embedding"]["timeemb"]
        self.emb_cat_feature_dim = config["embedding"]["featureemb"]  

        self.mts_emb_dim = 1+2*config["embedding"]["channels"]
        self.mask_method = config["model"]["mix_masking_strategy"]
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_cat_feature_dim
        )
        
        config_diff = config["diffusion"]
        config_diff["mts_emb_dim"] = self.mts_emb_dim
        config_diff["seq_len"] = self.seq_len
        config_emb = config["embedding"]
        self.cond_method = config["embedding"]["method"]

        # data: seq_len: 100  num_feat: 38
        L = config["data"]["seq_len"]
        K = config["data"]["num_feat"]

        # self.diffmodel = diff_Block(config_diff)
        # self.diffmodel = TSLANet(config_diff )
        self.diffmodel = TSLANet_with_residual(config_diff)
        self.embdmodel = embedding_MTS(config_emb)
        # self.maskmodel = ConditionMask(config_emb["timeemb"],config_emb["featureemb"],config_emb["num_feat"])
        self.maskmodel = ConditionMask_old(config_emb["timeemb"])
        self.cl_model = CL_model(config_emb["timeemb"],K=K,L=L)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat) 
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        
        L = config_emb["num_timestamps"]
        K = config_emb["num_feat"]

        # Number of classes for classification experiments
        num_classes = config_emb["classes"]
        
        ## Classifier head
        self.mlp = nn.Sequential(
            nn.Linear(L*K*self.mts_emb_dim, 256),  # Adjust as necessary
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
        ## projection to reconstruct MTS for Anomaly Detection
        self.conv = nn.Linear((self.mts_emb_dim-1)*K, K, bias=True)

        
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, mts_emb, is_train,cl_loss
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, mts_emb, is_train, set_t=t
            )
            loss_sum += loss.detach()
               
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, mts_emb, is_train, cl_loss,set_t=-1
    ):
        '''
                    observed_data 原始数据
                    cond_mask     # 掩码  有0有1
                    observed_mask  # 全为1的掩码
                    mts_emb        # f(x_obs)+ x_mask
                    is_train
        '''

        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:  # for training
            t = torch.randint(0, self.num_steps, [B]).to(self.device)  # (B) 32个随机数，取值0-时间步
        current_alpha = self.alpha_torch[t]  # (B,1,1)  当时随机时间步的alpha值
        
        noise = torch.randn_like(observed_data) # 生成shape一样的标准正态分布（均值为0，标准差为1）
        # 生成噪声数据 原始数据+噪声数据
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)   # 只含有mask位置的加噪后的数据

        # predicted = self.diffmodel(total_input, mts_emb, t)  # (B,K,L)  返回的是TS的预测值
        # predicted = noise - predicted  # 加噪数据 - 预测数据 = 预测误差
        # target_mask = observed_mask - cond_mask  # mask位置为1
        # residual = (noise - predicted) * target_mask   # 被mask位置的噪声数据减去预测值的残差
        # num_eval = target_mask.sum()    # 计算被mask位置的数量
        # loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)  # 计算损失
        # return loss

        predicted_noise, predicted_data = self.diffmodel(total_input, mts_emb, t)   # (B,K,L)   # 返回的是直接预测的原始数据


        # 添加一个频域的损失
        fft1 = torch.fft.fft(predicted_data.transpose(1, 2), norm='forward')
        fft2 = torch.fft.fft(observed_data.transpose(1, 2), norm='forward')
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)


        target_mask = observed_mask - cond_mask  # mask位置为1
        residual = (noise - predicted_noise) * target_mask   # 被mask位置的噪声数据减去预测值的残差
        fourier_loss = (torch.real(fft1)- torch.real(fft2)) + (torch.imag(fft1)- torch.imag(fft2)) * target_mask
        num_eval = target_mask.sum()  # 计算被mask位置的数量
        loss = ((residual ** 2 ).sum() + (fourier_loss**2).sum())/ (num_eval if num_eval > 0 else 1) + 0.1*cl_loss # 计算损失
        # loss = (residual ** 2).sum()  / (num_eval if num_eval > 0 else 1)  # 计算损失

        return loss

 


    def get_data_impute(self,observed_data, cond_mask, mts_emb, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        imputed_middle_samples = torch.zeros(B, self.num_steps, K, L)

        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)  # random_noise 标准正态分布

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)  # observed部分
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)  # 加噪的目标, mask部分

                # 去噪 预测噪声
                predicted = self.diffmodel(noisy_target, mts_emb, torch.tensor([t]).to(self.device))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)  # 去噪

                if t > 0:  # 若当前时间步，不是第一个时间步，添加噪声到当前样本
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
                imputed_middle_samples[:, t] = current_sample.detach()  # 保存当前生成的插值样本到imputed_samples
            imputed_samples[:, i] = current_sample.detach()  # 保存当前生成的插值样本到imputed_samples
        return imputed_samples, imputed_middle_samples

    def impute(self, observed_data, cond_mask, mts_emb, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        imputed_middle_samples = torch.zeros(B, self.num_steps, K, L)
        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)  # random_noise 标准正态分布

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data).unsqueeze(1)  # observed部分
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)  # 加噪的目标, mask部分

                # 去噪 预测噪声
                predicted, predicted_data= self.diffmodel(noisy_target, mts_emb, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)   # 去噪

                if t > 0:  # 若当前时间步，不是第一个时间步，添加噪声到当前样本
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
                imputed_middle_samples[:, t] = current_sample.detach()  # 保存当前生成的插值样本到imputed_samples
            imputed_samples[:, i] = current_sample.detach()  # 保存当前生成的插值样本到imputed_samples
        return imputed_samples, imputed_middle_samples


    def forward(self, batch, is_train=1, task='pretraining', normalize_for_ad=False):
        ## is_train = 1 for pretraining and for finetuning but task should be specified and = 0 for evaluation
        
        (
            observed_data,  # 原始数据
            observed_mask,  # 全为1的掩码
            feature_id,     #  None 特征id
            observed_tp,    # 时间步 (B, L)
            gt_mask,        # 全为1的掩码
            for_pattern_mask, # 全为1的掩码
            _,
            _,
            _
        ) = self.process_data(batch, sample_feat=self.sample_feat, train=is_train)
        if is_train == 0:   # 非训练
            if self.mask_method == 'patch_mask':
                cond_mask = patch_mask(observed_data)
            elif self.mask_method =='hist_mask':
                cond_mask = hist_mask(observed_data)
            elif self.mask_method =='random_mask':
                cond_mask = random_mask(observed_data)
            elif self.mask_method == 'cluster_patch_mask':
                cond_mask = cluster_patch_mask(observed_data,10)
                # print("the mask method is : ", self.mask_method)
            else:
                print(
                    'Please choose one of the following masking strategy in the config: patch_mask,hist_mask,random_mask,cluster_patch_mask')
        else:
            if task == 'pretraining':
                if self.mix_masking_strategy =='patch_mask':
                    cond_mask = patch_mask(observed_data)
                elif self.mask_method == 'hist_mask':
                    cond_mask = hist_mask(observed_data)
                elif self.mix_masking_strategy=='cluster_patch_mask':
                    cond_mask = cluster_patch_mask(observed_data,10)
                elif self.mask_method == 'random_mask':
                    cond_mask = random_mask(observed_data)
                elif self.mix_masking_strategy == 'equal_p':
                    cond_mask = get_mask_equal_p_sample(observed_mask)   # 生成掩码 cond_mask 元素为0或1
                elif self.mix_masking_strategy == 'probabilistic_layering':
                    cond_mask = get_mask_probabilistic_layering(observed_mask)
                else:
                    print('Please choose one of the following masking strategy in the config: patch_mask,hist_mask,random_mask,cluster_patch_mask')
            elif task == 'Imputation':
                cond_mask = imputation_mask_batch(observed_mask)
            elif task == 'Interpolation':
                cond_mask = interpolation_mask_batch(observed_mask)
            elif task == 'Imputation with pattern':
                cond_mask = pattern_mask_batch(observed_mask)
            elif task == 'Forecasting':
                cond_mask = gt_mask
            else:
                print('Please choose the right masking to be applied during finetuning')

        if normalize_for_ad:
            ## Normalization from non-stationary Transformer
            means = observed_data.mean(2, keepdim=True)  # 沿着时间维度求均值
            observed_data = observed_data-means  # 减去均值
            stdev = torch.sqrt(torch.var(observed_data, dim=2, keepdim=True, unbiased=False) + 1e-5)  # 沿着时间维度求标准差
            observed_data /= stdev  # 除以标准差

        x_co = (cond_mask * observed_data).unsqueeze(1)      # 掩码后的数据  也就是可见的数据
        mts_emb,cl_loss = self.get_mts_emb(observed_tp, cond_mask, observed_data, feature_id, method=self.mask_method) # 时间步， 条件掩码， 掩码后的数据， 特征id
        
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        '''
            observed_data 原始数据   
            cond_mask     # 掩码  有0有1
            observed_mask  # 全为1的掩码
            mts_emb        # f(x_obs)+ x_mask
            is_train
        '''
        return loss_func(observed_data, cond_mask, observed_mask, mts_emb, is_train,cl_loss)

    def forward_finetuning(self, batch, criterion, task='classification', normalize_for_ad=False):
        ## task should be either, classification or anomaly_detection
        
        (
            observed_data,
            observed_mask,
            feature_id,
            observed_tp,
            gt_mask,
            _,
            _,
            _,
            classes
        ) = self.process_data(batch, sample_feat=self.sample_feat, train=False)
        
        if normalize_for_ad:
            ## Normalization from non-stationary Transformer
            original_observed_data = observed_data.clone()
            means = observed_data.mean(2, keepdim=True)
            observed_data = observed_data-means
            stdev = torch.sqrt(torch.var(observed_data, dim=2, keepdim=True, unbiased=False) + 1e-5)
            observed_data /= stdev

        x_co = (observed_mask * observed_data).unsqueeze(1)
        cond_mask = cluster_patch_mask(observed_data, 10)
        target_mask = observed_mask - cond_mask

        mts_emb,cl_loss = self.get_mts_emb(observed_tp, cond_mask, observed_data, feature_id, method=self.mask_method)
        
        if task == 'classification':
            outputs = self.mlp(mts_emb.reshape(mts_emb.shape[0],-1)) 
            classes = classes.to(self.device)
            loss = criterion(outputs, classes)
            return outputs, loss
        elif task == 'anomaly_detection':
            B, C, K, L =mts_emb.shape
            # print("mts_emb.shape:    ",mts_emb.shape)
            #outputs = self.projection(mts_emb.permute(0,2,3,1)).squeeze(-1)
            outputs = self.conv(mts_emb[:, :C-1, :, :].reshape(B, (C-1)*K, L).permute(0,2,1)).permute(0,2,1)
            # print("outputs.shape:    ",outputs.shape)
            if normalize_for_ad:
                dec_out = outputs * \
                      (stdev[:, :, 0].unsqueeze(2).repeat(
                          1, 1, L))
                outputs = dec_out + \
                      (means[:, :, 0].unsqueeze(2).repeat(
                          1, 1, L))

            loss = criterion(outputs, original_observed_data)
            return outputs, loss




    def evaluate_finetuned_model(self, batch, criterion= nn.MSELoss(reduction='none'), n_samples=20, normalize_for_ad=False):

        (
            observed_data,
            observed_mask,
            feature_id,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            _,
            labels
        ) = self.process_data(batch, sample_feat=self.sample_feat, train=False)

        with torch.no_grad():

            if self.mask_method == 'patch_mask':
                cond_mask = patch_mask(observed_data)
            elif self.mask_method == 'hist_mask':
                cond_mask = hist_mask(observed_data)
                # print("the mask method is : ", self.mask_method)
            elif self.mask_method == 'random_mask':
                cond_mask = random_mask(observed_data)
            elif self.mask_method == 'cluster_patch_mask':
                cond_mask = cluster_patch_mask(observed_data, 10)
                # print("the mask method is : ", self.mask_method)
            else:
                print(
                    'Please choose one of the following masking strategy in the config: patch_mask,hist_mask,random_mask,cluster_patch_mask')
            target_mask = observed_mask - cond_mask

            if normalize_for_ad:
                ## Normalization from non-stationary Transformer
                means = observed_data.mean(2, keepdim=True)
                original_observed_data = observed_data.clone()
                observed_data = observed_data - means
                stdev = torch.sqrt(torch.var(observed_data, dim=2, keepdim=True, unbiased=False) + 1e-5)
                observed_data /= stdev

            x_co = (cond_mask * observed_data).unsqueeze(1)
            # mts_emb = self.get_mts_emb(observed_tp, cond_mask, x_co, feature_id)
            mts_emb,_ = self.get_mts_emb(observed_tp, cond_mask, observed_data, feature_id, method=self.mask_method)

            x = observed_data * (1-cond_mask)
            samples, imputed_middle_samples = self.impute(x, cond_mask, mts_emb, n_samples)


            outputs= samples.median(dim=1).values
            # for i in range(len(cut_length)):  # to avoid double evaluation
            #     target_mask[i, ..., 0: cut_length[i].item()] = 0


            observed_data = observed_data * (1-cond_mask)
            outputs=  outputs*(1-cond_mask)
            score = torch.mean(criterion(observed_data, outputs), dim=1)

            # print("score.shape:    ",score.shape)

            # score = torch.mean(criterion(x, y), dim=1)
            score = score.detach().cpu().numpy()
            return samples, score, observed_data, target_mask, observed_mask, observed_tp, imputed_middle_samples
            # return outputs, score


        
    def evaluate(self, batch, n_samples, normalize_for_ad=False):
        (
            observed_data,
            observed_mask,
            feature_id,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            _,
            labels
        ) = self.process_data(batch, sample_feat=self.sample_feat, train=False)

        with torch.no_grad():
            if self.mask_method == 'patch_mask':
                cond_mask = patch_mask(observed_data)
            elif self.mask_method == 'hist_mask':
                cond_mask = hist_mask(observed_data)
                # print("the mask method is : ", self.mask_method)
            elif self.mask_method =='random_mask':
                cond_mask = random_mask(observed_data)
            elif self.mask_method == 'cluster_patch_mask':
                cond_mask = cluster_patch_mask(observed_data, 10)
                # print("the mask method is : ", self.mask_method)
            else:
                print(
                    'Please choose one of the following masking strategy in the config: patch_mask,hist_mask,random_mask,cluster_patch_mask')
            target_mask = observed_mask - cond_mask

            if normalize_for_ad:
                ## Normalization from non-stationary Transformer
                original_observed_data = observed_data.clone()
                means = observed_data.mean(2, keepdim=True)
                observed_data = observed_data-means
                stdev = torch.sqrt(torch.var(observed_data, dim=2, keepdim=True, unbiased=False) + 1e-5)
                observed_data /= stdev

            x_co = (cond_mask * observed_data).unsqueeze(1)  # observed部分
            # mts_emb = self.get_mts_emb(observed_tp, cond_mask, x_co, feature_id)  # observerd部分embedding
            mts_emb,_ = self.get_mts_emb(observed_tp, cond_mask, observed_data, feature_id, method=self.mask_method)
            samples = self.impute(observed_data, cond_mask, mts_emb, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        if normalize_for_ad:
            if labels is not None:
                return samples, original_observed_data, target_mask, observed_mask, observed_tp, labels
            else:
                return samples, original_observed_data, target_mask, observed_mask, observed_tp
        else:
            if labels is not None:
                return samples, observed_data, target_mask, observed_mask, observed_tp, labels
            else:
                return samples, observed_data, target_mask, observed_mask, observed_tp
    




class CLEDAD_PM25(CLEDAD_base):
    """
    Specialized CLEDAD model for PM2.5 environmental data.
    
    Designed to handle and process PM2.5 data for imputation.
    """
    def __init__(self, config, device, target_dim=36, sample_feat=False):
        super(CLEDAD_PM25, self).__init__(target_dim, config, device, sample_feat)

    def process_data(self, batch, train, sample_feat):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            None,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            None,
            None,
        )


class CLEDAD_Physio(CLEDAD_base):
    """
    Specialized CLEDAD model for PhysioNet dataset.
    
    Adapts the CLEDAD_base model for tasks involving PhysioNet data, including imputation and interpolation.
    """
    def __init__(self, config, device, target_dim=35, sample_feat=False):
        super(CLEDAD_Physio, self).__init__(target_dim, config, device, sample_feat)

    def process_data(self, batch, train, sample_feat):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        labels = batch["labels"]
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            None,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            None,
            labels,

        )


class CLEDAD_AD(CLEDAD_base):
    """
    Specialized CLEDAD model for anomaly detection datasets.
    
    Tailors the CLEDAD_base model for anomaly detection datasets, including MSL, SMD, PSM, SMAP and SWaT.
    """
    def __init__(self, config, device, target_dim=55, sample_feat=False):
        super(CLEDAD_AD, self).__init__(target_dim, config, device, sample_feat)

    def process_data(self, batch, train, sample_feat):
        observed_data = batch["observed_data"].to(self.device).float()   # 原始数据
        observed_mask = batch["observed_mask"].to(self.device).float()   # 全为1的掩码
        observed_tp = batch["timepoints"].to(self.device).float()        # 时间步长度
        gt_mask = batch["gt_mask"].to(self.device).float()               # 全为1的掩码
        label = batch["label"].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)     # (B, F, C) -> (B, C, F)
        observed_mask = observed_mask.permute(0, 2, 1)     # (B, F, C) -> (B, C, F)
        gt_mask = gt_mask.permute(0, 2, 1)                 # (B, F, C) -> (B, C, F)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)  # (B)
        for_pattern_mask = observed_mask  # (B, C, F)
        return (
            observed_data,  # 原始数据
            observed_mask,  # 全为1的掩码
            None,
            observed_tp,    # 时间步长度
            gt_mask,
            for_pattern_mask, #
            cut_length,  # B
            None,
            label,  # 标签
        )


if __name__ == '__main__':
    data = torch.randn(32, 20, 25, 100)
    median_data = data.median(dim=1)
