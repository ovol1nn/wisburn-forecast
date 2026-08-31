import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        # iTransformer uses "inverted embedding": treat each variate as a token, and project the time axis length (c_in)
        # into d_model. In our training pipeline we may extend encoder input to seq_len + pred_len (known future
        # covariates for non-targets; placeholders for targets), so c_in must match the actual encoder length.
        enc_in_len = configs.seq_len
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            enc_in_len = configs.seq_len + configs.pred_len
        self.enc_embedding = DataEmbedding_inverted(enc_in_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

        # Mask embedding: 标识占位符位置（用于 known-future-covariate 模式）
        # 对 iTransformer 默认启用，当 enc_mask 传入时自动使用
        # 形状 [1, 1, d_model]，会加到占位符对应变量的 embedding 上
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, configs.d_model))
        nn.init.normal_(self.mask_embedding, mean=0, std=0.02)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_mask=None):
        """
        Args:
            x_enc: [B, L, N] encoder 输入（L = seq_len 或 seq_len + pred_len）
            x_mark_enc: [B, L, mark_dim] 时间特征
            x_dec: decoder 输入（iTransformer 不使用）
            x_mark_dec: decoder 时间特征（iTransformer 不使用）
            enc_mask: [B, L, N] 布尔张量，True 表示该位置是占位符（可选）
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N, d_model]

        # 如果有 mask，对占位符位置添加 mask embedding
        if enc_mask is not None:
            # enc_mask: [B, L, N] -> permute -> [B, N, L]
            mask_perm = enc_mask.permute(0, 2, 1).float()  # [B, N, L]
            # 计算每个变量的占位符比例，用于加权 mask embedding
            # 这样内生变量（有占位符）会得到较大的 mask embedding 权重
            # 外生变量（无占位符）权重为 0，不受影响
            mask_ratio = mask_perm.mean(dim=-1, keepdim=True)  # [B, N, 1]
            # enc_out 可能包含额外的时间特征 token（当 x_mark 不为 None 时，DataEmbedding_inverted
            # 会将变量 token 和时间特征 token concat 起来），需要对 mask_ratio 进行 padding
            # 时间特征 token 的 mask_ratio 应该是 0（不需要 mask embedding）
            B, num_tokens, _ = enc_out.shape
            N_vars = mask_ratio.shape[1]
            if num_tokens > N_vars:
                padding = torch.zeros(B, num_tokens - N_vars, 1, device=mask_ratio.device, dtype=mask_ratio.dtype)
                mask_ratio = torch.cat([mask_ratio, padding], dim=1)
            enc_out = enc_out + self.mask_embedding * mask_ratio

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, enc_mask=None):
        """
        Args:
            x_enc: [B, L, N] encoder 输入
            x_mark_enc: [B, L, mark_dim] 时间特征
            x_dec: decoder 输入
            x_mark_dec: decoder 时间特征
            mask: imputation 任务的 mask
            enc_mask: [B, L, N] 布尔张量，True 表示该位置是占位符（用于 forecast 任务）
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_mask=enc_mask)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
