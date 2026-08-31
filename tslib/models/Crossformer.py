import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead


from math import ceil


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # NOTE:
        # - 原版 Crossformer 这里写死 seg_len=12。
        # - 为了让外部可通过 Time-Series-Library/run.py 的 --patch_len 来调整 patch 长度，
        #   我们将 seg_len 绑定到 configs.patch_len（未显式设置时仍保持默认 12）。
        # - run.py 已做 Crossformer 场景下的默认回填：若未传 --patch_len，则 patch_len=12。
        self.seg_len = int(getattr(configs, "patch_len", 12))
        if self.seg_len <= 0:
            raise ValueError(f"Invalid patch_len/seg_len: {self.seg_len}. It must be a positive integer.")
        self.win_size = 2
        self.task_name = configs.task_name

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        # Decoder value embedding (optional): used to inject known future covariates / target placeholders.
        # We embed only the last pred_len steps of x_dec and then add learnable dec_pos_embedding.
        self.dec_value_embedding = PatchEmbedding(
            configs.d_model,
            self.seg_len,
            self.seg_len,
            self.pad_out_len - configs.pred_len,
            0,
        )
        # Crossformer decoder placeholder mask embedding (optional)
        # - Enabled by default; can be disabled via configs.use_dec_mask_embedding (or CLI: --no_dec_mask_embedding)
        self.use_dec_mask_embedding = bool(getattr(configs, "use_dec_mask_embedding", True))
        self.dec_mask_embedding = nn.Parameter(torch.zeros(1, 1, 1, configs.d_model))
        nn.init.normal_(self.dec_mask_embedding, mean=0, std=0.02)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l == 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    # activation=configs.activation,
                )
                for l in range(configs.e_layers + 1)
            ],
        )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dec_mask=None, *, return_hidden: bool = False):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        # Decoder input:
        # - baseline: learnable dec_pos_embedding (standard Crossformer query tokens)
        # - enhanced: add value embedding from x_dec (last pred_len), preserving positional encoding
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        if x_dec is not None:
            try:
                x_dec_seg = x_dec[:, -self.pred_len:, :]
                x_dec_emb, n_vars_dec = self.dec_value_embedding(x_dec_seg.permute(0, 2, 1))
                x_dec_emb = rearrange(x_dec_emb, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars_dec)
                # Optional: add learnable mask embedding for placeholder positions (target vars in future segment)
                if dec_mask is not None and self.use_dec_mask_embedding:
                    try:
                        # dec_mask: [B, pred_len, N] -> [B, N, pred_len]
                        m = dec_mask.permute(0, 2, 1).float()
                        pad = int(self.pad_out_len - self.pred_len)
                        if pad > 0:
                            m = F.pad(m, (0, pad), mode='replicate')  # [B, N, pad_out_len]
                        # patchify: [B, N, seg_num, seg_len]
                        m = m.unfold(dimension=-1, size=self.seg_len, step=self.seg_len)
                        mask_ratio = m.mean(dim=-1, keepdim=True)  # [B, N, seg_num, 1]
                        x_dec_emb = x_dec_emb + self.dec_mask_embedding * mask_ratio
                    except Exception:
                        pass
                dec_in = dec_in + x_dec_emb
            except Exception:
                # Fallback to standard behavior if shapes are incompatible
                pass
        if return_hidden:
            dec_out, dec_hidden = self.decoder(dec_in, enc_out, return_last_hidden=True)
            return dec_out, dec_hidden
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

        return dec_out

    def anomaly_detection(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_hidden: bool = False, dec_mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if return_hidden:
                dec_out, dec_hidden = self.forecast(
                    x_enc, x_mark_enc, x_dec, x_mark_dec, dec_mask=dec_mask, return_hidden=True
                )
                # dec_out: [B, pad_out_len, D] -> [B, pred_len, D]
                return dec_out[:, -self.pred_len:, :], dec_hidden
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, dec_mask=dec_mask)
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