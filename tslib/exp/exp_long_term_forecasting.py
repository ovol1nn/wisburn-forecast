from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import csv
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _maybe_extend_itransformer_encoder_input(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        """
        For iTransformer only:
        - Non-target variables: encoder sees seq_len + pred_len true values (known future covariates)
        - Target variables: encoder sees seq_len true values + pred_len placeholders
          placeholders are the mean over the history window (seq_len) for each target variable, per-sample.

        Note: this uses args.loss_target_indices (variable indices in the last dimension).

        Returns:
            batch_x: [B, seq_len+pred_len, N] 扩展后的 encoder 输入
            batch_x_mark: [B, seq_len+pred_len, mark_dim] 扩展后的时间特征
            enc_mask: [B, seq_len+pred_len, N] 布尔张量，True 表示占位符位置（仅 iTransformer 返回非 None）
        """
        if getattr(self.args, 'model', None) != 'iTransformer':
            return batch_x, batch_x_mark, None
        if not getattr(self.args, 'pred_len', None):
            return batch_x, batch_x_mark, None
        if not getattr(self.args, 'loss_target_indices', None):
            # Without explicit target indices we cannot separate targets vs non-targets safely.
            return batch_x, batch_x_mark, None

        pred_len = int(self.args.pred_len)
        if pred_len <= 0:
            return batch_x, batch_x_mark, None

        B, seq_len, N = batch_x.shape

        # Future true segment (aligned to [s_end : s_end + pred_len]) lives in the last pred_len of batch_y
        future = batch_y[:, -pred_len:, :].clone()
        idx = torch.as_tensor(self.args.loss_target_indices, device=future.device, dtype=torch.long)
        # Placeholder value for targets: mean over the seq_len history window
        hist_mean = batch_x[:, :, idx].mean(dim=1, keepdim=True)  # [B, 1, n_targets]
        future[:, :, idx] = hist_mean.repeat(1, pred_len, 1)

        # 构建 enc_mask：seq_len 段全 False，pred_len 段的 target 位置为 True
        # enc_mask[b, t, v] = True 表示样本 b、时间步 t、变量 v 的值是占位符
        enc_mask = torch.zeros(B, seq_len + pred_len, N, dtype=torch.bool, device=batch_x.device)
        enc_mask[:, seq_len:, idx] = True  # pred_len 段的 target 列标记为占位符

        # Extend encoder input and its time features
        batch_x = torch.cat([batch_x, future], dim=1)  # [B, seq_len + pred_len, D]
        batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, -pred_len:, :]], dim=1)
        return batch_x, batch_x_mark, enc_mask

    def _build_timexer_exogenous_input(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        """
        为 TimeXer MS 模式构建扩展的外生变量输入。

        内生变量（最后一列）：使用 seq_len 长度的历史数据
        外生变量（其余列）：使用 seq_len 历史 + pred_len 已知未来（从 batch_y 获取）

        Args:
            batch_x: [B, seq_len, N] - encoder 输入
            batch_x_mark: [B, seq_len, mark_dim] - encoder 时间特征
            batch_y: [B, label_len+pred_len, N] - decoder 输入（包含未来真值）
            batch_y_mark: [B, label_len+pred_len, mark_dim] - decoder 时间特征

        Returns:
            x_enc_ex: [B, seq_len+pred_len, N-1] 扩展的外生变量输入，或 None（非 TimeXer MS 模式）
            x_mark_ex: [B, seq_len+pred_len, mark_dim] 扩展的时间特征，或 None
        """
        if getattr(self.args, 'model', None) != 'TimeXer':
            return None, None
        if getattr(self.args, 'features', None) != 'MS':
            return None, None

        pred_len = int(getattr(self.args, 'pred_len', 0) or 0)
        if pred_len <= 0:
            return None, None

        # batch_x: [B, seq_len, N]，外生变量是前 N-1 列
        exo_hist = batch_x[:, :, :-1]  # [B, seq_len, N-1]

        # batch_y: [B, label_len+pred_len, N]，未来外生变量在最后 pred_len 行
        exo_future = batch_y[:, -pred_len:, :-1]  # [B, pred_len, N-1]

        # 拼接外生变量：[B, seq_len+pred_len, N-1]
        x_enc_ex = torch.cat([exo_hist, exo_future], dim=1)

        # 拼接时间特征：[B, seq_len+pred_len, mark_dim]
        x_mark_ex = torch.cat([batch_x_mark, batch_y_mark[:, -pred_len:, :]], dim=1)

        return x_enc_ex, x_mark_ex

    def _build_crossformer_decoder_input_and_mask(self, batch_y):
        """
        Crossformer decoder 输入构造（自定义）：
        - 输入长度：label_len + pred_len
        - 外生变量（non-target）：未来 pred_len 段使用真实值
        - 内生变量（target=loss_target_indices）：未来 pred_len 段用 label_len 段均值作为占位符

        Returns:
            dec_inp: [B, label_len+pred_len, N]
            dec_mask: [B, pred_len, N]，True 表示占位符位置（仅 target 变量在 pred 段为 True）
        """
        label_len = int(getattr(self.args, 'label_len', 0) or 0)
        pred_len = int(getattr(self.args, 'pred_len', 0) or 0)
        if pred_len <= 0:
            dec_inp = batch_y[:, :label_len, :].clone().float()
            B, _, N = dec_inp.shape
            dec_mask = torch.zeros(B, 0, N, dtype=torch.bool, device=dec_inp.device)
            return dec_inp, dec_mask

        dec_inp = batch_y[:, : label_len + pred_len, :].clone().float()
        idx_list = getattr(self.args, 'loss_target_indices', None)
        if not idx_list:
            # 兜底：没有 target 索引时，退化为 TSL 默认逻辑（pred 段全 0），并且不产生 mask
            dec_inp[:, -pred_len:, :] = 0.0
            return dec_inp, None

        idx = torch.as_tensor(idx_list, device=dec_inp.device, dtype=torch.long)
        if label_len > 0:
            hist_mean = dec_inp[:, :label_len, :].index_select(dim=-1, index=idx).mean(dim=1, keepdim=True)
        else:
            hist_mean = torch.zeros(dec_inp.shape[0], 1, int(idx.numel()), device=dec_inp.device, dtype=dec_inp.dtype)
        dec_inp[:, -pred_len:, idx] = hist_mean.repeat(1, pred_len, 1)

        B, _, N = dec_inp.shape
        dec_mask = torch.zeros(B, pred_len, N, dtype=torch.bool, device=dec_inp.device)
        dec_mask[:, :, idx] = True
        return dec_inp, dec_mask

    def _build_decoder_input(self, batch_y):
        """
        构造 decoder 输入（dec_inp）。

        重要：为避免对 encoder-only 模型（如 TimeXer / iTransformer）产生任何潜在副作用，
        这里将 Crossformer 的 decoder 输入特殊逻辑仅限定给 Crossformer。
        其它模型走 Time-Series-Library 默认做法：label 段喂真值 + pred 段全 0。
        """
        label_len = int(getattr(self.args, 'label_len', 0) or 0)
        pred_len = int(getattr(self.args, 'pred_len', 0) or 0)
        model_name = getattr(self.args, 'model', None)

        if pred_len <= 0:
            # 兜底：没有 pred_len 时，直接喂 label 段
            return batch_y[:, :label_len, :].clone().float()

        # Crossformer 特殊需求：
        # - non-target（外生变量）：输入 label_len+pred_len 真值；
        # - target（内生变量）：输入 label_len 真值 + pred_len 用 label 段均值占位符
        if model_name == 'Crossformer' and getattr(self.args, 'loss_target_indices', None):
            dec_inp, _ = self._build_crossformer_decoder_input_and_mask(batch_y)
            return dec_inp

        # 默认（原版 TSL）：label 段真值 + pred 段全 0
        dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        return torch.cat([batch_y[:, :label_len, :].float(), dec_zeros], dim=1)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _compute_time_weights(self, batch_idx, train_series_len, device):
        if not getattr(self.args, 'loss_time_weighted', False):
            return None
        if batch_idx is None or train_series_len is None:
            return None
        seq_len = int(getattr(self.args, 'seq_len', 0) or 0)
        pred_len = int(getattr(self.args, 'pred_len', 0) or 0)
        if seq_len <= 0 or pred_len <= 0:
            return None
        if not torch.is_tensor(batch_idx):
            batch_idx = torch.as_tensor(batch_idx, device=device, dtype=torch.long)
        else:
            batch_idx = batch_idx.to(device=device, dtype=torch.long)
        distance = train_series_len - (batch_idx + seq_len + pred_len)
        distance = torch.clamp(distance, min=0).float()
        decay = float(getattr(self.args, 'loss_time_weight_decay', 0.0) or 0.0)
        weights = torch.exp(-decay * distance)
        min_weight = float(getattr(self.args, 'loss_time_weight_min', 0.0) or 0.0)
        if min_weight > 0.0:
            weights = torch.clamp(weights, min=min_weight)
        return weights

    def _compute_loss_with_optional_time_weight(self, outputs, batch_y, criterion, batch_idx=None, train_series_len=None):
        if getattr(self.args, 'loss_target_indices', None):
            idx = torch.as_tensor(self.args.loss_target_indices, device=outputs.device, dtype=torch.long)
            outputs = torch.index_select(outputs, dim=-1, index=idx)
            batch_y = torch.index_select(batch_y, dim=-1, index=idx)
        weights = self._compute_time_weights(batch_idx, train_series_len, outputs.device)
        if weights is None:
            base_loss = criterion(outputs, batch_y)
        else:
            diff = outputs - batch_y
            if diff.dim() == 3:
                per_sample = diff.pow(2).mean(dim=(1, 2))
            elif diff.dim() == 2:
                per_sample = diff.pow(2).mean(dim=1)
            else:
                per_sample = diff.pow(2).view(diff.shape[0], -1).mean(dim=1)
            denom = weights.sum()
            base_loss = per_sample.mean() if denom.item() <= 0 else (per_sample * weights).sum() / denom

        trend_weight = float(getattr(self.args, 'loss_trend_weight', 0.0) or 0.0)
        endpoint_weight = float(getattr(self.args, 'loss_endpoint_weight', 0.0) or 0.0)
        if trend_weight < 0.0 or endpoint_weight < 0.0:
            raise ValueError('loss_trend_weight and loss_endpoint_weight must be non-negative.')
        trend_loss = (outputs[:, -1, :] - outputs[:, 0, :] - (batch_y[:, -1, :] - batch_y[:, 0, :])).pow(2).mean()
        endpoint_loss = (outputs[:, -1, :] - batch_y[:, -1, :]).pow(2).mean()
        return base_loss + trend_weight * trend_loss + endpoint_weight * endpoint_loss
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # iTransformer: extend encoder input to seq_len + pred_len as requested
                batch_x, batch_x_mark, enc_mask = self._maybe_extend_itransformer_encoder_input(
                    batch_x, batch_x_mark, batch_y, batch_y_mark
                )

                # TimeXer MS: build extended exogenous input
                x_enc_ex, x_mark_ex = self._build_timexer_exogenous_input(
                    batch_x, batch_x_mark, batch_y, batch_y_mark
                )

                # decoder input
                dec_mask = None
                if getattr(self.args, 'model', None) == 'Crossformer' and getattr(self.args, 'loss_target_indices', None):
                    dec_inp, dec_mask = self._build_crossformer_decoder_input_and_mask(batch_y)
                    if not bool(getattr(self.args, 'use_dec_mask_embedding', True)):
                        dec_mask = None
                else:
                    dec_inp = self._build_decoder_input(batch_y)
                dec_inp = dec_inp.to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if x_enc_ex is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 x_enc_ex=x_enc_ex, x_mark_ex=x_mark_ex)
                        elif enc_mask is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 enc_mask=enc_mask)
                        else:
                            if dec_mask is not None:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dec_mask=dec_mask)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if x_enc_ex is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             x_enc_ex=x_enc_ex, x_mark_ex=x_mark_ex)
                    elif enc_mask is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             enc_mask=enc_mask)
                    else:
                        if dec_mask is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dec_mask=dec_mask)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = self._compute_loss_with_optional_time_weight(pred, true, criterion)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_series_len = len(getattr(train_data, 'data_x', [])) or None

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        results_path = os.path.join('./results', setting)
        os.makedirs(results_path, exist_ok=True)
        loss_history = []

        def save_loss_artifacts():
            history_path = os.path.join(results_path, 'loss_history.csv')
            try:
                with open(history_path, 'w', newline='') as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=['epoch', 'train_loss', 'vali_loss', 'test_loss']
                    )
                    writer.writeheader()
                    for row in loss_history:
                        writer.writerow(row)
                print("loss history saved to {}".format(history_path), flush=True)
            except Exception as e:
                print("warning: failed to save loss history: {}".format(e), flush=True)

            if not loss_history:
                return
            plot_path = os.path.join(results_path, 'loss_history.png')
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                epochs = [row['epoch'] for row in loss_history]
                plt.figure(figsize=(8, 4.5))
                plt.plot(epochs, [row['train_loss'] for row in loss_history], marker='o', label='train')
                plt.plot(epochs, [row['vali_loss'] for row in loss_history], marker='o', label='val')
                plt.plot(epochs, [row['test_loss'] for row in loss_history], marker='o', label='test')
                plt.xlabel('epoch')
                plt.ylabel('Training objective')
                plt.title('Training objective history')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print("loss curve saved to {}".format(plot_path), flush=True)
            except Exception as e:
                print("warning: failed to save loss curve: {}".format(e), flush=True)

        resume_state_path = os.path.join(path, 'resume_state.pth')

        def save_resume_state(epoch_done):
            state = {
                'epoch': int(epoch_done),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'loss_history': loss_history,
                'early_stopping': {
                    'counter': early_stopping.counter,
                    'best_score': None if early_stopping.best_score is None else float(early_stopping.best_score),
                    'early_stop': early_stopping.early_stop,
                    'val_loss_min': float(early_stopping.val_loss_min),
                },
            }
            if self.args.use_amp:
                state['amp_scaler_state_dict'] = scaler.state_dict()
            torch.save(state, resume_state_path)

        start_epoch = 0
        if getattr(self.args, 'resume', False):
            resume_path = getattr(self.args, 'resume_checkpoint', '') or resume_state_path
            if not os.path.exists(resume_path):
                raise FileNotFoundError('resume checkpoint not found: {}'.format(resume_path))
            checkpoint = torch.load(resume_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            if 'optimizer_state_dict' in checkpoint:
                model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.args.use_amp and 'amp_scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
            start_epoch = int(checkpoint.get('epoch', 0))
            loss_history = list(checkpoint.get('loss_history', []))
            early_state = checkpoint.get('early_stopping', {})
            early_stopping.counter = int(early_state.get('counter', early_stopping.counter))
            early_stopping.best_score = early_state.get('best_score', early_stopping.best_score)
            early_stopping.early_stop = bool(early_state.get('early_stop', False))
            early_stopping.val_loss_min = early_state.get('val_loss_min', early_stopping.val_loss_min)
            print('Resumed from {} at completed epoch {}; next epoch {}'.format(
                resume_path, start_epoch, start_epoch + 1
            ), flush=True)

        print_every = max(0, int(getattr(self.args, 'print_every', 100) or 0))
        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_idx = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_idx = None
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # iTransformer: extend encoder input to seq_len + pred_len as requested
                batch_x, batch_x_mark, enc_mask = self._maybe_extend_itransformer_encoder_input(
                    batch_x, batch_x_mark, batch_y, batch_y_mark
                )

                # TimeXer MS: build extended exogenous input
                x_enc_ex, x_mark_ex = self._build_timexer_exogenous_input(
                    batch_x, batch_x_mark, batch_y, batch_y_mark
                )

                # decoder input
                dec_mask = None
                if getattr(self.args, 'model', None) == 'Crossformer' and getattr(self.args, 'loss_target_indices', None):
                    dec_inp, dec_mask = self._build_crossformer_decoder_input_and_mask(batch_y)
                    if not bool(getattr(self.args, 'use_dec_mask_embedding', True)):
                        dec_mask = None
                else:
                    dec_inp = self._build_decoder_input(batch_y)
                dec_inp = dec_inp.to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if x_enc_ex is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 x_enc_ex=x_enc_ex, x_mark_ex=x_mark_ex)
                        elif enc_mask is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 enc_mask=enc_mask)
                        else:
                            if dec_mask is not None:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dec_mask=dec_mask)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = self._compute_loss_with_optional_time_weight(
                            outputs, batch_y, criterion, batch_idx=batch_idx, train_series_len=train_series_len
                        )
                        train_loss.append(loss.item())
                else:
                    if x_enc_ex is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             x_enc_ex=x_enc_ex, x_mark_ex=x_mark_ex)
                    elif enc_mask is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             enc_mask=enc_mask)
                    else:
                        if dec_mask is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dec_mask=dec_mask)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._compute_loss_with_optional_time_weight(
                        outputs, batch_y, criterion, batch_idx=batch_idx, train_series_len=train_series_len
                    )
                    train_loss.append(loss.item())

                if print_every > 0 and ((i + 1) % print_every == 0 or i == 0 or (i + 1) == train_steps):
                    progress = 100.0 * (i + 1) / max(train_steps, 1)
                    print("\tepoch: {0}/{1} | batch: {2}/{3} ({4:.1f}%) | loss: {5:.7f}".format(
                        epoch + 1, self.args.train_epochs, i + 1, train_steps, progress, loss.item()
                    ), flush=True)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time), flush=True)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if float(getattr(self.args, 'grad_clip_norm', 0.0) or 0.0) > 0:
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.args.grad_clip_norm))
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if float(getattr(self.args, 'grad_clip_norm', 0.0) or 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.args.grad_clip_norm))
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            loss_history.append(
                {
                    "epoch": int(epoch + 1),
                    "train_loss": float(train_loss),
                    "vali_loss": float(vali_loss),
                    "test_loss": float(test_loss),
                }
            )
            save_loss_artifacts()
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                save_resume_state(epoch + 1)
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            save_resume_state(epoch + 1)

        save_loss_artifacts()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # iTransformer: extend encoder input to seq_len + pred_len as requested
                batch_x, batch_x_mark, enc_mask = self._maybe_extend_itransformer_encoder_input(
                    batch_x, batch_x_mark, batch_y, batch_y_mark
                )

                # TimeXer MS: build extended exogenous input
                x_enc_ex, x_mark_ex = self._build_timexer_exogenous_input(
                    batch_x, batch_x_mark, batch_y, batch_y_mark
                )

                # decoder input
                dec_mask = None
                if getattr(self.args, 'model', None) == 'Crossformer' and getattr(self.args, 'loss_target_indices', None):
                    dec_inp, dec_mask = self._build_crossformer_decoder_input_and_mask(batch_y)
                    if not bool(getattr(self.args, 'use_dec_mask_embedding', True)):
                        dec_mask = None
                else:
                    dec_inp = self._build_decoder_input(batch_y)
                dec_inp = dec_inp.to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if x_enc_ex is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 x_enc_ex=x_enc_ex, x_mark_ex=x_mark_ex)
                        elif enc_mask is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                 enc_mask=enc_mask)
                        else:
                            if dec_mask is not None:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dec_mask=dec_mask)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if x_enc_ex is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             x_enc_ex=x_enc_ex, x_mark_ex=x_mark_ex)
                    elif enc_mask is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             enc_mask=enc_mask)
                    else:
                        if dec_mask is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, dec_mask=dec_mask)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
