"""Standalone FP32 history-only forecasting; never emits control commands."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from short_horizon_policy import apply_policy


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_history(frame, features, as_of):
    if len(frame) != 120 or not frame.columns.is_unique:
        raise ValueError('Expected exactly 120 rows with unique column names')
    if 'timestamp' not in frame.columns:
        raise ValueError('Missing timestamp column')
    extra = set(frame.columns) - set(features) - {'timestamp'}
    if extra:
        raise ValueError('Unexpected columns: ' + ','.join(sorted(extra)))
    times = pd.to_datetime(frame['timestamp'], errors='raise')
    if times.isna().any() or not times.diff().iloc[1:].eq(pd.Timedelta(seconds=10)).all():
        raise ValueError('History must be a strictly increasing, gap-free 10-second grid')
    origin = pd.Timestamp(as_of)
    if times.iloc[-1] != origin:
        raise ValueError('Last history timestamp must equal the declared forecast origin (as_of)')
    values = frame.reindex(columns=features).apply(pd.to_numeric, errors='raise').to_numpy(dtype='float32')
    return values, origin


class HistoryOnlyPredictor:
    def __init__(self, bundle, device='cpu'):
        import torch

        self.torch = torch
        self.bundle = Path(bundle).resolve()
        self.manifest = json.loads((self.bundle / 'manifest.json').read_text(encoding='utf-8'))
        if not self.manifest.get('history_only') or self.manifest.get('usage') != 'offline_shadow_candidate':
            raise ValueError('Not an approved history-only offline candidate contract')
        required = ['checkpoint.pth', 'metadata.json', 'scaler.json', 'model_config.json', 'policy.json']
        if any(n not in self.manifest['files'] for n in required):
            raise ValueError('Manifest is missing required assets')
        for name, digest in self.manifest['files'].items():
            path = (self.bundle / name).resolve()
            if not path.is_relative_to(self.bundle) or sha256_file(path) != digest:
                raise ValueError('Bundle file hash mismatch or unsafe path: ' + name)
        self.meta = json.loads((self.bundle / 'metadata.json').read_text(encoding='utf-8'))
        scaler = json.loads((self.bundle / 'scaler.json').read_text(encoding='utf-8'))
        config = json.loads((self.bundle / 'model_config.json').read_text(encoding='utf-8'))
        self.policy = json.loads((self.bundle / 'policy.json').read_text(encoding='utf-8'))
        self.features, self.targets = self.meta['feature_columns'], self.meta['target_columns']
        self.inds = self.meta['target_indices']
        if (self.features != scaler['columns'] or len(set(self.features)) != len(self.features)
                or len(self.targets) != 14 or len(set(self.inds)) != 14
                or [self.features[i] for i in self.inds] != self.targets
                or (self.meta['seq_len'], self.meta['pred_len'], self.meta['label_len'], self.meta['freq_seconds']) != (120, 60, 60, 10)
                or config['enc_in'] != len(self.features)):
            raise ValueError('Metadata, scaler, model configuration or target order mismatch')
        self.mean, self.scale = np.asarray(scaler['mean']), np.asarray(scaler['scale'])
        if (self.mean.shape != (len(self.features),) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any()):
            raise ValueError('Invalid scaler')
        tslib = self.bundle / 'tslib'
        # TSLib uses absolute models/layers/utils imports. Refuse namespace collisions.
        for namespace in ['models', 'layers', 'utils']:
            loaded = sys.modules.get(namespace)
            if loaded is not None:
                location = Path(getattr(loaded, '__file__', '') or '.').resolve()
                if not location.is_relative_to(tslib):
                    raise RuntimeError('TSLib namespace collision; run this candidate in a separate Python process')
        sys.path.insert(0, str(tslib))
        from models.Crossformer import Model
        self.device = torch.device(device)
        self.model = Model(SimpleNamespace(**config))
        state = torch.load(self.bundle / 'checkpoint.pth', map_location='cpu', weights_only=True)
        if not all(torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('Nonfinite checkpoint parameters')
        self.model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.model.to(self.device).eval()

    def predict(self, frame, *, as_of):
        start = time.perf_counter()
        values, origin = validate_history(frame, self.features, as_of)
        finite = np.isfinite(values)
        missing_rates = 1. - finite.mean(axis=0)
        warnings = []
        if not finite.all():
            warnings.append('Missing/invalid inputs mapped to training mean; information is not recovered')
        coverage = float(finite.mean())
        z = np.nan_to_num((values - self.mean) / self.scale, nan=0., posinf=0., neginf=0.).astype('float32')
        extreme = int((np.abs(z) > 50).sum())
        if extreme:
            warnings.append('Some standardized inputs exceed 50; inspect upstream signal quality')
        raw = np.full((60, len(self.targets)), np.nan)
        if coverage >= .5:
            torch = self.torch
            with torch.inference_mode():
                x = torch.from_numpy(z[None]).to(self.device)
                future = x[:, -1:, :].repeat(1, 60, 1)
                future[:, :, self.inds] = x[:, -60:, self.inds].mean(dim=1, keepdim=True)
                dec = torch.cat([x[:, -60:, :], future], dim=1)
                mask = torch.zeros(1, 60, len(self.features), dtype=torch.bool, device=self.device)
                mask[:, :, self.inds] = True
                output = self.model(x, None, dec, None, dec_mask=mask).cpu().numpy()[0]
                raw = output[:, self.inds].astype('float64') * self.scale[self.inds] + self.mean[self.inds]
        else:
            warnings.append('Less than 50% input cells available; neural prediction skipped')
        final = raw.copy()
        status, reasons = {}, {}
        for j, (code, index) in enumerate(zip(self.targets, self.inds)):
            if not finite[-1, index]:
                final[:, j] = np.nan
                status[code], reasons[code] = 'unavailable', 'current_target_missing'
            elif coverage < .5 or missing_rates[index] > .2 or not np.isfinite(raw[:, j]).all():
                final[:, j] = values[-1, index]
                status[code] = 'degraded_persistence'
                reasons[code] = 'insufficient_inputs_or_target_history_or_nonfinite_model_output'
            else:
                status[code], reasons[code] = 'model', 'original_history_only_prediction'
        gate = 0.
        j, index = self.targets.index('B3TE03'), self.features.index('B3TE03')
        if status['B3TE03'] == 'model':
            corrected, gates = apply_policy(values[None, :, index], raw[None, :, j], self.policy)
            final[:, j], gate = corrected[0], float(gates[0])
            if gate > 0:
                status['B3TE03'], reasons['B3TE03'] = 'short_horizon_corrected', 'stable_observed_temperature_history'
        def clean(array):
            return [[float(v) if np.isfinite(v) else None for v in row] for row in array]
        return dict(model_version=self.manifest['version'], forecast_origin=origin.isoformat(),
                    usage='offline_shadow_candidate', usable_for_control=False, history_only=True,
                    target_columns=self.targets, step_seconds=10,
                    timestamps=[(origin + pd.Timedelta(seconds=10 * (h + 1))).isoformat() for h in range(60)],
                    prediction=clean(final), raw_prediction=clean(raw), target_status=status, target_reason=reasons,
                    preheater_gate=gate, quality=dict(observed_cell_fraction=coverage,
                        missing_input_fraction={c: float(r) for c, r in zip(self.features, missing_rates) if r > 0},
                        extreme_standardized_cells=extreme, warnings=warnings,
                        preprocessing='caller-provided cleaned grid; no interpolation or raw quality-rule relaxation'),
                    inference_seconds=time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description='History-only model1 offline candidate')
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--as-of', required=True, help='Declared prediction origin, equal to last history timestamp')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing output: ' + str(args.output))
    with args.history.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream))
        if len(header) != len(set(header)):
            raise ValueError('Duplicate CSV column names')
    import torch
    torch.set_num_threads(4)
    predictor = HistoryOnlyPredictor(args.bundle, args.device)
    result = predictor.predict(pd.read_csv(args.history), as_of=args.as_of)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(version=result['model_version'], origin=result['forecast_origin'],
                         target_status=result['target_status'], preheater_gate=result['preheater_gate'],
                         inference_seconds=result['inference_seconds']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
