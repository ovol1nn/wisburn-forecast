import importlib
import os

import torch


MODEL_MODULES = {
    'TimesNet': 'TimesNet',
    'Autoformer': 'Autoformer',
    'Transformer': 'Transformer',
    'Nonstationary_Transformer': 'Nonstationary_Transformer',
    'DLinear': 'DLinear',
    'FEDformer': 'FEDformer',
    'Informer': 'Informer',
    'LightTS': 'LightTS',
    'Reformer': 'Reformer',
    'ETSformer': 'ETSformer',
    'PatchTST': 'PatchTST',
    'Pyraformer': 'Pyraformer',
    'MICN': 'MICN',
    'Crossformer': 'Crossformer',
    'FiLM': 'FiLM',
    'iTransformer': 'iTransformer',
    'Koopa': 'Koopa',
    'TiDE': 'TiDE',
    'FreTS': 'FreTS',
    'MambaSimple': 'Mamba',
    'Mamba': 'Mamba',
    'TimeMixer': 'TimeMixer',
    'TSMixer': 'TSMixer',
    'SegRNN': 'SegRNN',
    'TemporalFusionTransformer': 'TemporalFusionTransformer',
    'SCINet': 'SCINet',
    'PAttn': 'PAttn',
    'TimeXer': 'TimeXer',
    'WPMixer': 'WPMixer',
    'MultiPatchFormer': 'MultiPatchFormer',
    'KANAD': 'KANAD',
    'MSGNet': 'MSGNet',
    'TimeFilter': 'TimeFilter',
    'Sundial': 'Sundial',
    'TimeMoE': 'TimeMoE',
    'Chronos': 'Chronos',
    'Moirai': 'Moirai',
    'TiRex': 'TiRex',
    'TimesFM': 'TimesFM',
    'Chronos2': 'Chronos2',
}


def load_model_module(model_name):
    module_name = MODEL_MODULES.get(model_name)
    if module_name is None:
        raise ValueError(f'Unknown model: {model_name}')
    if model_name in {'Mamba', 'MambaSimple'}:
        print('Please make sure you have successfully installed mamba_ssm')
    return importlib.import_module(f'models.{module_name}')


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {args.model: load_model_module(args.model)}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda' and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass