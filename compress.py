import warnings
import argparse
import yaml
import copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune

warnings.filterwarnings('ignore')


def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path


def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    trainer.train()


if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '',
        'data': '',
        'imgsz': 640,
        'epochs': 800,
        'batch': 32,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'project': '',
        'name': '',


        'prune_method': '',
        'global_pruning': False,
        'speed_up': 3,
        'reg': 0.02,
        'reg_decay': 0.05,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': r''
    }

    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)
