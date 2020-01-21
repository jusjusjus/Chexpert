#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('model', type=str, help="Path to the trained models")
parser.add_argument('dataset', type=str,
                    help="Path to the input image path in csv")
parser.add_argument('--out_csv_path', type=str, default='test/test.csv',
                    help="Path to the ouput predictions in csv")
parser.add_argument('--num_workers', type=int, default=8,
                    help="Number of workers for each data loader")
parser.add_argument('--device_ids', type=str, default='0',
                    help="GPU indices comma separated, e.g. '0,1'")
parser.add_argument('--cpu', action='store_true', help="Use cpu")
parser.add_argument('--ignore-missing-files', action='store_true',
                    help="Ignore missing files in the test set")
parser.add_argument('--batch-size', type=int, default=16,
                    help="Number of examples to process per batch")
opt = parser.parse_args()
device_ids = list(map(int, opt.device_ids.split(',')))

from os import makedirs
from os.path import dirname, abspath, join
from sys import path
path.insert(0, '.')

import json
import time
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa


makedirs('test', exist_ok=True)


def get_pred(output, cfg):
    if cfg.criterion in ('BCE', 'FL'):
        assert all(n == 1 for n in cfg.num_classes)
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    elif cfg.criterion == 'CE':
        assert all(n >= 1 for n in cfg.num_classes)
        prob = F.softmax(output)
        pred = prob[:, 1].cpu().detach().numpy()
    else:
        raise ValueError(f"Unknown criterion: {{cfg.criterion}}")

    return pred


def test_epoch(cfg, args, model, dataloader, out_csv_path):
    torch.set_grad_enabled(False)
    model.eval()
    device = next(model.parameters()).device
    num_tasks = len(cfg.num_classes)

    csv_header = [
        'Path',
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Atelectasis',
        'Pleural Effusion']

    # Predict and write probabilities to `out_csv_path`

    with open(out_csv_path, 'w') as f:
        f.write(','.join(csv_header) + '\n')
        for step, (images, paths) in enumerate(dataloader):
            print(f"Processing batch {step + 1} of {len(dataloader)}")
            images = images.to(device)
            output, _ = model(images)
            batch_size = len(paths)
            pred = np.zeros((num_tasks, batch_size))

            for j in range(num_tasks):
                pred[j] = get_pred(output[j], cfg)

            for i in range(batch_size):
                batch = ','.join(map(str, pred[:, i]))
                result = paths[i] + ',' + batch
                f.write(result + '\n')


def build_model(cfg, paramsfile, device):
    model = Classifier(cfg)
    model = DataParallel(model, device_ids=device_ids).to(device).eval()
    ckpt = torch.load(paramsfile, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.module.load_state_dict(state_dict)
    if 'step' in ckpt and 'auc_dev_best' in ckpt:
        print(f"Using model '{paramsfile}' at step: {ckpt['step']} "
              f"with AUC: {ckpt['auc_dev_best']}")
    return model

with open(join(opt.model, 'cfg.json'), 'r') as fp:
    cfg = edict(json.load(fp))

if not opt.cpu:
    num_devices = torch.cuda.device_count()
    assert num_devices >= len(device_ids), f"""
    #available gpu : {num_devices} < --device_ids : {len(device_ids)}"""
    device = torch.device(f"cuda:{device_ids[0]}")
else:
    num_devices = 0
    device = torch.device('cpu')

model = build_model(cfg, join(opt.model, 'best.ckpt'), device)
dataset = ImageDataset(opt.dataset, cfg, mode='test')
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                        drop_last=False, num_workers=opt.num_workers)

test_epoch(cfg, opt, model, dataloader, opt.out_csv_path)
