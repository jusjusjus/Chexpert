#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--model_path', default='./', metavar='MODEL_PATH',
                    type=str, help="Path to the trained models")
parser.add_argument('--in_csv_path', default='dev.csv', metavar='IN_CSV_PATH',
                    type=str, help="Path to the input image path in csv")
parser.add_argument('--out_csv_path', default='test/test.csv',
                    metavar='OUT_CSV_PATH', type=str,
                    help="Path to the ouput predictions in csv")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0', type=str, help="GPU indices "
                    "comma separated, e.g. '0,1' ")
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
    if cfg.criterion == 'BCE' or cfg.criterion == "FL":
        for num_class in cfg.num_classes:
            assert num_class == 1
        pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
    elif cfg.criterion == 'CE':
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output)
        pred = prob[:, 1].cpu().detach().numpy()
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return pred


def test_epoch(cfg, args, model, dataloader, out_csv_path):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    test_header = [
        'Path',
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Atelectasis',
        'Pleural Effusion']

    with open(out_csv_path, 'w') as f:
        f.write(','.join(test_header) + '\n')
        for step in range(steps):
            image, path = next(dataiter)
            image = image.to(device)
            output, __ = model(image)
            batch_size = len(path)
            pred = np.zeros((num_tasks, batch_size))

            for i in range(num_tasks):
                pred[i] = get_pred(output[i], cfg)

            for i in range(batch_size):
                batch = ','.join(map(lambda x: '{}'.format(x), pred[:, i]))
                result = path[i] + ',' + batch
                f.write(result + '\n')
                print(f"Image: {path[i]}, Prob: {batch}")



with open(join(opt.model_path, 'cfg.json'), 'r') as fp:
    cfg = edict(json.load(fp))

num_devices = torch.cuda.device_count()
if num_devices < len(device_ids):
    raise Exception(
        '#available gpu : {} < --device_ids : {}'
        .format(num_devices, len(device_ids)))
device = torch.device('cuda:{}'.format(device_ids[0]))

model = Classifier(cfg)
model = DataParallel(model, device_ids=device_ids).to(device).eval()
ckpt_path = join(opt.model_path, 'best.ckpt')
ckpt = torch.load(ckpt_path, map_location=device)
model.module.load_state_dict(ckpt['state_dict'])

dataloader = DataLoader(
    ImageDataset(opt.in_csv_path, cfg, mode='test'),
    batch_size=cfg.dev_batch_size, num_workers=opt.num_workers,
    drop_last=False, shuffle=False)

test_epoch(cfg, opt, model, dataloader, opt.out_csv_path)

print(f"Save best is step:", ckpt['step'], 'AUC :', ckpt['auc_dev_best'])
