#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('model', type=str, help="Path to the trained models")
parser.add_argument('dataset', type=str,
                    help="Path to the input image path in csv")
parser.add_argument('--output', type=str, default="cache/gradcam",
                    help="Output folder")
opt = parser.parse_args()

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
import cv2

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa
from grad_cam import GradCAM, save_gradcam


makedirs(opt.output, exist_ok=True)

def imread(imagepath, dsize=(512, 512)):
    img = cv2.imread(imagepath, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    return img

def build_model(cfg, paramsfile):
    model = Classifier(cfg)
    model = model.to('cpu')
    ckpt = torch.load(paramsfile, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    if 'step' in ckpt and 'auc_dev_best' in ckpt:
        print(f"Using model '{paramsfile}' at step: {ckpt['step']} "
              f"with AUC: {ckpt['auc_dev_best']}")
    return model.eval()


with open(join(opt.model, 'cfg.json'), 'r') as fp:
    cfg = edict(json.load(fp))

classes = ['Cardiomegaly', 'Edema', 'Consolidation',
          'Atelectasis', 'Pleural Effusion']

num_tasks = len(cfg.num_classes)

model = build_model(cfg, join(opt.model, 'best.ckpt'))
dataset = ImageDataset(opt.dataset, cfg, mode='heatmap')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

gcam = GradCAM(model=model)
for images, paths, labels in dataloader:
    break

probs, ids = gcam.forward(images)
layer = 'backbone.features.denseblock4.denselayer15'
for i in range(ids.shape[1]):
    gcam.backward(ids[:, [i]])
    regions = gcam.generate(target_layer=layer)
    for j in range(len(images)):
        print(f"#{j}: {classes[ids[j, i]]} ({probs[j, i]:.5f})")
        # Grad-CAM
        c, p, t = classes[ids[j, i]], 100*probs[j,i], bool(labels[j,i])
        imagepath = join(opt.output, f"{c}-{p:.0f}-{t}.png")
        raw_image = imread(paths[j])
        save_gradcam(filename=imagepath, gcam=regions[j, 0], raw_image=raw_image) 
