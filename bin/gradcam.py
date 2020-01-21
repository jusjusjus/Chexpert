#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('model', type=str, help="Path to the trained models")
parser.add_argument('dataset', type=str,
                    help="Path to the input image path in csv")
parser.add_argument('--output', type=str, default="cache/gradcam",
                    help="Output folder")
parser.add_argument('--batch-size', type=int, default=1)
opt = parser.parse_args()

from os import makedirs
from os.path import dirname, abspath, join, splitext
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
import matplotlib.pyplot as plt

from data.dataset import ImageDataset  # noqa
from model.classifier import Classifier  # noqa
from grad_cam import GradCAM, combine_image_and_gcam


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
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                        drop_last=False)

gcam = GradCAM(model=model)
layer = 'backbone.features.denseblock4.denselayer15'

for images, paths, labels in dataloader:
    probs, ids = gcam.forward(images)
    processed_images = [[] for _ in range(len(images))]
    for i in range(ids.shape[1]):
        gcam.backward(ids[:, [i]])
        regions = gcam.generate(target_layer=layer)
        for j in range(len(images)):
            print(f"#{j}: {classes[ids[j, i]]} ({probs[j, i]:.5f})")
            # Grad-CAM
            raw_image = imread(paths[j])
            combined = combine_image_and_gcam(regions[j, 0], raw_image)
            processed_images[j].append(combined.astype(np.uint8))

    for j, (image_list, path) in enumerate(zip(processed_images, paths)):
        plt.figure(figsize=(16, 4))
        for i, image in enumerate(image_list):
            plt.subplot(1, len(image_list), i + 1, xticks=[], yticks=[],
                        frameon=False)

            c, p, t = classes[ids[j, i]], 100 * probs[j, i], bool(labels[j, i])
            plt.title(f"{c} {p:.0f}% ({t})", fontsize=10)
            plt.imshow(image)

        plt.tight_layout()

        filename = '-'.join(path.split('/')[-3:])
        filename = splitext(filename)[0] + '.png'
        plt.savefig(join(opt.output, filename))
        plt.clf()
