#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', type=str, default=None,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', type=str, default=None,
                    help="Path to the saved models")
parser.add_argument('--num_workers', type=int, default=3,
                    help="Number of workers for each data loader")
parser.add_argument('--device_ids', type=str, default='0,1',
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train', type=str, default=None,
                    help="If get parameters from pretrained model")
parser.add_argument('--resume', type=int, default=0,
                    help="If resume from previous run")
parser.add_argument('--verbose', action='store_true', help="Detail info")
args = parser.parse_args()

import json
import subprocess
from os import makedirs
from os.path import dirname, abspath, join, exists
from sys import path
path.append(join(dirname(abspath(__file__)), '..'))
from time import time
from shutil import copyfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from sklearn import metrics
from easydict import EasyDict as edict

from tensorboardX import SummaryWriter

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # type: ignore

from data.dataset import ImageDataset  # noqa
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa
from model.classifier import Classifier  # noqa


def tostr(l, acc=3):
    fmt_str = "{:." + str(acc) + "f}"
    s = ' '.join(map(lambda x: fmt_str.format(x), l))
    return '(' + s + ')'


def get_loss(output, target, index, device, cfg):
    """return binary loss and accuracy between output and target

    The binary loss is computed from logits.  So, don't include a
    sigmoid layer in the network.

    Parameters
    ----------
        output:  network logits
        target:  binary ground-truth classes
        index:  column for which to compute the binary loss
        device:  device where to compute the logit
        cfg:  configuration namespace
    """
    assert cfg.criterion == 'BCE', f"Unknown criterion '{cfg.criterion}'"
    assert all(n == 1 for n in cfg.num_classes)
    target = target[:, index].view(-1)
    pos_weight = torch.from_numpy(np.array(cfg.pos_weight,
        dtype=np.float32)).to(device).type_as(target)
    if cfg.batch_weight:
        if target.sum() == 0:
            loss = torch.tensor(0., requires_grad=True).to(device)
        else:
            weight = (target.size()[0] - target.sum()) / target.sum()
            loss = F.binary_cross_entropy_with_logits(
                output[index].view(-1), target, pos_weight=weight)

    else:
        loss = F.binary_cross_entropy_with_logits(
            output[index].view(-1), target, pos_weight=pos_weight[index])

    label = torch.sigmoid(output[index].view(-1)).ge(0.5).float()
    acc = (target == label).float().sum() / len(label)

    return loss, acc


def train_epoch(summary, summary_dev, cfg, args, model, trainloader,
                testloader, optimizer, summary_writer, best_dict, dev_header):
    """"""
    torch.set_grad_enabled(True)
    model.train()

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device(f"cuda:{device_ids[0]}")
    label_header = trainloader.dataset._label_header
    num_tasks = len(cfg.num_classes)

    t0 = time()
    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    for step, (image, target) in enumerate(trainloader):

        # step is the step within this epoch, whereas `summary['step']` gives
        # us the global step of the algorithm

        image = image.to(device)
        target = target.to(device)
        output, logit_map = model(image)

        # Train network on batch.  Note that each class is implicitely weighted
        # using the array, `cfg.pos_weight`.

        loss = 0
        for t in range(num_tasks):
            loss_t, acc_t = get_loss(output, target, t, device, cfg)
            loss += loss_t
            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # The rest is testing and logging

        summary['step'] += 1

        if (step + 1) % cfg.log_every == 0:
            loss_sum /= cfg.log_every
            acc_sum /= cfg.log_every

            print(f"TRAIN (step {summary['step']}) > Loss: {tostr(loss_sum)},"
                    f" Acc: {tostr(acc_sum)}, {time() - t0:.2f} sec.")

            # global step
            gs = summary['step']
            for label, loss, acc in zip(label_header, loss_sum, acc_sum):
                summary_writer.add_scalar("train/loss_" + label, loss, gs)
                summary_writer.add_scalar("train/acc_" + label, acc, gs)

            loss_sum = np.zeros(num_tasks)
            acc_sum = np.zeros(num_tasks)

        if (step + 1) % cfg.test_every == 0:
            t0 = time()
            summary_dev, predlist, true_list = test_epoch(
                summary_dev, cfg, args, model, testloader)
            auclist = []
            for y_pred, y_true in zip(predlist, true_list):
                fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
                auclist.append(metrics.auc(fpr, tpr))

            summary_dev['auc'] = np.array(auclist)

            loss_dev_str = tostr(summary_dev['loss'])
            acc_dev_str = tostr(summary_dev['acc'])
            auc_dev_str = tostr(summary_dev['auc'])
            print(f"TEST  (step {summary['step']}) > Loss: {loss_dev_str}, "
                  f"Acc: {acc_dev_str}, Auc: {auc_dev_str}, "
                  f"Mean auc: {summary_dev['auc'].mean():.3f}, {time() - t0:.2f} sec.")

            # global step
            gs = summary['step']
            for label, loss, acc, auc in zip(dev_header, summary_dev['loss'],
                                     summary_dev['acc'], summary_dev['auc']):
                summary_writer.add_scalar('val/loss_' + label, loss, gs)
                summary_writer.add_scalar('val/acc_' + label, acc, gs)
                summary_writer.add_scalar('val/auc_' + label, auc, gs)

            save_best = False
            mean_acc = summary_dev['acc'][cfg.save_index].mean()
            if mean_acc >= best_dict['acc_dev_best']:
                best_dict['acc_dev_best'] = mean_acc
                if cfg.best_target == 'acc':
                    save_best = True

            mean_auc = summary_dev['auc'][cfg.save_index].mean()
            if mean_auc >= best_dict['auc_dev_best']:
                best_dict['auc_dev_best'] = mean_auc
                if cfg.best_target == 'auc':
                    save_best = True

            mean_loss = summary_dev['loss'][cfg.save_index].mean()
            if mean_loss <= best_dict['loss_dev_best']:
                best_dict['loss_dev_best'] = mean_loss
                if cfg.best_target == 'loss':
                    save_best = True

            if save_best:
                torch.save({
                    'epoch': summary['epoch'],
                     'step': summary['step'],
                     'acc_dev_best': best_dict['acc_dev_best'],
                     'auc_dev_best': best_dict['auc_dev_best'],
                     'loss_dev_best': best_dict['loss_dev_best'],
                     'state_dict': model.module.state_dict()
                 }, join(args.save_path, f"best_{best_dict['best_idx']}.ckpt"))
                print(f"BEST  (step {summary['step']}) > Loss: {loss_dev_str}, "
                    f"Acc: {acc_dev_str}, Auc: {auc_dev_str}, "
                    f"Best auc: {best_dict['auc_dev_best']:.3f}")
                best_dict['best_idx'] += 1
                if best_dict['best_idx'] > cfg.save_top_k:
                    best_dict['best_idx'] = 1

        model.train()
        torch.set_grad_enabled(True)

    summary['epoch'] += 1

    return summary, best_dict


def test_epoch(summary, cfg, args, model, dataloader):
    """"""
    torch.set_grad_enabled(False)
    model.eval()

    device_ids = list(map(int, args.device_ids.split(',')))
    device = torch.device(f"cuda:{device_ids[0]}")

    num_tasks = len(cfg.num_classes)
    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    predictions = [[] for _ in range(num_tasks)]
    targets_np = [[] for _ in range(num_tasks)]

    for step, (image, targets_th) in enumerate(dataloader):
        image = image.to(device)
        targets_th = targets_th.to(device)
        outputs, logit_map = model(image)
        for t, output in enumerate(outputs):
            output = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
            target = targets_th[:, t].view(-1).cpu().detach().numpy()
            predictions[t].append(output)
            targets_np[t].append(target)

            loss_t, acc_t = get_loss(outputs, targets_th, t, device, cfg)
            acc_sum[t] += acc_t.item()
            loss_sum[t] += loss_t.item()

    predictions = [np.concatenate(l) for l in predictions]
    targets_np = [np.concatenate(l) for l in targets_np]

    # This isn't 100% correct since the last batch can be smaller than the
    # rest, but we don't weight according to batchsize.

    summary['loss'] = loss_sum / len(dataloader)
    summary['acc'] = acc_sum / len(dataloader)

    return summary, predictions, targets_np


# Beginning of the script

print('Using args:')
print(args)

makedirs(args.save_path, exist_ok=True)
with open(args.cfg_path, 'r') as f:
    cfg = edict(json.load(f))
    if args.verbose is True:
        print(json.dumps(cfg, indent=4))

if not args.resume:
    with open(join(args.save_path, 'cfg.json'), 'w') as f:
        json.dump(cfg, f, indent=1)

device_ids = list(map(int, args.device_ids.split(',')))
num_devices = torch.cuda.device_count()
assert num_devices >= len(device_ids), f"""
#available gpu : {num_devices} < --device_ids : {len(device_ids)}"""

device = torch.device(f"cuda:{device_ids[0]}")

model = Classifier(cfg)
if args.verbose:
    from torchsummary import summary
    h, w = (cfg.long_side, cfg.long_side) if cfg.fix_ratio \
           else (cfg.height, cfg.width)
    summary(model.to(device), (3, h, w))

model = DataParallel(model, device_ids=device_ids).to(device)
if args.pre_train is not None:
    if exists(args.pre_train):
        ckpt = torch.load(args.pre_train, map_location=device)
        model.module.load_state_dict(ckpt)

optimizer = get_optimizer(model.parameters(), cfg)

trainset = ImageDataset(cfg.train_csv, cfg, mode='train')
testset = ImageDataset(cfg.dev_csv, cfg, mode='val')

trainloader = DataLoader(trainset, batch_size=cfg.train_batch_size,
    num_workers=args.num_workers, drop_last=True, shuffle=True)
testloader = DataLoader(testset, batch_size=cfg.dev_batch_size,
    num_workers=args.num_workers, drop_last=False, shuffle=False)

dev_header = testloader.dataset._label_header

# Initialize parameters to log training output

summary_train = {'epoch': 0, 'step': 0}
summary_dev = {'loss': float('inf'), 'acc': 0.0}
summary_writer = SummaryWriter(args.save_path)
epoch_start = 0
best_dict = {
    "acc_dev_best": 0.0,
    "auc_dev_best": 0.0,
    "loss_dev_best": float('inf'),
    "fused_dev_best": 0.0,
    "best_idx": 1
}

# Load checkpoint to resume from

if args.resume:
    ckpt_path = join(args.save_path, 'train.ckpt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.module.load_state_dict(ckpt['state_dict'])
    summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
    best_dict['acc_dev_best'] = ckpt['acc_dev_best']
    best_dict['loss_dev_best'] = ckpt['loss_dev_best']
    best_dict['auc_dev_best'] = ckpt['auc_dev_best']
    epoch_start = ckpt['epoch']

for epoch in range(epoch_start, cfg.epoch):
    print(f"Training in epoch {summary_train['epoch'] + 1}")

    # Update learning rate from Schedule.  Here we use a schedule that decays
    # the learning rate with a power of `cfg.lr_factor`.

    lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                     cfg.lr_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Train the model for on all minibatches in `trainloader`.  During the
    # training, the model is evaluated every `cfg.test_every` steps.

    summary_train, best_dict = train_epoch(
        summary_train, summary_dev, cfg, args, model,
        trainloader, testloader, optimizer,
        summary_writer, best_dict, dev_header)

    # Test the model on all minibatches of the testloader.

    t0 = time()
    summary_dev, predlist, true_list = test_epoch(
        summary_dev, cfg, args, model, testloader)

    # Using the test output, compute AUC statistics for each class, as well as
    # its mean across classes.

    auclist = []
    for y_pred, y_true in zip(predlist, true_list):
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auclist.append(metrics.auc(fpr, tpr))

    summary_dev['auc'] = np.array(auclist)

    loss_dev_str = tostr(summary_dev['loss'], 5)
    acc_dev_str = tostr(summary_dev['acc'])
    auc_dev_str = tostr(summary_dev['auc'])
    print(f"VAL > Step: {summary_train['step']}, Loss: {loss_dev_str}, "
            f"Acc: {acc_dev_str}, Auc: {auc_dev_str}, "
            f"Mean auc: {summary_dev['auc'].mean():.3f}, {time() - t0:.2f} sec.")

    step = summary_train['step']
    for label, loss, acc, auc in zip(dev_header, summary_dev['loss'],
                                 summary_dev['acc'], summary_dev['auc']):
        summary_writer.add_scalar( 'val/loss_' + label, loss, step)
        summary_writer.add_scalar( 'val/acc_' + label, acc, step)
        summary_writer.add_scalar( 'val/auc_' + label, auc, step)

    # If testing reveals that the current model performs best in either acc,
    # auc, or loss, save the model as the new best model.

    save_best = False
    mean_acc = summary_dev['acc'][cfg.save_index].mean()
    if mean_acc >= best_dict['acc_dev_best']:
        best_dict['acc_dev_best'] = mean_acc
        if cfg.best_target == 'acc':
            save_best = True

    mean_auc = summary_dev['auc'][cfg.save_index].mean()
    if mean_auc >= best_dict['auc_dev_best']:
        best_dict['auc_dev_best'] = mean_auc
        if cfg.best_target == 'auc':
            save_best = True

    mean_loss = summary_dev['loss'][cfg.save_index].mean()
    if mean_loss <= best_dict['loss_dev_best']:
        best_dict['loss_dev_best'] = mean_loss
        if cfg.best_target == 'loss':
            save_best = True

    if save_best:
        torch.save({
            'epoch': summary_train['epoch'],
            'step': summary_train['step'],
            'acc_dev_best': best_dict['acc_dev_best'],
            'auc_dev_best': best_dict['auc_dev_best'],
            'loss_dev_best': best_dict['loss_dev_best'],
            'state_dict': model.module.state_dict()
        }, join(args.save_path, f"best_{best_dict['best_idx']}.ckpt"))
        print(f"Best, Step: {summary_train['step']}, Loss : {loss_dev_str}, "
                f"Acc : {acc_dev_str}, Auc :{auc_dev_str}, "
                f"Best Auc: {best_dict['auc_dev_best']:.3f}")
        best_dict['best_idx'] += 1
        if best_dict['best_idx'] > cfg.save_top_k:
            best_dict['best_idx'] = 1

    torch.save({
        'epoch': summary_train['epoch'],
        'step': summary_train['step'],
        'acc_dev_best': best_dict['acc_dev_best'],
        'auc_dev_best': best_dict['auc_dev_best'],
        'loss_dev_best': best_dict['loss_dev_best'],
        'state_dict': model.module.state_dict()
    }, join(args.save_path, 'train.ckpt'))

summary_writer.close()
