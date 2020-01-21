#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Plot ROC')
parser.add_argument('true_csv_path', type=str,
                    help="Path to the ground truth in csv")
parser.add_argument('--pred_csv_path', type=str, default='test/test.csv',
                    help="Path to the prediction in csv")
parser.add_argument('--plot_path', type=str, default='test',
                    help="Path to the ROC plots")
parser.add_argument('--output', type=str, default='roc.png',
                    help="Basename of the ROC plots")
parser.add_argument('--threshold', type=float, default=0.5,
                    help="Probability threshold")
args = parser.parse_args()

from sys import path
path.insert(0, '.')
from os import environ as env
from os.path import join, splitext, dirname

import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt


def read_csv(csv_path, true_csv=False):
    """return data in `csv_path`

    If `true_csv`, csv_path is treated as part of the original csvs in the
    ground truth dataset.  Else the the csv is treated as an output of
    "/bin/test.py".
    """
    label_map = {
        2: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
        5: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'},
        6: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
        8: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'},
        10: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
    }
    image_paths = []
    probs = []
    with open(csv_path) as f:
        header = f.readline().strip('\n').split(',')
        for line in f:
            fields = line.strip('\n').split(',')
            image_paths.append(fields[0])
            if true_csv is False:
                probs.append(list(map(float, fields[1:])))
            else:
                prob = []
                for index, value in enumerate(fields[5:]):
                    lm = label_map.get(index, None)
                    if lm:
                        prob.append(lm[value])

                prob = list(map(int, prob))
                probs.append(prob)

    probs = np.array(probs)
    return image_paths, probs, header


def transform_csv(input_path, output_path):
    """to transform the first column of the original csv or test csv from Path
    to Study
    """
    infile = pd.read_csv(input_path)
    infile = infile.fillna('Unknown')
    infile['Study'] = infile.Path.apply(dirname)
    outfile = infile.drop('Path', axis=1).groupby('Study').max().reset_index()
    outfile.to_csv(output_path, index=False)


def transform_csv_en(input_path, output_path):
    """to transform the first column of the original
    csv or test csv from Path to Study
    """
    infile = pd.read_csv(input_path)
    infile = infile.fillna('Unknown')
    infile['Study'] = infile.Path.apply(dirname)
    outfile = infile.drop('Path', axis=1).groupby('Study').mean().reset_index()
    groups = infile.drop('Path', axis=1).groupby('Study')
    outfile['Cardiomegaly'] = groups['Cardiomegaly'].min().reset_index()[
        'Cardiomegaly']
    outfile['Edema'] = groups['Edema'].max().reset_index()['Edema']
    outfile['Consolidation'] = groups['Consolidation'].mean().reset_index()[
        'Consolidation']
    outfile['Atelectasis'] = groups['Atelectasis'].mean().reset_index()[
        'Atelectasis']
    outfile['Pleural Effusion'] = groups['Pleural Effusion'].mean(
    ).reset_index()['Pleural Effusion']
    outfile.to_csv(output_path, index=False)


pred_csv_path = join(args.plot_path, 'pred_csv_done.csv')
true_csv_path = join(args.plot_path, 'true_csv_done.csv')

transform_csv_en(args.pred_csv_path, pred_csv_path)
transform_csv(args.true_csv_path, true_csv_path)

images_pred, probs_pred, header_pred = read_csv(pred_csv_path, true_csv=False)
images_true, probs_true, header_true = read_csv(true_csv_path, true_csv=True)

# Prepend "$DATA/challenges" to `images_true`

images_true = [join(env['DATA'], 'challenges', I) for I in images_true]

assert images_pred == images_true

header = [header_true[7], header_true[10], header_true[11],
          header_true[13], header_true[15]]
num_labels = len(header)

probs_true = probs_true.transpose()
probs_pred = probs_pred.transpose()

for label, y_true, y_pred in zip(header, probs_true, probs_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(label, 'auc', auc)
    acc = metrics.accuracy_score(
        y_true, (y_pred >= args.threshold).astype(int), normalize=True)

    plt.figure(figsize=(8, 8), dpi=150)
    plt.subplot(111)
    plt.title(f"{label} ROC, AUC : {auc:.3f}, Acc : {acc:.3f}")
    plt.plot(fpr, tpr, '-b')
    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.xticks(np.linspace(0.0, 1.0, num=11))
    plt.yticks(np.linspace(0.1, 1.0, num=10))
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.grid()

    name, ext = splitext(args.output)
    outputname = join(args.plot_path, name + '-' + label + ext)
    plt.savefig(outputname, bbox_inches='tight')
