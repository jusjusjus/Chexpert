
from os import environ as env
from os.path import join, exists

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import get_transforms

np.random.seed(0)

BASEDIR = join(env['DATA'], 'challenges')

def parse_csv(filename, cfg, mode, ignore_missing_files=False):
    label_map = {
        2: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
        6: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
        10: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
        5: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'},
        8: {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}
    }
    _image_paths = []
    _labels = []
    with open(filename, 'r') as f:

        # Parse column header

        header = f.readline().strip('\n').split(',')
        selected_label_indices = [2, 5, 6, 8, 10]
        _label_header = [
            l.replace(' ', '_') for l in [
                header[5 + i] for i in selected_label_indices]
        ]

        # Parse data rows

        for row in f:
            row = row.strip('\n').split(',')
            image_path = join(BASEDIR, row[0])
            if ignore_missing_files:
                if not exists(image_path):
                    continue
            assert exists(image_path), image_path
            flg_enhance = False
            labels = row[5:]
            processed_labels = []
            for index, label in enumerate(labels):
                lm = label_map.get(index, None)
                if lm is not None:
                    label = lm[label]
                    processed_labels.append(label)
                    if label == '1' and index in cfg.enhance_index:
                        flg_enhance = True

            processed_labels = list(map(int, processed_labels))
            _image_paths.append(image_path)
            _labels.append(processed_labels)
            if flg_enhance and mode == 'train':
                for i in range(cfg.enhance_times):
                    _image_paths.append(image_path)
                    _labels.append(processed_labels)

    return _label_header, _image_paths, _labels

# This would be a much faster implementation using pandas.  The code is not
# finished in it's current form.

# import pandas as pd
#
# def parse_csv(filename, cfg, mode):
#     csv = pd.read_csv(filename)
#     csv['Path'] = csv.Path.apply(lambda x: join(BASEDIR, x))
#     csv.columns = [c.replace(' ', '_') for c in csv]
#     labels = csv[['Cardiomegaly', 'Edema', 'Consolidation',
#                     'Atelectasis', 'Pleural_Effusion']]
#     labels = labels.replace(np.nan, 0).astype(int)
#     _image_paths = csv.Path.tolist()
#     _label_header = list(labels.columns)
#     _labels = labels.values.tolist()
#     return _label_header, _image_paths, _labels

class ImageDataset(Dataset):

    def __init__(self, label_path, cfg, mode='train', ignore_missing_files=False):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self._label_header, self._image_paths, self._labels = parse_csv(
            label_path, cfg, mode, ignore_missing_files)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def _border_pad(self, image):
        h, w, c = image.shape

        if self.cfg.border_pad == 'zero':
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode='constant', constant_values=0.0
            )

        elif self.cfg.border_pad == 'pixel_mean':
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode='constant', constant_values=self.cfg.pixel_mean
            )

        else:
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode=self.cfg.border_pad
            )

        return image

    def _fix_ratio(self, image):
        """return resized image while keeping ratio fixed"""
        h, w, c = image.shape
        if h >= w:
            ratio = h * 1.0 / w
            h_ = self.cfg.long_side
            w_ = round(h_ / ratio)

        else:
            ratio = w * 1.0 / h
            w_ = self.cfg.long_side
            h_ = round(w_ / ratio)

        image = cv2.resize(image, dsize=(w_, h_),
                           interpolation=cv2.INTER_LINEAR)
        image = self._border_pad(image)
        return image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = get_transforms(image, ttype=self.cfg.use_transforms_type)

        image = np.array(image)
        if self.cfg.use_equalizeHist:
            image = cv2.equalizeHist(image)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)

        if self.cfg.fix_ratio:
            image = self._fix_ratio(image)
        else:
            image = cv2.resize(image, dsize=(self.cfg.width, self.cfg.height),
                               interpolation=cv2.INTER_LINEAR)

        if self.cfg.gaussian_blur > 0:
            image = cv2.GaussianBlur(image, (self.cfg.gaussian_blur,
                                             self.cfg.gaussian_blur), 0)

        # Normalization.  vgg and resnet do not use pixel_std, densenet and
        # inception use.

        image -= self.cfg.pixel_mean
        if self.cfg.use_pixel_std:
            image /= self.cfg.pixel_std

        # normal image tensor : H x W x C
        # torch image tensor  : C x H x W

        image = image.transpose((2, 0, 1))
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode in ('train', 'val'):
            return image, labels
        elif self._mode == 'test':
            return image, path
        elif self._mode == 'heatmap':
            return image, path, labels
        else:
            raise ValueError(f"Unknown mode : '{self._mode}'")
