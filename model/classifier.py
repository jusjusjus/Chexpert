from torch import nn

import torch.nn.functional as F
from model.backbone.vgg import (vgg19, vgg19_bn)
from model.backbone.densenet import (densenet121, densenet169, densenet201)
from model.backbone.inception import (inception_v3)
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap


BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}


BACKBONES_TYPES = {'vgg19': 'vgg',
                   'vgg19_bn': 'vgg',
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   'inception_v3': 'inception'}


class Classifier(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = BACKBONES[cfg.backbone](cfg)
        self.global_pool = GlobalPool(cfg)
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3
        else:
            self.expand = 1

        self._init_classifier()
        self._init_batchnorm()
        self._init_attention_map()

    def forward(self, x):
        """return logits and logits map

        Parameter
        ---------
            x: tensor of shape (N, C, H, W)
        """
        assert self.cfg.attention_map != "None"
        feat_map = self.backbone(x)
        feat_map = self.attention_map(feat_map)

        logits = []
        logit_maps = []
        for index, num_class in enumerate(self.cfg.num_classes):
            classifier = self.get_classifier(index)
            batchnorm = self.get_batchnorm(index)

            # (N, num_class, H, W)
            logit_map = None
            if self.cfg.global_pool not in ('AVG_MAX', 'AVG_MAX_LSE'):
                logit_map = classifier(feat_map)
                logit_maps.append(logit_map)

            # (N, num_class, H, W) -> (N, C, 1, 1)
            feat = self.global_pool(feat_map, logit_map)
            feat = batchnorm(feat)
            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)

            # (N, num_class, 1, 1)
            logit = classifier(feat)

            # (N, num_class, 1, 1) -> (N, num_class)
            logit = logit.squeeze()
            logits.append(logit)

        return logits, logit_maps

    def get_classifier(self, index):
        """return classification layer for column `index`"""
        return getattr(self, "fc_" + str(index))

    def get_batchnorm(self, index):
        """return batch-norm layer for column `index`

        If batch norm is not computed, the method returns the identity
        function.
        """
        return getattr(self, "bn_" + str(index)) if self.cfg.fc_bn \
               else (lambda x: x)

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                num_features = 512
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                num_features = self.backbone.num_features
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                num_features = 2048
            else:
                raise ValueError(f"Unknown backbone type: {self.cfg.backbone}")

            clf = nn.Conv2d(num_features * self.expand, num_class,
                            kernel_size=1, stride=1, padding=0, bias=True)
            clf.weight.data.normal_(0, 0.01)
            clf.bias.data.zero_()
            setattr(self, f"fc_{index}", clf)

    def _init_batchnorm(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
                num_features = 512
            elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
                num_features = self.backbone.num_features
            elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
                num_features = 2048
            else:
                raise ValueError(f"Unknown backbone type: {self.cfg.backbone}")

            setattr(self, f"bn_{index}",
                    nn.BatchNorm2d(self.expand * num_features))


    def _init_attention_map(self):
        if BACKBONES_TYPES[self.cfg.backbone] == 'vgg':
            num_features = 512
        elif BACKBONES_TYPES[self.cfg.backbone] == 'densenet':
            num_features = self.backbone.num_features
        elif BACKBONES_TYPES[self.cfg.backbone] == 'inception':
            num_features = 2048
        else:
            raise ValueError(f"Unknown backbone type : {self.cfg.backbone}")

        self.attention_map = AttentionMap(self.cfg, num_features)

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))
