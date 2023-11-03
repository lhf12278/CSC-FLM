# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


class CrossEntropySmooth(nn.Module):

    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropySmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1])).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / size[1]
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class CrossEntropySmooth2(nn.Module):

    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropySmooth2, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = targets.repeat(16, 1)

        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / size[1]
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    # feat_dim = 2048
    # center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    # C.DATALOADER.SAMPLER = 'softmax'

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            # recent use
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(score, target) for score in score[1:]]  # score为list
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                           cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(score, target) for score in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                           cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))

    ide_creiteron1 = CrossEntropySmooth()
    ide_creiteron = CrossEntropyLabelSmooth(num_classes=500).cuda()
    criterion_tri = TripletLoss()
    ide_creiteron2 = CrossEntropySmooth2()
    return loss_func, ide_creiteron, ide_creiteron1, criterion_tri, ide_creiteron2

def kl_loss(a,b):
    kl_loss = F.kl_div(a.softmax(dim=-1).log(), b.softmax(dim=-1), reduction='mean')
    return kl_loss
