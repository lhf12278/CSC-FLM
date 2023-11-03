from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import *
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)

    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        #resnet50
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)


            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        # patch 16 X 16 X 3
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)

            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        cls_score = self.classifier(feat, label)
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4], [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4]  # global feature for triplet loss

        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

# recent use
# divide 3 parts to catch feature

class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(BottleClassifier, self).__init__()

        self.bottleneck = nn.BatchNorm1d(in_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck1 = nn.BatchNorm1d(in_dim)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)

        self.bottleneck2 = nn.BatchNorm1d(in_dim)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

        self.bottleneck3 = nn.BatchNorm1d(in_dim)
        self.bottleneck3.bias.requires_grad_(False)
        self.bottleneck3.apply(weights_init_kaiming)

        self.classifier = nn.Linear(in_dim, out_dim, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier1 = nn.Linear(in_dim, out_dim, bias=False)
        self.classifier1.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(in_dim, out_dim, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.classifier3 = nn.Linear(in_dim, out_dim, bias=False)
        self.classifier3.apply(weights_init_classifier)

    def forward(self, x,x1,x2,x3):
        x = self.bottleneck(x)
        x = self.classifier(x)

        x1 = self.bottleneck1(x1)
        x1 = self.classifier1(x1)

        x2 = self.bottleneck2(x2)
        x2 = self.classifier2(x2)

        x3 = self.bottleneck3(x3)
        x3 = self.classifier3(x3)
        return [x,x1,x2,x3]

class build_transformer_three(nn.Module):
    def __init__(self, num_classes, t_num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer_three, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        # 16 X 16 X 3 = 768
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # TRANSFORMER_TYPE: 'vit_base_patch16_224_TransReID'
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.THREE_DOMAIN, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)


        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm

        # 全局网络
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        # 分块网络
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.t_num_classes = t_num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)


        else:
            # self.num_classes = num_classes+t_num_classes
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

            # target
            for i in range(len(self.t_num_classes)):
                name = 't_classifier' + str(i)
                setattr(self, name, BottleClassifier(768, self.t_num_classes[i]))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)

        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)

        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        # self.rearrange = rearrange

    def forward(self, x, label=None, modal=0, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)
        # global branch
        # origin b1_feat = self.b1(features)
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        token = features[:, 0:1]

        x = features[:, 1:]

        # if share the same parameters use self.b1
        # in contrast, use self.b2

        # lf_1
        b1_local_feat = x[:, :23]
        # b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        b1_local_feat = self.b1(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, 24:68]
        # b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        b2_local_feat = self.b1(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, 69:]
        # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        b3_local_feat = self.b1(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        feat = self.bottleneck(global_feat)
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3) #16,25,768

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                if modal==1:
                    i = cam_label.cpu().numpy()[0]
                    classifier_i = getattr(self, 't_classifier'+str(i))
                    t_score = classifier_i(global_feat, local_feat_1, local_feat_2, local_feat_3)
                    return  [global_feat, local_feat_1, local_feat_2, local_feat_3],t_score

                elif modal==2: # 全局和局部特征
                    return [global_feat, local_feat_1, local_feat_2, local_feat_3], [b1_local_feat, b2_local_feat, b3_local_feat]

                elif modal==3:
                    cls_score = self.classifier(feat)
                    cls_score_1 = self.classifier_1(local_feat_1_bn)
                    cls_score_2 = self.classifier_2(local_feat_2_bn)
                    cls_score_3 = self.classifier_3(local_feat_3_bn)
                    # for FEM
                    return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                            ], [global_feat, local_feat_1, local_feat_2, local_feat_3], [b1_local_feat, b2_local_feat,
                                                                                         b3_local_feat]  # global feature for triplet loss

                else:
                    cls_score = self.classifier(feat)
                    sort, idx1 = torch.max(cls_score, 1)
                    weight = self.classifier.weight
                    weights = weight.T
                    weight_mask = weights.clone()
                    weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
                    sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                             dim=1)  # descending=True为降序
                    sort2 = sort2.T
                    a, idx = sort2.topk(k=384, dim=1)  #idx为topk的索引
                    a_min = torch.min(a, dim=-1).values
                    a_min = a_min.unsqueeze(-1).repeat(1, 768)
                    ge = torch.ge(sort2, a_min)
                    zero = torch.zeros_like(sort2)
                    result = torch.where(ge, sort2, zero)
                    weight_idx_vec = result.T
                    idx1 = idx1.T  # 16*1
                    weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
                    self.classifier.weight.data = weights.T
                    self.classifier.weight.data.copy_(F.normalize(self.classifier.weight.data, dim=1))
                    cls_score = self.classifier(feat)

                    cls_score_1 = self.classifier_1(local_feat_1_bn)
                    sort, idx1 = torch.max(cls_score_1, 1)
                    weight = self.classifier_1.weight
                    weights = weight.T
                    weight_mask = weights.clone()
                    weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
                    sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                             dim=1)  # descending=True为降序
                    sort2 = sort2.T
                    a, idx = sort2.topk(k=384, dim=1)
                    values, drop = sort2.topk(384, dim=1, largest=False, sorted=True)
                    a_min = torch.min(a, dim=-1).values
                    a_min = a_min.unsqueeze(-1).repeat(1, 768)
                    ge = torch.ge(sort2, a_min)
                    zero = torch.zeros_like(sort2)
                    result = torch.where(ge, sort2, zero)
                    weight_idx_vec = result.T
                    idx1 = idx1.T  # 16*1
                    weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
                    self.classifier_1.weight.data = weights.T
                    self.classifier_1.weight.data.copy_(F.normalize(self.classifier_1.weight.data, dim=1))
                    cls_score_1 = self.classifier_1(local_feat_1_bn)

                    cls_score_2 = self.classifier_2(local_feat_2_bn)
                    sort, idx1 = torch.max(cls_score_2, 1)
                    weight = self.classifier_2.weight
                    weights = weight.T
                    weight_mask = weights.clone()
                    weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
                    sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                             dim=1)  # descending=True为降序
                    sort2 = sort2.T
                    a, idx = sort2.topk(k=384, dim=1)
                    values, drop = sort2.topk(384, dim=1, largest=False, sorted=True)
                    a_min = torch.min(a, dim=-1).values
                    a_min = a_min.unsqueeze(-1).repeat(1, 768)
                    ge = torch.ge(sort2, a_min)
                    zero = torch.zeros_like(sort2)
                    result = torch.where(ge, sort2, zero)
                    weight_idx_vec = result.T
                    idx1 = idx1.T
                    weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
                    self.classifier_2.weight.data = weights.T
                    self.classifier_2.weight.data.copy_(F.normalize(self.classifier_2.weight.data, dim=1))
                    cls_score_2 = self.classifier_2(local_feat_2_bn)

                    cls_score_3 = self.classifier_3(local_feat_3_bn)
                    sort, idx1 = torch.max(cls_score_3, 1)
                    weight = self.classifier_3.weight
                    weights = weight.T
                    weight_mask = weights.clone()
                    weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
                    sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                             dim=1)  # descending=True为降序
                    sort2 = sort2.T
                    a, idx = sort2.topk(k=384, dim=1)
                    values, drop = sort2.topk(384, dim=1, largest=False, sorted=True)
                    a_min = torch.min(a, dim=-1).values
                    a_min = a_min.unsqueeze(-1).repeat(1, 768)
                    ge = torch.ge(sort2, a_min)
                    zero = torch.zeros_like(sort2)
                    result = torch.where(ge, sort2, zero)
                    weight_idx_vec = result.T
                    idx1 = idx1.T  # 16*1
                    weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
                    self.classifier_3.weight.data = weights.T
                    self.classifier_3.weight.data.copy_(F.normalize(self.classifier_3.weight.data, dim=1))
                    local_adv = local_feat_3_bn.scatter_(1, drop, 0)
                    cls_score_3 = self.classifier_3(local_feat_3_bn)

            # for FEM
            return idx, drop, local_adv, [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3] ,[b1_local_feat,b2_local_feat,b3_local_feat] # global feature for triplet loss


        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 3, local_feat_2_bn / 3, local_feat_3_bn / 3], dim=1)

            else:
                return torch.cat([global_feat, local_feat_1 / 3, local_feat_2 / 3, local_feat_3 / 3], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class DomainNet(nn.Module):
    def __init__(self, targetName):
        super(DomainNet, self).__init__()
        # patch_Num = 16 x 16 x 3
        patch_Num = 768

        if targetName == 'market1501':
            CamView_Num = 9
        elif targetName == 'marketSCT':
            CamView_Num = 9
        elif targetName == 'msmt17':
            CamView_Num = 18
        elif targetName == 'msmt17_sct':
            CamView_Num = 18

        self.classifier = self._make_layer(patch_Num, CamView_Num)
        self.classifier2 = self._make_layer2(patch_Num, CamView_Num)

        # Classifier Trans Set
        self.atten1 = vit_base_Domain_net(drop_path_rate=0.1,sie_xishu=3)
        self.atten2 = vit_base_Domain_net(drop_path_rate=0.1,sie_xishu=3)

        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.classifier2.apply(weights_init_classifier)
        self.classifier.apply(weights_init_classifier)

        self.Linear = nn.Linear(patch_Num, CamView_Num)
        self.Linear2 = nn.Linear(patch_Num, CamView_Num)

    # the fully Connect of Classifier 1
    def _make_layer(self, in_nc, out_nc):
        block = [
                 nn.Linear(in_nc, 384, bias=False),
                 nn.BatchNorm1d(384),
                 nn.ReLU(),
                 nn.Linear(384, 192, bias=False),
                 nn.BatchNorm1d(192),
                 nn.ReLU(),
                 nn.Linear(192, 96, bias=False),
                 nn.BatchNorm1d(96),
                 nn.ReLU(),
                 nn.Linear(96, out_nc, bias=False),
                 nn.BatchNorm1d(out_nc),
        ]

        return   nn.Sequential(*block)

    # the fully Connect of Classifier 2
    def _make_layer2(self, in_nc, out_nc):
        block = [
                 nn.Linear(in_nc, 400, bias=False),
                 nn.BatchNorm1d(400),
                 nn.ReLU(),
                 nn.Linear(400, 200, bias=False),
                 nn.BatchNorm1d(200),
                 nn.ReLU(),
                 nn.Linear(200, 100, bias=False),
                 nn.BatchNorm1d(100),
                 nn.ReLU(),
                 nn.Linear(100, out_nc, bias=False),
                 nn.BatchNorm1d(out_nc),
        ]

        return  nn.Sequential(*block)

    def forward(self, feature):
        # for cls1
        feature1 = self.dropout(feature)
        feature1 = self.atten1(feature1)
        cls_score1 = self.Linear(feature1)
        sort, idx1 = torch.max(cls_score1, 1)
        weight = self.Linear.weight
        weights = weight.T
        weight_mask = weights.clone()
        weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
        sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                 dim=1)  # descending=True为降序
        sort2 = sort2.T
        a, _ = sort2.topk(k=384, dim=1)
        a_min = torch.min(a, dim=-1).values
        a_min = a_min.unsqueeze(-1).repeat(1, 768)
        ge = torch.ge(sort2, a_min)
        zero = torch.zeros_like(sort2)
        result = torch.where(ge, sort2, zero)
        weight_idx_vec = result.T
        idx1 = idx1.T
        weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
        self.Linear.weight.data = weights.T
        self.Linear.weight.data.copy_(F.normalize(self.Linear.weight.data, dim=1))
        cls_score1 = self.classifier(feature1)

        # for cls2
        feature2 = self.dropout2(feature)
        feature2 = self.atten2(feature2)
        cls_score2 = self.Linear2(feature2)
        sort, idx1 = torch.max(cls_score2, 1)
        weight = self.Linear2.weight  #)
        weights = weight.T
        weight_mask = weights.clone()
        weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
        sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                 dim=1)  # descending=True为降序
        sort2 = sort2.T
        a, _ = sort2.topk(k=384, dim=1)
        a_min = torch.min(a, dim=-1).values
        a_min = a_min.unsqueeze(-1).repeat(1, 768)
        ge = torch.ge(sort2, a_min)
        zero = torch.zeros_like(sort2)
        result = torch.where(ge, sort2, zero)
        weight_idx_vec = result.T
        idx1 = idx1.T  # 16*1
        weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
        self.Linear2.weight.data = weights.T
        self.Linear2.weight.data.copy_(F.normalize(self.Linear2.weight.data, dim=1))
        cls_score2 = self.classifier2(feature2)
        # for sum between cls1 and cls2
        cls_score = (cls_score1 + cls_score2)/2

        return cls_score1, cls_score2, cls_score


# just freeze module
class DomainNet2(nn.Module):
    def __init__(self):
        super(DomainNet2, self).__init__()
        self.classifier = self._make_layer(768,9)
        self.dropout = nn.Dropout(0.5)
        self.Linear = nn.Linear(768, 9)
        self.Linear2 = nn.Linear(768, 9)
    def _make_layer(self, in_nc, out_nc):
        block = [
            nn.Linear(in_nc, 384, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, 192, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Linear(192, 96, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(96, out_nc, bias=False),
            nn.BatchNorm1d(out_nc),
            # nn.ReLU()
        ]
        return nn.Sequential(*block)
    def forward(self, feature):
        cls_score1 = self.classifier(feature)
        sort, idx1 = torch.max(cls_score1, 1)
        weight = self.Linear.weight
        weights = weight.T
        weight_mask = weights.clone()
        weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
        sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                 dim=1)  # descending=True为降序
        sort2 = sort2.T
        a, _ = sort2.topk(k=384,dim=1)
        a_min = torch.min(a, dim=-1).values
        a_min = a_min.unsqueeze(-1).repeat(1, 768)  # 16*2048
        ge = torch.ge(sort2, a_min)
        zero = torch.zeros_like(sort2)
        result = torch.where(ge, sort2, zero)
        weight_idx_vec = result.T
        idx1 = idx1.T  # 16*1
        weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
        self.Linear.weight.data = weights.T
        self.Linear.weight.data.copy_(F.normalize(self.Linear.weight.data, dim=1))
        cls_score1 = self.classifier(feature)
        cls_score2 = self.classifier(feature)
        sort, idx1 = torch.max(cls_score2, 1)
        weight = self.Linear.weight
        weights = weight.T
        weight_mask = weights.clone()
        weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
        sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                 dim=1)  # descending=True为降序
        sort2 = sort2.T
        a, _ = sort2.topk(k=384, dim=1)
        a_min = torch.min(a, dim=-1).values
        a_min = a_min.unsqueeze(-1).repeat(1, 768)
        ge = torch.ge(sort2, a_min)
        zero = torch.zeros_like(sort2)
        result = torch.where(ge, sort2, zero)
        weight_idx_vec = result.T
        idx1 = idx1.T  # 16*1
        weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
        self.Linear.weight.data = weights.T
        self.Linear.weight.data.copy_(F.normalize(self.Linear.weight.data, dim=1))
        cls_score2 = self.classifier(feature)
        cls_score = cls_score1 + cls_score2
        return cls_score1, cls_score2, cls_score

class Person_classifier(nn.Module):
    def __init__(self, num_classes,t_num_classes,cfg):
        super(Person_classifier, self).__init__()
        # patch_Num = 16 x 16 x 3
        self.in_planes = 768
        self.num_classes = num_classes
        self.t_num_classes = t_num_classes

        # Classifier Trans Set
        self.atten1 = vit_base_Domain_net(drop_path_rate=0.1,sie_xishu=3)
        self.atten2 = vit_base_Domain_net(drop_path_rate=0.1,sie_xishu=3)

        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
    # the fully Connect of Classifier 1
    def _make_layer(self, in_nc, out_nc):
        block = [
                 nn.Linear(in_nc, 384, bias=False),
                 nn.BatchNorm1d(384),
                 nn.ReLU(),
                 nn.Linear(384, 192, bias=False),
                 nn.BatchNorm1d(192),
                 nn.ReLU(),
                 nn.Linear(192, 96, bias=False),
                 nn.BatchNorm1d(96),
                 nn.ReLU(),
                 nn.Linear(96, out_nc, bias=False),
                 nn.BatchNorm1d(out_nc),
        ]
        return nn.Sequential(*block)

    def forward(self, feature):
        # for cls1
        feature1 = self.dropout(feature)
        cls_score = self.classifier(feature1)
        sort, idx1 = torch.max(cls_score, 1)
        weight = self.classifier.weight
        weights = weight.T
        weight_mask = weights.clone()
        weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
        sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                 dim=1)  # descending=True为降序
        sort2 = sort2.T
        a, _ = sort2.topk(k=384, dim=1)
        a_min = torch.min(a, dim=-1).values
        a_min = a_min.unsqueeze(-1).repeat(1, 768)
        ge = torch.ge(sort2, a_min)
        zero = torch.zeros_like(sort2)
        result = torch.where(ge, sort2, zero)
        weight_idx_vec = result.T
        idx1 = idx1.T  # 16*1
        weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
        self.classifier.weight.data = weights.T
        self.classifier.weight.data.copy_(F.normalize(self.classifier.weight.data, dim=1))
        cls_score = self.classifier(feature1)
        return feature1, cls_score

class Person_classifier_Refine(nn.Module):
    def __init__(self, num_classes, t_num_classes, cfg):
        super(Person_classifier_Refine, self).__init__()
        # patch_Num = 16 x 16 x 3
        self.in_planes = 768
        self.num_classes = num_classes
        self.t_num_classes = t_num_classes
        # Classifier Trans Set
        self.atten1 = vit_base_Domain_net(drop_path_rate=0.1,sie_xishu=3)
        self.atten2 = vit_base_Domain_net(drop_path_rate=0.1,sie_xishu=3)

        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
    # the fully Connect of Classifier 1
    def _make_layer(self, in_nc, out_nc):
        block = [
                 nn.Linear(in_nc, 384, bias=False),
                 nn.BatchNorm1d(384),
                 nn.ReLU(),
                 nn.Linear(384, 192, bias=False),
                 nn.BatchNorm1d(192),
                 nn.ReLU(),
                 nn.Linear(192, 96, bias=False),
                 nn.BatchNorm1d(96),
                 nn.ReLU(),
                 nn.Linear(96, out_nc, bias=False),
                 nn.BatchNorm1d(out_nc),
        ]
        return nn.Sequential(*block)

    def forward(self, feature):
        # for cls1
        feature1 = self.dropout(feature)
        cls_score = self.classifier(feature1)
        sort, idx1 = torch.max(cls_score, 1)
        weight = self.classifier.weight
        weights = weight.T
        weight_mask = weights.clone()
        weight_idx_vec = torch.index_select(weight_mask, 1, idx1)
        sort2, idx2 = torch.sort(weight_idx_vec, descending=True,
                                 dim=1)  # descending=True为降序
        sort2 = sort2.T
        a, _ = sort2.topk(k=384, dim=1)
        a_min = torch.min(a, dim=-1).values
        a_min = a_min.unsqueeze(-1).repeat(1, 768)
        ge = torch.ge(sort2, a_min)
        zero = torch.zeros_like(sort2)
        result = torch.where(ge, sort2, zero)
        weight_idx_vec = result.T
        idx1 = idx1.T  # 16*1
        weights.data[:, idx1].copy_(F.normalize(weight_idx_vec, dim=1).float().cuda())
        self.classifier.weight.data = weights.T
        self.classifier.weight.data.copy_(F.normalize(self.classifier.weight.data, dim=1))
        cls_score = self.classifier(feature1)
        return feature1, cls_score




__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model(cfg, num_class, num_class_t, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_three(num_class, num_class_t, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer with JPM module ===========')
        elif cfg.MODEL.THREE_DOMAIN:
            model = build_transformer_three(num_class, num_class_t, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer with three_domian module ===========')
            coarse_classi = Person_classifier(num_class, num_class_t, cfg)
            print('===========building coarse person classifier with coarse_classifier ===========')
            refine_classi = Person_classifier_Refine(num_class, num_class_t, cfg)
            print('===========building refine person classifier with refine_classifier ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    if cfg.MODEL.THREE_DOMAIN:
        class_net = DomainNet(cfg.DATASETS.TARGET)
        class_net_2 = DomainNet2()
    return  model, coarse_classi, refine_classi, class_net, class_net_2
