import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import collections
from sklearn.cluster import DBSCAN
from cluster.reranking import compute_jaccard_dist
from datasets.bases import ImageDataset
from loss.make_loss import kl_loss
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.make_dataloader import train_collate_fn

def do_train(cfg,
             model, coarse_classi, refine_classi,
             class_net,
             class_net2,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank,
             coarse_classi_optimizer, refine_classi_optimizer, class_optimizer, class2_optimizer,
             coarse_classi_scheduer, refine_classi_scheduer, class_scheduler, class2_scheduler,
             target_train_loader, ide_creiteron, ide_creiteron1, loss_classifier, criterion_tri, t_loader):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        coarse_classi.to(local_rank)
        refine_classi.to(local_rank)
        class_net.to(local_rank)
        class_net2.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            coarse_classi = torch.nn.parallel.DistributedDataParallel(coarse_classi, device_ids=[local_rank], find_unused_parameters=True)
            refine_classi = torch.nn.parallel.DistributedDataParallel(refine_classi, device_ids=[local_rank], find_unused_parameters=True)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # train
    # num of batch
    num_Batch = 100

    # self-train Num
    s0_train_num = 1
    s1_train_num = cfg.MODEL.S1_NUM
    s2_train_num = cfg.MODEL.S2_NUM
    s3_train_num = cfg.MODEL.S3_NUM
    s4_train_num = cfg.MODEL.S4_NUM
    # Num of source batch
    Num_Batch = 300

    # Print best result
    best_top1 = 0
    best_top5 = 0
    best_top10 = 0
    best_map = 0
    best_epoch = 0

    logger.info("The loss classifier para is C1_{},C2_{}, and Cunion_{}".format(loss_classifier[0], loss_classifier[1], loss_classifier[2]))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        coarse_classi_scheduer.step(epoch)
        refine_classi_scheduer.step(epoch)
        class_scheduler.step(epoch)
        class2_scheduler.step(epoch)
        model.train()
        coarse_classi.train()
        refine_classi.train()

        if epoch < s1_train_num:

            for i in range(Num_Batch):
                n_iter = i
                img, vid, target_cam, target_view = train_loader[0].next_one()
                optimizer.zero_grad()
                coarse_classi_optimizer.zero_grad()
                refine_classi_optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device) # vid = p_id
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)
                with amp.autocast(enabled=True):
                    _,_,local_adv,score, feat, _ = model(img, target, cam_label=target_cam, view_label=target_view )
                    feat1, feat2, feat3, feat4 = feat[0], feat[1], feat[2], feat[3]
                    feature1, cls_score1 = coarse_classi(feat1.detach())
                    feature2, cls_score2 = refine_classi(feat1.detach())
                    feat_adv, cls_score,_ = class_net2(local_adv.detach(),img)
                    loss1 = loss_fn(score, feat, target, target_cam)
                    adv_score = torch.ones(cfg.INPUT.C_CAM)/cfg.INPUT.C_CAM
                    loss_adv = ide_creiteron1(cls_score,adv_score)
                    loss1= loss1 + loss_adv
                    loss2 = ide_creiteron(cls_score1, target)
                    loss3 = ide_creiteron(cls_score2, target)
                scaler.scale(loss1).backward()
                scaler.scale(loss2).backward()
                scaler.scale(loss3).backward()
                scaler.step(optimizer)
                scaler.step(coarse_classi_optimizer)
                scaler.step(refine_classi_optimizer)
                scaler.update()

                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                loss_meter.update(loss1.item(), img.shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader[1]),
                                        loss_meter.avg, acc_meter.avg, optimizer.state_dict()['param_groups'][0]['lr']))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader[1].batch_size / time_per_batch))

        # cluster
        if epoch >= s3_train_num and epoch < s4_train_num:

            target_iters = t_loader[0]
            for i in range(len(target_iters)):
                target_iters = t_loader[0]
                # t_img, t_vid, t_target_cam, t_target_view = target_iters[i].next_one()
                optimizer.zero_grad()
                coarse_classi_optimizer.zero_grad()
                refine_classi_optimizer.zero_grad()

                img, vid, target_cam, target_view = train_loader[0].next_one()
                t_img, t_vid, t_target_cam, t_target_view = target_iters[i].next_one()
                optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)

                _, local_adv, _, score, feat, _ = model(img, target, cam_label=target_cam, view_label=target_view)
                t_img = t_img.to(device)
                t_target = t_vid.to(device)

                with amp.autocast(enabled=True):
                    t_feat, t_score = model(t_img, modal=1, cam_label=t_target_cam)
                    _, _, local_adv, t_score, dict_f, l_tfeat = model(t_img, t_target, cam_label=t_target_cam,
                                                                      view_label=t_target_view)
                    dict_f, _, _, _ = dict_f[0], dict_f[1], dict_f[2], dict_f[3]
                    feat_adv, cls_score, _ = class_net2(local_adv.detach(), t_img)
                    c_feature, c_cls_score = coarse_classi(dict_f.detach())
                    loss_3 = ide_creiteron1(c_cls_score, t_target)
                    loss11 = loss_fn(t_score, t_feat, t_target, t_target_cam)
                scaler.scale(loss_3).backward()
                scaler.step(coarse_classi_optimizer)
                scaler.scale(loss11).backward()
                scaler.step(optimizer)
                scaler.update()

                for i in range(len(target_iters)):
                    with amp.autocast(enabled=True):

                        _, _, local_adv, _, dict_f, l_tfeat = model(t_img, t_target, cam_label=t_target_cam,
                                                                    view_label=t_target_view)
                        dict_f, _, _, _ = dict_f[0], dict_f[1], dict_f[2], dict_f[3]

                        feat_adv, cls_score, _ = class_net2(local_adv.detach(), t_img)
                        adv_score = torch.ones(cfg.INPUT.C_CAM) / cfg.INPUT.C_CAM
                        loss_7 = ide_creiteron1(cls_score, adv_score)
                        rerank_dist = compute_jaccard_dist(dict_f, lambda_value=cfg.LAMBDA_VALUE,
                                                           source_features=dict_f,
                                                           use_gpu=cfg.MODEL.GPU).detach().numpy()
                        if (epoch == 0):

                            tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
                            tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
                            tri_mat = np.sort(tri_mat, axis=None)
                            rho = 1.6e-3
                            top_num = np.round(rho * tri_mat.size).astype(int)
                            eps = tri_mat[:top_num].mean()
                            print('eps for cluster: {:.3f}'.format(eps))

                            cluster = DBSCAN(eps=0.5, min_samples=4, metric='precomputed', n_jobs=-1)
                            print('Clustering and labeling...')
                            labels = cluster.fit_predict(rerank_dist)
                            print(labels)
                            num_ids = len(set(labels)) - (1 if -1 in labels else 0)
                            num_clusters = num_ids
                            print('\n Clustered into {} classes \n'.format(num_clusters))
                            # generate new dataset and calculate cluster centers
                            new_dataset = []
                            cluster_centers = collections.defaultdict(list)
                            for i, ((fname, _, cid), label) in enumerate(zip(t_img, labels)):
                                if label == -1: continue
                                new_dataset.append((fname, label, cid))
                                cluster_centers[label].append(dict_f[i])
                            cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in
                                               sorted(cluster_centers.keys())]
                            cluster_centers = torch.stack(cluster_centers)
                            model.classifier.weight.data[:num_clusters].copy_(
                                F.normalize(cluster_centers, dim=1).float().cuda())
                            dataset_c_intra = []
                            camera_train_loader = []
                            camera_intra_train_set = ImageDataset(dataset_c_intra)
                            camera_train_loader.append(DataLoader(
                                camera_intra_train_set, batch_size=cfg.SOLVER.CITRA_IMS_PER_BATCH,
                                sampler=RandomIdentitySampler(camera_intra_train_set.dataset,
                                                              cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                                num_workers=8, collate_fn=train_collate_fn
                            ))
                            t_dataset = []
                            t_datat_train_set = ImageDataset(t_dataset)
                            t_loader.append(DataLoader(
                                t_datat_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                num_workers=8, collate_fn=train_collate_fn
                            ))
                            # Optimizer
                            params = []
                            for key, value in model.named_parameters():
                                if not value.requires_grad:
                                    continue
                                params += [{"params": [value], "lr": cfg.SOLVER.C_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}]
                            optimizer = torch.optim.Adam(params)
                            model.classifier.weight.data[:num_clusters].copy_(
                                F.normalize(cluster_centers, dim=1).float().cuda())
                            labels = torch.tensor(labels)
                            num_clusters = t_target
                            cls_score2 = cls_score2[:, :num_clusters]
                            loss_cluster = ide_creiteron(cls_score2, labels)
                            loss_cluster = loss_cluster
                            loss_cluster1 = loss_cluster + loss_7
                            scaler.scale(loss_cluster).backward()
                            scaler.step(refine_classi_optimizer)
                            scaler.scale(loss_cluster1).backward()
                            scaler.step(optimizer)
                            scaler.update()


        if epoch >= s2_train_num and epoch < s3_train_num:
            for i in range(Num_Batch):
                n_iter = i
                img, vid, target_cam, target_view = train_loader[0].next_one()
                optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)
                with amp.autocast(enabled=True):

                    _,_,local_adv,score, feat,_ = model(img, target, cam_label=target_cam, view_label=target_view )
                    feat_adv, cls_score, _ = class_net2(local_adv.detach(), img)
                    loss = loss_fn(score, feat, target, target_cam)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                target_iters=t_loader[0]
                for i in range(len(target_iters)):
                    img, vid, target_cam, target_view = train_loader[0].next_one()
                    t_img, t_vid, t_target_cam, t_target_view = target_iters[i].next_one()
                    optimizer.zero_grad()
                    img = img.to(device)
                    target = vid.to(device)
                    target_cam = target_cam.to(device)
                    target_view = target_view.to(device)

                    _,_,_,score, feat, _ = model(img, target, cam_label=target_cam, view_label=target_view)
                    feat1, feat2, feat3, feat4 = feat[0], feat[1], feat[2], feat[3]
                    t_img = t_img.to(device)
                    t_target = t_vid.to(device)

                    with amp.autocast(enabled=True):
                        t_feat, t_score = model(t_img, modal=1, cam_label=t_target_cam) #目标域
                        _,_,_,_, dict_f, l_tfeat = model(t_img, t_target, cam_label=t_target_cam, view_label=t_target_view)
                        dict_f, _, _, _ = dict_f[0], dict_f[1], dict_f[2], dict_f[3]
                        # Transfer
                        # B = feat1.size(0)
                        s_mu = feat1.mean(dim=[-2, 1], keepdim=True)
                        s_var = feat1.var(dim=[-2, 1], keepdim=True)
                        eps = 1e-6
                        s_sig = (s_var + eps).sqrt()
                        s_mu, s_sig = s_mu.detach(), s_sig.detach()
                        s_normed = (feat1 - s_mu) / s_sig
                        t_mu = dict_f.mean(dim=[-2, 1], keepdim=True)
                        t_var = dict_f.var(dim=[-2, 1], keepdim=True)
                        eps = 1e-6
                        t_sig = (t_var + eps).sqrt()
                        t_mu, t_sig = t_mu.detach(), t_sig.detach()
                        t_normed = (dict_f - t_mu) / t_sig
                        mixstyle_s_t = s_normed * t_sig + t_mu
                        mixstyle_t_s = t_normed * s_sig + s_mu
                        loss_1 = kl_loss(feat1, mixstyle_s_t)
                        loss_2 = kl_loss(dict_f, mixstyle_t_s)
                        loss_3 = loss_fn(t_score, t_feat, t_target, t_target_cam)
                        feat_adv, cls_score, _ = class_net2(local_adv.detach(), t_img)
                        adv_score = torch.ones(cfg.INPUT.C_CAM) / cfg.INPUT.C_CAM
                        loss_adv = ide_creiteron1(cls_score, adv_score)
                        t_loss = loss_1 + loss_2 + loss_3 + loss_adv
                    scaler.scale(t_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                loss_meter.update(loss.item(), img.shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader[1]),
                                        loss_meter.avg, acc_meter.avg, optimizer.state_dict()['param_groups'][0]['lr']))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader[1].batch_size / time_per_batch))

        else:
            for i in range(Num_Batch):
                n_iter = i
                img, vid, target_cam, target_view = train_loader[0].next_one() #源域
                optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)

                with amp.autocast(enabled=True):
                    _,_,local_adv, score, feat,_ = model(img, target, cam_label=target_cam, view_label=target_view )
                    loss = loss_fn(score, feat, target, target_cam)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                target_iters=t_loader[0]
                for i in range(len(target_iters)):
                    img, vid, target_cam, target_view = train_loader[0].next_one()
                    t_img, t_vid, t_target_cam, t_target_view = target_iters[i].next_one()
                    optimizer.zero_grad()
                    img = img.to(device)
                    target = vid.to(device)
                    target_cam = target_cam.to(device)
                    target_view = target_view.to(device)

                    _,local_adv,_, score, feat, _ = model(img, target, cam_label=target_cam, view_label=target_view)
                    t_img = t_img.to(device)
                    t_target = t_vid.to(device)

                    with amp.autocast(enabled=True):
                        t_feat, t_score = model(t_img, modal=1, cam_label=t_target_cam) #目标域
                        #目标域
                        _,_,local_adv,_, dict_f, l_tfeat = model(t_img, t_target, cam_label=t_target_cam, view_label=t_target_view)
                        dict_f, _, _, _ = dict_f[0], dict_f[1], dict_f[2], dict_f[3]
                        adv_score = torch.ones(cfg.INPUT.C_CAM) / cfg.INPUT.C_CAM
                        loss_3 = loss_fn(t_score, t_feat, t_target, t_target_cam)

                    scaler.scale(loss_3).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                loss_meter.update(loss.item(), img.shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader[1]),
                                        loss_meter.avg, acc_meter.avg, optimizer.state_dict()['param_groups'][0]['lr']))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader[1].batch_size / time_per_batch))

        if epoch >= s1_train_num and epoch < s2_train_num:
            class_net.train()
            class_net2.train()
            for i in range(num_Batch):
                n_iter = i
                train_iters = train_loader[0]
                target_iters=target_train_loader[0]
                img, vid, target_cam, target_view = train_iters.next_one()
                t_img, t_vid, t_target_cam, t_target_view = target_iters.next_one()
                optimizer.zero_grad()
                class_optimizer.zero_grad()
                class2_optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)
                t_img = t_img.to(device)
                t_target = t_vid.to(device)
                t_target_cam = t_target_cam.to(device)
                t_target_view = t_target_view.to(device)


                if cfg.DATASETS.TARGET == 'marketSCT':
                    Cam_Nero = [6, 7, 8 ]
                elif cfg.DATASETS.TARGET == 'market1501':
                    Cam_Nero = [6, 7, 8 ]
                elif cfg.DATASETS.TARGET == 'msmt17':
                    Cam_Nero = [15, 16, 17]
                elif cfg.DATASETS.TARGET == 'msmt17_sct':
                    Cam_Nero = [15, 16, 17]

                top_label = (torch.ones(t_target_cam.size()) * Cam_Nero[0]).int()
                m_label = (torch.ones(t_target_cam.size()) * Cam_Nero[1]).int()
                b_label = (torch.ones(t_target_cam.size()) * Cam_Nero[2]).int()

                top_label = top_label.to(device)
                m_label = m_label.to(device)
                b_label = b_label.to(device)

                with amp.autocast(enabled=True):

                    _,_,_,_, score, l_tfeat = model(t_img, t_target, cam_label=t_target_cam, view_label=t_target_view)
                    t_top_feat, t_m_feat, t_b_feat = l_tfeat[0], l_tfeat[1], l_tfeat[2] #
                    t_top_cls, t_top_cls2, _ = class_net(t_top_feat.detach())
                    t_m_cls, t_m_cls2, _ = class_net(t_m_feat.detach())
                    t_b_cls, t_b_cls2, _ = class_net(t_b_feat.detach())
                    loss_t = loss_classifier[0]*(ide_creiteron1(t_top_cls, t_target_cam)+ide_creiteron1(t_m_cls, t_target_cam)+ide_creiteron1(t_b_cls, t_target_cam))\
                             +loss_classifier[1]*(ide_creiteron1(t_top_cls2, t_target_cam)+ide_creiteron1(t_m_cls2, t_target_cam)+ide_creiteron1(t_b_cls2, t_target_cam))

                    _,_,local_adv,_, _, l_feat = model(img, target, cam_label=target_cam, view_label=target_view)
                    top_feat, m_feat, b_feat = l_feat[0], l_feat[1], l_feat[2]
                    top_cls, top_cls2, _ = class_net(top_feat.detach())
                    m_cls, m_cls2, _ = class_net(m_feat.detach())
                    b_cls, b_cls2, _ = class_net(b_feat.detach())
                    feat_adv, cls_score, _ = class_net2(local_adv.detach(), img)
                    adv_score = torch.ones(cfg.INPUT.C_CAM) / cfg.INPUT.C_CAM
                    loss_adv = ide_creiteron1(cls_score, adv_score)

                    try:
                        loss_body = loss_classifier[0]*(ide_creiteron1(top_cls, top_label)+ide_creiteron1(m_cls, m_label)+ide_creiteron1(b_cls, b_label))\
                               +loss_classifier[1]*(ide_creiteron1(top_cls2, top_label)+ide_creiteron1( m_cls2, m_label)+ide_creiteron1(b_cls2, b_label))

                    except:
                        print('Error in loss_body')
                    loss = loss_t + loss_body + loss_adv

                scaler.scale(loss).backward()
                scaler.step(class_optimizer)
                scaler.update()

                # source
                with amp.autocast(enabled=True):
                    _,_,local_adv,score, feat,l_tfeat = model(img, target, cam_label=target_cam, view_label=target_view)
                    _,l_tfeat = model(t_img, t_target,modal=2, cam_label=target_cam, view_label=target_view )


                    id_loss = loss_fn(score, feat, target, target_cam)
                    t_top_feat,t_m_feat,t_b_feat = l_tfeat[0],l_tfeat[1],l_tfeat[2]
                    _,_,t_top_clss = class_net(t_top_feat)
                    _,_,t_m_clss = class_net(t_m_feat)
                    _,_,t_b_clss = class_net(t_b_feat)
                    loss = loss_classifier[2]*(ide_creiteron1(t_top_clss, top_label) + ide_creiteron1(t_m_clss, m_label) + ide_creiteron1(t_b_clss, b_label))\
                            +id_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                target_iters=t_loader[0]
                for i in range(len(target_iters)):
                    t_img, t_vid, t_target_cam, t_target_view = target_iters[i].next_one()
                    optimizer.zero_grad()
                    t_img = t_img.to(device)
                    t_target = t_vid.to(device)
                    with amp.autocast(enabled=True):
                        t_feat,t_score = model(t_img,modal=1,cam_label=t_target_cam)
                        t_loss = loss_fn(t_score, t_feat, t_target, t_target_cam)
                    scaler.scale(t_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                loss_meter.update(loss.item(), img.shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()

                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, optimizer.state_dict()['param_groups'][0]['lr']))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)

            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                            .format(epoch, time_per_batch, train_loader[1].batch_size / time_per_batch))

        if epoch >= s0_train_num:
            if epoch % eval_period == 0:
                # # If train with multi-gpu ddp mode, options: 'True', 'False'
                # not use in windows
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        model.eval()
                        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                            with torch.no_grad():
                                img = img.to(device)
                                camids = camids.to(device)
                                target_view = target_view.to(device)
                                feat = model(img, cam_label=camids, view_label=target_view)
                                evaluator.update((feat, vid, camid))
                        cmc, mAP, _, _, _, _, _ = evaluator.compute()
                        logger.info("Validation Results - Ep+och: {}".format(epoch))
                        logger.info("mAP: {:.1%}".format(mAP))
                        for r in [1, 5, 10]:
                            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                        if cmc[0] > best_top1:
                            best_top1 = max(cmc[0], best_top1)
                            best_top5 = cmc[4]
                            best_top10 = cmc[9]
                            best_map = mAP
                            best_epoch = epoch
                            # save the model
                            torch.save(model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, 'bestResult_{}.pth'.format(best_epoch)))
                        torch.cuda.empty_cache()
                else:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                    # save the best result
                    if cmc[0] > best_top1:
                        best_top1 = max(cmc[0], best_top1)
                        best_top5 = cmc[4]
                        best_top10 = cmc[9]
                        best_map = mAP
                        best_epoch = epoch
                        # save the model
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, 'bestResult_{}.pth'.format(best_epoch)))



                    logger.info("Best Results - Epoch: {}".format(best_epoch))
                    logger.info("Best Rank-1  :{:.1%} and mAP: {:.1%}".format(best_top1, best_map))
                    torch.cuda.empty_cache()
    # show the best result
    logger.info("Best Results - Epoch: {}".format(best_epoch))
    logger.info("mAP: {:.1%}".format(best_map))
    logger.info("CMC curve, Rank-1  :{:.1%}".format(best_top1))
    logger.info("CMC curve, Rank-5  :{:.1%}".format(best_top5))
    logger.info("CMC curve, Rank-10 :{:.1%}".format(best_top10))

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    # refresh model
    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]




def mean_weight(model):
    # softmax = nn.Softmax(dim=0)
    w1_0 = model.classifier[0].weight
    w2_0 = model.classifier2[0].weight
    w1_3 = model.classifier[3].weight
    w2_3 = model.classifier2[3].weight
    w1_6 = model.classifier[6].weight
    w2_6 = model.classifier2[6].weight
    w1_9 = model.classifier[9].weight
    w2_9 = model.classifier2[9].weight


    loss_weight = weight_diff(w1_0, w2_0) + weight_diff(w1_3, w2_3) + \
                  weight_diff(w1_6, w2_6) + weight_diff(w1_9, w2_9)
    loss_weight = loss_weight/4

    return loss_weight

def weight_diff(w1,w2):
    w2 = w2.view(-1)
    w1 = w1.view(-1)
    loss = (torch.matmul(w1,w2)/(torch.norm(w1)*torch.norm(w2))+1)
    return loss

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)