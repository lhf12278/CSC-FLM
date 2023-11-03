from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="../vit_3_domain.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)

    # set the loss para
    parser.add_argument("--loss1", default=1, type=float)
    parser.add_argument("--loss2", default=0.5, type=float)
    parser.add_argument("--lossU", default=2, type=float)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loss_classifier = [0, 0, 0]
    loss_classifier[0] = args.loss1
    loss_classifier[1] = args.loss2
    loss_classifier[2] = args.lossU

    logger = setup_logger("transreid", output_dir, loss_classifier, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    # the log path setting
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # dataloader for source
    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # dataloader for target
    target_train_loader, t_loader, target_val_loader, target_num_query, \
    target_num_classes, target_camera_num, target_view_num, c_pids = make_dataloader(cfg, True)

    model, coarse_classi, refine_classi, class_net, class_net2 = make_model(cfg, num_class=num_classes,
                                              num_class_t=c_pids, camera_num=camera_num, view_num=view_num)

    loss_func, ide_creiteron, ide_creiteron1, criterion_tri,ide_creiteron2  = make_loss(cfg, num_classes=num_classes)


    optimizer, coarse_classi_optimizer, refine_classi_optimizer, class_optimizer, class2_optimizer = make_optimizer(cfg, model, coarse_classi, refine_classi, class_net, class_net2)

    scheduler, coarse_classi_scheduer, refine_classi_scheduer,  class_scheduler, class2_scheduler = create_scheduler(cfg, optimizer, coarse_classi_optimizer, refine_classi_optimizer, class_optimizer, class2_optimizer)

    do_train(
        cfg,
        model, coarse_classi, refine_classi,
        class_net,
        class_net2,
        train_loader,
        target_val_loader,
        optimizer,
        scheduler,
        loss_func,
        target_num_query, args.local_rank,
        coarse_classi_optimizer, refine_classi_optimizer, class_optimizer, class2_optimizer,
        coarse_classi_scheduer, refine_classi_scheduer, class_scheduler, class2_scheduler,
        target_train_loader, ide_creiteron, ide_creiteron1, loss_classifier, criterion_tri, t_loader
    )

