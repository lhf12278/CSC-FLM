import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="../vit_3_domain.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Wang 211003
    # loss para
    loss_classifier = [0.08, 0.59, 1.45]
    # just for writing

    logger = setup_logger("transreid", output_dir, loss_classifier, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    # ori
    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # dataloader for target
    target_train_loader, t_loader, target_val_loader, target_num_query, \
    target_num_classes, target_camera_num, target_view_num, c_pids = make_dataloader(cfg, True)


    model, coarse_classi, refine_classi, class_net, class_net2 = make_model(cfg, num_class=num_classes,
                                              num_class_t=c_pids, camera_num=camera_num, view_num=view_num)

    model.load_param(cfg.TEST.WEIGHT)

    # init dataset
    # _datasets = [val_query_loader.dataset.dataset, val_gallery_loader.dataset.dataset]
    # _loader = [val_query_loader, val_gallery_loader]

    # Compute query and gallery features
    # with torch.no_grad():
    #     for loader_id, loader in enumerate(_loader):
    #         for data in loader:

    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(
            #     cfg)
            train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

            # dataloader for target
            target_train_loader, t_loader, target_val_loader, target_num_query, \
            target_num_classes, target_camera_num, target_view_num, c_pids = make_dataloader(cfg, True)
            rank_1, rank5 = do_inference(cfg,
                                         model,
                                         target_val_loader,
                                         target_num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum() / 10.0, all_rank_5.sum() / 10.0))
    else:
        do_inference(cfg,
                  model,
                  target_val_loader,
                  target_num_query)



