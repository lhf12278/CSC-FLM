import torch


def make_optimizer(cfg, model,coarse_classi, refine_classi, class_net,class_net2):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    coarse_classi_optimizer = torch.optim.SGD(coarse_classi.parameters(), lr=cfg.SOLVER.C_LR)
    refine_classi_optimizer = torch.optim.SGD(refine_classi.parameters(), lr=cfg.SOLVER.C_LR)
    class_optimizer = torch.optim.SGD(class_net.parameters(), lr=cfg.SOLVER.C_LR)
    class2_optimizer = torch.optim.SGD(class_net2.parameters(), lr=cfg.SOLVER.C_LR)
    return  optimizer, coarse_classi_optimizer, refine_classi_optimizer, class_optimizer, class2_optimizer

