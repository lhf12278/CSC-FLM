""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import  WarmupMultiStepLR


def create_scheduler(cfg, optimizer, coarse_classi_optimizer, refine_classi_optimizer, class_optimizer, class2_optimizer):
    # num_epochs = cfg.SOLVER.MAX_EPOCHS
    # # type 1
    # # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # # type 2
    # lr_min = 0.002 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    # # type 3
    # # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    #
    # warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    # noise_range = None
    #
    # lr_scheduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=num_epochs,
    #         lr_min=lr_min,
    #         t_mul= 1.,
    #         decay_rate=0.1,
    #         warmup_lr_init=warmup_lr_init,
    #         warmup_t=warmup_t,
    #         cycle_limit=1,
    #         t_in_epochs=True,
    #         noise_range_t=noise_range,
    #         noise_pct= 0.67,
    #         noise_std= 1.,
    #         noise_seed=42,
    #     )
    # class_lr_scheduler = CosineLRScheduler(
    #     class_optimizer,
    #     t_initial=num_epochs,
    #     lr_min=lr_min,
    #     t_mul= 1.,
    #     decay_rate=0.1,
    #     warmup_lr_init=warmup_lr_init,
    #     warmup_t=warmup_t,
    #     cycle_limit=1,
    #     t_in_epochs=True,
    #     noise_range_t=noise_range,
    #     noise_pct= 0.67,
    #     noise_std= 1.,
    #     noise_seed=42,
    # )

    # def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
    #              warmup_method="linear", last_epoch=-1)


    lr_scheduler = WarmupMultiStepLR(optimizer, [40], gamma=0.1, warmup_factor=0.01,
                                          warmup_iters=10)
    coarse_classi_lr_scheduer = WarmupMultiStepLR(coarse_classi_optimizer, [40], gamma=0.1, warmup_factor=0.01,
                                     warmup_iters=10)
    refine_classi_lr_scheduer = WarmupMultiStepLR(refine_classi_optimizer, [40], gamma=0.1, warmup_factor=0.01,
                                     warmup_iters=10)
    class_lr_scheduler = WarmupMultiStepLR(class_optimizer, [40], gamma=0.1, warmup_factor=0.01,
                                     warmup_iters=10)
    class2_lr_scheduler = WarmupMultiStepLR(class2_optimizer, [40], gamma=0.1, warmup_factor=0.01,
                                           warmup_iters=10)

    return  lr_scheduler, coarse_classi_lr_scheduer, refine_classi_lr_scheduer, class_lr_scheduler, class2_lr_scheduler
