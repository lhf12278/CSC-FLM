import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .cuhk03 import cuhk03_np
from .vehicleid import VehicleID
from .veri import VeRi
__factory = {
    'market1501': Market1501,
    # 'marketSCT': Market1501,
    # 'cuhk03': cuhk03_np,
    # 'veri': VeRi,
    # 'VehicleID': VehicleID,
    # 'msmt17': MSMT17,
    'msmt17_sct': MSMT17,
}

class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg, is_target=False):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    if is_target:
        dataset = __factory[cfg.DATASETS.TARGET](root=cfg.DATASETS.ROOT_DIR) #target
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)  #source

    train_set = ImageDataset(dataset.train, train_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    # TARGET CAMERA NUM
    range_CAM = 0

    if is_target:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=num_workers, collate_fn=train_collate_fn
        )

        if cfg.DATASETS.TARGET == 'marketSCT':
            range_CAM = 6
        if cfg.DATASETS.TARGET == 'marketmix':
            range_CAM = 14
        if cfg.DATASETS.TARGET == 'market1501':
            range_CAM = 6
        if cfg.DATASETS.TARGET == 'dukeSCT':
            range_CAM = 8
        if cfg.DATASETS.TARGET == 'dukemtmc':
            range_CAM = 8
        if cfg.DATASETS.TARGET == 'msmt17_sct':
            range_CAM = 15
        if cfg.DATASETS.TARGET == 'msmt17':
            range_CAM = 15

        t_trains = []
        for i in range(range_CAM):
            t_trains.append(ImageDataset(dataset.c_train[i], train_transforms))
        t_loader = []
        for i in range(range_CAM):
            t_loader.append(DataLoader(
                t_trains[i], batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.c_train[i], cfg.SOLVER.IMS_PER_BATCH,
                                              cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            ))
        cids = dataset.c_pids
        t_iters = []
        for i in range(range_CAM):
            t_iters.append(IterLoader(t_loader[i]))
    elif 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH,
                                                     cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    # test dataset combine with query and gallery
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    iters = IterLoader(train_loader)

    # for ori test and train
    if is_target:
        return [iters, train_loader], [t_iters, t_loader], val_loader, len(
            dataset.query), num_classes, cam_num, view_num, cids
    else:
        return [iters, train_loader], val_loader, len(dataset.query), num_classes, cam_num, view_num
