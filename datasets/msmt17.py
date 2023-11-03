import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17/MSMT17_V1'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train, c_train = self._process_dir(self.train_dir, self.list_train_path)

        val, c_val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        for i in range(len(c_train)):
            c_train[i] += c_val[i]
            for j in range(len(c_train[i])):
                c_train[i][j] = list(c_train[i][j])
            c_train[i] = self._relabels(c_train[i], 1)
            for k in range(len(c_train[i])):
                c_train[i][k] = tuple(c_train[i][k])

        query, _ = self._process_dir(self.test_dir, self.list_query_path)
        gallery, _ = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> msmt17_sct loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.c_train = c_train
        self.query = query
        self.gallery = gallery

        self.c_pids = []
        for i in range(15):
            self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
                self.c_train[i])
            self.c_pids.append(self.num_train_pids)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _relabels(self, samples, label_index):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        a = ids
        b = set(ids)
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        c_datasets_dict = {}
        for i in range(15):
            c_datasets_dict[i] = []

        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin + pid, camid - 1, 1))
            pid_container.add(pid)
            cam_container.add(camid)
            c_datasets_dict[camid - 1].append((img_path, self.pid_begin + pid, camid - 1, 1))

        c_datasets = []
        for i in range(15):
            c_datasets.append(c_datasets_dict[i])
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        # for idx, pid in enumerate(pid_container):
        #     assert idx == pid, "See code comment for explanation"
        return dataset, c_datasets
