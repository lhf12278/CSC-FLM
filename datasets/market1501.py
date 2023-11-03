# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'marketSCT'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train_sct')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train,c_trains = self._process_dir(self.train_dir, relabel=True)
        query,_ = self._process_dir(self.query_dir, relabel=False)
        gallery,_ = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.c_train = c_trains
        self.query = query
        self.gallery = gallery

        self.c_pids=[]
        #for i in range(6):
        for i in range(6):
            self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
                self.c_train[i])
            self.c_pids.append(self.num_train_pids)
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        dataset = self._process_c_dir(img_paths,relabel)
        c_datasets = self.camera_datasets(img_paths,relabel)
        return dataset,c_datasets

    def camera_datasets(self,img_paths,relabel):
        cimg_paths = []
        for i in range(6):
            cimg_paths.append([])
        pattern = re.compile(r'([-\d]+)_c(\d)')

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            #cam_id = 0-5
            camid -= 1
            for i in range(6):
                if camid==i:
                    cimg_paths[i].append(img_path)

        c_datasets = []
        for i in range(6):
            c_datasets.append(self._process_c_dir(cimg_paths[i], relabel))
        return c_datasets

    def _process_c_dir(self,img_paths,relabel=False):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background   1501
            assert 1 <= camid <= 6   #6
            camid -= 1  # index starts from 0 camera_id = 0-5
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset