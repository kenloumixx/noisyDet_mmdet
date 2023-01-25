# Copyright (c) OpenMMLab. All rights reserved.

from torch.utils.data import Dataset

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
from mmcv.runner import get_dist_info


import numpy as np


@DATASETS.register_module()
class GMMCOCO(CustomDataset):
    def __init__(self,
                 splitnet_data, samples_per_gpu, workers_per_gpu, pipeline, test_mode=False):
        self.box_ids = splitnet_data[0]
        self.loss_bbox = splitnet_data[1]
        self.logits = splitnet_data[2]
        self.cls_labels = splitnet_data[3]
        self.gmm_labels = splitnet_data[4]
        self.logits_delta = splitnet_data[5]
        self.loss_bbox_delta = splitnet_data[6]
        # self.GMM_GT_idx = splitnet_data[7]


        self.test_mode = test_mode  # 의미없음. 그냥 계속 test_mode=False로 두기
        self.data_infos = splitnet_data
        self.pipeline = Compose(pipeline)
        self._set_group_flag()
    
    
    def __len__(self):
        """Total number of samples of data."""
        return len(self.cls_labels)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 0

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        logits = self.logits[idx]
        cls_labels = self.cls_labels[idx]
        loss_bbox = self.loss_bbox[idx]
        box_ids = self.box_ids[idx]
        gmm_labels = self.gmm_labels[idx]
        logits_delta = self.logits_delta[idx]
        loss_bbox_delta = self.loss_bbox_delta[idx]
        # GMM_GT_idx = self.GMM_GT_idx[idx]

        data = dict(logits=logits,
                    cls_labels=cls_labels,
                    loss_bbox=loss_bbox,
                    box_ids=box_ids,
                    logits_delta=logits_delta,
                    loss_bbox_delta=loss_bbox_delta,
                    # GMM_GT_idx=GMM_GT_idx,
                    gmm_labels=gmm_labels)
        return self.pipeline(data)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        pass