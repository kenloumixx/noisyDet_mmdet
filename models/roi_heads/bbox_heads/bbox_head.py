# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer
from mmcv.runner import get_dist_info

@HEADS.register_module()
class BBoxHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg=None):
        super(BBoxHead, self).__init__(init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        rank, world_size = get_dist_info()
        if rank == 0:
            self.log_file = open('compare_sup_unsup.txt', 'w')
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=in_channels,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=in_channels,
                out_features=out_dim_reg)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.01, override=dict(name='fc_cls'))
                ]
            if self.with_reg:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.001, override=dict(name='fc_reg'))
                ]

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, pos_gmm_labels, cfg, unsup):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)

        if (unsup[0] == True) and num_pos != 0:
            rank, world_size = get_dist_info()
            if rank == 0:
                self.log_file.write(f'unsup and num_pose {num_pos}\n')
            
        if num_pos > 0:   # ids
            max_vals, max_ids = torch.max(pos_gmm_labels, dim=-1)
            if unsup == False:  # adaptive threshold
                pos_gt_ids_output = torch.where(max_vals >= 0.8)  
            else:   # automatic filtering
                tau_auto = (1 - max_vals) * 0.95 + max_vals * 0.5
                pos_gt_ids_output = torch.where(max_vals >= tau_auto)   
            pos_gt_ids = pos_gt_ids_output[0]   # tuple to tensor. 
            
            if len(pos_gt_ids) > 0:  # 이미지에 gt가 1개 이상은 있고 > 0 and max_vals >= threshold -> pred를 loss 계산에 반영할 준비가 되었다. 
                try:    # thres를 넘은 box의 ids에 대해, 그 box의 gmm val의 idx가 [0, 1, 2, 3] 중 어디에 해당? 
                    label_pos_ids = [ids.item() for ids in pos_gt_ids if (max_ids[ids] == 0) or (max_ids[ids] == 1)]
                    bbox_pos_ids = [ids.item() for ids in pos_gt_ids if (max_ids[ids] == 0) or (max_ids[ids] == 2)]

                except: # when same gt boxes are assigned to the predicted pos boxes
                    label_pos_ids = [ids.item() for ids in pos_gt_ids if (max_ids[ids][0] == 0) or (max_ids[ids][0] == 1)]
                    bbox_pos_ids = [ids.item() for ids in pos_gt_ids if (max_ids[ids][0] == 0) or (max_ids[ids][0] == 2)]

                # label weight
                labels[label_pos_ids] = pos_gt_labels[[label_pos_ids]]    # 여기도 확인해야함!
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                label_weights[label_pos_ids] = pos_weight
                if not self.reg_decoded_bbox:
                    pos_bbox_targets = self.bbox_coder.encode(
                        pos_bboxes, pos_gt_bboxes)
                else:
                    # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                    # is applied directly on the decoded bounding boxes, both
                    # the predicted boxes and regression targets should be with
                    # absolute coordinate format.
                    pos_bbox_targets = pos_gt_bboxes
                        
                # bbox weight
                bbox_targets[bbox_pos_ids] = pos_bbox_targets[[bbox_pos_ids]]
                bbox_weights[bbox_pos_ids] = 1
                
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights


    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    unsup=False,
                    gmm_labels=None, 
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        # 다 3개씩 들어가는 중
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]      
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]          # 한 bbox 당 512개 sample 중 gt 제외하고 
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]    # gt 갯수만큼 나옴  -> 19개? 21개?
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]    # gt 갯수만큼 나옴

        if gmm_labels == None:
            pos_gmm_labels = [pos_bboxes.new_full((pos_bboxes.size(0), ),
                                True,
                                dtype=torch.long).bool()
                                for pos_bboxes in pos_bboxes_list]
        else:
            pos_gmm_labels = [gmm[res.pos_assigned_gt_inds] for gmm, res in zip(gmm_labels, sampling_results)] 

        unsup_list = [unsup for _ in range(len(gmm_labels))]
        
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gmm_labels,
            unsup=unsup_list,
            cfg=rcnn_train_cfg)
        
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def smooth_L1(pred, target, beta=1.0):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                        diff - 0.5 * beta)
        return loss

    # ann inds -> 17개
    # [125737, 1209135, 1217974, 1237725, 1260661, 1280377, 1287342, 1336430, 1475393, 1709486, 1743150, 1745379, 1755035, 2017209, 2018401, 2159863, 900100087038]
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def gmm_loss(self,
             cls_score,
             bbox_pred,
             rois,
             gt_labels, 
             gt_bboxes, 
             ):
        gt_labels, gt_bboxes = gt_labels[0], gt_bboxes[0]
        cls_loss = F.cross_entropy(cls_score, gt_labels, reduction='none')

        reshaped_gt_labels = gt_labels.reshape(gt_labels.size(-1), 1, 1).repeat(1, 1, 4)
        pos_bbox_pred = torch.gather(bbox_pred.view(bbox_pred.size(0), -1, 4), 1, reshaped_gt_labels).squeeze(1)

        bbox_loss = self.loss_bbox(
                    pos_bbox_pred,
                    gt_bboxes,
                    reduction_override='none').mean(dim=-1)
        return cls_loss, gt_labels, bbox_loss, cls_score



    # # ann inds -> 17개
    # # [125737, 1209135, 1217974, 1237725, 1260661, 1280377, 1287342, 1336430, 1475393, 1709486, 1743150, 1745379, 1755035, 2017209, 2018401, 2159863, 900100087038]
    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    # def gmm_loss_splitnet(self,
    #          cls_score,
    #          bbox_pred,
    #          rois,
    #          box_ids,
    #          gt_pos_ids, 
    #          labels,
    #          label_weights,
    #          bbox_targets,
    #          bbox_weights,
    #          reduction_override=None):
    #         #  gt_labels, 
    #         #  gt_bboxes, 
    #         #  ):

    #     # gt_pos_ids = sampled bbox들이 어떤 gt bbox idx(0~len(label))와 매핑되는지
    #     import pdb 
    #     pdb.set_trace()
    #     losses = dict()
    #     if cls_score is not None:
    #         avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
    #         if cls_score.numel() > 0:
    #             loss_cls_ = self.loss_cls(
    #                 cls_score,
    #                 labels,
    #                 label_weights,
    #                 avg_factor=avg_factor,
    #                 reduction_override=reduction_override)
    #             if isinstance(loss_cls_, dict):
    #                 losses.update(loss_cls_)
    #             else:
    #                 losses['loss_cls'] = loss_cls_
    #             if self.custom_activation:
    #                 acc_ = self.loss_cls.get_accuracy(cls_score, labels)
    #                 losses.update(acc_)
    #             else:
    #                 losses['acc'] = accuracy(cls_score, labels)
    #     if bbox_pred is not None:
    #         bg_class_ind = self.num_classes
    #         # 0~self.num_classes-1 are FG, self.num_classes is BG
    #         pos_inds = (labels >= 0) & (labels < bg_class_ind)
    #         # do not perform bounding box regression for BG anymore.
    #         if pos_inds.any():
    #             if self.reg_decoded_bbox:
    #                 # When the regression loss (e.g. `IouLoss`,
    #                 # `GIouLoss`, `DIouLoss`) is applied directly on
    #                 # the decoded bounding boxes, it decodes the
    #                 # already encoded coordinates to absolute format.
    #                 bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
    #             if self.reg_class_agnostic:
    #                 pos_bbox_pred = bbox_pred.view(
    #                     bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
    #             else:
    #                 pos_bbox_pred = bbox_pred.view(
    #                     bbox_pred.size(0), -1,
    #                     4)[pos_inds.type(torch.bool),
    #                        labels[pos_inds.type(torch.bool)]]
    #             losses['loss_bbox'] = self.loss_bbox(
    #                 pos_bbox_pred,
    #                 bbox_targets[pos_inds.type(torch.bool)],
    #                 bbox_weights[pos_inds.type(torch.bool)],
    #                 avg_factor=bbox_targets.size(0),
    #                 reduction_override=reduction_override)
    #         else:
    #             losses['loss_bbox'] = bbox_pred[pos_inds].sum()
            
    #     '''
    #     gt_labels, gt_bboxes = gt_labels[0], gt_bboxes[0]
    #     cls_loss = F.cross_entropy(cls_score, gt_labels, reduction='none')

    #     reshaped_gt_labels = gt_labels.reshape(gt_labels.size(-1), 1, 1).repeat(1, 1, 4)
    #     pos_bbox_pred = torch.gather(bbox_pred.view(bbox_pred.size(0), -1, 4), 1, reshaped_gt_labels).squeeze(1)

    #     bbox_loss = self.loss_bbox(
    #                 pos_bbox_pred,
    #                 gt_bboxes,
    #                 reduction_override='none').mean(dim=-1)
    #     '''
    #     cls_loss, cls_labels, bbox_loss, logits = None, None, None, None
    #     # return cls_loss, gt_labels, bbox_loss, cls_score
    #     return  cls_loss, cls_labels, bbox_loss, logits



    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): Rois from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4) or
                (num_proposals, 5).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """

        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        max_shape = img_meta['img_shape']

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=max_shape)
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=max_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

    def onnx_export(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    cfg=None,
                    **kwargs):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed.
                Has shape (B, num_boxes, 5)
            cls_score (Tensor): Box scores. has shape
                (B, num_boxes, num_classes + 1), 1 represent the background.
            bbox_pred (Tensor, optional): Box energies / deltas for,
                has shape (B, num_boxes, num_classes * 4) when.
            img_shape (torch.Tensor): Shape of image.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """

        assert rois.ndim == 3, 'Only support export two stage ' \
                               'model to ONNX ' \
                               'with batch dimension. '
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class',
                                                 cfg.max_per_img)
        iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        score_threshold = cfg.score_thr
        nms_pre = cfg.get('deploy_nms_pre', -1)

        scores = scores[..., :self.num_classes]
        if self.reg_class_agnostic:
            return add_dummy_nms_for_onnx(
                bboxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img)
        else:
            batch_size = scores.shape[0]
            labels = torch.arange(
                self.num_classes, dtype=torch.long).to(scores.device)
            labels = labels.view(1, 1, -1).expand_as(scores)
            labels = labels.reshape(batch_size, -1)
            scores = scores.reshape(batch_size, -1)
            bboxes = bboxes.reshape(batch_size, -1, 4)

            max_size = torch.max(img_shape)
            # Offset bboxes of each class so that bboxes of different labels
            #  do not overlap.
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes_for_nms = bboxes + offsets

            batch_dets, labels = add_dummy_nms_for_onnx(
                bboxes_for_nms,
                scores.unsqueeze(2),
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img,
                labels=labels)
            # Offset the bboxes back after dummy nms.
            offsets = (labels * max_size + 1).unsqueeze(2)
            # Indexing + inplace operation fails with dynamic shape in ONNX
            # original style: batch_dets[..., :4] -= offsets
            bboxes, scores = batch_dets[..., 0:4], batch_dets[..., 4:5]
            bboxes -= offsets
            batch_dets = torch.cat([bboxes, scores], dim=2)
            return batch_dets, labels
