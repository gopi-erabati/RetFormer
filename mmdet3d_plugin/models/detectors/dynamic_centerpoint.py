import numpy as np
import time
import torch
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.dynamic_voxelnet import DynamicVoxelNet
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d


@DETECTORS.register_module()
class DynamicCenterPoint(DynamicVoxelNet):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_.
    """

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DynamicCenterPoint, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.time_list = []

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.
        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.bbox_head.loss(*loss_inputs)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        # st = time.time()
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        # et = time.time()
        # self.time_list.append(et-st)
        # if len(self.time_list) > 1100:
        #     print(f"avg time: {np.mean(self.time_list[100:1099])}")
        #     np.save("/home/gopi/PhD/workdirs/RetFormer/waymo"
        #             "/retformer_waymo_D1_2x_3class/timearray.npy", np.array(
        #         self.time_list))
        bbox_list = self.bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]