import os
from os import path as osp
import tempfile
import numpy as np
from mmcv.utils import print_log
from mmdet3d.datasets import DATASETS
from mmdet3d.datasets import WaymoDataset
from mmdet3d.core.bbox import Box3DMode, Coord3DMode
from ..core.visualizer import (show_result, show_multi_modality_result,
                               show_bev_result_waymo)


@DATASETS.register_module()
class WaymoDatasetCustom(WaymoDataset):
    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        assert ('waymo' in metric or 'kitti' in metric), \
            f'invalid metric {metric}'
        if 'kitti' in metric:
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='kitti')
            from mmdet3d.core.evaluation import kitti_eval
            gt_annos = [info['annos'] for info in self.data_infos]

            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bev', '3d']
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(
                            '{:.4f}'.format(ap))

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos,
                    result_files,
                    self.CLASSES,
                    eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
        if 'waymo' in metric:
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='waymo')
            import subprocess
            ret_bytes = subprocess.check_output(
                'mmdet3d_plugin/'
                f'core/evaluation/waymo_utils/compute_detection_metrics_main {pklfile_prefix}.bin ' +
                f'{waymo_root}/gt.bin',
                shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts)
            # parse the text to get ap_dict
            ap_dict = {
                'Vehicle/L1 mAP': 0,
                'Vehicle/L1 mAPH': 0,
                'Vehicle/L2 mAP': 0,
                'Vehicle/L2 mAPH': 0,
                'Pedestrian/L1 mAP': 0,
                'Pedestrian/L1 mAPH': 0,
                'Pedestrian/L2 mAP': 0,
                'Pedestrian/L2 mAPH': 0,
                'Sign/L1 mAP': 0,
                'Sign/L1 mAPH': 0,
                'Sign/L2 mAP': 0,
                'Sign/L2 mAPH': 0,
                'Cyclist/L1 mAP': 0,
                'Cyclist/L1 mAPH': 0,
                'Cyclist/L2 mAP': 0,
                'Cyclist/L2 mAPH': 0,
                'Overall/L1 mAP': 0,
                'Overall/L1 mAPH': 0,
                'Overall/L2 mAP': 0,
                'Overall/L2 mAPH': 0
            }
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 2) + 1
                if idx % 2 == 0:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            ap_dict['Overall/L1 mAP'] = \
                (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
                 ap_dict['Cyclist/L1 mAP']) / 3
            ap_dict['Overall/L1 mAPH'] = \
                (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
                 ap_dict['Cyclist/L1 mAPH']) / 3
            ap_dict['Overall/L2 mAP'] = \
                (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
                 ap_dict['Cyclist/L2 mAP']) / 3
            ap_dict['Overall/L2 mAPH'] = \
                (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
                 ap_dict['Cyclist/L2 mAPH']) / 3
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        show_threshold = 0.35
        save_imgs = True
        from tqdm import tqdm
        for i, result in tqdm(enumerate(results)):
            if i not in [4723, 7664, 8694, 8700, 8790, 10110, 12119, 13730,
                         15627, 16319, 18095, 18125, 18794, 18965]:
                continue
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points_org = points.numpy()

            # Show boxes on point cloud
            # for now we convert points into depth mode
            points_depth = Coord3DMode.convert_point(points_org,
                                                     Coord3DMode.LIDAR,
                                                     Coord3DMode.DEPTH)

            # Get GT Boxes and Filter by range and name
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            mask = gt_bboxes.in_range_bev([-76.8, -76.8, 76.8, 76.8])
            gt_bboxes = gt_bboxes[mask]
            gt_bboxes.limit_yaw(offset=0.5, period=2 * np.pi)
            gt_labels = gt_labels[mask.numpy().astype(np.bool)]
            # name filtering
            labels = list(range(3))
            gt_bboxes_mask = np.array([n in labels for n in gt_labels],
                                      dtype=np.bool_)
            gt_bboxes = gt_bboxes[gt_bboxes_mask]
            gt_labels = gt_labels[gt_bboxes_mask]

            # Convert GT boxes to Depth Mode to show with Visualizer
            gt_bboxes_numpy = gt_bboxes.tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes_numpy,
                                               Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)

            # Get Prediction Boxes
            inds = result['scores_3d'] > show_threshold
            pred_bboxes = result['boxes_3d'][inds]
            pred_bboxes_numpy = pred_bboxes.tensor.numpy()
            pred_labels = result['labels_3d'][inds]
            show_pred_bboxes = Box3DMode.convert(pred_bboxes_numpy,
                                                 Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points_depth, show_gt_bboxes, show_pred_bboxes,
                        out_dir,
                        file_name, show, pred_labels=pred_labels,
                        gt_labels=gt_labels)

            # BEV Show and Save
            # show_bev_result_waymo(points_org, coord_type='LIDAR',
            #                       gt_bboxes=gt_bboxes,
            #                       pred_bboxes=pred_bboxes,
            #                       out_dir=out_dir, filename=str(i),
            #                       show=show,
            #                       pred_labels=pred_labels,
            #                       gt_labels=gt_labels,
            #                       save=save_imgs, voxel_size=0.2,
            #                       bev_img_size=1024)

@DATASETS.register_module()
class MultiSweepsWaymoDataset(WaymoDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 save_training=False):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            load_interval=load_interval,
            pcd_limit_range=pcd_limit_range, )

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img,
            sweeps=info['sweeps'],
            timestamp=info['timestamp'],
            pose=info['pose'],
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        show_threshold = 0.35
        save_imgs = True
        from tqdm import tqdm
        for i, result in tqdm(enumerate(results)):
            if i not in [4723, 7664, 8694, 8700, 8790, 10110, 12119, 13730,
                         15627, 16319, 18095, 18125, 18794, 18965]:
                continue
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points_org = points.numpy()

            # Show boxes on point cloud
            # for now we convert points into depth mode
            points_depth = Coord3DMode.convert_point(points_org,
                                                     Coord3DMode.LIDAR,
                                                     Coord3DMode.DEPTH)

            # Get GT Boxes and Filter by range and name
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            mask = gt_bboxes.in_range_bev([-76.8, -76.8, 76.8, 76.8])
            gt_bboxes = gt_bboxes[mask]
            gt_bboxes.limit_yaw(offset=0.5, period=2 * np.pi)
            gt_labels = gt_labels[mask.numpy().astype(np.bool)]
            # name filtering
            labels = list(range(3))
            gt_bboxes_mask = np.array([n in labels for n in gt_labels],
                                      dtype=np.bool_)
            gt_bboxes = gt_bboxes[gt_bboxes_mask]
            gt_labels = gt_labels[gt_bboxes_mask]

            # Convert GT boxes to Depth Mode to show with Visualizer
            gt_bboxes_numpy = gt_bboxes.tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes_numpy,
                                               Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)

            # Get Prediction Boxes
            inds = result['scores_3d'] > show_threshold
            pred_bboxes = result['boxes_3d'][inds]
            pred_bboxes_numpy = pred_bboxes.tensor.numpy()
            pred_labels = result['labels_3d'][inds]
            show_pred_bboxes = Box3DMode.convert(pred_bboxes_numpy,
                                                 Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points_depth, show_gt_bboxes, show_pred_bboxes,
                        out_dir,
                        file_name, show, pred_labels=pred_labels,
                        gt_labels=gt_labels)

            # BEV Show and Save
            # show_bev_result_waymo(points_org, coord_type='LIDAR',
            #                       gt_bboxes=gt_bboxes,
            #                       pred_bboxes=pred_bboxes,
            #                       out_dir=out_dir, filename=str(i),
            #                       show=show,
            #                       pred_labels=pred_labels,
            #                       gt_labels=gt_labels,
            #                       save=save_imgs, voxel_size=0.2,
            #                       bev_img_size=1024)
