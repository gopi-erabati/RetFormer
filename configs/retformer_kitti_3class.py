plugin = True
plugin_dir = 'mmdet3d_plugin'

voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 79.36, 39.68, 1]
sparse_shape = [496, 496, 1]
window_shape = [18, 18, 1]
group_size = 276

model = dict(
    type='DynamicCenterPoint',
    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    voxel_encoder=dict(
        type='DynamicVFECustom',
        in_channels=4,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),
    middle_encoder=dict(
        type='RetFormer',
        embed_dim=128,
        num_heads=4,
        num_blocks=2,
        activation="gelu",
        window_shape=window_shape,
        sparse_shape=sparse_shape,
        output_shape=sparse_shape[:2],
        pos_temperature=10000,
        normalize_pos=False,
        group_size=group_size,
    ),
    backbone=dict(
        type='SECONDCustom',
        in_channels=128,
        out_channels=[64, 128],
        layer_nums=[3, 3],
        layer_strides=[1, 2],
        conv_cfg=dict(type='Conv2d', bias=False),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        residual=True,
    ),
    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128],
        upsample_strides=[1, 2],
        out_channels=[128, 128]
    ),
bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=3, class_names=['pedestrian', 'cyclist', 'car']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), iou=(1, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[0, -40, -5, 72, 40, 5],
            max_num=4096,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True
    ),
    # model training and testing settings
    train_cfg=dict(
        grid_size=sparse_shape,
        voxel_size=voxel_size,
        out_size_factor=1,
        dense_reg=1,  # not used
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        point_cloud_range=point_cloud_range,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        iou_weight=1.0
    ),
    test_cfg=dict(
        post_center_limit_range=[0, -50, -5, 80, 50, 5],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        # not used in normal nms, task-wise
        score_threshold=0.1,
        pc_range=point_cloud_range[:2],  # seems not used
        out_size_factor=1,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        pre_max_size=4096,
        post_max_size=500,
        nms_thr=0.7,
        iou_pow=2.0
    ),
)

# dataset settings
# dataset settings

dataset_type = 'CustomKittiDataset'
data_root = 'data/kitti/'
file_client_args = dict(backend='disk')
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -39.68, -2.999, 69.12, 39.68, 0.999]
# add a small margin in z-axis
input_modality = dict(use_lidar=True, use_camera=False)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    file_client_args=file_client_args)

db_sampler_cyc = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Cyclist=10)),
    classes=['Cyclist'],
    sample_groups=dict(Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    file_client_args=file_client_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            file_client_args=file_client_args)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=file_client_args))

evaluation = dict(interval=1, pipeline=eval_pipeline)

lr = 1e-5
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(100, 1e-3),
    cyclic_times=1,
    step_ratio_up=0.1,
)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=40)

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
cudnn_benchmark = False
workflow = [('train', 1)]
seed = 0
deterministic = False

find_unused_parameters = True
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
