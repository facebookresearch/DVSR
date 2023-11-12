# Copyright (c) Meta Platforms, Inc. and affiliates.

exp_name = 'dvsr_tartan'

# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='DVSR',
        mid_channels=64,
        num_blocks=7,
        scale=16,
        is_low_res_input=True,
        spynet_pretrained='pretrained/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['MAE_masked'], crop_border=0) ## TODO

# dataset settings
train_dataset_type = 'CustomRGBDMultiFrameDataset'
val_dataset_type = 'CustomRGBDMultiFrameDataset'
test_dataset_type = 'CustomRGBDMultiFrameDataset'

ds_scale = 16
train_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='guide',
        channel_order='rgb'),
    dict(
        type='LoadDFromFileList',
        io_backend='disk',
        key='gt'),
    dict(
        type='DToFSimulator',
        scale = ds_scale,
        temp_res = 1024,
        dtof_sampler = 'peak',
        key='lq'),
    dict(type='ColorJitter', keys=['guide'], 
        brightness=0.01, contrast=0.3, saturation=0.3, hue=0.5 / 3.14),
    dict(type='RescaleToZeroOne', keys=['guide']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'guide', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'guide', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'guide', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'guide', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'guide', 'gt'],
        meta_keys=['guide_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='guide',
        channel_order='rgb'),
    dict(
        type='LoadDFromFileList',
        with_conf=True,
        io_backend='disk',
        key='gt'),
    dict(
        type='DToFSimulator',
        scale = ds_scale,
        temp_res = 1024,
        dtof_sampler = 'peak',
        with_conf=True,
        key='lq'),
    dict(type='RescaleToZeroOne', keys=['guide']),
    dict(type='FramesToTensor', keys=['lq', 'guide', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'guide', 'gt'],
        meta_keys=['guide_path', 'gt_path'])
]

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type=train_dataset_type,
        guide_folder='data/tartanair/color',
        gt_folder = 'data/tartanair/depth',
        split_file = 'data/tartanair/tartanair_train.txt',
        rgb_prefix = '', 
        rgb_suffix = '_left.png',
        d_prefix = '',
        d_suffix = '_left_depth.npy',
        num_input_frames=7,
        pipeline=train_pipeline,
        test_mode=False),
    # val ## TODO
    val=dict(
        type=val_dataset_type,
        guide_folder='data/tartanair/color',
        gt_folder = 'data/tartanair/depth',
        split_file = 'data/tartanair/tartanair_valsmall.txt',
        rgb_prefix = '', 
        rgb_suffix = '_left.png',
        d_prefix = '',
        d_suffix = '_left_depth.npy',
        num_input_frames=7,
        pipeline=test_pipeline,
        test_mode=True,
        test_all=False,
        test_idx_start=120),
    # test
    test=dict(
        type=test_dataset_type,
        guide_folder='data/demo_dvsr/color',
        gt_folder='data/demo_dvsr/depth',
        split_file='data/demo_dvsr/demo.txt',
        rgb_prefix = '', 
        rgb_suffix = '.png',
        d_prefix = '',
        d_suffix = '.npy',
        pipeline=test_pipeline,
        num_input_frames=100,
        test_mode=True,
        test_all=False),
)

# optimizer
optimizers = dict(
    type='Adam',
    lr=1e-4,
    betas=(0.9, 0.99),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.1)}))

# learning policy
total_iters = 150000
lr_config = dict(
    policy='Step',
    step=[100000,125000],
    gamma=0.2,
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5, save_image=True, gpu_collect=True)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
