model = dict(
    backbone = 'BaFPR',
    args=dict(
        num_classes = 1,
    ),
    loss_decode = 'structure_loss',
    )

# dataset settings
dataset_type = 'polyp'
augmentation = False
data_path = './data/dataset/'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    ratio_range= None,
    train=dict(
        type=dataset_type,
        args = None,
         ),
    val=dict(
        type=dataset_type,
        args = None,
        ),
    test=dict(
        type=dataset_type,
        args = None,
        ))

# experimental
save_to = None#'./exp/' # None for not recording the exp
exp_name = 'BaFPR_polyp_debug_mgpu'
seed = 0

#dist_params = dict(backend='nccl')
log_level = 'INFO'
log_config = dict(img2save=0)

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-4)
optimizer_config = dict(grad_clip = 0.5)
enable_polar_consist = True
boundary_prior = True
# learning policy
lr_scheduler = dict(type = 'decay', decay_rate = 0.1, decay_epoch= 50)
# runtime settings
runner = dict(type='IterBasedRunner', max_epoch=40, resume=False)
checkpoint_config = dict(by_epoch=False, interval=10)
evaluation = dict(interval=10, metric='default')
