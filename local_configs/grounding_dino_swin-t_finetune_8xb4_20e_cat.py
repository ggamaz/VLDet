_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

data_root = '/home/legion/Pictures/'
backend_args = None
dataset_type = 'PairedCocoDataset'


model = dict(
    bbox_head=dict(num_classes=11),
    
    fusion_module = dict(
        use_fusion=True,
        
        type ="msd",
        msd_cfg=dict(
            num_layers=2,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)
            )
        ),
        position="after_backbone",
    ),
)


train_pipeline = [
    dict(
        type='LoadMultiChannelImageFromFiles',
        color_type='color',
        backend_args=None,
        imdecode_backend='cv2'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

datasets = [
    # dict(
    #     type='PairedCocoDataset',
    #     metainfo=dict(
    #         classes=("Bus", "Car", "Lamp", "Motorcycle", "People", "Truck"),
    #         palette=[
    #             (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    #             (255, 0, 255), (0, 255, 255)
    #         ],
    #     ),
    #     data_root=data_root+'M3FD_Detection/',
    #     return_classes=True,
    #     pipeline=train_pipeline,
    #     filter_cfg=dict(filter_empty_gt=False, min_size=32),
    #     ann_file='annotations/instances_default.json',
    #     data_prefix=dict(imga='vi',imgb='ir')
    # ),
    dict(
        metainfo = dict(
            classes= ('car', 'bus', 'freight_car', 'truck', 'van'),
            palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),]
        ),
        type='PairedCocoDataset',
        data_root= data_root+'DroneVehicleNoBorder/',
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='coco_annotations_noborder/DV_train_ir.json',
        data_prefix=dict(imga='train/rgb/images/',imgb='train/ir/images/')
    )
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=datasets,
    )
)

test_pipeline = [
    dict(
        type='LoadMultiChannelImageFromFiles',
        color_type='color',
        backend_args=None,
        imdecode_backend='cv2'),
    dict(
        type='SpiltMultiChannel',
        channels=[3, 3],
        transforms=[
            dict(
                type='FixScaleResize',
                scale=(512, 640),
                keep_ratio=True,
                backend='pillow'),
            ]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    dataset=dict(
        _delete_=True,
        type='PairedCocoDataset',
        return_classes=True,
        pipeline=test_pipeline,
        data_root=data_root+'DroneVehicleNoBorder/',
        ann_file='coco_annotations_noborder/DV_test_ir.json',
        data_prefix=dict(imga='test/rgb/images/',imgb='test/ir/images/'),
        # ann_file='annotations/instances_default.json',
        # data_prefix=dict(imga='vi',imgb='ir')
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root+'DroneVehicleNoBorder/coco_annotations_noborder/DV_test_ir.json',)
# val_evaluator = dict(ann_file=data_root + 'annotations/instances_default.json')
test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0),
            # 'encoder': dict(lr_mult=0.0),
            # 'decoder': dict(lr_mult=0.0),
            # 'bbox_head': dict(lr_mult=0.0),
        }))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
