_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

data_root = '/root/autodl-tmp/'
backend_args = None
dataset_type = 'PairedCocoDataset'

inner_model = dict(
    type='GroundingDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetMultiChannelDataPreprocessor',
        mean=[123.675,116.28,103.53, 123.675,116.28,103.53, 123.675,116.28,103.53, 123.675,116.28,103.53],
        std=[58.395,57.12,57.375, 58.395,57.12,57.375, 58.395,57.12,57.375, 58.395,57.12,57.375],
        bgr_to_rgb=True,
        pad_mask=False,),
    language_model=dict(
        type='BertModel',
        name='bert-base-uncased',
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
        ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=11,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0)),
    # bbox_head=dict(
    #     type='GroundingDFINEHead',
    #     num_classes=11,
    #     sync_cls_avg_factor=True,
    #     contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
    #     # loss_cls=dict(
    #     #     _delete_=True,
    #     #     type='QualityFocalLoss',
    #     #     use_sigmoid=True,
    #     #     beta=2.0,
    #     #     loss_weight=1.0),
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),  # 2.0 in DeformDETR
    #     loss_bbox=dict(type='L1Loss', loss_weight=5.0),
    #     reg_max=32,
    #     # loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
    #     loss_ld=dict(type='KnowledgeDistillationKLDivLoss',loss_weight=0.25,T=10)),
    
    fusion_module = dict(
        use_fusion=True,      
        # type ="msd",
        # msd_cfg=dict(
        #     num_layers=4,
        #     num_cp=4,
        #     layer_cfg=dict(
        #         self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
        #         ffn_cfg=dict(
        #             embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
        type ="moe",
        # type ="add",
        position="after_backbone"),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

model = dict(
    _delete_=True,
    type='DistillOptimizer',
    model_cfg = inner_model,
    sft_type='mimicking'
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
                   'custom_entities'))]

DV_Dataset = dict(
    _delete_=True,
    metainfo = dict(
        classes= ('car', 'bus', 'freight_car', 'truck', 'van'),
        palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),]
    ),
    type='PairedCocoDataset',
    data_root= data_root+'DroneVehicleNoBorder/',
    return_classes=True,
    pipeline=train_pipeline,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    ann_file='coco_annotations/DV_train_ir.json',
    data_prefix=dict(
        imga='degraded_train/rgb/images/',
        imgb='degraded_train/ir/images/',
        imgc='train/rgb/images/',
        imgd='train/ir/images/'
        ))

M3FD_Dataset = dict(
        _delete_=True,
        type='PairedCocoDataset',
        metainfo=dict(
            classes=("Bus", "Car", "Lamp", "Motorcycle", "People", "Truck"),
            palette=[
                (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0),
                (255, 0, 255), (0, 255, 255)
            ],
        ),
        data_root=data_root+'M3FD_Detection/',
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations/instances_default.json',
        data_prefix=dict(imga='vi',imgb='ir'))

datasets = [DV_Dataset, M3FD_Dataset]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    # dataset=dict(
    #     _delete_=True,
    #     type='ConcatDataset',
    #     datasets=datasets,
    # )
    dataset = DV_Dataset,
)

test_pipeline = [
    dict(
        type='LoadMultiChannelImageFromFiles',
        color_type='color',
        backend_args=None,
        imdecode_backend='cv2'),
    dict(
        type='SpiltMultiChannel',
        channels=[3, 3, 3, 3],
        transforms=[
            dict(
                type='FixScaleResize',
                scale=(512, 640),
                # scale=(712, 840),
                keep_ratio=True,
                backend='pillow'),
            ]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))]

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='PairedCocoDataset',
        pipeline=test_pipeline,
        data_root=data_root+'DroneVehicleNoBorder/',
        ann_file='coco_annotations/DV_test_ir.json',
        data_prefix=dict(
            _delete_=True,
            imga='degraded_test/rgb/images/',
            imgb='degraded_test/ir/images/',
            imgc='train/rgb/images/',
            imgd='train/ir/images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root+'DroneVehicleNoBorder/coco_annotations/DV_test_ir.json',)
# val_evaluator = dict(ann_file=data_root + 'annotations/instances_default.json')
test_evaluator = val_evaluator

max_epoch = 10

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best=['coco/bbox_mAP_50']),
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

# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa
# load_from = "/root/VLDet/work_dirs/afterbackbone_a+b_grounding_dino_swin-t_finetune_8xb4_20e_cat/best_coco_car_precision_epoch_11.pth"
load_from = '/root/VLDet/work_dirs/moe_grounding_dino_swin-t_finetune_8xb4_20e_cat/best_coco_car_precision_epoch_20.pth'