from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    # CLASS_LABELS_200,
    CLASS_LABELS_BASE,
)

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus
num_worker = 1
mix_prob = 0.8
empty_cache = False
enable_amp = True

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# model settings
model = dict(
    type="DefaultSegmentorV2", # 模型的类型，使用 "DefaultSegmentorV2"
    num_classes=len(CLASS_LABELS_BASE), # 类别数量
    backbone_out_channels=64, # 骨干网络输出通道数
    backbone=dict(
        type="PT-v3m1",  # 使用PT-v3m1作为骨干网络
        in_channels=6,  # 输入通道数为6
        order=["z", "z-trans", "hilbert", "hilbert-trans"],  # 特征的顺序
        stride=(2, 2, 2, 2),  # 卷积步幅
        enc_depths=(2, 2, 2, 6, 2),  # 编码器的深度
        enc_channels=(32, 64, 128, 256, 512),  # 编码器各层的通道数
        enc_num_head=(2, 4, 8, 16, 32),  # 编码器每层的头数
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),  # 每个patch的大小
        dec_depths=(2, 2, 2, 2),  # 解码器的深度
        dec_channels=(64, 64, 128, 256),  # 解码器各层的通道数
        dec_num_head=(4, 4, 8, 16),  # 解码器每层的头数
        dec_patch_size=(1024, 1024, 1024, 1024),  # 每个patch的大小
        mlp_ratio=4,  # 多层感知机的比率
        qkv_bias=True,  # 是否使用qkv偏置
        qk_scale=None,  # qk的缩放
        attn_drop=0.0,  # 注意力dropout比率
        proj_drop=0.0,  # 投影dropout比率
        drop_path=0.3,  # 路径dropout比率
        shuffle_orders=True,  # 是否打乱顺序
        pre_norm=True,  # 是否使用预规范化
        enable_rpe=False,  # 是否启用位置编码
        enable_flash=True,  # 是否启用Flash注意力机制
        upcast_attention=False,  # 是否提升注意力计算精度
        upcast_softmax=False,  # 是否提升softmax精度
        cls_mode=False,  # 是否使用分类模式
        pdnorm_bn=False,  # 是否使用批量归一化
        pdnorm_ln=False,  # 是否使用层归一化
        pdnorm_decouple=True,  # 是否解耦pdnorm
        pdnorm_adaptive=False,  # 是否使用自适应pdnorm
        pdnorm_affine=True,  # 是否启用仿射变换
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),  # 使用的标准数据集
    ),
    criteria=[
        # 交叉熵损失
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        #  Lovasz 损失
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
)

# scheduler settings
epoch = 800
#  AdamW优化器，学习率为 0.006，权重衰减为 0.05
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
# 调度器，OneCycleLR调度策略
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "ScanNet200Dataset_BASETrain"
data_root = "data/ScanNet200"

data = dict(
    num_classes=len(CLASS_LABELS_BASE),
    ignore_index=-1, # 在处理标签时，如果遇到标签为 -1，表示该点云数据不需要进行分割（通常用于背景或无关类别）
    names=CLASS_LABELS_BASE,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True), # 将点云的中心移到原点
            dict(
                type="RandomDropout",
                dropout_ratio=0.2,
                dropout_application_ratio=0.2,
            ), # 随机丢弃部分点云数据
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(
                type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5
            ),
            dict(
                type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5
            ), # 对点云进行随机旋转
            dict(type="RandomScale", scale=[0.9, 1.1]), # 对点云进行随机缩放
            dict(type="RandomFlip", p=0.5), # 随机翻转点云
            dict(type="RandomJitter", sigma=0.005, clip=0.02), # 对点云进行小幅度的随机抖动
            dict(
                type="ElasticDistortion",
                distortion_params=[[0.2, 0.4], [0.8, 1.6]],
            ), # 对点云进行弹性变形，使得点云结构发生非线性变形
            # 对点云的颜色进行增强的操作，通过自动对比度、颜色平移和颜色抖动来增加数据多样性
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # 对点云进行网格采样，设置网格大小为 0.02
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # 对点云进行球形裁剪
            dict(type="SphereCrop", point_max=102400, mode="random"),
            # # 将点云的中心移到原点
            dict(type="CenterShift", apply_z=False),
            # 对点云的颜色进行归一化处理，使得颜色的数值分布统一
            dict(type="NormalizeColor"),
            # 将点云转换为张量形式
            dict(type="ToTensor"),
            # 将点云的坐标、网格坐标、分割标签以及颜色、法线特征收集到一起
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    # 验证集通常不包含像 RandomDropout 和 RandomRotate 这类随机数据增强操作，而是更多地关注点云的规范化处理
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            # 对测试数据进行体素化,是把连续的点云坐标映射到一个三维网格中，每个网格单元（体素 voxel）表示一个小立方体
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "normal"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)
