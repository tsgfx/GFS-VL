"""
Main Training Script

Author: Zhaochong An (anzhaochong@outlook.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    # 加载训练参数
    cfg = default_setup(cfg)
    # 构建训练器实例
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    # 启动训练
    trainer.train()


def main():
    # 解析输入的参数
    args = default_argument_parser().parse_args()
    # 读取配置文件信息，传入配置文件名和附加选项
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker, # 主训练函数
        num_gpus_per_machine=args.num_gpus, # 每台机器上的gpu数
        num_machines=args.num_machines, # 机器数
        machine_rank=args.machine_rank, # 机器编号
        dist_url=args.dist_url, # 用于分布式训练的url
        cfg=(cfg,), # 可选参数
    )


if __name__ == "__main__":
    main()
