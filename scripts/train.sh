#!/bin/sh
set -x

# 变量初始化
PYTHON=python

TRAIN_CODE=train.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
GPU=4
OPT=""

while getopts "p:d:c:n:w:g:r:o:" opt; do
  case $opt in
    # 指定解释器
    p)
      PYTHON=$OPTARG
      ;;
    # 指定数据集
    d)
      DATASET=$OPTARG
      ;;
    # 指定配置文件
    c)
      CONFIG=$OPTARG
      ;;
    # 实验名称
    n)
      EXP_NAME=$OPTARG
      ;;
    # 指定预训练权重
    w)
      WEIGHT=$OPTARG
      ;;
    # 恢复训练的标志
    r)
      RESUME=$OPTARG
      ;;
    # 使用的GPU数量
    g)
      GPU=$OPTARG
      ;;
    # 附加选项
    o)
      OPT="$OPT $OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

# 检测GPU数量
if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"

EXP_DIR=./exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $EXP_DIR"
if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=$CODE_DIR
echo $PYTHONPATH
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" $OPT
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
fi