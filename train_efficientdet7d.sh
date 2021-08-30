#!/bin/sh

MODEL_DIR="EfficientDet_D7/model"
python3 model_main_tf2.py \
  --strategy_type=one_device \
  --num_gpus=1 \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --config_file="EfficientDet_D7/pipeline.config"