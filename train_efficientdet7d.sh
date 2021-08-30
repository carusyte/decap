#!/bin/sh

# MODEL_DIR="EfficientDet_D7/model"
# python3 model_main_tf2.py \
#   --strategy_type=one_device \
#   --num_gpus=1 \
#   --model_dir="${MODEL_DIR?}" \
#   --mode=train \
#   --config_file="EfficientDet_D7/pipeline.config"


PIPELINE_CONFIG_PATH=EfficientDet_D7/pipeline.config
MODEL_DIR=EfficientDet_D7/model
NUM_TRAIN_STEPS=60000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr