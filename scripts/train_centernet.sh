#!/bin/sh

PIPELINE_CONFIG_PATH=model/centernet_hourglass104_coco17/pipeline.config
MODEL_DIR=model/centernet_hourglass104_coco17/model
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr