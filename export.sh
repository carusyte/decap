#!/bin/sh

python export_inference_graph.py \
    --pipeline_config_path "ssd_mobilenet_v2_fpnlite/pipeline.config" \
    --input_type image_tensor \
    --trained_checkpoint_dir "ssd_mobilenet_v2_fpnlite/model" \
    --output_directory "ssd_mobilenet_v2_fpnlite/model/saved"