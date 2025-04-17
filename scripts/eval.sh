#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python runners/evaulation.py \
--data_path ./data \
--batch_size 1 \
--dino pointwise \
--num_worker 32 \
--pretrained_score_model_path path/to/ckpt \
--pred_path './result' \
--infer_split 'test_intra' \
--eval \
--visual
