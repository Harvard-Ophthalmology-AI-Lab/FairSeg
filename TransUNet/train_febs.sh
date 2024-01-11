#!/bin/bash
BASE_DIR=/data/home/tiany/Datasets/FairSeg_for_publish
LIST_DIR=lists/FairSeg_final
BATCH_SIZE=24
CENTER_CROP_SIZE=512
NUM_CLASS=3
MAX_EPOCHS=200
STOP_EPOCH=130

CUDA_VISIBLE_DEVICES=6 python train_febs.py \
	--root_path ${BASE_DIR} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE}
