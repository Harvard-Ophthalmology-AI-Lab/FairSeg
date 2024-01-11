#!/bin/bash
BASE_DIR=/data/home/tiany/Datasets/FairSeg_for_publish
OUTPUT_DIR=./model/FairSeg_Output
LIST_DIR=lists/FairSeg_final
BATCH_SIZE=24
CENTER_CROP_SIZE=512
NUM_CLASS=3
MAX_EPOCHS=200
STOP_EPOCH=130
# for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
# do
# for (( i=0; i<10; i++ ));
# do
CUDA_VISIBLE_DEVICES=5 python train.py \
	--root_path ${BASE_DIR} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE}
# done
# done