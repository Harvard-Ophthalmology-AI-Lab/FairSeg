#!/bin/bash
BASE_DIR=./XXX
OUTPUT_DIR=./results/
LIST_DIR=./lists/FairSeg_final
LORA_CKPT=./XXX
CENTER_CROP_SIZE=420
NUM_CLASS=2
ATTRIBUTE=race
python test.py \
	--datadir ${BASE_DIR} \
	--output ${OUTPUT_DIR} \
	--list_dir ${LIST_DIR} \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--lora_ckpt ${LORA_CKPT}
