#!/bin/bash
BASE_DIR=XXX
LIST_DIR=./lists/FairSeg_final
BATCH_SIZE=42
CENTER_CROP_SIZE=512
NUM_CLASS=2
MAX_EPOCHS=10
STOP_EPOCH=5
ATTRIBUTE=( race language hispanic gender)
for (( j=0; j<${#ATTRIBUTE[@]}; j++ ));
do
CUDA_VISIBLE_DEVICES=0,1 python train_finetune.py \
	--root_path ${BASE_DIR} \
	--output ./outputs/FairSeg_fair_loss/${ATTRIBUTE[$j]} \
	--list_dir ${LIST_DIR} \
	--max_epochs ${MAX_EPOCHS} \
	--stop_epoch ${STOP_EPOCH} \
	--base_lr 0.0005 \
	--AdamW \
	--center_crop_size ${CENTER_CROP_SIZE} \
	--num_classes ${NUM_CLASS} \
	--batch_size ${BATCH_SIZE}
done
