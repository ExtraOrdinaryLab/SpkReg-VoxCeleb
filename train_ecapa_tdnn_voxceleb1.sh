#!/bin/bash

DEVICE="0"

OUTPUT_DIR="/mnt/data4_HDD_14TB/yang/voxceleb-checkpoints/ecapa-tdnn/voxceleb1/finetune"

MAX_LENGTH="3"
BATCH_SIZE="256"
LEARNING_RATE="1e-3"
NUM_EPOCHS="10"

EXP_NAME="len$MAX_LENGTH-bs$BATCH_SIZE-lr$LEARNING_RATE"

CUDA_VISIBLE_DEVICES=$DEVICE /home/yang/miniconda3/envs/confit/bin/python train_ecapa_tdnn_voxceleb1.py \
    --model_name_or_path "confit/ecapa-tdnn-voxceleb1" \
    --config_name "/home/yang/data/audio/spkreg-voxceleb/configs/ecapa-tdnn-voxceleb1/config.json" \
    --dataset_name "confit/voxceleb" \
    --data_dir "/mnt/data1_HDD_14TB/yang/corpus/audio/VoxCeleb1" \
    --audio_column_name "audio" \
    --label_column_name "speaker" \
    --output_dir $OUTPUT_DIR/$EXP_NAME \
    --overwrite_output_dir \
    --trust_remote_code "True" \
    --remove_unused_columns "False" \
    --freeze_feature_encoder "True" \
    --eval_split_name "validation" \
    --do_train \
    --do_eval \
    --fp16 \
    --learning_rate $LEARNING_RATE \
    --max_length_seconds $MAX_LENGTH \
    --return_attention_mask "False" \
    --warmup_ratio "0.1" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps "1" \
    --per_device_eval_batch_size "1" \
    --dataloader_num_workers "8" \
    --logging_strategy "steps" \
    --logging_steps "20" \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end "True" \
    --metric_for_best_model "accuracy" \
    --save_total_limit "1" \
    --report_to "none" \
    --seed "914"