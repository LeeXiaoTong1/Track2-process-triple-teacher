#!/usr/bin/env bash
set -e

# next_step_type_specialist_pipeline.sh
# This is a command template. Review paths before running.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT=/root/autodl-tmp/AT-ADD-Baseline-track2-R2
DATA=$ROOT/AT_ADD_data/Track2
XLSR=$ROOT/huggingface/wav2vec2-xls-r-300m
MERT=$ROOT/huggingface/MERT-v1-330M
BEATS=$ROOT/huggingface/OpenBEATs-ICME

python make_track2_type_labels.py \
  --train_label $DATA/label/train.csv \
  --dev_label $DATA/label/dev.csv \
  --out_dir $DATA/label_by_type

# Speech specialist: XLSR-based, type-filtered.
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ft-w2v2aasist \
  --xlsr $XLSR \
  --atadd_t2_train_audio $DATA/train \
  --atadd_t2_train_label $DATA/label_by_type/train_speech.csv \
  --atadd_t2_dev_audio $DATA/dev \
  --atadd_t2_dev_label $DATA/label_by_type/dev_speech.csv \
  --seed 2027 \
  --batch_size 8 \
  --lr 0.000003 \
  --num_epochs 5 \
  --interval 5 \
  --save_best_by f1 \
  --out_fold ./ckpt_t2/speech_ftxlsr_specialist

# Sound specialist: UFM initialized from previous specialist, sound-only supervision.
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --init_from ./ckpt_t2/ufm_vocal_anchor_soundmusic/atadd_model.pt \
  --xlsr $XLSR \
  --mert $MERT \
  --beats $BEATS \
  --atadd_t2_train_audio $DATA/train \
  --atadd_t2_train_label $DATA/label_by_type/train_sound.csv \
  --atadd_t2_dev_audio $DATA/dev \
  --atadd_t2_dev_label $DATA/label_by_type/dev_sound.csv \
  --ufm_freeze_xlsr \
  --ufm_freeze_mert \
  --ufm_freeze_beats \
  --ufm_dim 512 \
  --ufm_mem_slots 16 \
  --ufm_heads 8 \
  --ufm_layers 1 \
  --ufm_dropout 0.0 \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --seed 2028 \
  --batch_size 32 \
  --lr 0.0000002 \
  --num_epochs 4 \
  --interval 4 \
  --save_best_by f1 \
  --out_fold ./ckpt_t2/sound_ufm_specialist

# Music specialist: MERT-based, type-filtered. If OOM, reduce batch_size to 4.
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ft-mertaasist \
  --mert $MERT \
  --atadd_t2_train_audio $DATA/train \
  --atadd_t2_train_label $DATA/label_by_type/train_music.csv \
  --atadd_t2_dev_audio $DATA/dev \
  --atadd_t2_dev_label $DATA/label_by_type/dev_music.csv \
  --seed 2029 \
  --batch_size 6 \
  --lr 0.000003 \
  --num_epochs 6 \
  --interval 6 \
  --save_best_by f1 \
  --out_fold ./ckpt_t2/music_mert_specialist
