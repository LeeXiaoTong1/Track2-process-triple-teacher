**完整 Track2 改进流程**：从代码准备、baseline/teacher、UFM、type classifier、specialist、fusion 到提交。

---

# 0. 固定路径约定

下面所有命令默认在项目根目录执行：

```bash
cd /root/autodl-tmp/AT-ADD-Baseline-track2-R2
```

统一环境变量：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

统一路径：

```bash
ROOT=/root/autodl-tmp/AT-ADD-Baseline-track2-R2
DATA=/root/autodl-tmp/AT-ADD-Baseline-track2-R2/AT_ADD_data/Track2

XLSR=/root/autodl-tmp/AT-ADD-Baseline-track2-R2/huggingface/wav2vec2-xls-r-300m
MERT=/root/autodl-tmp/AT-ADD-Baseline-track2-R2/huggingface/MERT-v1-330M
BEATS=/root/autodl-tmp/AT-ADD-Baseline-track2-R2/huggingface/OpenBEATs-ICME

TEACHER_CKPT=/root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/checkpoint/atadd_model_10.pt
TEACHER_DIR=/root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist
```

---

# 2. 整理 baseline / teacher checkpoint

你决定继续使用：

```bash
/root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/checkpoint/atadd_model_10.pt
```

我们把它整理成标准评分目录：

```bash
mkdir -p ./ckpt_t2/ft_xlsr_baseline

cp $TEACHER_DIR/args.json \
   ./ckpt_t2/ft_xlsr_baseline/args.json

cp $TEACHER_CKPT \
   ./ckpt_t2/ft_xlsr_baseline/atadd_model.pt
```

这个 baseline 的用途：

```text
主要保护 Singing；
不要用它强行 anchor Speech，因为它 Speech 不够强。
```

---

# 3. 生成 type-filtered label

这是后续 Speech/Sound/Music specialist 的基础。

```bash
python make_track2_type_labels.py \
  --train_label $DATA/label/train.csv \
  --dev_label $DATA/label/dev.csv \
  --out_dir $DATA/label_by_type
```

生成后检查：

```bash
ls $DATA/label_by_type
```

应该看到：

```text
train_speech.csv
train_sound.csv
train_singing.csv
train_music.csv
dev_speech.csv
dev_sound.csv
dev_singing.csv
dev_music.csv
```

---

# 4. 训练独立 type classifier

这个模型只负责判断音频类型：

```text
speech / sound / singing / music
```

它替代 UFM 内部那个不可靠的 type posterior。

```bash
python train_type_classifier_track2.py \
  --gpu 0 \
  --train_audio $DATA/train \
  --train_label $DATA/label/train.csv \
  --dev_audio $DATA/dev \
  --dev_label $DATA/label/dev.csv \
  --xlsr $XLSR \
  --mert $MERT \
  --beats $BEATS \
  --out_dir ./ckpt_t2/type_classifier_xmb \
  --batch_size 32 \
  --epochs 5 \
  --lr 0.0001 \
  --seed 1234
```

生成 Dev type probabilities：

```bash
python score_type_classifier_track2.py \
  --gpu 0 \
  --model_dir ./ckpt_t2/type_classifier_xmb \
  --eval_audio $DATA/dev \
  --out_csv ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv \
  --batch_size 32 \
  --num_workers 8
```

生成 Progress type probabilities：

```bash
python score_type_classifier_track2.py \
  --gpu 0 \
  --model_dir ./ckpt_t2/type_classifier_xmb \
  --eval_audio $DATA/eval_progress \
  --out_csv ./ckpt_t2/type_classifier_xmb/progress_type_probs.csv \
  --batch_size 32 \
  --num_workers 8
```

---

# 5. 训练基础 UFM：all95 stage1

这一阶段训练一个通用 UFM。
注意：teacher 只 anchor Singing，即：

```bash
--t2_teacher_anchor_types 2
```

不要 anchor Speech。

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --xlsr $XLSR \
  --mert $MERT \
  --beats $BEATS \
  --ufm_freeze_xlsr \
  --ufm_freeze_mert \
  --ufm_freeze_beats \
  --ufm_dim 512 \
  --ufm_mem_slots 16 \
  --ufm_heads 8 \
  --ufm_layers 1 \
  --ufm_dropout 0.0 \
  --t2_return_type \
  --t2_gdro \
  --t2_gdro_active_types 0,1,3 \
  --t2_gdro_eta 0.15 \
  --ufm_type_loss 0.003 \
  --ufm_router_entropy 0.0 \
  --t2_teacher_model ft-w2v2aasist \
  --t2_sing_teacher_ckpt $TEACHER_CKPT \
  --t2_teacher_anchor_types 2 \
  --t2_sing_anchor_weight 0.5 \
  --t2_sing_anchor_temp 2.0 \
  --t2_sing_anchor_margin_weight 0.2 \
  --t2_sing_anchor_correct_only \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --t2_target_floor 0.95 \
  --t2_floor_penalty 2.0 \
  --seed 1234 \
  --batch_size 32 \
  --lr 0.000001 \
  --num_epochs 5 \
  --interval 4 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_all95_stage1_teacher_weak3
```

---

# 6. 训练基础 UFM：all95 stage2

从 stage1 继续低学习率细调。

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --init_from ./ckpt_t2/ufm_all95_stage1_teacher_weak3/atadd_model.pt \
  --xlsr $XLSR \
  --mert $MERT \
  --beats $BEATS \
  --ufm_freeze_xlsr \
  --ufm_freeze_mert \
  --ufm_freeze_beats \
  --ufm_dim 512 \
  --ufm_mem_slots 16 \
  --ufm_heads 8 \
  --ufm_layers 1 \
  --ufm_dropout 0.0 \
  --t2_return_type \
  --t2_gdro \
  --t2_gdro_active_types 0,1,3 \
  --t2_gdro_eta 0.10 \
  --ufm_type_loss 0.001 \
  --ufm_router_entropy 0.0 \
  --t2_teacher_model ft-w2v2aasist \
  --t2_sing_teacher_ckpt $TEACHER_CKPT \
  --t2_teacher_anchor_types 2 \
  --t2_sing_anchor_weight 0.5 \
  --t2_sing_anchor_temp 2.0 \
  --t2_sing_anchor_margin_weight 0.2 \
  --t2_sing_anchor_correct_only \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --t2_target_floor 0.95 \
  --t2_floor_penalty 2.0 \
  --seed 1234 \
  --batch_size 32 \
  --lr 0.0000002 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_all95_stage2_teacher_weak3
```

如果你已经有：

```text
./ckpt_t2/ufm_all95_stage2_teacher_weak3/atadd_model.pt
```

可以跳过第 5、6 步。

---

# 7. 训练 Sound/Music UFM branch

这一阶段得到：

```text
./ckpt_t2/ufm_vocal_anchor_soundmusic
```

它是后续 Sound/Music 相关分支的基础。

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --init_from ./ckpt_t2/ufm_all95_stage2_teacher_weak3/atadd_model.pt \
  --xlsr $XLSR \
  --mert $MERT \
  --beats $BEATS \
  --ufm_freeze_xlsr \
  --ufm_freeze_mert \
  --ufm_freeze_beats \
  --ufm_dim 512 \
  --ufm_mem_slots 16 \
  --ufm_heads 8 \
  --ufm_layers 1 \
  --ufm_dropout 0.0 \
  --t2_return_type \
  --t2_gdro \
  --t2_gdro_active_types 1,3 \
  --t2_gdro_eta 0.35 \
  --ufm_type_loss 0.001 \
  --ufm_router_entropy 0.0 \
  --t2_teacher_model ft-w2v2aasist \
  --t2_sing_teacher_ckpt $TEACHER_CKPT \
  --t2_teacher_anchor_types 2 \
  --t2_sing_anchor_weight 1.0 \
  --t2_sing_anchor_temp 2.0 \
  --t2_sing_anchor_margin_weight 0.2 \
  --t2_sing_anchor_correct_only \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --t2_target_floor 0.95 \
  --t2_floor_penalty 2.0 \
  --seed 1234 \
  --batch_size 32 \
  --lr 0.0000002 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_vocal_anchor_soundmusic
```

---

# 8. 训练 Music UFM specialist

这个是第一代 Music specialist。

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --init_from ./ckpt_t2/ufm_vocal_anchor_soundmusic/atadd_model.pt \
  --xlsr $XLSR \
  --mert $MERT \
  --beats $BEATS \
  --ufm_freeze_xlsr \
  --ufm_freeze_mert \
  --ufm_freeze_beats \
  --ufm_dim 512 \
  --ufm_mem_slots 16 \
  --ufm_heads 8 \
  --ufm_layers 1 \
  --ufm_dropout 0.0 \
  --t2_return_type \
  --t2_gdro \
  --t2_gdro_active_types 3 \
  --t2_gdro_eta 0.60 \
  --ufm_type_loss 0.001 \
  --ufm_router_entropy 0.0 \
  --t2_teacher_model ft-w2v2aasist \
  --t2_sing_teacher_ckpt $TEACHER_CKPT \
  --t2_teacher_anchor_types 2 \
  --t2_sing_anchor_weight 1.0 \
  --t2_sing_anchor_temp 2.0 \
  --t2_sing_anchor_margin_weight 0.2 \
  --t2_sing_anchor_correct_only \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --t2_target_floor 0.95 \
  --t2_floor_penalty 2.0 \
  --seed 1234 \
  --batch_size 32 \
  --lr 0.00000015 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_music_specialist
```

---

# 9. 可选：训练 Music UFM specialist v2

这版使用轻量 multi-crop，训练更慢，但对 Music 长音频可能更友好。

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --init_from ./ckpt_t2/ufm_music_specialist/atadd_model.pt \
  --xlsr $XLSR \
  --mert $MERT \
  --beats $BEATS \
  --ufm_freeze_xlsr \
  --ufm_freeze_mert \
  --ufm_freeze_beats \
  --ufm_dim 512 \
  --ufm_mem_slots 16 \
  --ufm_heads 8 \
  --ufm_layers 1 \
  --ufm_dropout 0.05 \
  --t2_return_type \
  --t2_gdro \
  --t2_gdro_active_types 3 \
  --t2_gdro_eta 0.80 \
  --ufm_type_loss 0.001 \
  --ufm_router_entropy 0.0 \
  --t2_teacher_model ft-w2v2aasist \
  --t2_sing_teacher_ckpt $TEACHER_CKPT \
  --t2_teacher_anchor_types 2 \
  --t2_sing_anchor_weight 1.2 \
  --t2_sing_anchor_temp 2.0 \
  --t2_sing_anchor_margin_weight 0.2 \
  --t2_sing_anchor_correct_only \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 2 \
  --crop_consistency_weight 0.01 \
  --t2_target_floor 0.95 \
  --t2_floor_penalty 2.0 \
  --seed 2026 \
  --batch_size 16 \
  --lr 0.0000001 \
  --num_epochs 2 \
  --interval 2 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_music_specialist_v2
```

如果时间紧，可以跳过这一版最好是训练一下试试。

---

# 10. 训练 Speech specialist

当前 Progress 最大短板是 Speech，所以现在必须引入异构 Speech specialist。
用 type-filtered speech 数据训练 FT-XLSR。

```bash
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
```

如果显存不足：

```bash
--batch_size 4
```

---

# 11. 训练 Sound specialist

用 type-filtered sound 数据训练 UFM sound-only specialist。

```bash
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
```

---

# 12. 训练 Music MERT specialist

这是最终阶段建议补上的异构 Music specialist。
它和 UFM music specialist 不同，使用 MERT-only。

```bash
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
```

如果显存不足：

```bash
--batch_size 4
```

---

# 13. 生成 baseline Dev / Progress score

Dev：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ft_xlsr_baseline \
  --eval_task atadd-track2 \
  --eval_audio $DATA/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ft_xlsr_baseline/result/dev_baseline_plus.csv
```

Progress：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ft_xlsr_baseline \
  --eval_task atadd-track2 \
  --eval_audio $DATA/eval_progress \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ft_xlsr_baseline/result/progress_baseline_plus.csv
```

---

# 14. 生成 UFM Sound/Music branch score

## UFM vocal_anchor_soundmusic

Dev：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_vocal_anchor_soundmusic \
  --eval_task atadd-track2 \
  --eval_audio $DATA/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/dev_ufm_plus.csv
```

Progress：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_vocal_anchor_soundmusic \
  --eval_task atadd-track2 \
  --eval_audio $DATA/eval_progress \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/progress_ufm_plus.csv
```

## UFM music specialist

Dev：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_music_specialist \
  --eval_task atadd-track2 \
  --eval_audio $DATA/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_music_specialist/result/dev_music_plus.csv
```

Progress：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_music_specialist \
  --eval_task atadd-track2 \
  --eval_audio $DATA/eval_progress \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_music_specialist/result/progress_music_plus.csv
```

## UFM music specialist v2，可选

Dev：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_music_specialist_v2 \
  --eval_task atadd-track2 \
  --eval_audio $DATA/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_music_specialist_v2/result/dev_music_v2_plus.csv
```

Progress：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_music_specialist_v2 \
  --eval_task atadd-track2 \
  --eval_audio $DATA/eval_progress \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_music_specialist_v2/result/progress_music_v2_plus.csv
```

---

# 15. 生成 type-specific specialists score

## Speech specialist

Dev：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/speech_ftxlsr_specialist \
  --eval_task atadd-track2 \
  --eval_audio $DATA/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/speech_ftxlsr_specialist/result/dev_speech_plus.csv
```

Progress：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/speech_ftxlsr_specialist \
  --eval_task atadd-track2 \
  --eval_audio $DATA/eval_progress \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/speech_ftxlsr_specialist/result/progress_speech_plus.csv
```

## Sound specialist

Dev：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/sound_ufm_specialist \
  --eval_task atadd-track2 \
  --eval_audio $DATA/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/sound_ufm_specialist/result/dev_sound_plus.csv
```

Progress：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/sound_ufm_specialist \
  --eval_task atadd-track2 \
  --eval_audio $DATA/eval_progress \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/sound_ufm_specialist/result/progress_sound_plus.csv
```

## Music MERT specialist

Dev：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/music_mert_specialist \
  --eval_task atadd-track2 \
  --eval_audio $DATA/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/music_mert_specialist/result/dev_music_mert_plus.csv
```

Progress：

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/music_mert_specialist \
  --eval_task atadd-track2 \
  --eval_audio $DATA/eval_progress \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/music_mert_specialist/result/progress_music_mert_plus.csv
```

---

# 16. 两分支 / 三分支 fusion 历史方案

这是你之前从 88.20 到 90.17 的关键路径。

## 16.1 二分支 fusion

```bash
python tune_branch_fusion.py \
  --ufm_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/dev_ufm_plus.csv \
  --baseline_csv ./ckpt_t2/ft_xlsr_baseline/result/dev_baseline_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv \
  --label_csv $DATA/label/dev.csv \
  --out_json ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/branch_fusion_typeclf.json \
  --trials 12000 \
  --seed 1234
```

应用：

```bash
python apply_branch_fusion.py \
  --ufm_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/progress_ufm_plus.csv \
  --baseline_csv ./ckpt_t2/ft_xlsr_baseline/result/progress_baseline_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/progress_type_probs.csv \
  --calib_json ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/branch_fusion_typeclf.json \
  --out_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/predict.csv
```

## 16.2 三分支 fusion：baseline + UFM + music-specialist

```bash
python tune_three_branch_fusion_holdout.py \
  --baseline_csv ./ckpt_t2/ft_xlsr_baseline/result/dev_baseline_plus.csv \
  --ufm_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/dev_ufm_plus.csv \
  --music_csv ./ckpt_t2/ufm_music_specialist/result/dev_music_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv \
  --label_csv $DATA/label/dev.csv \
  --out_json ./ckpt_t2/ufm_music_specialist/result/three_branch_fusion_holdout.json \
  --trials 6000 \
  --holdout_frac 0.35 \
  --mode music_boost \
  --seed 1234
```

应用：

```bash
python apply_three_branch_fusion.py \
  --baseline_csv ./ckpt_t2/ft_xlsr_baseline/result/progress_baseline_plus.csv \
  --ufm_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/progress_ufm_plus.csv \
  --music_csv ./ckpt_t2/ufm_music_specialist/result/progress_music_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/progress_type_probs.csv \
  --calib_json ./ckpt_t2/ufm_music_specialist/result/three_branch_fusion_holdout.json \
  --out_csv ./ckpt_t2/ufm_music_specialist/result/predict.csv \
  --debug_csv ./ckpt_t2/ufm_music_specialist/result/progress_three_branch_debug.csv
```

打包：

```bash
cd ./ckpt_t2/ufm_music_specialist/result
zip submit_three_branch_music.zip predict.csv
```

我自己训练完大约 **90.17** 的F1。

---

# 17. 推荐：多分支 fusion

以下建议将17.1和17.2的两个方案都尝试一下：

```text
baseline
speech specialist
sound specialist
music specialist
type classifier
```

## 17.1 如果使用 Music MERT specialist

```bash
python tune_multi_branch_fusion_holdout.py \
  --branch baseline:./ckpt_t2/ft_xlsr_baseline/result/dev_baseline_plus.csv \
  --branch speech:./ckpt_t2/speech_ftxlsr_specialist/result/dev_speech_plus.csv \
  --branch sound:./ckpt_t2/sound_ufm_specialist/result/dev_sound_plus.csv \
  --branch music:./ckpt_t2/music_mert_specialist/result/dev_music_mert_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv \
  --label_csv $DATA/label/dev.csv \
  --out_json ./ckpt_t2/multibranch_fusion_typespecialists.json \
  --trials 8000 \
  --holdout_frac 0.35 \
  --mode speech_music_boost \
  --seed 1234
```

应用到 Progress：

```bash
python apply_multi_branch_fusion.py \
  --branch baseline:./ckpt_t2/ft_xlsr_baseline/result/progress_baseline_plus.csv \
  --branch speech:./ckpt_t2/speech_ftxlsr_specialist/result/progress_speech_plus.csv \
  --branch sound:./ckpt_t2/sound_ufm_specialist/result/progress_sound_plus.csv \
  --branch music:./ckpt_t2/music_mert_specialist/result/progress_music_mert_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/progress_type_probs.csv \
  --calib_json ./ckpt_t2/multibranch_fusion_typespecialists.json \
  --out_csv ./ckpt_t2/predict_multibranch_typespecialists.csv \
  --debug_csv ./ckpt_t2/debug_multibranch_typespecialists.csv
```

打包提交：

```bash
cd ./ckpt_t2
cp predict_multibranch_typespecialists.csv predict.csv
zip submit_multibranch_typespecialists.zip predict.csv
```

## 17.2 如果使用 Music UFM specialist v2

把上面 `music` 分支替换为：

```bash
--branch music:./ckpt_t2/ufm_music_specialist_v2/result/dev_music_v2_plus.csv
```

Progress 应用时替换为：

```bash
--branch music:./ckpt_t2/ufm_music_specialist_v2/result/progress_music_v2_plus.csv
```

---


# 19. 推荐复现实验顺序

从零开始完整复现顺序：

```text
1. 安装最终代码；
2. 整理 ft_xlsr_baseline；
3. 生成 label_by_type；
4. 训练独立 type classifier；
5. 训练 ufm_all95_stage1_teacher_weak3；
6. 训练 ufm_all95_stage2_teacher_weak3；
7. 训练 ufm_vocal_anchor_soundmusic；
8. 训练 ufm_music_specialist；
9. 训练 speech_ftxlsr_specialist；
10. 训练 sound_ufm_specialist；
11. 训练 music_mert_specialist；
12. 生成所有 Dev / Progress score；
13. 跑 multi-branch fusion；
14. 生成 predict.csv；
15. 提交。
```

---

# 20. 最终主线是：

```text
稳定 UFM + 独立 type classifier + type-specific specialists + multi-branch fusion
```
