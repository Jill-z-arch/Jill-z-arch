#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/synology/wen.zhou/2026.05/Coordinate_Attention"
source "/home/wen.zhou/miniforge3/etc/profile.d/conda.sh"
conda activate "/home/wen.zhou/miniforge3/envs/py310_local"

echo "[Info] WORKDIR=/mnt/synology/wen.zhou/2026.05/Coordinate_Attention"
echo "[Info] CONDA=$CONDA_PREFIX"
echo "[Info] LOG=/mnt/synology/wen.zhou/2026.05/Coordinate_Attention/logs_tmux/train_2026-01-24_005155.log"
echo "[Info] OUT=/tmp/output_torch"

python -u train_Coordinate_Attention.py \
  --image_root "/mnt/synology/wen.zhou/2026.01" \
  --train_txt  "/mnt/synology/wen.zhou/2026.04/ds2_train_labeled.txt" \
  --val_txt    "/mnt/synology/wen.zhou/2026.04/ds2_val_labeled.txt" \
  --num_classes "4" \
  --epochs "100" \
  --batch_size "16" \
  --num_workers "0" \
  --lr "0.025" \
  --topk "1,2,4" \
  --pretrained \
  --output_dir "/tmp/output_torch" \
  --skip_broken \
  --use_coord_attn \
  --coord_reduction "32" \
  --use_agent_attn \
  --agent_tokens "16" \
  --agent_heads "8" \
  --use_agent_bias \
  --agent_bias_max_hw "32" \
  --attn_order "coord_then_agent" \
  2>&1 | tee -a "/mnt/synology/wen.zhou/2026.05/Coordinate_Attention/logs_tmux/train_2026-01-24_005155.log" \
  || { echo "[Error] python failed. Keeping tmux alive for debugging..."; sleep 365d; }
