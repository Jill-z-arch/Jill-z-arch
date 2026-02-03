#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "/mnt/synology/wen.zhou/2026.05/Sparge_Attention"
source "/home/wen.zhou/miniforge3/etc/profile.d/conda.sh"
conda activate "/home/wen.zhou/miniforge3/envs/py310_local"

python -m py_compile "/mnt/synology/wen.zhou/2026.05/Sparge_Attention/train_Sparge_Agent_Attention.py"

python "/mnt/synology/wen.zhou/2026.05/Sparge_Attention/train_Sparge_Agent_Attention.py"   --image_root "/mnt/synology/wen.zhou/2026.01/dataset2/DATASET/Segmentation"   --train_txt "/mnt/synology/wen.zhou/2026.04/ds2_train_labeled.txt"   --val_txt "/mnt/synology/wen.zhou/2026.04/ds2_val_labeled.txt"   --num_classes 4   --epochs 100   --batch_size 32   --num_workers 0   --lr 0.025   --momentum 0.9   --weight_decay 1e-4   --print_batch_step 20   --topk "1,2,4"   --pretrained   --output_dir "/mnt/synology/wen.zhou/2026.05/Sparge_Attention/output"   --seed 42   --use_sparge_attn   --sparge_topk 64   --attn_order "sparge_then_agent"   --use_agent_attn   --agent_tokens 16   --agent_heads 8   --use_agent_bias   --agent_bias_max_hw 32   --skip_broken
