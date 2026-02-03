#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend"
source "/home/wen.zhou/miniforge3/etc/profile.d/conda.sh"
conda activate "/home/wen.zhou/miniforge3/envs/py310_local"

echo "[Info] WORKDIR=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend"
echo "[Info] CONDA=${CONDA_PREFIX}"
echo "[Info] LOG=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/logs_tmux/train_2026-01-24_120756.log"
echo "[Info] OUT=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/output"

echo "[Meta] launcher=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/run_Rotate_Agent_Attention_tmux.sh"
echo "[Meta] run_file=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/runs_tmux/run_2026-01-24_120756.sh"
echo "[Meta] python_script=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/train_Rotate_Agent_Attention.py"
echo "[Meta] image_root=/mnt/synology/wen.zhou/2026.01/dataset2/DATASET/Segmentation"
echo "[Meta] train_txt=/mnt/synology/wen.zhou/2026.04/ds2_train_labeled.txt"
echo "[Meta] val_txt=/mnt/synology/wen.zhou/2026.04/ds2_val_labeled.txt"

# Keep a copy of scripts inside output for reproducibility
cp -f "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/train_Rotate_Agent_Attention.py" "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/output/train_Rotate_Agent_Attention_2026-01-24_120756.py"
cp -f "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/run_Rotate_Agent_Attention_tmux.sh" "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/output/run_Rotate_Agent_Attention_tmux_2026-01-24_120756.sh" 2>/dev/null || true
echo "[Meta] saved_py=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/output/train_Rotate_Agent_Attention_2026-01-24_120756.py"
echo "[Meta] saved_sh=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/output/run_Rotate_Agent_Attention_tmux_2026-01-24_120756.sh"

python -m py_compile "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/train_Rotate_Agent_Attention.py" 2>&1 | tee -a "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/logs_tmux/train_2026-01-24_120756.log"

CMD=(python -u "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/train_Rotate_Agent_Attention.py"
  --image_root "/mnt/synology/wen.zhou/2026.01/dataset2/DATASET/Segmentation"
  --train_txt "/mnt/synology/wen.zhou/2026.04/ds2_train_labeled.txt"
  --val_txt "/mnt/synology/wen.zhou/2026.04/ds2_val_labeled.txt"
  --num_classes "4"
  --epochs "100"
  --batch_size "32"
  --num_workers "0"
  --lr "0.025"
  --momentum "0.9"
  --weight_decay "1e-4"
  --print_batch_step "20"
  --topk "1,2,4"
  --output_dir "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/output"
  --seed "42"
  --skip_broken
)

if [ "1" = "1" ]; then
  CMD+=(--pretrained)
fi

if [ "1" = "1" ]; then
  CMD+=(--use_triplet_attn --triplet_kernel_size "7" --attn_order "triplet_then_agent")
  if [ "0" = "1" ]; then
    CMD+=(--triplet_no_spatial)
  fi
fi

if [ "1" = "1" ]; then
  CMD+=(--use_agent_attn --agent_tokens "16" --agent_heads "8")
  if [ "1" = "1" ]; then
    CMD+=(--use_agent_bias --agent_bias_max_hw "32")
  fi
fi

echo "[Meta] cmd=${CMD[*]}" | tee -a "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/logs_tmux/train_2026-01-24_120756.log"

set +e
"${CMD[@]}" 2>&1 | tee -a "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/logs_tmux/train_2026-01-24_120756.log"
status=${PIPESTATUS[0]}
set -e
echo "[Meta] python_exit=${status}" | tee -a "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/logs_tmux/train_2026-01-24_120756.log"
if [ "${status}" -ne 0 ]; then
  echo "[Error] python failed; keeping tmux alive for debugging..." | tee -a "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/logs_tmux/train_2026-01-24_120756.log"
fi

echo "[Done] log=/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/logs_tmux/train_2026-01-24_120756.log" | tee -a "/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend/logs_tmux/train_2026-01-24_120756.log"
exec bash
