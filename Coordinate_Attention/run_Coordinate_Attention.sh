#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/mnt/synology/wen.zhou/2026.05/Coordinate_Attention"

CONDA_SH="/home/wen.zhou/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="/home/wen.zhou/miniforge3/envs/py310_local"

SESSION="train_coord_agent_attn"
LOG_DIR="${WORKDIR}/logs_tmux"
RUN_DIR="${WORKDIR}/runs_tmux"
OUT_DIR="/mnt/synology/wen.zhou/2026.05/Coordinate_Attention/output"

# 数据路径：按你真实情况填写（你之前已经能跑，沿用你原来的）
IMAGE_ROOT="/mnt/synology/wen.zhou/2026.01/dataset2/DATASET/Segmentation"
TRAIN_TXT="/mnt/synology/wen.zhou/2026.04/ds2_train_labeled.txt"
VAL_TXT="/mnt/synology/wen.zhou/2026.04/ds2_val_labeled.txt"

NUM_CLASSES=4
EPOCHS=100
BATCH_SIZE=16
NUM_WORKERS=0
LR=0.025
TOPK="1,2,4"

# Coordinate Attention
COORD_REDUCTION=32

# Agent Attention
AGENT_TOKENS=16
AGENT_HEADS=8
AGENT_BIAS_MAX_HW=32

# 叠加顺序：coord_then_agent / agent_then_coord
ATTN_ORDER="coord_then_agent"

mkdir -p "$LOG_DIR" "$RUN_DIR"
STAMP="$(date +%F_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${STAMP}.log"
RUN_FILE="${RUN_DIR}/run_${STAMP}.sh"

cat > "$RUN_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "${WORKDIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

echo "[Info] WORKDIR=${WORKDIR}"
echo "[Info] CONDA=\$CONDA_PREFIX"
echo "[Info] LOG=${LOG_FILE}"
echo "[Info] OUT=${OUT_DIR}"

python -u train_Coordinate_Attention.py \\
  --image_root "${IMAGE_ROOT}" \\
  --train_txt  "${TRAIN_TXT}" \\
  --val_txt    "${VAL_TXT}" \\
  --num_classes "${NUM_CLASSES}" \\
  --epochs "${EPOCHS}" \\
  --batch_size "${BATCH_SIZE}" \\
  --num_workers "${NUM_WORKERS}" \\
  --lr "${LR}" \\
  --topk "${TOPK}" \\
  --pretrained \\
  --print_batch_step 1 \\
  --output_dir "${OUT_DIR}" \\
  --skip_broken \\
  --use_coord_attn \\
  --coord_reduction "${COORD_REDUCTION}" \\
  --use_agent_attn \\
  --agent_tokens "${AGENT_TOKENS}" \\
  --agent_heads "${AGENT_HEADS}" \\
  --use_agent_bias \\
  --agent_bias_max_hw "${AGENT_BIAS_MAX_HW}" \\
  --attn_order "${ATTN_ORDER}" \\
  2>&1 | tee -a "${LOG_FILE}" \\
  || { echo "[Error] python failed. Keeping tmux alive for debugging..."; sleep 365d; }
EOF

chmod +x "$RUN_FILE"

# 启动 tmux（稳定写法：直接跑 RUN_FILE）
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[Warn] tmux session '$SESSION' already exists. Kill it first:"
  echo "       tmux kill-session -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$WORKDIR" "bash \"$RUN_FILE\""

echo "[OK] Started tmux session: $SESSION"
echo "     Log: $LOG_FILE"
echo "     Run: $RUN_FILE"
echo "     Attach: tmux attach -t $SESSION"
