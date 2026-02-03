#!/usr/bin/env bash
set -euo pipefail

# -------- paths (固定到当前工程目录，避免 WORKDIR 写错导致“无日志闪退”) --------
WORKDIR="/mnt/synology/wen.zhou/2026.05/Rotate_to_Attend"
PY_SCRIPT="${WORKDIR}/train_Rotate_to_Attend.py"

CONDA_SH="/home/wen.zhou/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="/home/wen.zhou/miniforge3/envs/py310_local"

SESSION="train_rotate_to_attend"

LOG_DIR="${WORKDIR}/logs_tmux"
RUN_DIR="${WORKDIR}/runs_tmux"
OUT_DIR="${WORKDIR}/output"

# -------- dataset --------
IMAGE_ROOT="/mnt/synology/wen.zhou/2026.01/dataset2/DATASET/Segmentation"
TRAIN_TXT="/mnt/synology/wen.zhou/2026.04/ds2_train_labeled.txt"
VAL_TXT="/mnt/synology/wen.zhou/2026.04/ds2_val_labeled.txt"

# -------- train --------
NUM_CLASSES=4
EPOCHS=100
BATCH_SIZE=32
NUM_WORKERS=0
LR=0.025
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
PRINT_BATCH_STEP=20
TOPK="1,2,4"
SEED=42
PRETRAINED=1

# -------- agent attn --------
USE_AGENT_ATTN=1
AGENT_TOKENS=16
AGENT_HEADS=8
USE_AGENT_BIAS=1
AGENT_BIAS_MAX_HW=32

# -------- rotate-to-attend (triplet attention) --------
USE_TRIPLET_ATTN=1
TRIPLET_KERNEL=7
TRIPLET_NO_SPATIAL=0
ATTN_ORDER="triplet_then_agent"   # triplet_then_agent | agent_then_triplet

mkdir -p "${LOG_DIR}" "${RUN_DIR}" "${OUT_DIR}"

STAMP="$(date +%F_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${STAMP}.log"
RUN_FILE="${RUN_DIR}/run_${STAMP}.sh"

# 先写一行日志，保证你就算 tmux 起不来也能看到“脚本至少运行到了哪一步”
echo "[Boot] $(date) starting launcher" > "${LOG_FILE}"
echo "[Meta] launcher=${WORKDIR}/run_Rotate_to_Attend_tmux.sh" | tee -a "${LOG_FILE}"
echo "[Meta] python_script=${PY_SCRIPT}" | tee -a "${LOG_FILE}"
echo "[Meta] image_root=${IMAGE_ROOT}" | tee -a "${LOG_FILE}"
echo "[Meta] train_txt=${TRAIN_TXT}" | tee -a "${LOG_FILE}"
echo "[Meta] val_txt=${VAL_TXT}" | tee -a "${LOG_FILE}"
echo "[Meta] output_dir=${OUT_DIR}" | tee -a "${LOG_FILE}"

# 生成真正执行的 run 脚本（避免 tmux quoting 坑）
cat > "${RUN_FILE}" <<RUN_EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "${WORKDIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

echo "[Info] WORKDIR=${WORKDIR}" | tee -a "${LOG_FILE}"
echo "[Info] CONDA=\${CONDA_PREFIX}" | tee -a "${LOG_FILE}"
echo "[Info] LOG=${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "[Info] OUT=${OUT_DIR}" | tee -a "${LOG_FILE}"

echo "[Meta] launcher=${WORKDIR}/run_Rotate_to_Attend_tmux.sh" | tee -a "${LOG_FILE}"
echo "[Meta] run_file=${RUN_FILE}" | tee -a "${LOG_FILE}"
echo "[Meta] python_script=${PY_SCRIPT}" | tee -a "${LOG_FILE}"
echo "[Meta] image_root=${IMAGE_ROOT}" | tee -a "${LOG_FILE}"
echo "[Meta] train_txt=${TRAIN_TXT}" | tee -a "${LOG_FILE}"
echo "[Meta] val_txt=${VAL_TXT}" | tee -a "${LOG_FILE}"

# 备份本次运行的 py/sh 到 output，方便复现
cp -f "${PY_SCRIPT}" "${OUT_DIR}/train_Rotate_to_Attend_${STAMP}.py" || true
cp -f "${WORKDIR}/run_Rotate_to_Attend_tmux.sh" "${OUT_DIR}/run_Rotate_to_Attend_tmux_${STAMP}.sh" || true
echo "[Meta] saved_py=${OUT_DIR}/train_Rotate_to_Attend_${STAMP}.py" | tee -a "${LOG_FILE}"
echo "[Meta] saved_sh=${OUT_DIR}/run_Rotate_to_Attend_tmux_${STAMP}.sh" | tee -a "${LOG_FILE}"

# 先编译检查，编译失败也会写入 log
python -m py_compile "${PY_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"

CMD=(python -u "${PY_SCRIPT}"
  --image_root "${IMAGE_ROOT}"
  --train_txt "${TRAIN_TXT}"
  --val_txt "${VAL_TXT}"
  --num_classes "${NUM_CLASSES}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --momentum "${MOMENTUM}"
  --weight_decay "${WEIGHT_DECAY}"
  --print_batch_step "${PRINT_BATCH_STEP}"
  --topk "${TOPK}"
  --output_dir "${OUT_DIR}"
  --seed "${SEED}"
  --skip_broken
)

if [ "${PRETRAINED}" = "1" ]; then
  CMD+=(--pretrained)
fi

if [ "${USE_TRIPLET_ATTN}" = "1" ]; then
  CMD+=(--use_triplet_attn --triplet_kernel_size "${TRIPLET_KERNEL}" --attn_order "${ATTN_ORDER}")
  if [ "${TRIPLET_NO_SPATIAL}" = "1" ]; then
    CMD+=(--triplet_no_spatial)
  fi
fi

if [ "${USE_AGENT_ATTN}" = "1" ]; then
  CMD+=(--use_agent_attn --agent_tokens "${AGENT_TOKENS}" --agent_heads "${AGENT_HEADS}")
  if [ "${USE_AGENT_BIAS}" = "1" ]; then
    CMD+=(--use_agent_bias --agent_bias_max_hw "${AGENT_BIAS_MAX_HW}")
  fi
fi

echo "[Meta] cmd=\${CMD[*]}" | tee -a "${LOG_FILE}"

set +e
"\${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
status=\${PIPESTATUS[0]}
set -e
echo "[Meta] python_exit=\${status}" | tee -a "${LOG_FILE}"
if [ "\${status}" -ne 0 ]; then
  echo "[Error] python failed; keeping tmux alive for debugging..." | tee -a "${LOG_FILE}"
fi

echo "[Done] log=${LOG_FILE}" | tee -a "${LOG_FILE}"
exec bash
RUN_EOF

chmod +x "${RUN_FILE}"

# 防止“旧 session 冲突”
tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" "bash '${RUN_FILE}'"

echo "[OK] Started tmux session: ${SESSION}"
echo "     Log: ${LOG_FILE}"
echo "     Run: ${RUN_FILE}"
echo "     Attach: tmux attach -t ${SESSION}"
