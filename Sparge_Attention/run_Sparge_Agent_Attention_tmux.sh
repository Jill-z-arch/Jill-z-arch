#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/mnt/synology/wen.zhou/2026.05/Sparge_Attention"
CONDA="/home/wen.zhou/miniforge3/envs/py310_local"
SESSION="train_sparge_agent_attn"

LOG_DIR="${WORKDIR}/logs_tmux"
RUN_DIR="${WORKDIR}/runs_tmux"
OUT_DIR="${WORKDIR}/output"

mkdir -p "${LOG_DIR}" "${RUN_DIR}" "${OUT_DIR}"

TS="$(date +%F_%H%M%S)"
LOG="${LOG_DIR}/train_${TS}.log"
RUN="${RUN_DIR}/run_${TS}.sh"

cat > "${RUN}" <<RUN_EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "${WORKDIR}"
source /home/wen.zhou/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA}"

echo "[Info] WORKDIR=${WORKDIR}"
echo "[Info] CONDA=${CONDA}"
echo "[Info] LOG=${LOG}"
echo "[Info] OUT=${OUT_DIR}"

python -m py_compile "${WORKDIR}/train_Sparge_Agent_Attention.py"

python -u "${WORKDIR}/train_Sparge_Agent_Attention.py" \\
  --image_root "/mnt/synology/wen.zhou/2026.01/dataset2/DATASET/Segmentation" \\
  --train_txt  "/mnt/synology/wen.zhou/2026.04/ds2_train_labeled.txt" \\
  --val_txt    "/mnt/synology/wen.zhou/2026.04/ds2_val_labeled.txt" \\
  --num_classes 4 \\
  --epochs 100 \\
  --batch_size 32 \\
  --num_workers 0 \\
  --lr 0.025 \\
  --momentum 0.9 \\
  --weight_decay 1e-4 \\
  --print_batch_step 20 \\
  --topk "1,2,4" \\
  --pretrained \\
  --output_dir "${OUT_DIR}" \\
  --seed 42 \\
  --use_sparge_attn \\
  --sparge_topk 64 \\
  --attn_order "sparge_then_agent" \\
  --use_agent_attn \\
  --agent_tokens 16 \\
  --agent_heads 8 \\
  --use_agent_bias \\
  --agent_bias_max_hw 32 \\
  --skip_broken \\
  2>&1 | tee -a "${LOG}"
RUN_EOF

chmod +x "${RUN}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" "bash '${RUN}' || { echo '[Error] python failed; keeping tmux alive...'; exec bash; }"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[OK] Started tmux session: ${SESSION}"
  echo "     Log: ${LOG}"
  echo "     Run: ${RUN}"
  echo "     Attach: tmux attach -t ${SESSION}"
else
  echo "[Error] tmux session not found right after start. Showing log tail:"
  tail -n 120 "${LOG}" || true
  exit 1
fi
