export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STREAMVLN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HABITAT_LAB_ROOT="/home/yuchang/yc_code/habitat-lab-v0.2.4/habitat-lab"
HABITAT_BASELINES_ROOT="/home/yuchang/yc_code/habitat-lab-v0.2.4/habitat-baselines"

CHECKPOINT="/home/yuchang/yc_code/StreamVLN/data/pretrained_models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
OUTPUT_PATH="/home/yuchang/yc_code/StreamVLN/outputs/cpu_eval"

mkdir -p "$OUTPUT_PATH"

echo "CHECKPOINT: ${CHECKPOINT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "HABITAT_LAB_ROOT: ${HABITAT_LAB_ROOT}"

echo "Running a single-episode demo with video output enabled."

PYTHONPATH="$STREAMVLN_ROOT:$HABITAT_LAB_ROOT:$HABITAT_BASELINES_ROOT:${PYTHONPATH:-}" \
python "$STREAMVLN_ROOT/streamvln/streamvln_eval.py" \
  --model_path "$CHECKPOINT" \
  --habitat_config_path "$STREAMVLN_ROOT/config/vln_r2r.yaml" \
  --eval_split val_unseen \
  --output_path "$OUTPUT_PATH" \
  --num_future_steps 2 \
  --num_frames 4 \
  --num_history 8 \
  --save_video \
  --device cpu \
  --max_episodes 1
