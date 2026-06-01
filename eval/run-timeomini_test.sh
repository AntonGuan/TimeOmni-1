set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export ROOT_DIR
WORKSPACE_DIR="$(cd "$ROOT_DIR/.." && pwd)"
HF_CACHE_DIR=${HF_CACHE_DIR:-"$WORKSPACE_DIR/.hf"}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$WORKSPACE_DIR/.cache"}
export XDG_CACHE_HOME
# Resolve Hugging Face repo ids (e.g. TimeOmni-1/TimeOmni-1-4B) from the shared
# cache so MODEL_DIR may be either a local snapshot path or a repo id.
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_CACHE_DIR"}

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=${PYTHON_BIN:-python}
else
  PYTHON_BIN=${PYTHON_BIN:-python3}
fi

if [ -d "$ROOT_DIR/.venv/bin" ]; then
  export PATH="$ROOT_DIR/.venv/bin:$PATH"
fi

"$PYTHON_BIN" "$ROOT_DIR/install/download_testbed.py" --out_dir "$ROOT_DIR/data" --cache_dir "$HF_CACHE_DIR"

MODEL_DIR=${MODEL_DIR:-"Your Local Model Path"}
ANS_ID_PATH=${ANS_ID_PATH:-"$ROOT_DIR/answer/timeomni1_test/id_outputs.json"}
RES_ID_PATH=${RES_ID_PATH:-"$ROOT_DIR/answer/timeomni1_test/id_results.json"}
ANS_OOD_PATH=${ANS_OOD_PATH:-"$ROOT_DIR/answer/timeomni1_test/ood_outputs.json"}
RES_OOD_PATH=${RES_OOD_PATH:-"$ROOT_DIR/answer/timeomni1_test/ood_results.json"}
SAMPLE_N=${SAMPLE_N:-0}
GPU_IDS=${GPU_IDS:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-4}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
DTYPE=${DTYPE:-auto}
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-0}
# CUDA graphs (enforce_eager=0) are much faster, especially for the Qwen3.5-based
# 4B/9B linear-attention models. Set ENFORCE_EAGER=1 to fall back to eager mode.
ENFORCE_EAGER=${ENFORCE_EAGER:-0}

ID_TEST_FILE="$ROOT_DIR/data/timeomni1_id_test.json"
OOD_TEST_FILE="$ROOT_DIR/data/timeomni1_ood_test.json"

SAMPLE_SEED=${SAMPLE_SEED:-42}
export SAMPLE_N SAMPLE_SEED
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
PROC_TOTAL=${PROC_TOTAL:-${#GPU_ARRAY[@]}}

if [ "$SAMPLE_N" -gt 0 ]; then
  ID_TEST_FILE="$ROOT_DIR/data/timeomni1_id_test_sample_${SAMPLE_N}.json"
  OOD_TEST_FILE="$ROOT_DIR/data/timeomni1_ood_test_sample_${SAMPLE_N}.json"
  "$PYTHON_BIN" - <<'PY'
import json, os, random
root = os.environ["ROOT_DIR"]
n = int(os.environ["SAMPLE_N"])
seed = int(os.environ["SAMPLE_SEED"])
for src, dst in [
    (os.path.join(root, "data", "timeomni1_id_test.json"), os.path.join(root, "data", f"timeomni1_id_test_sample_{n}.json")),
    (os.path.join(root, "data", "timeomni1_ood_test.json"), os.path.join(root, "data", f"timeomni1_ood_test_sample_{n}.json")),
]:
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.Random(seed).shuffle(data)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data[:n], f, ensure_ascii=False)
PY
fi

run_split() {
  local test_file="$1"
  local output_path="$2"
  local proc_id
  local gpu_index
  local gpu_id
  local status=0
  local -a extra_args=()
  local -a pids=()

  if [ "$LANGUAGE_MODEL_ONLY" = "1" ]; then
    extra_args+=(--language_model_only)
  fi
  if [ "$ENFORCE_EAGER" = "1" ]; then
    extra_args+=(--enforce_eager)
  fi

  for ((proc_id=0; proc_id<PROC_TOTAL; proc_id++)); do
    gpu_index=$((proc_id % ${#GPU_ARRAY[@]}))
    gpu_id="${GPU_ARRAY[$gpu_index]}"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$ROOT_DIR/eval/inference.py" \
      --model_dir "$MODEL_DIR" \
      --test_file "$test_file" \
      --output_path "$output_path" \
      --proc_total "$PROC_TOTAL" \
      --proc_id "$proc_id" \
      --batch_size "$BATCH_SIZE" \
      --workers "$WORKERS" \
      --parallel_size "$PARALLEL_SIZE" \
      --max_model_len "$MAX_MODEL_LEN" \
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
      --dtype "$DTYPE" \
      "${extra_args[@]}" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

run_split "$ID_TEST_FILE" "$ANS_ID_PATH"
run_split "$OOD_TEST_FILE" "$ANS_OOD_PATH"

# Calculate scores for ID test results
"$PYTHON_BIN" "$ROOT_DIR/eval/get_score.py" \
    --input_path "$ANS_ID_PATH" \
    --output_path "$RES_ID_PATH" \
    --proc_total "$PROC_TOTAL"

# Calculate scores for OOD test results
"$PYTHON_BIN" "$ROOT_DIR/eval/get_score.py" \
    --input_path "$ANS_OOD_PATH" \
    --output_path "$RES_OOD_PATH" \
    --proc_total "$PROC_TOTAL"