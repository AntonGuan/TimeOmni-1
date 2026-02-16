set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export ROOT_DIR
python "$ROOT_DIR/install/download_testbed.py" --out_dir "$ROOT_DIR/data"

MODEL_DIR=${MODEL_DIR:-"Your Local Model Path"}
ANS_ID_PATH=${ANS_ID_PATH:-"$ROOT_DIR/answer/timeomni1_test/id_outputs.json"}
RES_ID_PATH=${RES_ID_PATH:-"$ROOT_DIR/answer/timeomni1_test/id_results.json"}
ANS_OOD_PATH=${ANS_OOD_PATH:-"$ROOT_DIR/answer/timeomni1_test/ood_outputs.json"}
RES_OOD_PATH=${RES_OOD_PATH:-"$ROOT_DIR/answer/timeomni1_test/ood_results.json"}
SAMPLE_N=${SAMPLE_N:-0}

ID_TEST_FILE="$ROOT_DIR/data/timeomni1_id_test.json"
OOD_TEST_FILE="$ROOT_DIR/data/timeomni1_ood_test.json"

SAMPLE_SEED=${SAMPLE_SEED:-42}
export SAMPLE_N SAMPLE_SEED

if [ "$SAMPLE_N" -gt 0 ]; then
  ID_TEST_FILE="$ROOT_DIR/data/timeomni1_id_test_sample_${SAMPLE_N}.json"
  OOD_TEST_FILE="$ROOT_DIR/data/timeomni1_ood_test_sample_${SAMPLE_N}.json"
  python - <<'PY'
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

# Run inference for ID test data -- 4 processes in parallel on 4 GPUs
CUDA_VISIBLE_DEVICES=0 python "$ROOT_DIR/eval/inference.py" \
    --model_dir "$MODEL_DIR" \
    --test_file "$ID_TEST_FILE" \
    --output_path "$ANS_ID_PATH" \
    --proc_total 4 \
    --proc_id 0 \
    --batch_size 8 \
    --workers 4 \
    --parallel_size 1 &

CUDA_VISIBLE_DEVICES=1 python "$ROOT_DIR/eval/inference.py" \
    --model_dir "$MODEL_DIR" \
    --test_file "$ID_TEST_FILE" \
    --output_path "$ANS_ID_PATH" \
    --proc_total 4 \
    --proc_id 1 \
    --batch_size 8 \
    --workers 4 \
    --parallel_size 1 &

CUDA_VISIBLE_DEVICES=2 python "$ROOT_DIR/eval/inference.py" \
    --model_dir "$MODEL_DIR" \
    --test_file "$ID_TEST_FILE" \
    --output_path "$ANS_ID_PATH" \
    --proc_total 4 \
    --proc_id 2 \
    --batch_size 8 \
    --workers 4 \
    --parallel_size 1 &

CUDA_VISIBLE_DEVICES=3 python "$ROOT_DIR/eval/inference.py" \
    --model_dir "$MODEL_DIR" \
    --test_file "$ID_TEST_FILE" \
    --output_path "$ANS_ID_PATH" \
    --proc_total 4 \
    --proc_id 3 \
    --batch_size 8 \
    --workers 4 \
    --parallel_size 1 &

wait

# Run inference for OOD test data - 4 processes in parallel on 4 GPUs
CUDA_VISIBLE_DEVICES=0 python "$ROOT_DIR/eval/inference.py" \
    --model_dir "$MODEL_DIR" \
    --test_file "$OOD_TEST_FILE" \
    --output_path "$ANS_OOD_PATH" \
    --proc_total 4 \
    --proc_id 0 \
    --batch_size 8 \
    --workers 4 \
    --parallel_size 1 &

CUDA_VISIBLE_DEVICES=1 python "$ROOT_DIR/eval/inference.py" \
    --model_dir "$MODEL_DIR" \
    --test_file "$OOD_TEST_FILE" \
    --output_path "$ANS_OOD_PATH" \
    --proc_total 4 \
    --proc_id 1 \
    --batch_size 8 \
    --workers 4 \
    --parallel_size 1 &

CUDA_VISIBLE_DEVICES=2 python "$ROOT_DIR/eval/inference.py" \
    --model_dir "$MODEL_DIR" \
    --test_file "$OOD_TEST_FILE" \
    --output_path "$ANS_OOD_PATH" \
    --proc_total 4 \
    --proc_id 2 \
    --batch_size 8 \
    --workers 4 \
    --parallel_size 1 &

CUDA_VISIBLE_DEVICES=3 python "$ROOT_DIR/eval/inference.py" \
    --model_dir "$MODEL_DIR" \
    --test_file "$OOD_TEST_FILE" \
    --output_path "$ANS_OOD_PATH" \
    --proc_total 4 \
    --proc_id 3 \
    --batch_size 8 \
    --workers 4 \
    --parallel_size 1 &

wait

# Calculate scores for ID test results
python "$ROOT_DIR/eval/get_score.py" \
    --input_path "$ANS_ID_PATH" \
    --output_path "$RES_ID_PATH" \
    --proc_total 4

# Calculate scores for OOD test results
python "$ROOT_DIR/eval/get_score.py" \
    --input_path "$ANS_OOD_PATH" \
    --output_path "$RES_OOD_PATH" \
    --proc_total 4