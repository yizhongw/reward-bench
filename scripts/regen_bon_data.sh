#!/bin/bash
set -x

# set variables
GENERATOR="allenai/tulu-2-13b"
N_COMPLETIONS=16
TEMPERATURE=0.7
TOP_P=1.0
GPU_COUNT=2
BATCH_SIZE_VLLM=1000
PER_TASK_MAX_EXAMPLES=1000
EXAMPLE_SAMPLING_SEED=42
MODEL_MAX_LENGTH=8192
MAX_GEN_TOKS=4096
OUTPUT_DIR="output/oe-eval-bon-candidates-new"
SUBDIR="${GENERATOR##*/}-n-${N_COMPLETIONS}"
UPLOAD_TO_HF=true
HF_REPO_ID="ai2-adapt-dev/oe-eval-bon-candidates"
HF_PATH_IN_REPO="${GENERATOR##*/}-n-${N_COMPLETIONS}"

TASKS=(
    'gsm8k'
    'codex_humaneval'
    'codex_humanevalplus'
    'ifeval'
    'popqa'
    'alpaca_eval_v1'
    'alpaca_eval_v2'
    'bbh:cot::none'  # This is a suite of tasks. Run this sepeartely with PER_TASK_MAX_EXAMPLES=50.
)

GENERATION_KWARGS="{
    \"repeats\": 1,
    \"temperature\": ${TEMPERATURE},
    \"top_p\": ${TOP_P},
    \"max_gen_toks\": ${MAX_GEN_TOKS},
    \"do_sample\": true,
    \"truncate_context\": true
}"

# Here for each task, we run oe-eval N_COMPLETIONS times to get N_COMPLETIONS generations for each task.
# We do this outside oe-eval because in this way we can get the score for each generation.
# These scores will be used in bon evaluation, without rerunning the oe-eval.
for TASK_NAME in "${TASKS[@]}"; do
    for i in $(seq 3 3); do
        echo "Running oe-eval for task ${TASK_NAME} for the ${i}th completion."
        # Randomly generate a seed to initialize VLLM
        # Otherwise, oe-eval will use the default seed 1234, which causes the same generations for every run.
        VLLM_INIT_SEED=${i}
        oe-eval \
            --task "${TASK_NAME}"  \
            --task-args "{\"generation_kwargs\":${GENERATION_KWARGS}}" \
            --use-chat-format true \
            --model ${GENERATOR} \
            --model-args "{\"model_path\":\"${GENERATOR}\", \"seed\":${VLLM_INIT_SEED}, \"max_length\":${MODEL_MAX_LENGTH}}" \
            --model-type vllm \
            --gpus ${GPU_COUNT} \
            --batch-size ${BATCH_SIZE_VLLM} \
            --save-raw-requests true \
            --output-dir ${OUTPUT_DIR}/${SUBDIR}/${TASK_NAME}/run-${i} \
            --limit ${PER_TASK_MAX_EXAMPLES} \
            --random-subsample-seed ${EXAMPLE_SAMPLING_SEED} \
            --run-local
    done
done

PROCESSED_DIR=${OUTPUT_DIR}/${SUBDIR}/processed
for TASK_NAME in "${TASKS[@]}"; do
    mkdir -p ${PROCESSED_DIR}/${TASK_NAME}
    python scripts/process_bon_data.py \
        --task_name ${TASK_NAME} \
        --oe_eval_result_dir ${OUTPUT_DIR}/${SUBDIR}/${TASK_NAME} \
        --output_path ${PROCESSED_DIR}/${TASK_NAME}/bon_candidates.jsonl
done

# (optionally) upload the data to huggingface hub
if $UPLOAD_TO_HF; then
    huggingface-cli upload --private --repo-type dataset ${HF_REPO_ID} ${OUTPUT_DIR}/${SUBDIR}/processed $HF_PATH_IN_REPO
fi
