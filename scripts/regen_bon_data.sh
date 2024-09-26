#!/bin/bash
# set -x

# set variables
GENERATOR="allenai/tulu-2-13b"
N_COMPLETIONS=16
TEMPERATURE=0.7
TOP_P=1.0
GPU_COUNT=4
BATCH_SIZE_VLLM=1000
PER_TASK_MAX_EXAMPLES=1000
EXAMPLE_SAMPLING_SEED=42
MODEL_MAX_LENGTH=8192
MAX_GEN_TOKS=4096
OUTPUT_DIR="output/oe-eval-bon-candidates"
SUBDIR="${GENERATOR##*/}-n-${N_COMPLETIONS}"
UPLOAD_TO_HF=true
HF_REPO_ID="ai2-adapt-dev/oe-eval-bon-candidates"
HF_PATH_IN_REPO="${GENERATOR##*/}-n-${N_COMPLETIONS}"

GENERATION_KWARGS="{
    \"repeats\": 1,
    \"temperature\": ${TEMPERATURE},
    \"top_p\": ${TOP_P},
    \"max_gen_toks\": ${MAX_GEN_TOKS},
    \"do_sample\": true,
    \"truncate_context\": true
}"

declare -A TASK_SPECS=(
    ["gsm8k"]="{
        \"task_name\": \"gsm8k\",
        \"split\": \"test\",
        \"primary_metric\": \"exact_match\",
        \"use_chat_format\": true,
        \"num_shots\": 8,
        \"fewshot_source\": \"STD:GSM8k\",
        \"chat_overrides\": {
            \"context_kwargs\": {
                \"fewshot_as_multiturn\": false
            }
        },
        \"generation_kwargs\": ${GENERATION_KWARGS}
    }"
    ["codex_humaneval"]="{
        \"task_name\": \"codex_humaneval\",
        \"primary_metric\": \"pass_at_1\",
        \"use_chat_format\": true,
        \"metric_kwargs\": {
            \"pass_at_ks\": [1]
        },
        \"generation_kwargs\": ${GENERATION_KWARGS}
    }"
    ["codex_humanevalplus"]="{
        \"task_name\": \"codex_humanevalplus\",
        \"primary_metric\": \"pass_at_1\",
        \"use_chat_format\": true,
        \"metric_kwargs\": {
            \"pass_at_ks\": [1]
        },
        \"generation_kwargs\": ${GENERATION_KWARGS}
    }"
    ["ifeval"]="{
        \"task_name\": \"ifeval\",
        \"primary_metric\": \"prompt_level_loose_acc\",
        \"split\": \"train\",
        \"use_chat_format\": true,
        \"metric_kwargs\": {
            \"aggregation_levels\": [\"prompt\", \"inst\"],
            \"strictness_levels\": [\"strict\", \"loose\"],
            \"output_individual_metrics\": true
        },
        \"generation_kwargs\": ${GENERATION_KWARGS}
    }"
    ["popqa"]="{
        \"task_name\": \"popqa\",
        \"use_chat_format\": true,
        \"chat_overrides\": {
            \"context_kwargs\": {
                \"fewshot_as_multiturn\": false
            }
        },
        \"generation_kwargs\": ${GENERATION_KWARGS}
    }"
    ["alpaca_eval_v1"]="{
        \"task_name\": \"alpaca_eval\",
        \"primary_metric\": \"win_rate\",
        \"metric_kwargs\": {
            \"alpaca_eval_version\": 1
        },
        \"use_chat_format\": true,
        \"generation_kwargs\": ${GENERATION_KWARGS}
    }"
    ["alpaca_eval_v2"]="{
        \"task_name\": \"alpaca_eval\",
        \"primary_metric\": \"length_controlled_winrate\",
        \"metric_kwargs\": {
            \"alpaca_eval_version\": 2
        },
        \"use_chat_format\": true,
        \"generation_kwargs\": ${GENERATION_KWARGS}
    }"
    ["bbh"]="{
        \"task_name\": \"bbh:cot::tulu\",
        \"primary_metric\": \"exact_match\",
        \"split\": \"test\",
        \"use_chat_format\": true,
        \"num_shots\": 3,
        \"chat_overrides\": {
            \"context_kwargs\": {
                \"fewshot_as_multiturn\": false
            }
        },
        \"generation_kwargs\": ${GENERATION_KWARGS}
    }"
)

TASKS_TO_RUN=(
    'gsm8k'
    'codex_humaneval'
    'codex_humanevalplus'
    'ifeval'
    'popqa'
    'alpaca_eval_v1'
    'alpaca_eval_v2'
    'bbh'  # This is a suite of tasks. Run this sepeartely with PER_TASK_MAX_EXAMPLES=50.
)

# Here for each task, we run oe-eval N_COMPLETIONS times to get N_COMPLETIONS generations for each task.
# We do this outside oe-eval because in this way we can get the score for each generation.
# These scores will be used in bon evaluation, without rerunning the oe-eval.
for TASK_NAME in "${TASKS_TO_RUN[@]}"; do
    TASK_SPEC_ARGS=${TASK_SPECS[${TASK_NAME}]}
    ACTUAL_TASK_NAME=$(echo "${TASK_SPEC_ARGS}" | jq -r '.task_name')
    TASK_SPEC_ARGS=$(echo "${TASK_SPEC_ARGS}" | jq 'del(.task_name)')

    for i in $(seq 1 ${N_COMPLETIONS}); do
        echo "Running oe-eval for task ${TASK_NAME} for the ${i}th completion."
        echo "Task spec args: ${TASK_SPEC_ARGS}"
        # Randomly generate a seed to initialize VLLM
        # Otherwise, oe-eval will use the default seed 1234, which causes the same generations for every run.
        VLLM_INIT_SEED=${i}
        echo "VLLM init seed: ${VLLM_INIT_SEED}"
        oe-eval \
            --task "${ACTUAL_TASK_NAME}"  \
            --task-args "${TASK_SPEC_ARGS}" \
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
for TASK_NAME in "${TASKS_TO_RUN[@]}"; do
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
