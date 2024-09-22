#!/bin/bash
set -x

# set variables
GENERATOR="allenai/tulu-2-13b"
N_COMPLETIONS=16
TEMPERATURE=0.7
TOP_P=1.0
GPU_COUNT=2
BATCH_SIZE_VLLM=100
PER_TASK_MAX_EXAMPLES=1000
MODEL_MAX_LENGTH=8192
MAX_GEN_TOKS=4096
OUTPUT_DIR="output/oe-eval-bon-candidates"
SUBDIR="${GENERATOR##*/}-n-${N_COMPLETIONS}"
RUN_LOCAL=true
UPLOAD_TO_HF=false

TASKS=(
    'gsm8k'
    'codex_humaneval'
    'codex_humanevalplus'
    'ifeval'
    'popqa'
    'alpaca_eval_v1'
    'alpaca_eval_v2'
    # 'bbh:cot::none'
    # 'minerva_math::tulu'
)

GENERATION_KWARGS="{
    \"repeats\": ${N_COMPLETIONS},
    \"temperature\": ${TEMPERATURE},
    \"top_p\": ${TOP_P},
    \"max_gen_toks\": ${MAX_GEN_TOKS},
    \"do_sample\": true,
    \"truncate_context\": true
}"

for TASK_NAME in "${TASKS[@]}"; do
    if $RUN_LOCAL; then
        # run oe-eval locally if beaker is not installed
        echo "Running oe-eval locally"
        oe-eval \
            --task "${TASK_NAME}"  \
            --task-args "{\"generation_kwargs\":${GENERATION_KWARGS}}" \
            --use-chat-format true \
            --model ${GENERATOR} \
            --model-args "{\"model_path\":\"${GENERATOR}\", \"max_length\":${MODEL_MAX_LENGTH}}" \
            --model-type vllm \
            --gpus ${GPU_COUNT} \
            --batch-size ${BATCH_SIZE_VLLM} \
            --save-raw-requests true \
            --output-dir ${OUTPUT_DIR}/${SUBDIR}/${TASK_NAME} \
            --limit ${PER_TASK_MAX_EXAMPLES} \
            --random-subsample-seed 42 \
            --run-local
    else
        # otherwise we assume it's ai2 internal and will run on beaker
        echo "Running oe-eval on beaker"
        oe-eval \
            --task "${TASK_NAME}"  \
            --task-args "{\"generation_kwargs\":${GENERATION_KWARGS}}" \
            --use-chat-format true \
            --model ${GENERATOR} \
            --model-args "{\"model_path\":\"${GENERATOR}\", \"max_length\":${MODEL_MAX_LENGTH}}" \
            --model-type vllm \
            --gpus ${GPU_COUNT} \
            --batch-size ${BATCH_SIZE_VLLM} \
            --save-raw-requests true \
            --output-dir /results/ \
            --limit ${PER_TASK_MAX_EXAMPLES} \
            --random-subsample-seed 42 \
            --beaker-workspace ai2/tulu-3-dev \
            --beaker-budget ai2/oe-adapt \
            --beaker-image Yizhongw03/oe_eval \
            --beaker-priority high \
            --gantry-args '{"env-secret": "OPENAI_API_KEY=Yizhongw03_OPENAI_API_KEY"}' \
            --cluster ai2/allennlp-cirrascale
    fi
done

# # if using beaker, we need to copy the results back to local
# if ! $RUN_LOCAL; then
#     declare -A BEAKER_DATASET_IDS=(
#         ["gsm8k"]="01J89XZP2N59Y3ARZB2GPYEH0K"
#         ["codex_humaneval"]="01J89XZPRZPJEFJH10A54V3FN4"
#         ["codex_humanevalplus"]="01J89XZQEJ14WJHDAHYH3ED1TM"
#         ["ifeval"]="01J89XZR3WY88KM30EPS6YQJ68"
#         ["alpaca_eval_v1"]="12137"
#         ["alpaca_eval_v2"]="12138"
#         ["popqa"]="01J89XZTYZRJG7AYKGPNRG47GQ"
#         ["bbh:cot::none"]="01J89Y3T85G25RHQT8DN6Y9GFV"
#         ["minerva_math::tulu"]="01J89Y6CZ0QDA60T7T7EM57NTQ"
#     )
#     mkdir -p ${OUTPUT_DIR}/${SUBDIR}
#     for TASK_NAME in "${TASKS[@]}"; do
#         BEAKER_DATASET_ID=${BEAKER_DATASET_IDS[$TASK_NAME]}
#         beaker dataset fetch -o ${OUTPUT_DIR}/${SUBDIR}/${TASK_NAME} ${BEAKER_DATASET_ID}
#     done
# fi

# # (optionally) upload the data to huggingface hub
# if $UPLOAD_TO_HF; then
#     HF_REPO_ID="ai2-adapt-dev/oe-eval-bon-candidates"
#     huggingface-cli upload --private --repo-type dataset ${HF_REPO_ID} "${OUTPUT_DIR}/${SUBDIR}/"
# fi
