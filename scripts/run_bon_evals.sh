#!/bin/bash
# set -x

REWARD_MODELS=(
    'checkpoints/hs2_0_human'
    'checkpoints/hs2_100_human'
    'checkpoints/hs2_optimal'
    'checkpoints/multipref_0_human'
    'checkpoints/multipref_100_human'
    'checkpoints/multipref_optimal'
)

TASKS=(
    'gsm8k'
    'codex_humaneval'
    'codex_humanevalplus'
    'ifeval'
    'popqa'
    'alpaca_eval_v1'
    'alpaca_eval_v2'
    'bbh'
)

for TASK in ${TASKS[@]}; do
    for REWARD_MODEL in ${REWARD_MODELS[@]}; do
        echo "Running BON eval for task ${TASK} with reward model ${REWARD_MODEL}"
        MODEL_NAME=${REWARD_MODEL#"checkpoints/"}
        OUTPUT_DIR="bon_evals/${TASK}/${MODEL_NAME}"
        python scripts/run_bon_oe.py --model ${REWARD_MODEL} --task ${TASK} --output_dir ${OUTPUT_DIR} --batch_size 16 --debug
    done
done
