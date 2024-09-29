# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runs best of n (BoN) ranking
# TODO: implement this for DPO models

import argparse
import logging
import os
import sys
import json
from typing import List

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, pipeline
from datasets import load_dataset, Dataset

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    save_to_hub,
)

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bon_candidates_hf_repo", 
        type=str, 
        default="ai2-adapt-dev/oe-eval-bon-candidates", 
        help="path to huggingface dataset repo containing bon candidates"
    )
    parser.add_argument(
        "--bon_candidates_subdir",
        type=str,
        default="tulu-2-13b-n-16",
        help="If specified, will load the BON candidates from this subdirectory in the hub."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save results to."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "alpaca_eval_v1", 
            "alpaca_eval_v2", 
            "bbh", 
            "codex_humaneval", 
            "codex_humanevalplus", 
            "ifeval", 
            "gsm8k", 
            "popqa"
        ],
        help="Task to run best of n evaluation on. This must be the name of the subset in the bon_candidates_hf_repo."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="path to model"
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        help="path to non-matching tokenizer to model"
    )
    parser.add_argument(
        "--chat_template_jinja_file", 
        type=str,
        help="By default, the tokenizer chat template is used to format the messages."
        "If this is specified, this jinja template in this file will be used instead."
    )
    parser.add_argument(
        "--trust_remote_code", 
        action="store_true", 
        help="directly load model instead of pipeline"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="batch size for inference"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="run on common preference sets instead of our custom eval set"
    )
    args = parser.parse_args()
    return args



def load_bon_dataset(
    hf_repo: str,
    task_subdir: str,
    tokenizer: PreTrainedTokenizer = None,
    chat_template_jinja_file: str = None,
    logger: logging.Logger = None,
    remove_columns: List[str] = None,
):
    """
    Loads the BON candidates dataset.
    """
    logger.info(f"Loading dataset from {hf_repo} with subdirectory {task_subdir}")
    dataset = load_dataset(hf_repo, data_files=os.path.join(task_subdir, "bon_candidates.jsonl"))["train"]
    # features: 
    # ['args.task_name', 'doc_id', 'request_messages', 'request_assistant_prefix', 'continuation', 
    # 'metrics', 'generation_kwargs']
    
    if chat_template_jinja_file:
        chat_template = open(chat_template_jinja_file, "r").read()
        tokenizer.chat_template = chat_template

    def reformat_conversatoin(example):
        """
        Reformat the conversation to add the request_assistant_prefix and continuation.
        """
        messages = example["request_messages"]
        assert messages[-1]["role"] == "user", "Last message in `request_messages` must be from user"
        messages.append({
            "role": "assistant",
            "content": example["request_assistant_prefix"] + example["continuation"]
        })
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return example
    
    dataset = dataset.map(reformat_conversatoin, batched=False, num_proc=8)
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)
    return dataset


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if args.model in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    quantized = config["quantized"]  # only Starling isn't quantized for now
    custom_dialogue = config["custom_dialogue"]
    _ = config["model_type"]  # todo will be needed to add PairRM and SteamSHP
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token}}"
        "{% endif %}"
        "{% if message['role'] == 'assistant' and not loop.last %}"
        "{{ '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    )
    # this is critical as tulu template as an <eos> token at the end of the conversation
    # and we need to use a separate pad toekn to differentiate between the eos and pad tokens
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    dataset = load_bon_dataset(
        hf_repo=args.bon_candidates_hf_repo,
        task_subdir=os.path.join(args.bon_candidates_subdir, args.task) if args.bon_candidates_subdir else args.task,
        tokenizer=tokenizer,
        chat_template_jinja_file=args.chat_template_jinja_file,
        logger=logger,
        remove_columns=None,
    )
    
    metrics = dataset["metrics"]
    dataset = dataset.remove_columns(["metrics"])

    # debug: use only 10 records
    if args.debug:
        dataset = dataset.select(range(16*50))
        metrics = metrics[:16*50]
        
    # print(dataset[0])

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": 2048,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {"device_map": {"": current_device}}

    model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    model.resize_token_embeddings(len(tokenizer))
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    if pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        # this setup can be optimized slightly with one pipeline call
        # prepare for inference
        reward_pipe = accelerator.prepare(reward_pipe)

        results = reward_pipe(dataset["text"], **reward_pipeline_kwargs)

        # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        scores = [r["score"] for r in results]

    ############################
    # Run inference [2/2] custom pipelines
    ############################
    else:
        logger.info("*** Running dataloader to collect results ***")
        # TODO make more custom pipelines work with pre-tokenized data
        from torch.utils.data.dataloader import default_collate

        # for PairRM, hmm, will move all of this later
        def custom_collate_fn(batch):
            # check if ['text_chosen'] is in first batch element
            # Check if the first element of the batch is a dictionary
            if isinstance(batch[0]["text"][0], dict):
                return batch  # Return the batch as-is if it's a list of dicts
            else:
                return default_collate(batch)  # Use the default collate behavior otherwise

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn,  # if not args.pref_sets else None,
            shuffle=False,
            drop_last=False,
        )

        model = accelerator.prepare(reward_pipe.model)
        reward_pipe.model = model

        scores = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if "PairRM" in args.model or "SteamSHP" in args.model:
                raise NotImplementedError("PairRM and SteamSHP are not yet supported for batched inference")
            else:
                rewards = reward_pipe(batch["text"], **reward_pipeline_kwargs)

                # for each item in batch, record 1 if chosen > rejected
                # extra score from dict within batched results (e.g. logits)
                # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
                if isinstance(rewards[0], dict):
                    scores_batch = [result["score"] for result in rewards]
                # for classes that directly output scores (custom code)
                else:
                    scores_batch = rewards.cpu().numpy().tolist()

                scores.extend(scores_batch)

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("rm_score", scores)
    out_dataset = out_dataset.add_column("metrics", metrics)

    # select the top scored instance per doc_id
    top_scored_completions = {}
    for i, row in enumerate(out_dataset):
        doc_id = row["doc_id"]
        if doc_id not in top_scored_completions or row["rm_score"] > top_scored_completions[doc_id]["rm_score"]:
            top_scored_completions[doc_id] = row

    # convert to dataset
    top_scored_dataset = Dataset.from_list(
        [top_scored_completions[doc_id] for doc_id in sorted(top_scored_completions.keys())]
    )

    # average the metrics across the top scored completions
    metrics = {}
    
    if args.task in ["gsm8k", "codex_humaneval", "codex_humanevalplus", "popqa", "bbh"]:
        for metric in top_scored_dataset["metrics"][0]:
            metrics[metric] = sum([row["metrics"][metric] for row in top_scored_dataset]) / len(top_scored_dataset)
    elif args.task == "ifeval":
        for metric in ["prompt_level_strict_acc", "inst_level_strict_acc", "prompt_level_loose_acc", "inst_level_loose_acc"]:
            metrics[metric] = sum([row["metrics"][metric] for row in top_scored_dataset]) / len(top_scored_dataset)
    elif args.task == "alpaca_eval_v1":
        # in alpaca eval v1, 1 means the model output loses to the reference output, 2 means the model output wins, 1.5 is a draw
        preferences = [row["metrics"]["preference"]-1 for row in top_scored_dataset if row["metrics"]["preference"] is not None]
        metrics["win_rate"] = sum(preferences) / len(preferences)
    elif args.task == "alpaca_eval_v2":
        # alpaca eval v2 uses a regression metric
        # need to double check again
        # logger.warning("alpaca eval v2 metrics are not well supported, please take them with a grain of salt")
        preferences = [row["metrics"]["preference"] for row in top_scored_dataset if row["metrics"]["preference"] is not None]
        metrics["average_preference"] = sum(preferences) / len(preferences)
    logger.info(f"Metrics: {metrics}")

    # save the results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(args.output_dir, f"bon_candidates_scores.jsonl"), "w") as f:
        for row in out_dataset:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(args.output_dir, f"bon_selections.jsonl"), "w") as f:
        for row in top_scored_dataset:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
