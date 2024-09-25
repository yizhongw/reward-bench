import json
import os
import re
import glob
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task to process",
    )
    parser.add_argument(
        "--oe_eval_result_dir",
        type=str,
        required=True,
        help="Path to output directory of oe_eval. "
        "This is the directory that contains multiple runs of oe_eval for the specified task. "
        "Each run directory should contain requests and predictions."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed bon data",
    )
    return parser.parse_args()


def main():
    args = get_args()

    print(f"Processing {args.task_name}...")
    task_raw_requests, task_raw_predictions = [], []
    # We assume there are multiple run directories, each containing requests and predictions
    for run_dir in glob.glob(f"{args.oe_eval_result_dir}/*"):
        print(f"Processing {run_dir}...")
        for file in glob.glob(f"{run_dir}/*-requests.jsonl"):
            task_id = file.split("/")[-1].split("-requests.jsonl")[0]
            with open(file, "r") as f:
                requests = [json.loads(line) for line in f.readlines()]
                for request in requests:
                    # append task_id to doc_id to avoid duplicate doc_ids in different files
                    request["doc_id"] = f"{task_id}_doc_{request['doc_id']}"  
                task_raw_requests += requests
        for file in glob.glob(f"{run_dir}/*-predictions.jsonl"):
            task_id = file.split("/")[-1].split("-predictions.jsonl")[0]
            with open(file, "r") as f:
                predictions = [json.loads(line) for line in f.readlines()]
                for prediction in predictions:
                    # append task_id to doc_id to avoid duplicate doc_ids in different files
                    prediction["doc_id"] = f"{task_id}_doc_{prediction['doc_id']}"
                task_raw_predictions += predictions
        assert len(task_raw_requests) == len(task_raw_predictions), \
            f"Number of requests and predictions do not match for {args.task_name}, {run_dir}"
    
    requests_grouped_by_doc = {}
    for request in task_raw_requests:
        doc_id = request["doc_id"]
        if doc_id not in requests_grouped_by_doc:
            requests_grouped_by_doc[doc_id] = []
        requests_grouped_by_doc[doc_id].append(request)
        
    print(len(requests_grouped_by_doc))
    
    # Check that each document has 16 requests
    # assert all(len(requests_grouped_by_doc[doc_id]) == 16 for doc_id in requests_grouped_by_doc), \
        # f"Number of requests per document do not match for {args.task_name}, expected 16, got {len(requests_grouped_by_doc[doc_id])} for {doc_id}"
    for doc_id in requests_grouped_by_doc:
        if len(requests_grouped_by_doc[doc_id]) != 16:
            print(f"Number of requests per document do not match for {args.task_name}, expected 16, got {len(requests_grouped_by_doc[doc_id])} for {doc_id}")
            print(requests_grouped_by_doc[doc_id])
            exit()
    
    # Check that the requests are the same across runs
    for doc_id in requests_grouped_by_doc:
        request_ref = requests_grouped_by_doc[doc_id][0]
        for request in requests_grouped_by_doc[doc_id][1:]:
            assert len(request_ref["request"]["context"]["messages"]) == \
                len(request["request"]["context"]["messages"])
            assert all(m1["content"] == m2["content"] for m1, m2 in \
                zip(request_ref["request"]["context"]["messages"], \
                    request["request"]["context"]["messages"]))
            assert request["request"]["context"]["assistant_prefix"] == request_ref["request"]["context"]["assistant_prefix"]
    
    predictions_grouped_by_doc = {}
    for prediction in task_raw_predictions:
        doc_id = prediction["doc_id"]
        if doc_id not in predictions_grouped_by_doc:
            predictions_grouped_by_doc[doc_id] = []
        predictions_grouped_by_doc[doc_id].append(prediction)
    
    assert all(len(predictions_grouped_by_doc[doc_id]) == 16 for doc_id in predictions_grouped_by_doc), \
        f"Number of predictions per document do not match for {args.task_name}"
    
    bon_data = []
    for doc_id in requests_grouped_by_doc:
        bon_data += [
            {
                "args.task_name": args.task_name,
                "doc_id": doc_id,
                "request_messages": requests_grouped_by_doc[doc_id][i]["request"]["context"]["messages"],
                "request_assistant_prefix": requests_grouped_by_doc[doc_id][i]["request"]["context"]["assistant_prefix"],
                "continuation": predictions_grouped_by_doc[doc_id][i]["model_output"][0]["continuation"],
                "metrics": predictions_grouped_by_doc[doc_id][i]["metrics"],
                "generation_kwargs": requests_grouped_by_doc[doc_id][i]["request"]["generation_kwargs"]
            }
            for i in range(len(requests_grouped_by_doc[doc_id]))
        ]
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        for entry in bon_data:
            f.write(json.dumps(entry) + "\n")
        
        
if __name__ == "__main__":
    main()