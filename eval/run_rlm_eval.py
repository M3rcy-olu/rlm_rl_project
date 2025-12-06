import tinker
from tinker import types
from tinker_cookbook.model_info import get_recommended_renderer_name

from rlm_evaluator import RLMEvaluator
from rlm_evaluator_parallel import RLMEvaluatorParallel
from datasets import load_dataset
from typing import Literal, Iterator

from datasets import load_dataset
import jsonlines
import tiktoken
import dateutil

from datetime import datetime
import os
import ast
import sys
import re
from pathlib import Path
import asyncio


def synth_attempt_answer_parse(answer):
    parse_confidence = "low"
    if ":" not in answer:  # bad start
        if len(answer) < 20:  # it's short, return the whole thing
            return answer, parse_confidence
        else:
            return answer.split()[-1], parse_confidence
    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace(
        "*", ""
    )  # OpenAI models like bolding the answer


    candidate_answer = candidate_answer.replace(
        "[", ""
    ) 
    candidate_answer = candidate_answer.replace(
        "]", ""
    )  # Anthropic models like putting the answer in []
    parse_confidence = "med"
    if (
        "User:" in answer
        or "Answer:" in answer
        or "Date:" in answer
        or "Label" in answer
    ):
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"

    return candidate_answer, parse_confidence


def synth_process_response(datapoint, output):

    score = 0
    gold = (
        ast.literal_eval(datapoint["answer"])[0]
        if "datetime" not in datapoint["answer"]
        else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    )

    trimmed_output, parse_confidence =  synth_attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        score = 1
    elif str(trimmed_output) in ['more common', 'less common', 'same frequency']: # account for these being slightly different wordings
        if str(trimmed_output) in  str(gold):
            score = 1
    elif (
        datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC"
    ):  # partial credit for numbers
        try:
            trimmed_output = int(trimmed_output)
            gold = int(gold)
            score = 0.75 ** (abs(gold - trimmed_output))
        except Exception:
            parse_confidence = "low"  # didn't parse as a number, that's a bad sign
    elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        try:
            trimmed_output = dateutil.parser.parse(trimmed_output)
            score = trimmed_output == gold
        except Exception:
            parse_confidence = "low"  # didn't parse as a date, that's a bad sign


    # this_output = {
    #     "id": datapoint["id"],
    #     "context_window_id": datapoint["context_window_id"],
    #     "dataset": datapoint["dataset"],
    #     "model": model,
    #     "attempted_parse": str(trimmed_output),
    #     "parse_confidence": parse_confidence,
    #     "full_answer": output,
    #     "score": score,
    #     "answer": str(gold),
    # }

    return score


def main(
    model_name: str = "Qwen/Qwen3-8B",
    split: str = "validation",
    dataset_type: Literal["synth", "real"] = "synth",
    subset: str | list[str] | None = None,
    max_context_len: int = 131072,
    min_context_len: int = 0,
    max_examples: int | None = None,
    model_path: str | None = None,
    parallel: bool = False,
    concurrency: int = 5,
):
    dataset = RLMEvaluator.get_dataset(
        split=split,
        dataset_type=dataset_type,
        subset=subset,
        max_context_len=max_context_len,
        min_context_len=min_context_len,
        max_examples=max_examples
    )

    # Choose evaluator based on parallel flag
    EvaluatorClass = RLMEvaluatorParallel if parallel else RLMEvaluator
    
    evaluator = EvaluatorClass(
        dataset=dataset,
        grader_fn=synth_process_response,
        model_name=model_name,
        renderer_name=get_recommended_renderer_name(model_name),
    )

    service_client = tinker.ServiceClient()

    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)

    if parallel:
        result = asyncio.run(evaluator(sampling_client, max_concurrency=concurrency))
    else:
        result = asyncio.run(evaluator(sampling_client))

    print(result)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OOLONG evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--dataset-type", default="synth", choices=["synth", "real"])
    parser.add_argument("--subset", default=None, help="Comma-separated dataset subsets (e.g., trec_coarse,sample)")
    parser.add_argument("--max-context-len", type=int, default=4096)
    parser.add_argument("--min-context-len", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=1)
    parser.add_argument("--model-path", default=None, help="Tinker path to finetuned weights")
    parser.add_argument("--parallel", action="store_true", help="Run evaluations in parallel")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel evaluations (requires --parallel)")
    
    args = parser.parse_args()
    
    subset = args.subset.split(",") if args.subset else None
    
    metrics = main(
        model_name=args.model,
        split=args.split,
        dataset_type=args.dataset_type,
        subset=subset,
        max_context_len=args.max_context_len,
        min_context_len=args.min_context_len,
        max_examples=args.max_examples,
        model_path=args.model_path,
        parallel=args.parallel,
        concurrency=args.concurrency,
    )
    
    print("\nEvaluation complete!")

