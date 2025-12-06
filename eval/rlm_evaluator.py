import asyncio
import json
from typing import Any, Callable

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

from collections import defaultdict
from datasets import load_dataset
import jsonlines
import dateutil

from datetime import datetime
import os
import ast
import sys
import re
from pathlib import Path

from typing import Literal, Iterator, Optional


from rlm import RLM_REPL



class RLMEvaluator(SamplingClientEvaluator):
    """
        Evaluator that enables RLM inference.
    """

    def __init__(
        self,
        dataset: Any,
        grader_fn: Callable[[str, str], bool],
        model_name: str,
        renderer_name: str,
    ):
        """
        Initialize the RLMEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """
        self.dataset = dataset
        self.grader_fn = grader_fn

        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

    @staticmethod
    def get_dataset(
        split: str,
        dataset_type: Literal["synth", "real"] = "synth",
        subset: str | list[str] | None = None,
        max_context_len: int = 131072,
        min_context_len: int = 0,
        max_examples: int | None = None,
    ):
        """
        Load and filter the OOLONG dataset.
        
        Args:
            split: "validation" or "test"
            dataset_type: "synth" or "real"
            subset: Specific dataset(s) to include, or None for all
            max_context_len: Maximum context length (in tokens) to include
            min_context_len: Minimum context length (in tokens) to include
            max_examples: Maximum number of examples to return
        
        Returns:
            Iterator over filtered dataset examples
        """
        full_dataset = load_dataset(f"oolongbench/oolong-{dataset_type}")
        data = full_dataset[split]
        
        subset_set = None
        if subset is not None:
            if isinstance(subset, str):
                subset_set = {subset}
            else:
                subset_set = set(subset)
        
        data = data.filter(lambda x: x["context_len"] <= max_context_len)
        data = data.filter(lambda x: x["context_len"] >= min_context_len)
        
        if subset_set is not None:
            data = data.filter(lambda x: x["dataset"] in subset_set)
        
        if max_examples is not None:
            data = data.select(range(min(max_examples, len(data))))
        
        return data

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run custom evaluation on the given sampling client and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics from inspect evaluation
        """
        from tqdm import tqdm

        # Setup debug directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = Path(f"./debug/rlm_eval_{timestamp}")
        debug_path.mkdir(parents=True, exist_ok=True)

        metrics = {}
        num_examples = len(self.dataset)
        score = 0
        ctx_len_score = defaultdict(int)
        ctx_len_total = defaultdict(int)
        errors = []

        log_file = debug_path / f"rlm_{timestamp}.log"
        rlm = RLM_REPL(
                model="Qwen/Qwen3-8B",
                sampling_client=sampling_client,
                recursive_model="Qwen/Qwen3-8B",
                enable_logging=True,  # Enable logging to write to log file
                log_file=str(log_file),
                max_iterations=5
            )

        for idx, datum in enumerate(tqdm(self.dataset, desc="Evaluating")):
            ctx_len = datum.get("context_len", 0)
            ctx_len_total[ctx_len] += 1
            context = datum.get("context_window_text", "")
            query = datum.get("question", "")
            example_id = datum.get("id", idx)
            
            try:
                result = rlm.completion(context=context, query=query)
                grade = self.grader_fn(datum, result)
                score += grade
                ctx_len_score[ctx_len] += grade
                
                # Save result for this example
                with open(debug_path / f"result_{idx:04d}.json", "w") as f:
                    json.dump({
                        "example_id": example_id,
                        "ctx_len": ctx_len,
                        "query": query[:200],
                        "correct_answer": datum["answer"],
                        "result": result[:1000] if result else None,
                        "grade": grade,
                    }, f, indent=2)
                    
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                errors.append((idx, error_msg))
                with open(debug_path / f"error_{idx:04d}.log", "w") as f:
                    f.write(f"example_id = {example_id}\n")
                    f.write(f"ctx_len = {ctx_len}\n")
                    f.write(f"query = {query[:200]}...\n")
                    f.write(f"Exception: {error_msg}\n")

        # Close loggers after all evaluations complete
        rlm.close_loggers()
        
        metrics["overall_score"] = score
        metrics["overall_score_percent"] = score / num_examples if num_examples > 0 else 0
        metrics["num_examples"] = num_examples
        metrics["num_errors"] = len(errors)
        
        for ctx_len in ctx_len_total:
            metrics[f"{ctx_len}_score"] = ctx_len_score[ctx_len]
            metrics[f"{ctx_len}_total"] = ctx_len_total[ctx_len]
            metrics[f"{ctx_len}_accuracy"] = ctx_len_score[ctx_len] / ctx_len_total[ctx_len] if ctx_len_total[ctx_len] > 0 else 0.0
        
        # Save metrics summary
        with open(debug_path / "metrics_summary.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save error summary if any
        if errors:
            with open(debug_path / "errors_summary.log", "w") as f:
                for idx, err in errors:
                    f.write(f"[{idx:04d}] {err}\n")
        
        print(f"\nResults: {score}/{num_examples} ({100*score/num_examples if num_examples > 0 else 0:.1f}%)")
        print(f"Errors: {len(errors)}")
        print(f"Debug files saved to: {debug_path}")
        
        return metrics