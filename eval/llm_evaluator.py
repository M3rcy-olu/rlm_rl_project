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
from tqdm import tqdm

from typing import Literal, Iterator, Optional


from rlm import RLM_REPL



class LLMEvaluator(SamplingClientEvaluator):
    """
        Evaluator for LLM.
    """

    def __init__(
        self,
        dataset: Any,
        grader_fn: Callable[[str, str], bool],
        model_name: str,
        renderer_name: str,
    ):
        """
        Initialize the LLMEvaluator.
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
        max_context_len: int = 16384,
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

        metrics = {}

        num_examples = len(self.dataset)
        score = 0
        ctx_len_score = defaultdict(int)
        ctx_len_total = defaultdict(int)

        # Setup debug directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = Path(f"./debug/llm_eval_{timestamp}")
        debug_path.mkdir(parents=True, exist_ok=True)
        
        run_metadata = {
            "timestamp": timestamp,
            "num_examples": num_examples,
        }
        with open(debug_path / "run_metadata.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        # First pass: submit all sample requests and collect futures
        futures = []
        datum_list = []
        
        for datum in tqdm(self.dataset, desc="Submitting requests", unit="ex"):
            ctx_len = datum.get("context_len", 0)
            ctx_len_total[ctx_len] += 1
            # context = datum.get("context_window_text", "")
            context = datum.get("context_window_text", "")
            query = datum.get("question", "")

            sampling_params = types.SamplingParams(
                max_tokens=16384,
                temperature=0.0,
                top_p=1.0,
                stop=self.renderer.get_stop_sequences(),
            )

            model_input: types.ModelInput = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=context + query)]
            )

            future = sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
            futures.append(future)
            datum_list.append(datum)

        # Second pass: collect results and grade
        pbar = tqdm(enumerate(zip(futures, datum_list)), total=len(futures), desc="Evaluating", unit="ex")
        for idx, (future, datum) in pbar:
            ctx_len = datum.get("context_len", 0)
            query = datum.get("question", "")
            error_msg = None
            response_text = ""
            tokens = []
            
            try:
                r: types.SampleResponse = future.result(timeout=600)
                tokens: list[int] = r.sequences[0].tokens
                response: renderers.Message = self.renderer.parse_response(tokens)[0]
                response_text = response.get("content", str(response))
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                with open("llm_eval_errors.log", "a") as f:
                    f.write(f"[{datetime.now().isoformat()}] Error processing datum\n")
                    f.write(f"  example_id = {datum.get('id')} ctx_len={ctx_len}, query={query[:100]}...\n")
                    f.write(f"  Exception: {error_msg}\n\n")
                
                # Save debug info for error case
                debug_info = {
                    "example_id": datum.get("id", f"example_{idx}"),
                    "context_window_id": datum.get("context_window_id", ""),
                    "dataset": datum.get("dataset", "unknown"),
                    "question": query,
                    "gold_answer": datum.get("answer", ""),
                    "model_output": "",
                    "error": error_msg,
                    "score": 0,
                    "context_len": ctx_len,
                    "response_tokens": 0,
                }
                safe_id = str(datum.get("id", f"example_{idx}")).replace("/", "_")[:100]
                with open(debug_path / f"{idx:05d}_{safe_id}.json", "w") as f:
                    json.dump(debug_info, f, indent=2)
                continue

            grade = self.grader_fn(datum, response_text)
            score += grade
            ctx_len_score[ctx_len] += grade
            
            # Update progress bar with running accuracy
            current_acc = score / (idx + 1)
            pbar.set_postfix({"acc": f"{current_acc:.3f}"})

            # Save debug info for each example
            debug_info = {
                "example_id": datum.get("id", f"example_{idx}"),
                "context_window_id": datum.get("context_window_id", ""),
                "dataset": datum.get("dataset", "unknown"),
                "question": query,
                "gold_answer": datum.get("answer", ""),
                "model_output": response_text,
                "error": error_msg,
                "score": grade,
                "context_len": ctx_len,
                "response_tokens": len(tokens),
            }
            safe_id = str(datum.get("id", f"example_{idx}")).replace("/", "_")[:100]
            with open(debug_path / f"{idx:05d}_{safe_id}.json", "w") as f:
                json.dump(debug_info, f, indent=2)

        metrics["overall_score"] = score
        metrics["overall_score_percent"] = score / num_examples
        
        for ctx_len in ctx_len_total:
            metrics[f"{ctx_len}_score"] = ctx_len_score[ctx_len]
            metrics[f"{ctx_len}_total"] = ctx_len_total[ctx_len]
            metrics[f"{ctx_len}_accuracy"] = ctx_len_score[ctx_len] / ctx_len_total[ctx_len] if ctx_len_total[ctx_len] > 0 else 0.0
        
        # Save metrics summary
        with open(debug_path / "metrics_summary.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Debug files saved to: {debug_path}")
        
        return metrics