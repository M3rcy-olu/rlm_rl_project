import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from tqdm.asyncio import tqdm as async_tqdm

import tinker
from rlm import RLM_REPL
from rlm_evaluator import RLMEvaluator


class RLMEvaluatorParallel(RLMEvaluator):
    """
    Parallel version of RLMEvaluator that runs evaluations concurrently.
    """

    async def __call__(self, sampling_client: tinker.SamplingClient, max_concurrency: int = 5) -> dict[str, float]:
        """
        Run custom evaluation on the given sampling client in parallel and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
            max_concurrency: Maximum number of concurrent evaluations
        Returns:
            Dictionary of metrics from inspect evaluation
        """
        # Setup debug directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = Path(f"./debug/rlm_eval_parallel_{timestamp}").resolve()
        debug_path.mkdir(parents=True, exist_ok=True)

        metrics = {}
        num_examples = len(self.dataset)
        
        print(f"Dataset length: {num_examples}")
        print(f"Dataset type: {type(self.dataset)}")
        
        # Shared state for aggregation
        aggregated_results = {
            "score": 0,
            "ctx_len_score": defaultdict(int),
            "ctx_len_total": defaultdict(int),
            "errors": [],
            "completed_count": 0
        }
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        print(f"Starting parallel evaluation with concurrency={max_concurrency}")
        print(f"Debug files will be saved to: {debug_path}")

        async def process_single_example(idx: int, datum: Any) -> None:
            print(f"[Task {idx}] Starting...")
            async with semaphore:
                print(f"[Task {idx}] Acquired semaphore")
                ctx_len = datum.get("context_len", 0)
                context = datum.get("context_window_text", "")
                query = datum.get("question", "")
                example_id = datum.get("id", idx)
                
                print(f"[Task {idx}] Creating RLM instance...")
                # Create a separate RLM instance for each task to ensure thread safety
                # and avoid state collisions.
                log_file = debug_path / f"rlm_{idx:04d}.log"
                rlm = RLM_REPL(
                    model="Qwen/Qwen3-8B",
                    sampling_client=sampling_client,
                    recursive_model="Qwen/Qwen3-8B",
                    enable_logging=True,  # Enable logging to write to log file
                    log_file=str(log_file),
                    max_iterations=5
                )
                print(f"[Task {idx}] RLM instance created")

                try:
                    print(f"[Task {idx}] Starting RLM completion...")
                    # Run the synchronous RLM completion in a separate thread
                    result = await asyncio.to_thread(
                        rlm.completion, 
                        context=context, 
                        query=query
                    )
                    print(f"[Task {idx}] RLM completion done")
                    
                    grade = self.grader_fn(datum, result)
                    print(f"[Task {idx}] Graded: {grade}")
                    
                    # Update aggregated results
                    # Note: Dict operations are atomic in Python for single items, but let's be safe with data structure updates
                    aggregated_results["score"] += grade
                    aggregated_results["ctx_len_score"][ctx_len] += grade
                    aggregated_results["ctx_len_total"][ctx_len] += 1
                    aggregated_results["completed_count"] += 1

                    # Save individual result immediately
                    print(f"[Task {idx}] Saving result...")
                    with open(debug_path / f"result_{idx:04d}.json", "w") as f:
                        json.dump({
                            "example_id": example_id,
                            "ctx_len": ctx_len,
                            "query": query[:200],
                            "correct_answer": datum["answer"],
                            "result": result if result else None,
                            "grade": grade,
                        }, f, indent=2)
                    print(f"[Task {idx}] Result saved successfully")
                        
                except Exception as e:
                    print(f"[Task {idx}] ERROR: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    error_msg = f"{type(e).__name__}: {e}"
                    aggregated_results["errors"].append((idx, error_msg))
                    aggregated_results["completed_count"] += 1
                    
                    with open(debug_path / f"error_{idx:04d}.log", "w") as f:
                        f.write(f"example_id = {example_id}\n")
                        f.write(f"ctx_len = {ctx_len}\n")
                        f.write(f"query = {query[:200]}...\n")
                        f.write(f"Exception: {error_msg}\n")
                        f.write(f"\nTraceback:\n{traceback.format_exc()}\n")
                    print(f"[Task {idx}] Error logged")
                finally:
                    print(f"[Task {idx}] Closing loggers...")
                    # Close loggers to ensure files are flushed and closed properly
                    rlm.close_loggers()
                    print(f"[Task {idx}] Task complete")

        # Create tasks for all examples
        print(f"Creating {num_examples} tasks...")
        tasks = [
            process_single_example(idx, datum) 
            for idx, datum in enumerate(self.dataset)
        ]
        print(f"Created {len(tasks)} tasks")
        
        # Run tasks with progress bar
        # We wrap the gather in a tqdm for progress tracking
        print("Starting task execution...")
        for f in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating (Parallel)"):
            await f
        print("All tasks completed")

        # Compile final metrics
        score = aggregated_results["score"]
        errors = aggregated_results["errors"]
        ctx_len_score = aggregated_results["ctx_len_score"]
        ctx_len_total = aggregated_results["ctx_len_total"]

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
                for idx, err in sorted(errors):
                    f.write(f"[{idx:04d}] {err}\n")
        
        print(f"\nResults: {score}/{num_examples} ({100*score/num_examples if num_examples > 0 else 0:.1f}%)")
        print(f"Errors: {len(errors)}")
        print(f"Debug files saved to: {debug_path}")
        
        return metrics

