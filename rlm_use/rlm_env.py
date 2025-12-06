import asyncio
import logging
import os
import random
import re
import string
from functools import partial, reduce
from pathlib import Path
from typing import Literal, Sequence, TypedDict, cast


#RLM Imports
from rlm.rlm_repl import RLM_REPL
from rlm.repl import *
from rlm.utils import *
from rlm.utils.utils import check_for_final_answer

import chz
import pandas as pd
import tinker
from huggingface_hub import hf_hub_download
from datasets import load_dataset 
from tinker_cookbook import renderers
from tinker_cookbook.renderers import Message, ToolCall
from tinker_cookbook.completers import StopCondition
from .tools_m import RLMToolClient
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
# from tinker_cookbook.recipes.train_m import 


logger = logging.getLogger(__name__)

_CONNECTION_SEMAPHORE = asyncio.Semaphore(128)

RLM_TOOL_SYSTEM_PROMPT = """You are an assistant that can execute Python code in a REPL environment to analyze context and answer questions.

Tool calling: Execute tools by wrapping calls in <tool_call>...</tool_call> tags.

The execute_code tool you are given has the following schema:
```
{
    "name": "execute_code",
    "title": "REPL Code Execution",
    "description": "Executes Python code in a REPL environment with access to context, llm_query function, and other REPL capabilities. Use this to analyze context, query sub-LLMs, and build up your answer.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the REPL environment. The REPL has access to: context variable, llm_query() function, and standard Python built-ins. Code should be written as if it's being executed in a Python REPL."
            }
        },
        "required": ["code"],
    },
    "outputSchema": {
        "type": "string",
        "description": "Execution results including stdout, stderr, and available variables"
    },
}
```

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!
When you want to execute Python code, call the tool like this:
<tool_call>{"name": "execute_code", "args": {"code": "your_python_code_here"}}</tool_call>


**Example 1: Exploring the context**
<tool_call>{"name": "execute_code", "args": {"code": "print(type(context))\nprint(len(context) if isinstance(context, str) else 'Context is not a string')"}}</tool_call>

**Example 2: Chunking and querying with llm_query**
<tool_call>{"name": "execute_code", "args": {"code": "chunk = context[:10000] if isinstance(context, str) else str(context)[:10000]\nanswer = llm_query(f\"What is the key information in this chunk? {chunk}\")\nprint(answer)"}}</tool_call>

**Example 3: Building up an answer through multiple queries**
<tool_call>{"name": "execute_code", "args": {"code": "import re\nsections = re.split(r'### (.+)', context) if isinstance(context, str) else []\nbuffers = []\nfor i in range(1, len(sections), 2):\n    header = sections[i]\n    info = sections[i+1]\n    summary = llm_query(f\"Summarize this {header} section: {info}\")\n    buffers.append(f\"{header}: {summary}\")\nfinal_answer = llm_query(f\"Based on these summaries, answer the original query. Summaries:\\n\" + \"\\n\".join(buffers))\nprint(final_answer)"}}</tool_call>

**Important notes:**
- You will only see truncated outputs from the REPL environment, so use `llm_query()` on variables you want to analyze in detail
- The `llm_query()` function is powerful and can handle ~500K characters, so don't be afraid to put a lot of context into sub-LLM calls
- Make sure to explicitly look through the entire context before answering your query
- Use variables as buffers to build up your final answer across multiple tool calls

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""


def normalize_answer(s: str) -> str:
    """Normalize answer by lowercasing, removing punctuation, articles, and fixing whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    # Apply transformations in order using reduce
    transformations = [lower, remove_punc, remove_articles, white_space_fix]
    return reduce(lambda text, func: func(text), transformations, s)


class RLMEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: list[str],
        context: str | dict | list,
        rlm_tool_client: RLMToolClient,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32 * 1024,
        timeout: float = 1.0,
        max_num_calls: int = 4,
        max_iterations: int = 20,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem: str = problem
        self.answer: list[str] = answer
        self.context = context

        self.rlm_tool_client = rlm_tool_client
        self.timeout: float = timeout
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.past_messages: list[renderers.Message] = convo_prefix.copy() if convo_prefix else []
        self.current_num_calls: int = 0
        self.max_num_calls: int = max_num_calls
        self.max_iterations: int = max_iterations
        self.current_iteration: int = 0

        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json = context_data, 
            context_str = context_str,
            recursive_model = "Qwen/Qwen3-8B"
        )
    

    async def initial_observation(self) -> tuple[Observation, StopCondition]:

        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        self.past_messages = convo.copy()
        self.current_iteration = 0
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    def get_question(self) -> str:
        return self.problem

    def _extract_answer(self, sample_str: str) -> str | None:
        if "Answer:" not in sample_str:
            return None
        message_pars = sample_str.split("Answer:")
        if len(message_pars) != 2:
            return None
        return message_pars[1].strip()

    def check_answer(self, sample_str: str) -> bool:
        model_answer = self._extract_answer(sample_str) #when expecting to be in Answer: "<answer>" format
        # model_answer = sample_str
        logger.info("Model Answer Type")
        print(type(model_answer))
        print(model_answer)
        if model_answer is None or len(self.answer) == 0:
            return False

        for gold_answer in self.answer:
            if normalize_answer(model_answer) == normalize_answer(gold_answer):
                return True
        return False

    def check_format(self, sample_str: str) -> bool:
        return self._extract_answer(sample_str) is not None

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        return " OR ".join(self.answer) if self.answer else "N/A"

    def _check_for_final_answer_in_response(self, response_content: str) -> str | None:
        """
        Check if the response contains a FINAL() or FINAL_VAR() call.
        Similar to RLM_REPL's check_for_final_answer logic.
        """
        
        return check_for_final_answer(response_content, self.repl_env, logger)

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        logger.info("returned model output")
        self.past_messages.append(message)

        failure_result = StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
            )

        # Check iteration limit (like RLM_REPL's max_iterations)
        self.current_iteration += 1
        if self.current_iteration > self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            return failure_result
        if "tool_calls" in message:
            logger.info("Found Tool call in model output")
            logger.info("Model output:")
            # print(message)
            # if message["tool_calls"][0]["name"] == "execute_code":
            if type(message["tool_calls"][0]) == ToolCall:

                self.current_num_calls += 1
                if self.current_num_calls > self.max_num_calls:
                    return failure_result
                # NOTE(tianyi): seems wasteful: we should share the client somehow

                try:
                    tool_return_message = await self.rlm_tool_client.invoke(
                        tool_call = message["tool_calls"][0], 
                        rlm = self.repl_env
                    )
                    logger.info(f"Tool call executed, tool_return_response: {tool_return_message}")
                    self.past_messages.extend(tool_return_message)
                    # Check if the tool response contains a final answer
                    # (The agent might have used FINAL_VAR in the code)
                    tool_content = tool_return_message[0]["content"] if tool_return_message else ""
                    final_answer = self._check_for_final_answer_in_response(tool_content)

                    if final_answer:
                        # Agent found final answer via FINAL_VAR in code execution
                        # Check if it's correct
                        correct = self.check_answer(final_answer)
                        return StepResult(
                            reward=float(correct),  # Reward 1.0 if correct, 0.0 otherwise
                            episode_done=True,
                            next_observation=tinker.ModelInput.empty(),
                            next_stop_condition=self.stop_condition,
                            metrics={
                                "format": 1.0,
                                "correct": float(correct),
                                "found_via_final_var": 1.0,
                            },
                        )
                except Exception as e:
                    logger.error(f"Error calling search tool: {repr(e)}")
                    return failure_result
                
                # Continue the episode - agent can make more tool calls
                next_observation = self.renderer.build_generation_prompt(self.past_messages)
                if next_observation.length > self.max_trajectory_tokens:
                    return failure_result
                return StepResult(
                    reward=0.0,
                    episode_done=False,
                    next_observation=next_observation,
                    next_stop_condition=self.stop_condition,
                )
            else:
                return failure_result
        else:
            # Agent provided a text response (no tool call)
            response_content = message.get("content", "")
            
            # Check for FINAL() or FINAL_VAR() in the response
            final_answer = self._check_for_final_answer_in_response(response_content)
            
            if final_answer:
                # Agent used FINAL() or FINAL_VAR() in response
                correct = self.check_answer(final_answer)
                return StepResult(
                    reward=float(correct),
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={
                        "format": 1.0,
                        "correct": float(correct),
                        "found_via_final": 1.0,
                    },
                )

            # Check for "Answer:" format (traditional format)
            correct_format = float(parse_success) and float(self.check_format(response_content))
            correct_answer = float(self.check_answer(response_content))
            total_reward = self.format_coef * (correct_format - 1) + correct_answer
            
            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format": correct_format,
                    "correct": correct_answer,
                },
            )

            # return failure_result

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "system",
                "content": RLM_TOOL_SYSTEM_PROMPT,
            },
        ]

class SearchR1Datum(TypedDict):
    question: str
    answer: list[str]
    context: str



# def process_single_row(row_series: pd.Series) -> SearchR1Datum:
    # """
    # Process a single row of data for SearchR1-like format.

    # Args:
    #     row: DataFrame row containing the original data
    #     current_split_name: Name of the current split (train/test)
    #     row_index: Index of the row in the DataFrame

    # Returns:
    #     pd.Series: Processed row data in the required format
    # """
    # import numpy as np

    # row = row_series.to_dict()
    # question: str = row.get("question", "")

    # # Extract ground truth from reward_model or fallback to golden_answers
    # reward_model_data = row.get("reward_model")
    # if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
    #     ground_truth = reward_model_data.get("ground_truth")
    # else:
    #     ground_truth = row.get("golden_answers", [])

    # # NOTE(tianyi)
    # # I hate datasets with mixed types but it is what it is.
    # if isinstance(ground_truth, dict):
    #     ground_truth = ground_truth["target"]
    # if isinstance(ground_truth, np.ndarray):
    #     ground_truth = ground_truth.tolist()

    # assert isinstance(ground_truth, list)
    # for item in ground_truth:
    #     assert isinstance(item, str)
    # ground_truth = cast(list[str], ground_truth)
    # return {
    #     "question": question,
    #     "answer": ground_truth,
    #     "context": row["context_window_text"]
    # }


def download_search_r1_dataset(split: Literal["validation", "test"], subset: str, max_context_len, min_context_len, max_examples: int | None = None) -> list[SearchR1Datum]:
    # hf_repo_id: str = "PeterJinGo/nq_hotpotqa_train"
    # parquet_filename: str = f"{split}.parquet"
    # # TODO(tianyi): make download dir configurable for release
    # user = os.getenv("USER", "unknown")
    # assert user is not None
    # tmp_download_dir = Path("/tmp") / user / "data" / hf_repo_id / split
    # tmp_download_dir.mkdir(parents=True, exist_ok=True)

    # local_parquet_filepath = hf_hub_download(
    #     repo_id=hf_repo_id,
    #     filename=parquet_filename,
    #     repo_type="dataset",
    #     local_dir=tmp_download_dir,
    #     local_dir_use_symlinks=False,
    # )

    # df_raw = pd.read_parquet(local_parquet_filepath)

    # return df_raw.apply(process_single_row, axis=1).tolist()

#OOLONG Dataset
    #return as a list of dictionaries.
    dataset = load_dataset("oolongbench/oolong-synth", streaming=False)
    data = dataset[split]

    subset_set = None
    if subset is not None:
        if isinstance(subset, str):
            subset_set = {subset}
        else: 
            subset_set = set[str](subset)

    data = data.filter(lambda x: x["context_len"] <= max_context_len)
    data = data.filter(lambda x: x["context_len"] >= min_context_len)

    if subset_set is not None:
        data = data.filter(lambda x: x["dataset"] in subset_set)

    if max_examples is not None: 
        data = data.select(range(min(max_examples, len(data))))
    
    # print(type(data))
    data = data.to_list() 
    # print(type(data), type(data[0]))
    return data

class SearchR1Dataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        # tool args
        rlm_tool_client: RLMToolClient,
        # optional args
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        split: Literal["validation", "test"] = "validation",
        subset: Literal["spam", "trec_course"] = "spam",
        max_context_len: int = 16_000,
        min_context_len: int = 128,
        max_examples: int | None = None,
        max_trajectory_tokens: int = 32 * 1024,
    ):
        self.batch_size: int = batch_size
        self.group_size: int = group_size
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.renderer: renderers.Renderer = renderer
        self.convo_prefix: list[renderers.Message] | None = convo_prefix
        self.rlm_tool_client: RLMToolClient = rlm_tool_client
        self.seed: int = seed
        self.split: Literal["validation", "test"] = split
        self.ds: list[SearchR1Datum] = download_search_r1_dataset(split, subset, max_context_len, min_context_len, max_examples)
        # shuffle with seed
        rng = random.Random(self.seed)
        rng.shuffle(self.ds)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            self._make_env_group_builder(row, self.group_size)
            for row in self.ds[index * self.batch_size : (index + 1) * self.batch_size]
        ]

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size

    def _make_env_group_builder(self, row: SearchR1Datum, group_size: int) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                RLMEnv,
                row["question"],
                row["answer"],
                row["context_window_text"], 
                self.rlm_tool_client,
                self.renderer,
                convo_prefix=self.convo_prefix,
                max_trajectory_tokens=self.max_trajectory_tokens,
            ),
            num_envs=group_size,
        )


@chz.chz
class SearchR1DatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    # rlm_tool_config: RLMToolClientConfig
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    max_eval_size: int = 1024
    max_trajectory_tokens: int = 32 * 1024

    async def __call__(self) -> tuple[SearchR1Dataset, None]:
        if self.convo_prefix == "standard":
            convo_prefix = RLMEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        rlm_tool_client = RLMToolClient()

        train_dataset = SearchR1Dataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            rlm_tool_client=rlm_tool_client,
            convo_prefix=convo_prefix,
            split="validation", 
            seed=self.seed,
            max_trajectory_tokens=self.max_trajectory_tokens,
        )
        return (train_dataset, None)
