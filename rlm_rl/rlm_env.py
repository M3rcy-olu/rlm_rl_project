import asyncio
import logging
import os
import random
import re
import string
import ast
from functools import partial, reduce
from pathlib import Path
from typing import Literal, Sequence, TypedDict, cast, Any

import chz
import pandas as pd
import tinker
from huggingface_hub import hf_hub_download
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
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

from rlm.utils import prompts as rlm_prompts
import rlm.utils.utils as rlm_utils
from rlm.repl import REPLEnv
from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

from datasets import load_dataset

logger = logging.getLogger(__name__)

class RLMEnv(ProblemEnv):
    def __init__(
        self,
        context: str,
        query: str,
        answer: list[str],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        max_trajectory_tokens: int = 32768,
        max_iterations: int = 10,
        timeout: float = 1.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.context: str = context
        self.query: str = query
        self.answer: list[str] = answer
        self.timeout: float = timeout
        self.max_trajectory_tokens: int = max_trajectory_tokens
        self.past_messages: list[renderers.Message] = convo_prefix.copy() if convo_prefix else []
        self.max_iterations: int = max_iterations
        self.current_iteration: int = 0
        
        context_data, context_str = rlm_utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data, 
            context_str=context_str, 
            recursive_model="Qwen/Qwen3-8B",
        )

        self.logger = ColorfulLogger(enabled=False)
        self.repl_env_logger = REPLEnvLogger(enabled=False)
        self.USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original query: \"{query}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:""" 


    def next_action_prompt(self, query: str, iteration: int = 0) -> dict[str, str]:
        if iteration == self.max_iterations - 1:
            return {"role": "user", "content": "Based on all the information you have, provide a final answer to the user's query. IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options: 1. Use FINAL(your final answer here) to provide the answer directly 2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output"}
        else:
            return {"role": "user", "content": "The history before is your previous interactions with the REPL environment. " + self.USER_PROMPT.format(query=query)}

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        safeguard = "You have not interacted with the REPL environment or seen your context yet. Your next action should be to look through, don't just provide a final answer yet.\n\n"
        convo = self.convo_prefix + [
            {"role": "user", "content": safeguard + self.USER_PROMPT.format(query=self.query)},
        ]
        self.past_messages = convo.copy()
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    def get_query(self) -> str:
        return self.query

    def _extract_answer(self, sample_str: str) -> str | None:
        if ":" not in sample_str:  # bad start
            if len(sample_str) < 20:  # it's short, return the whole thing
                return sample_str
            else:
                return sample_str.split()[-1]
        candidate_answer = sample_str.split(":")[-1].strip()
        candidate_answer = candidate_answer.replace(
            "*", ""
        ) 
        candidate_answer = candidate_answer.replace(
            "[", ""
        ) 
        candidate_answer = candidate_answer.replace(
            "]", ""
        )

        if "more common" in candidate_answer:
            candidate_answer = "more common"
        elif "less common" in candidate_answer:
            candidate_answer = "less common"
        elif "same frequency" in candidate_answer:
            candidate_answer = "same frequency"

        return candidate_answer

    def check_answer(self, answer, output):
        score = 0

        try:
            gold = ast.literal_eval(answer)[0]
        except Exception:
            gold = answer

        # trimmed_output = synth_attempt_answer_parse(output)
        trimmed_output = output
        
        if str(trimmed_output) == str(gold):
            score = 1
        elif str(trimmed_output) in ['more common', 'less common', 'same frequency']: # account for these being slightly different wordings
            if str(trimmed_output) in  str(gold):
                score = 1
        
        return score

        # elif (
        #     datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC"
        # ):  # partial credit for numbers
        #     try:
        #         trimmed_output = int(trimmed_output)
        #         gold = int(gold)
        #         score = 0.75 ** (abs(gold - trimmed_output))
        #     except Exception:
        #         parse_confidence = "low"  # didn't parse as a number, that's a bad sign
        # elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        #     try:
        #         trimmed_output = dateutil.parser.parse(trimmed_output)
        #         score = trimmed_output == gold
        #     except Exception:
        #         parse_confidence = "low"  # didn't parse as a date, that's a bad sign

    def standard_prefix() -> list[renderers.Message]:
        return [
            {"role": "system", "content": rlm_prompts.REPL_SYSTEM_PROMPT},
        ]

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)

        self.past_messages.append(message)

        failure_result = StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
        )
            
        self.current_iteration += 1
        response = message["content"]
        code_blocks = rlm_utils.find_code_blocks(response)

        # Process code execution or add assistant message
        try:
            if code_blocks:
                self.past_messages = rlm_utils.process_code_execution(
                    response, self.past_messages, self.repl_env,
                    self.repl_env_logger, self.logger
                )
            # else:
            #     assistant_message = {"role": "assistant", "content": "You responded with:\n" + response}
            #     self.past_messages.append(assistant_message)
        except Exception as e:
            logger.error(f"Error processing code execution: {e}")
            return failure_result
        
        # Check for final answer
        final_answer = rlm_utils.check_for_final_answer(
            response, self.repl_env, self.logger,
        )

        if final_answer:
            potential_answer = self._extract_answer(final_answer)
            score = self.check_answer(self.answer, potential_answer)

            # Log full trajectory
            logtree.log_formatter(ConversationFormatter(self.past_messages))
            logtree.log_text(f"Final Answer: {potential_answer}")
            logtree.log_text(f"Correct Answer: {self.answer}")
            logtree.log_text(f"Score: {score}")

            logger.info(f"Final Answer: {potential_answer}")
            logger.info(f"Correct Answer: {self.answer}")
            logger.info(f"Score: {score}")
            logger.info(f"Num Iterations: {self.current_iteration}")

            return StepResult(
                reward = score,
                episode_done = True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "score": score,
                    "num_iterations": self.current_iteration,
                }
            )
        else:
            self.past_messages.append(self.next_action_prompt(self.query, iteration=self.current_iteration))
            next_observation = self.renderer.build_generation_prompt(self.past_messages)

            if next_observation.length > self.max_trajectory_tokens or  self.current_iteration >= self.max_iterations:
                logtree.log_formatter(ConversationFormatter(self.past_messages))
                logtree.log_text("Max iterations or tokens reached.", div_class="lt-exc")
                return failure_result

            return StepResult(
                reward = 0.0,
                episode_done=False,
                next_observation=next_observation,
                next_stop_condition=self.stop_condition,
            )
            
    def get_question(self) -> str:
        return self.query

    def check_format(self, sample_str: str) -> bool:
        return True

    def get_reference_answer(self) -> str:
        return str(self.answer)
            
        
class RLMDataset(RLDataset):
    """RLM DATASET"""
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,

        # optional args
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["validation", "test"] = "validation",
        max_trajectory_tokens: int = 32768,
        max_iterations: int = 10,
        timeout: float = 1.0,
        seed: int = 0
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.split = split
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.ds = load_dataset("oolongbench/oolong-synth", split="validation")
        self.seed = seed
        if split == "validation":
            self.ds = self.ds.shuffle(seed=seed)
            self.ds = self.ds.filter(lambda x: x["dataset"] == "spam")

            # REMOVE LATER
            self.ds = self.ds.select(range(10))
        elif split == "test":
            self.ds = self.ds.filter(lambda x: x["dataset"] == "trec_coarse")

            # REMOVE LATER
            self.ds = self.ds.select(range(10))




    
    def __len__(self) -> int:
        return (len(self.ds) + self.batch_size - 1) // self.batch_size
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.ds))

        if start >= end:
            raise IndexError("Incorrect batch index for DeepcoderDataset")

        # Basically calls the _make_env_group_builder method for each datum in the batch
        # This essentially creates a "group" of environments for each example.
        # For example, if group_size is 4, then it will create 4 environments for each example.
        # So we can think of our RL run as:
        #   total_overall_steps = dataset // batch_size
        #   where every step we carry out group_size trajectories
        #   and each trajectory has at most max_iterations steps

        builders: list[EnvGroupBuilder] = [] # arranges the group and the env for each example

        for example in self.ds.select(range(start, end)):
            builder = self._make_env_group_builder(cast(dict[str, Any], example), self.group_size)
            if builder is not None:
                builders.append(builder)

        return builders

    def _make_env_group_builder(self, datum, group_size: int ) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                RLMEnv,
                context=datum.get("context_window_text"),
                query=datum.get("question"),
                answer=datum.get("answer"),
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_iterations=self.max_iterations,
                timeout=self.timeout,
            ),
            num_envs=group_size,
            dataset_name="oolongbench/oolong-synth",
        )

@chz.chz
class RLMDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    convo_prefix: list[renderers.Message] | None = None
    max_trajectory_tokens: int = 32768
    max_iterations: int = 10
    timeout: float = 1.0
    seed: int = 0

    async def __call__(self) -> tuple[RLMDataset, None]:
        # if self.convo_prefix is None:
        #     convo_prefix = RLMEnv.standard_prefix()
        # else:
        #     convo_prefix = self.convo_prefix

        convo_prefix = RLMEnv.standard_prefix()
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)

        train_dataset = RLMDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_iterations=self.max_iterations,
            timeout=self.timeout,
            seed=self.seed,
        )

        test_dataset = RLMDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_iterations=self.max_iterations,
            timeout=self.timeout,
            seed=self.seed,
            split="test",
        )

        return train_dataset, test_dataset






        
            


    
    
