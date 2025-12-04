"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

import os
from typing import Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import tinker
from tinker import types

from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")

class TinkerClient:
    def __init__(self, sampling_client: Optional[tinker.SamplingClient] = None, model: str = "Qwen/Qwen3-8B"):
        self.service_client = tinker.ServiceClient()

        if sampling_client:
            self.sampling_client = sampling_client
        else:
            self.sampling_client = self.service_client.create_sampling_client(base_model=model)
        
        self.tokenizer = get_tokenizer(model)
        self.renderer = renderers.get_renderer(get_recommended_renderer_name(model), self.tokenizer)
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
    ) -> str:

        sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences()
        )

        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            try:
                with open("convo_log.txt", "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            except Exception as log_exc:
                pass  # Non-fatal: ignore any logging failures
            
            # prompt_text = self.tokenizer.apply_chat_template(
            #         messages,
            #         tokenize=False,
            #         add_generation_prompt=True,
            # )
            # tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            # model_input = types.ModelInput.from_ints(tokens)

            prompt = self.renderer.build_generation_prompt(messages)
            print(f"Making completion at {datetime.now().strftime('%H:%M:%S')}")
            response: types.SampleResponse = self.sampling_client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            ).result(timeout=300)
            # response_tokens = response.sequences[0].tokens
            # response_text = self.tokenizer.decode(response_tokens)
            sampled_message, parse_success = self.renderer.parse_response(response.sequences[0].tokens)

            return sampled_message["content"] # TODO MIGHT HAVE TO CUT OUT REASONING TRACES TO NOT EXPLODE ROOT CONTEXT ???
        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")