REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""


context = """The following lines contain 10 text messages, one per line. Each text message can be classified as spam or ham (i.e., not spam).

You will be asked to answer questions about the aggregate label statistics across all 10 examples in this dataset. Do not try to guess, estimate, or approximate the result. Calculate the exact answer given these datapoints.

Date: Dec 28, 2022 || User: 76063 || Instance: Todays Vodafone numbers ending with 4882 are selected to a receive a £350 award. If your number matches call 09064019014 to receive your £350 award.
Date: Jul 28, 2024 || User: 33845 || Instance: URGENT This is our 2nd attempt to contact U. Your £900 prize from YESTERDAY is still awaiting collection. To claim CALL NOW 09061702893
Date: Feb 05, 2025 || User: 76063 || Instance: 74355 XMAS iscoming & ur awarded either £500 CD gift vouchers & free entry 2 r £100 weekly draw txt MUSIC to 87066 TnC
Date: Jun 04, 2025 || User: 76063 || Instance: We left already we at orchard now.
Date: May 16, 2025 || User: 24151 || Instance: Guessin you ain't gonna be here before 9?
Date: Apr 06, 2024 || User: 76063 || Instance: GSOH? Good with SPAM the ladies?U could b a male gigolo? 2 join the uk's fastest growing mens club reply ONCALL. mjzgroup. 08714342399.2stop reply STOP. msg@£1.50rcvd
Date: Feb 07, 2024 || User: 76063 || Instance: No need for the drug anymore.
Date: Jul 24, 2024 || User: 76063 || Instance: Your bill at 3 is £33.65 so thats not bad!
Date: Jul 02, 2024 || User: 76063 || Instance: Final Chance! Claim ur £150 worth of discount vouchers today! Text YES to 85023 now! SavaMob, member offers mobile! T Cs SavaMob POBOX84, M263UZ. £3.00 Subs 16
Date: Oct 14, 2022 || User: 76063 || Instance: You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p p£3.99
Recall: the preceding lines contain 10 text messages, one per line. Each text message can be classified as spam or ham (i.e., not spam).

You will be asked to answer questions about the aggregate label statistics across all 10 examples in this dataset. Do not try to guess, estimate, or approximate the result. Calculate the exact answer given these datapoints.

"""


query = """
In the above data, which of the labels is the most common? Give your final answer in the form 'Label: answer' where answer is one of the labels: ham, spam.
"""

messages =[
    {'role': 'system', 'content': REPL_SYSTEM_PROMPT},
    {'role': 'user', 'content': context+query},
]

from tinker_cookbook import renderers, tokenizer_utils
tokenizer = tokenizer_utils.get_tokenizer('Qwen/Qwen3-8B')
renderer = renderers.get_renderer('qwen3', tokenizer)
prompt = renderer.build_generation_prompt(messages)
print(prompt)
print('-'*70)
print(tokenizer.decode(prompt.to_ints()))
print('-'*70)


import tinker
from tinker.types import SamplingParams
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model='Qwen/Qwen3-8B')
stop_sequences = renderer.get_stop_sequences()
# print(f"Stop sequences: {stop_sequences}")
sampling_params = SamplingParams(max_tokens=4000, temperature=0.0, stop=stop_sequences)
output = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
# print(f"Sampled tokens: {output.sequences[0].tokens}")
print(f"Decoded response = {tokenizer.decode(output.sequences[0].tokens)}")
print('-'*70)

sampled_message, parse_success = renderer.parse_response(output.sequences[0].tokens)
print(f"Sampled message: {sampled_message}")
print('-'*70)

print(f"Parse success: {parse_success}")