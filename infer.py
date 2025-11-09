import torch
import argparse
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
from llm_utils import read_json, read_jsonl, to_json

def repeat_data(data, n):
	if n == 1:
		return data
	final = []
	for one in data:
		final += [one] * n
	return final

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default='test/test.txt')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--file_type', type=str, default='json')
args = parser.parse_args()

if args.file_type == 'json':
	data = read_json(args.test_path)
else:
	data = read_jsonl(args.test_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
llm = LLM(
	model=args.model_path,
	tensor_parallel_size=1,
	max_model_len=2048,
	gpu_memory_utilization=0.5,
	enforce_eager=True
)
sampling_params = SamplingParams(
	n=args.n,
	temperature=0.7,
	top_p=0.95,
	top_k=20,
	repetition_penalty=1,
	max_tokens=800,
	stop_token_ids=[151643, 151645]
)
all_prompts, all_querys, all_labels = [], [], []
# only support singleturn infer for now
for sample in data:
	messages = sample['messages']
	assert len(messages) == 3, print(len(messages), messages)
	tools = sample.get('tools', None)
	text = tokenizer.apply_chat_template(messages[:-1], 
		tokenize=False, add_generation_prompt=True, 
		tools=tools, enable_thinking=False)
	all_prompts.append(text)
	all_querys.append(messages[1])
	all_labels.append(messages[-1])

all_prompts = repeat_data(all_prompts, args.n)
all_querys = repeat_data(all_querys, args.n)
all_labels = repeat_data(all_labels, args.n)
print('*' * 10)
print(all_prompts[0])
outputs = llm.generate(all_prompts, sampling_params)
responses = [one.text for output in outputs for one in output.outputs]
print(len(responses), len(data))
assert len(responses) == len(all_prompts) == len(all_querys) == len(all_labels)
results = []
for query, pred, label in zip(all_querys, responses, all_labels):
	results.append({'input': query, 'gt': label, 'pred': pred})
to_json(results, args.result_path)