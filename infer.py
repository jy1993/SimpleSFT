import torch
import argparse
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer, AutoProcessor
from llm_utils import read_json, read_jsonl, to_json
from qwen_vl_utils import process_vision_info
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

def repeat_data(data, n):
	if n == 1:
		return data
	final = []
	for one in data:
		final += [one] * n
	return final

def prepare_inputs_for_vllm(messages, processor):
	text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	# qwen_vl_utils 0.0.14+ reqired
	image_inputs, video_inputs, video_kwargs = process_vision_info(
		messages,
		image_patch_size=processor.image_processor.patch_size,
		return_video_kwargs=True,
		return_video_metadata=True
	)
	
	mm_data = {}
	if image_inputs is not None:
		mm_data['image'] = image_inputs
	if video_inputs is not None:
		mm_data['video'] = video_inputs
	return {
		'prompt': text,
		'multi_modal_data': mm_data,
		'mm_processor_kwargs': video_kwargs
	}


parser = argparse.ArgumentParser()
parser.add_argument('--llm_or_vlm', type=str, choices=['llm', 'vlm'])
parser.add_argument('--test_path', type=str, default='test/test.txt')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--max_model_len', type=int, default=2048)
parser.add_argument('--gpu_memory_utilization', type=float, default=0.5)
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--file_type', type=str, default='json')
args = parser.parse_args()

if args.file_type == 'json':
	data = read_json(args.test_path)
else:
	data = read_jsonl(args.test_path)
if args.llm_or_vlm == 'llm':
	processor_or_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
	llm = LLM(
		model=args.model_path,
		tensor_parallel_size=torch.cuda.device_count(),
		max_model_len=args.max_model_len,
		gpu_memory_utilization=args.gpu_memory_utilization,
		enforce_eager=True
	)
elif args.llm_or_vlm == 'vlm':
	processor_or_tokenizer = AutoProcessor.from_pretrained(args.model_path)
	llm = LLM(
        model=args.model_path,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        max_model_len=args.max_model_len,
		gpu_memory_utilization=args.gpu_memory_utilization,
		enforce_eager=True
    )

sampling_params = SamplingParams(
	n=1,
	temperature=0.7,
	top_p=0.95,
	top_k=20,
	repetition_penalty=1,
	max_tokens=800,
	stop_token_ids=[151643, 151645]
)
all_prompts, all_querys, all_labels = [], [], []
all_inputs, all_episode_ids = [], []
for sample in data:
	messages = sample['messages']
	# assert len(messages) == 3, print(len(messages), messages)
	tools = sample.get('tools', None)
	if args.llm_or_vlm == 'llm':
		text = processor_or_tokenizer.apply_chat_template(messages[:-1], 
			tokenize=False, add_generation_prompt=True, 
			tools=tools, enable_thinking=False)
		all_prompts.append(text)
	elif args.llm_or_vlm == 'vlm':
		inputs = prepare_inputs_for_vllm(messages[:-1], processor_or_tokenizer)
		all_inputs.append(inputs)
	all_querys.append(messages[-2])
	all_labels.append(messages[-1])
	all_episode_ids.append(sample.get('episode_id', None))

all_prompts = repeat_data(all_prompts, args.n)
all_querys = repeat_data(all_querys, args.n)
all_labels = repeat_data(all_labels, args.n)
all_inputs = repeat_data(all_inputs, args.n)
print('*' * 10)
if args.llm_or_vlm == 'llm':
	print(all_prompts[0])
elif args.llm_or_vlm:
	print(all_inputs[0])
outputs = llm.generate(all_prompts if args.llm_or_vlm == 'llm' else all_inputs, sampling_params)
responses = [one.text for output in outputs for one in output.outputs]
print(len(responses), len(data))
if args.llm_or_vlm == 'llm':
	assert len(responses) == len(all_prompts) == len(all_querys) == len(all_labels) == len(all_episode_ids)
elif args.llm_or_vlm:
	assert len(responses) == len(all_inputs) == len(all_querys) == len(all_labels) == len(all_episode_ids)
results = []
for query, pred, label, episode_id in zip(all_querys, responses, all_labels, all_episode_ids):
	results.append({'input': query, 'gt': label, 'pred': pred, 'episode_id': episode_id})
to_json(results, args.result_path)