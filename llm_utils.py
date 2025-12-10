import torch
import json
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
# import deepspeed
# from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

def read_json(filename):
	with open(filename, 'r', encoding='utf8') as f:
		data = json.load(f)
	return data

def read_jsonl(filename):
	data = []
	with open(filename, 'r', encoding='utf8') as f:
		for line in f.readlines():
			data.append(json.loads(line))
	return data

def to_json(data, fout):
	with open(fout, 'w', encoding='utf8') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

# class SFTDataset(Dataset):
# 	"""docstring for SFTDataset"""
# 	def __init__(self, train_filename, tokenizer, model_type):
# 		super(SFTDataset, self).__init__()
# 		self.data = read_json(train_filename)
# 		self.tokenizer = tokenizer
# 		self.model_type = model_type

# 	def __getitem__(self, index):
# 		# support models: qwen2.5 and qwen3-2507(no-thinking)
# 		if 'tools' in self.data[index]:
# 			messages = [
# 				{"role": "user", "content": self.data[index]['user']}
# 			]
# 			if self.model_type == 'instruct':
# 				text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=self.data[index]['tools'])
# 			else:
# 				text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=self.data[index]['tools'], enable_thinking=False)
# 		else:
# 			messages = [
# 				{"role": "system", "content": self.data[index]['system']},
# 				{"role": "user", "content": self.data[index]['user']}
# 			]
# 			if self.model_type == 'instruct':
# 				text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
# 			else:
# 				text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
# 		input_text = text + '<|im_start|>assistant\n'
# 		label_text = self.data[index]['assistant'] + '<|im_end|>\n<|endoftext|>'
# 		inputs = self.tokenizer(input_text, add_special_tokens=False, return_tensors='pt')
# 		outputs = self.tokenizer(label_text, add_special_tokens=False, return_tensors='pt')
# 		input_ids = torch.cat([inputs['input_ids'], outputs['input_ids']], dim=1)
# 		attention_mask = torch.cat([inputs['attention_mask'], outputs['attention_mask']], dim=1)
# 		labels = torch.cat([torch.ones(1, inputs['input_ids'].shape[1], dtype=torch.long) * -100, outputs['input_ids']], dim=1)
# 		return input_ids, attention_mask, labels

# 	def __len__(self):
# 		return len(self.data)

def apply_chat_template(messages):
	text = ''
	for message in messages:
		text += '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'
	return text

class MultiTurnSFTDataset(Dataset):
	"""docstring for MultiTurnSFTDataset"""
	def __init__(self, train_filename, tokenizer, model_type, tool_result_tags=None):
		super(MultiTurnSFTDataset, self).__init__()
		self.data = read_json(train_filename)
		self.tokenizer = tokenizer
		self.model_type = model_type
		assert self.model_type in ['instruct', 'mixed']
		if tool_result_tags is not None:
			self.tool_result_start_ids = self.tokenizer(tool_result_tags[0])['input_ids']
			self.tool_result_end_ids = self.tokenizer(tool_result_tags[1])['input_ids']
		else:
			self.tool_result_start_ids, self.tool_result_end_ids = None, None

	def __getitem__(self, index):
		one = self.data[index]
		messages = one['messages']
		tools = one.get('tools', None)
		if self.model_type == 'instruct':
			text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
			text = text.replace('<think>\n\n</think>\n\n', '')
		else:
			new_messages = self.add_no_think_tags(messages)
			# text = self.tokenizer.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=False, tools=tools, enable_thinking=False)
			text = apply_chat_template(new_messages)
		inputs = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
		labels = self.get_labels(inputs['input_ids'])
		return inputs['input_ids'], inputs['attention_mask'], labels

	def __len__(self):
		return len(self.data)

	def add_no_think_tags(self, messages):
		new_messages = []
		for m in messages:
			if m['role'] == 'assistant':
				new_messages.append({'role': 'assistant', 'content': '<think>\n\n</think>\n\n' + m['content']})
			else:
				new_messages.append(m)
		return new_messages

	def get_labels(self, input_ids):
		im_start_pos = (input_ids[0] == self.tokenizer.vocab['<|im_start|>']).nonzero().tolist()
		im_end_pos = (input_ids[0] == self.tokenizer.vocab['<|im_end|>']).nonzero().tolist()
		assert len(im_start_pos) == len(im_end_pos)
		labels = [-100] * input_ids.shape[1]
		input_ids_list = input_ids[0].tolist()
		if self.model_type == 'mixed':
			start_offset = 7
		else:
			start_offset = 3
		for i, (start, end) in enumerate(zip(im_start_pos, im_end_pos)):
			if i > 0 and i % 2 == 0:
				labels[start[0]+start_offset:end[0]+2] = input_ids_list[start[0]+start_offset:end[0]+2]
		for i in self.get_label_mask(labels):
			labels[i] = -100
		return torch.LongTensor(labels).unsqueeze(0)

	def get_label_mask(self, labels):
		if self.tool_result_start_ids is not None and self.tool_result_end_ids is not None:
			n = len(labels)
			start_len = len(self.tool_result_start_ids)
			end_len = len(self.tool_result_end_ids)
			if n < start_len + end_len:
				return []
			start_positions = []
			for i in range(n - start_len + 1):
				if labels[i:i+start_len] == self.tool_result_start_ids:
					start_positions.append(i)
			end_positions = []
			for j in range(end_len - 1, n):
				if j - end_len + 1 >= 0 and labels[j-end_len+1:j+1] == self.tool_result_end_ids:
					end_positions.append(j)
			intervals = []
			for start in start_positions:
				for end in end_positions:
					if end >= start + start_len + end_len - 1:
						intervals.append((start, end))
						break
			indexs = []
			for se in intervals:
				s, e = se
				for i in range(s, e+end_len):
					indexs.append(i)
			return indexs
		return []

def pad_and_cat(tensor_list, padding):
	max_len = max([tensor.shape[1] for tensor in tensor_list])
	return torch.cat([torch.cat([tensor, torch.ones(1, max_len - tensor.shape[1], dtype=torch.long) * padding], dim=1) for tensor in tensor_list], dim=0)

def collate_for_lm(batch):
	input_ids = pad_and_cat([item[0] for item in batch], 151643)
	attention_mask = pad_and_cat([item[1] for item in batch], 0)
	labels = pad_and_cat([item[2] for item in batch], -100)
	return input_ids, attention_mask, labels

def prepare_model_inputs(batch):
	inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
	return inputs

def get_simpo_loss(args, batch, logits):
	chosen_logits, rejected_logits = torch.split(logits, logits.shape[0] // 2, dim=0)
	shift_chosen_logits = chosen_logits[:, :-1]
	shift_rejected_logits = rejected_logits[:, :-1]
	shift_chosen_labels = batch[1][:, 1:]
	shift_rejected_labels = batch[3][:, 1:]
	chosen_mask = shift_chosen_labels.ne(-100).to(logits.dtype)
	rejected_mask = shift_rejected_labels.ne(-100).to(logits.dtype)
	loss_fct = nn.CrossEntropyLoss(reduction='none')
	# bsz, seq_len
	chosen_loss = - loss_fct(shift_chosen_logits.transpose(1, 2), shift_chosen_labels)
	rejected_loss = - loss_fct(shift_rejected_logits.transpose(1, 2), shift_rejected_labels)
	chosen_loss_avg = torch.sum(chosen_loss * chosen_mask, dim=-1) / chosen_mask.sum(dim=-1)
	rejected_loss_avg = torch.sum(rejected_loss * rejected_mask, dim=-1) / rejected_mask.sum(dim=-1)
	loss = - nn.LogSigmoid()(args.beta * (chosen_loss_avg - rejected_loss_avg) - args.margin)
	return loss.mean()

def get_train_ds_config(offload,
						stage,
						global_batch_size,
						micro_batch_size,
						grad_acc,
						bf16=False,
						job_name=None,
						enable_hybrid_engine=False,
						inference_tp_size=1,
						release_inference_cache=False,
						pin_parameters=True,
						tp_gather_partition_size=8,
						max_out_tokens=512):
	device = "cpu" if offload else "none"
	zero_opt_dict = {
		"stage": stage,
		"offload_param": {
			"device": device,
			"pin_memory": True
		},
		"offload_optimizer": {
			"device": device,
			"pin_memory": True
		},
		"stage3_param_persistence_threshold": 1e4,
		"stage3_max_live_parameters": 3e7,
		"stage3_prefetch_bucket_size": 3e7,
		"memory_efficient_linear": False,
		"contiguous_gradients": False,
		"overlap_comm": True,
		"reduce_scatter": False
	}
	return {
		"train_batch_size": global_batch_size,
		"train_micro_batch_size_per_gpu": micro_batch_size,
		"steps_per_print": 500,
		"zero_optimization": zero_opt_dict,
		"fp16": {
			"enabled": True if not bf16 else False,
			"auto_cast": False,
			"loss_scale": 0,
			"initial_scale_power": 16,
			"loss_scale_window": 1000,
			"hysteresis": 2,
			"consecutive_hysteresis": False,
			"min_loss_scale": 1
		},
		"bf16":{
			"enabled": True if bf16 else False
		},
		"gradient_clipping": 1.0,
		"prescale_gradients": False,
		"wall_clock_breakdown": False,
		"gradient_acculation_steps": grad_acc,
		# "tensorboard": {
		# 	"enabled": True,
		# 	"output_path": "logs",
		# 	"job_name": job_name
		# },
	}

def NEFTune(model, noise_alpha=5):
	def noised_embed(orig_embed, noise_alpha):
		def new_func(x):
			# during training, we add noise to the embedding
			# during generation, we don't add noise to the embedding
			if model.training:
				embed_init = orig_embed(x)
				dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
				mag_norm = noise_alpha/torch.sqrt(dims)
				return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
			else:
				return orig_embed(x)
		return new_func
	model.model.embed_tokens.forward = noised_embed(model.model.embed_tokens, noise_alpha)
	return model

def print_gpu_memory(rank):
	total_mem = torch.cuda.get_device_properties(rank).total_memory / (1024 ** 3)
	allocated = torch.cuda.memory_allocated(rank) / (1024 ** 3)  
	reserved = torch.cuda.memory_reserved(rank) / (1024 ** 3)    
	free = total_mem - (allocated + reserved) 
	print(f"GPU {rank}:")
	print(f"  ▸ 总显存: {total_mem:.2f} GB")
	print(f"  ▸ 已占用: {allocated:.2f} GB (Tensor+模型)")
	print(f"  ▸ 框架缓存: {reserved:.2f} GB (预分配Block)")
	print(f"  ▸ 剩余可用: {free:.2f} GB\n")
	torch.distributed.barrier()

# def _z3_params_to_fetch(param_list):
# 	return [
# 		p for p in param_list
# 		if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
# 	]

# def save_zero_three_model(model, args, path):
# 	os.makedirs(path, exist_ok=True)
# 	WEIGHTS_NAME = "pytorch_model.bin"
# 	output_model_file = os.path.join(path, WEIGHTS_NAME)
# 	model_to_save = model.module if hasattr(model, 'module') else model
# 	output_state_dict = {}
# 	for k, v in model_to_save.named_parameters():
# 		if hasattr(v, 'ds_id'):
# 			with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),enabled=True):
# 				v_p = v.data
# 		else:
# 			v_p = v
# 		if args.global_rank == 0:
# 			output_state_dict[k] = v_p
# 	if args.global_rank == 0:
# 		torch.save(output_state_dict, output_model_file)
# 	del output_state_dict

def save_zero_three_model(model, args, path):
	os.makedirs(path, exist_ok=True)
	WEIGHTS_NAME = "pytorch_model.bin"
	output_model_file = os.path.join(path, WEIGHTS_NAME)
	model_to_save = model.module if hasattr(model, 'module') else model
	if args.global_rank == 0:
		params_to_gather = []
		params_to_save = {}
		for k, v in model_to_save.named_parameters():
			if not hasattr(v, 'ds_id'):
				params_to_save[k] = v.data
		for k, v in model_to_save.named_parameters():
			if hasattr(v, 'ds_id'):
				params_to_gather.append(v)
		with deepspeed.zero.GatheredParameters(params_to_gather, enabled=True, modifier_rank=0):
			if args.global_rank == 0:
				for k, v in model_to_save.named_parameters():
					if hasattr(v, 'ds_id'):
						params_to_save[k] = v.data.cpu()
		torch.save(params_to_save, output_model_file)
		del params_to_save