import copy
import sys
import os
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import (
	get_linear_schedule_with_warmup, 
	get_constant_schedule_with_warmup, 
	get_scheduler, 
	AutoModelForCausalLM, 
	AutoTokenizer,
	AutoProcessor,
	AutoModelForImageTextToText
)
from tqdm import tqdm
from llm_utils import *
from vlm_utils import MultiTurnSFTDatasetForVL, collate_for_sft_vl, collate_for_dpo_vl, prepare_model_inputs_vl
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from safetensors.torch import save_model

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--task_type', type=str, default='sft', choices=['sft', 'dpo'])
parser.add_argument('--llm_or_vlm', type=str, default='llm', choices=['llm', 'vlm'])
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--ref_model_path', type=str, default=None)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--warmup_ratio', type=float, default=0.02)
parser.add_argument('--scheduler_type', type=str, default='cosine_with_warmup')
parser.add_argument('--per_device_train_batch_size', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--train_filename', type=str, default='train/train.txt')
parser.add_argument('--only_learn_last', action='store_true')
parser.add_argument('--save_steps', type=int, default=1000)
parser.add_argument('--eval_steps', type=int, default=5000)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--clip_grad_norm', type=float, default=1.0)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--offload', action='store_true')
parser.add_argument('--zero_stage', type=int, default=0)
parser.add_argument('--gradient_checkpointing', action='store_true')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--instruct_or_mixed', type=str, choices=['instruct', 'mixed'])
parser.add_argument('--tool_result_tags', type=str, default=None)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--max_vl_tokens', type=int, default=500)
parser.add_argument('--min_vl_tokens', type=int, default=200)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

def save_everything(model, processor_or_tokenizer, args, step):
	path = args.output_dir + '/sft_steps_%s' % step
	if args.zero_stage == 3:
		save_zero_three_model(model, args, path)
		if args.global_rank == 0:
			processor_or_tokenizer.save_pretrained(path)
			os.system('cp %s/config.json %s' % (args.model_path, path))
	else:
		# sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
		# torch.save(sd, os.path.join(path, 'pytorch_model.bin'))
		# save_model(model.module if hasattr(model, 'module') else model, os.path.join(path, 'model.safetensors'))
		# os.system('cp %s/config.json %s' % (pretrain_model_path, path))
		# os.system('cp %s/generation_config.json %s' % (pretrain_model_path, path))
		# os.system('cp %s/tokenizer.json %s' % (pretrain_model_path, path))
		# os.system('cp %s/tokenizer_config.json %s' % (pretrain_model_path, path))
		# os.system('cp %s/vocab.json %s' % (pretrain_model_path, path))
		# os.system('cp %s/merges.txt %s' % (pretrain_model_path, path))
		if args.global_rank == 0:
			os.makedirs(path, exist_ok=True)
			model.save_pretrained(
				path,
				safe_serialization=True,
				max_shard_size="2GB",
			)
			processor_or_tokenizer.save_pretrained(path)
			os.system('cp %s/config.json %s' % (args.model_path, path))

def train(model, ref_model, processor_or_tokenizer, train_loader, valid_loader, optimizer, scheduler, writer):
	global_step = 0
	for _ in trange(args.epochs):
		for i, batch in enumerate(tqdm(train_loader)):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			if args.llm_or_vlm == 'llm':
				inputs = prepare_model_inputs(batch, args.task_type)
			elif args.llm_or_vlm == 'vlm':
				inputs = prepare_model_inputs_vl(batch, args.task_type)
			# if i == 0 and args.global_rank <= 2:
			# 	print('*' * 10)
			# 	for k, v in inputs.items():
			# 		print(k, v.shape)
			if args.task_type == 'sft':
				loss = model(**inputs, use_cache=False).loss
			elif args.task_type == 'dpo':
				chosen_labels = inputs.pop('chosen_labels')
				rejected_labels = inputs.pop('rejected_labels')
				logits = model(**inputs, use_cache=False).logits
				with torch.no_grad():
					ref_logits = ref_model(**inputs, use_cache=False).logits
				chosen_logits, rejected_logits = torch.split(logits, logits.shape[0] // 2, dim=0)
				ref_chosen_logits, ref_rejected_logits = torch.split(ref_logits, ref_logits.shape[0] // 2, dim=0)
				chosen_logps = get_logps(chosen_logits, chosen_labels)
				rejected_logps = get_logps(rejected_logits, rejected_labels)
				ref_chosen_logps = get_logps(ref_chosen_logits, chosen_labels)
				ref_rejected_logits = get_logps(ref_rejected_logits, rejected_labels)
				loss, chosen_rewards, rejected_rewards = get_dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logits, args.beta)
			if args.global_rank <= 0 and i % args.gradient_accumulation_steps == 0:
				writer.add_scalar('Train/total_loss', loss.item(), global_step)
				if args.task_type == 'dpo':
					writer.add_scalar('Train/chosen_rewards', chosen_rewards.item(), global_step)
					writer.add_scalar('Train/rejected_rewards', rejected_rewards.item(), global_step)
			model.backward(loss)
			model.step()
			if i % args.gradient_accumulation_steps == 0:
				global_step += 1

			if global_step % args.save_steps == 0 and args.global_rank == 0:
				save_everything(model, processor_or_tokenizer, args, global_step)

	# save final model
	if args.global_rank == 0:
		save_everything(model, processor_or_tokenizer, args, global_step)

def main():
	if args.local_rank == -1:
		args.device = torch.device('cuda')
	else:
		torch.cuda.set_device(args.local_rank)
		args.device = torch.device('cuda', args.local_rank)
		deepspeed.init_distributed()
	args.global_rank = torch.distributed.get_rank()
	args.n_gpus = torch.distributed.get_world_size()
	torch.distributed.barrier()

	if args.llm_or_vlm == 'llm':
		processor_or_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
		model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
	elif args.llm_or_vlm == 'vlm':
		processor_or_tokenizer = AutoProcessor.from_pretrained(args.model_path)
		processor_or_tokenizer.image_processor.size = {"longest_edge": args.max_vl_tokens*32*32, "shortest_edge": args.min_vl_tokens*32*32}
		model = AutoModelForImageTextToText.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
	if args.task_type == 'dpo':
		if args.llm_or_vlm == 'llm':
			ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_path if args.ref_model_path else args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
		elif args.llm_or_vlm == 'vlm':
			ref_model = AutoModelForImageTextToText.from_pretrained(args.ref_model_path if args.ref_model_path else args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
	elif args.task_type == 'sft':
		ref_model = None
	if args.global_rank == 0:
		print(model)
	no_decay = ["bias", "norm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay, 
			"lr": args.lr
		},     
		{   
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
			"weight_decay": 0.0, 
			"lr": args.lr
		}           
	]
	AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
	optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95))
	if args.task_type == 'sft':
		if args.llm_or_vlm == 'llm':
			train_dataset = MultiTurnSFTDataset(args.train_filename, processor_or_tokenizer, args.instruct_or_mixed, args.tool_result_tags.split(',') if args.tool_result_tags else None)
		elif args.llm_or_vlm == 'vlm':
			train_dataset = MultiTurnSFTDatasetForVL(args.train_filename, processor_or_tokenizer, args.only_learn_last)
	else:
		if args.llm_or_vlm == 'llm':
			train_dataset = DPODataset(args.train_filename, processor_or_tokenizer, args.instruct_or_mixed, args.tool_result_tags.split(',') if args.tool_result_tags else None)
		elif args.llm_or_vlm == 'vlm':
			train_dataset = DPODatasetForVL(args.train_filename, processor_or_tokenizer)
	if args.global_rank == 0:
		print(processor_or_tokenizer.decode(train_dataset[0][0][0]))
		print('*' * 10)
		if args.task_type == 'sft':
			if args.llm_or_vlm == 'llm':
				print(processor_or_tokenizer.decode([l for l in train_dataset[0][2][0].tolist() if l != -100]))
			elif args.llm_or_vlm == 'vlm':
				print(processor_or_tokenizer.decode([l for l in train_dataset[0][4][0].tolist() if l != -100]))
		elif args.task_type == 'dpo':
			print('chosen:\n')
			if args.llm_or_vlm == 'llm':
				print(processor_or_tokenizer.decode([l for l in train_dataset[0][2][0].tolist() if l != -100]))
			elif args.llm_or_vlm == 'vlm':
				print(processor_or_tokenizer.decode([l for l in train_dataset[0][8][0].tolist() if l != -100]))
			print('*' * 10)
			print('rejected:\n')
			if args.llm_or_vlm == 'llm':
				print(processor_or_tokenizer.decode([l for l in train_dataset[0][-1][0].tolist() if l != -100]))
			elif args.llm_or_vlm == 'vlm':
				print(processor_or_tokenizer.decode([l for l in train_dataset[0][9][0].tolist() if l != -100]))
	if args.local_rank == -1:
		train_sampler = torch.utils.data.SequentialSampler(train_dataset)
	else:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
	if args.task_type == 'sft':
		collate_fn = collate_for_sft if args.llm_or_vlm == 'llm' else collate_for_sft_vl
	elif args.task_type == 'dapo':
		collate_fn = collate_for_dpo if args.llm_or_vlm == 'llm' else collate_for_dpo_vl
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, sampler=train_sampler, collate_fn=collate_fn)
	
	t_total = len(train_loader) * args.epochs 
	warmup_steps = args.warmup_ratio * t_total
	if args.scheduler_type == 'linear_with_warmup':
		lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
	elif args.scheduler_type == 'constant_with_warmup':
		lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
	elif args.scheduler_type == 'cosine_with_warmup':
		# lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total, min_lr_rate=0.1)
		lr_scheduler = get_scheduler('cosine_with_min_lr', optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total, scheduler_specific_kwargs={'min_lr_rate':0.1})
	os.makedirs(args.output_dir, exist_ok=True)
	if args.global_rank <= 0:
		writer = SummaryWriter('logs/%s' % args.exp_name)
	else:
		writer = None
	# writer = None

	ds_config = get_train_ds_config(offload=args.offload, 
		stage=args.zero_stage, 
		global_batch_size=args.per_device_train_batch_size*args.gradient_accumulation_steps*args.n_gpus,
		micro_batch_size=args.per_device_train_batch_size,
		grad_acc=args.gradient_accumulation_steps,
		bf16=args.bf16,
		job_name=args.exp_name)
	model, optimizer, _, lr_scheduler = deepspeed.initialize(
		model=model,
		optimizer=optimizer,
		args=args,
		config=ds_config,
		lr_scheduler=lr_scheduler,
		dist_init_required=True)
	if ref_model:
		ref_model = deepspeed.init_inference(model=ref_model, config={"dtype": 'bfloat16', "replace_with_kernel_inject": True})
	if args.gradient_checkpointing:
		model.gradient_checkpointing_enable()
	train(model, ref_model, processor_or_tokenizer, train_loader, None, optimizer, lr_scheduler, writer)
	if args.global_rank <= 0:
		writer.close()

if __name__ == '__main__':
	main()
