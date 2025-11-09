import copy
import sys
import os
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_scheduler, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from llm_utils import *
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from safetensors.torch import save_model

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--warmup_ratio', type=float, default=0.02)
parser.add_argument('--scheduler_type', type=str, default='cosine_with_warmup')
parser.add_argument('--per_device_train_batch_size', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--train_filename', type=str, default='train/train.txt')
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
parser.add_argument('--model_type', type=str, choices=['instruct', 'mixed'])
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

def save_everything(model, tokenizer, args, step):
	path = args.output_dir + '/sft_steps_%s' % step
	if args.zero_stage == 3:
		save_zero_three_model(model, args, path)
		if args.global_rank == 0:
			tokenizer.save_pretrained(path)
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
			tokenizer.save_pretrained(path)
			os.system('cp %s/config.json %s' % (args.model_path, path))

def train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, writer):
	global_step = 0
	for _ in trange(args.epochs):
		for i, batch in enumerate(tqdm(train_loader)):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			inputs = prepare_model_inputs(batch)
			# if i == 0 and args.global_rank <= 2:
			# 	print('*' * 10)
			# 	for k, v in inputs.items():
			# 		print(k, v.shape)
			loss = model(**inputs, use_cache=False).loss
			if args.global_rank <= 0 and i % args.gradient_accumulation_steps == 0:
				writer.add_scalar('Train/total_loss', loss.item(), global_step)
			model.backward(loss)
			model.step()
			if i % args.gradient_accumulation_steps == 0:
				global_step += 1

			if global_step % args.save_steps == 0 and args.global_rank == 0:
				save_everything(model, tokenizer, args, global_step)

	# save final model
	if args.global_rank == 0
		save_everything(model, tokenizer, args, global_step)

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

	tokenizer = AutoTokenizer.from_pretrained(args.model_path)
	model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
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
	# train_dataset = SFTDataset(args.train_filename, tokenizer, args.model_type)
	train_dataset = MultiTurnSFTDataset(args.train_filename, tokenizer, args.model_type)
	if args.global_rank == 0:
		print(tokenizer.decode(train_dataset[0][0][0]))
		print('*' * 10)
		print(tokenizer.decode([l for l in train_dataset[0][-1][0].tolist() if l != -100]))
	if args.local_rank == -1:
		train_sampler = torch.utils.data.SequentialSampler(train_dataset)
	else:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, sampler=train_sampler, collate_fn=collate_for_lm)
	
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
	if args.gradient_checkpointing:
		model.gradient_checkpointing_enable()
	train(model, tokenizer, train_loader, None, optimizer, lr_scheduler, writer)
	if args.global_rank <= 0:
		writer.close()

if __name__ == '__main__':
	main()
