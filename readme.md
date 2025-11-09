# SimpleRL
My implementations of llm sft.

# Features
+ Supported models
	+ Qwen2/Qwen2.5/Qwen3 language models
+ Supported situations
	+ single-turn/multiturn data
	+ agent training
+ Supported features
	+ zero stage 1-3
	+ offload
	+ gradient checkpointing
	+ gradient accumulation
	+ flash-attn

# Requirements
	pip install -r requirements

# Train
	bash scripts/run_sft.sh

# infer
	bash scripts/infer.sh





