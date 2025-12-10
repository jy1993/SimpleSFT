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

# Dataset format
refer to data/retool_sft_short.txt

# Train
Note: for mixed model(with both thinking mode and no thinking mode) like Qwen3-1.7B, set MODEL_TYPE to mixed, otherwise set it to instruct.  

	bash scripts/run_sft.sh

## train_curves_of_qwen2.5_7B_instruct_on_retool_sft
![train_curves_of_qwen2.5_7B_instruct_on_retool_sft](/assets/images/train_curves_of_qwen2.5_7B_instruct_on_retool_sft.png)

# infer
	bash scripts/infer.sh





