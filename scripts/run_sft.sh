MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
OUTPUT_DIR=checkpoints
TRAIN_PATH=data/retool_sft_short.txt
EXP_NAME=sft_v20251115
MODEL_TYPE=instruct
deepspeed --master_port 56789 --include localhost:0,1,2,3,4,5,6,7 train.py --model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --zero_stage 2 --bf16 --train_filename ${TRAIN_PATH} --exp_name ${EXP_NAME} --offload --model_type ${MODEL_TYPE} --tool_result_tags '<interpreter,</interpreter'