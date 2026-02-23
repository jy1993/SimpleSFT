MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
OUTPUT_DIR=checkpoints
TRAIN_PATH=data/train.txt
EXP_NAME=run_sft_vl_4b
deepspeed --master_port 56789 --include localhost:0,1,2,3,4,5,6,7 train.py --model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --zero_stage 2 --bf16 --train_filename ${TRAIN_PATH} --exp_name ${EXP_NAME} --offload --epochs 2 --task_type sft --llm_or_vlm vlm --only_learn_last