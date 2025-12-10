MODEL_PATH=checkpoints/sft_steps_100
RESULT_PATH=result/result.txt
CUDA_VISIBLE_DEVICES=0 python infer.py --model_path ${MODEL_PATH} --result_path {RESULT_PATH}