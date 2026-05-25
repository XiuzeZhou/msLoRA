echo $1
task=msrvtt
llm_model=../autodl-fs/Qwen2.5-7B
clip_model=./llm/clip-vit-base-patch32/
out_log_dir=./logs/
save_path=../autodl-tmp/
seed=11
r_val=4
m_scaling=4

mkdir -p ${out_log_dir}${task}

out_log=qwen7b_mslora_train.log
echo "task: $task"
TRANSFORMERS_CACHE=./llm/ \
HF_DATASETS_CACHE=./llm/ \
CUDA_VISIBLE_DEVICES=0 ../miniconda3/bin/python -u ./main.py \
    -task ${task} \
    -llm_model ${llm_model} \
    -clip_model ${clip_model} \
    -lr 2e-5 \
    -epochs 2 \
    -batch_size 32 \
    -r ${r_val} \
    -lora_modules 7 \
    -multimodal_scaling ${m_scaling} \
    -gpu 0 \
    -save_path ${save_path} \
    -seed $seed \
    > ${out_log_dir}${task}/${out_log}