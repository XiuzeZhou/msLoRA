echo $1
TASK=twitter17
LLM_MODEL=./llm/Qwen2.5-7B
CLIP_MODEL=./llm/clip-vit-base-patch32/
OUT_LOG_DIR=./logs/
SAVE_PATH=./checkpoints/

LR=2e-5
EPOCHS=10
BATCH_SIZE=16
R=8
LoRA_MODULES=7
LoRA_NAME=msLoRA
GPU=0
SEED=110

mkdir -p ${OUT_LOG_DIR}${TASK}

OUT_LOG=${LoRA_NAME}_train.log
echo "task: $TASK, r: $R"
TRANSFORMERS_CACHE=./llm/ \
HF_DATASETS_CACHE=./llm/ \
CUDA_VISIBLE_DEVICES=0 D:/Anaconda3/envs/torch2.2/python -u ./main.py \
    -task ${TASK} \
    -llm_model ${LLM_MODEL} \
    -clip_model ${CLIP_MODEL} \
    -lr $LR \
    -epochs $EPOCHS \
    -batch_size $BATCH_SIZE \
    -r $R \
    -lora_modules $LoRA_MODULES \
    -lora_name ${LoRA_NAME} \
    -gpu ${GPU} \
    -save_path ${SAVE_PATH} \
    -seed $SEED \
    > ${OUT_LOG_DIR}${TASK}/${OUT_LOG}
