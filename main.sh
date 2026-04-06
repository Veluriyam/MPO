export http_proxy="http://127.0.0.1:7897"
export https_proxy="http://127.0.0.1:7897"
export all_proxy="socks5://127.0.0.1:7897"
export no_proxy="localhost,127.0.0.1,::1"




BASE_MODEL='Qwen2.5-VL-7B' # Qwen2.5-VL-7B is implemented using vllm / or simply use gpt-4o-mini / gpt-4.1-nano
OPTIM_MODEL='gpt-4o-mini' # gpt-4.1-nano
MM_GENERATOR_MODEL='gpt-image'

ulimit -n 65535


METHOD=mpo
#在这里改实验名称
EXP_NAME=testRAG_2026_0406_1152

TASK="cuckoo" # CUB dataset
BUDGET_PER_PROMPT=100

LOG_DIR="./logs/$BASE_MODEL/$OPTIM_MODEL/$MM_GENERATOR_MODEL/${EXP_NAME}/${TASK}"

python main.py \
    --data_dir /workspace/yp/MPO/datasets \
    --task_name $TASK \
    --log_dir $LOG_DIR \
    --base_model_name $BASE_MODEL \
    --base_model_port 13141 \
    --optim_model_name $OPTIM_MODEL \
    --mm_generator_model_name $MM_GENERATOR_MODEL \
    --search_method $METHOD \
    --iteration 13 \
    --beam_width 3 \
    --model_responses_num 3 \
    --seed 42 \
    --budget_per_prompt $BUDGET_PER_PROMPT \
    --evaluation_method bayes-ucb \
    --bayes_prior_strength 10