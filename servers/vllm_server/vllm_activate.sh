MODEL_NAME="/workspace/yp/MPO/datasets/Qwen_Qwen2.5-VL-7B-Instruct"

echo "VLLM_API_KEY: $VLLM_API_KEY"
ulimit -n 65535

# Define multiple GPU IDs as an array (without commas/spaces)
GPU_NUMS=(3)

# Port: Use the first GPU number as the suffix
PORT_SUFFIX="${GPU_NUMS[0]}"

# Tensor parallel size = number of GPUs
TENSOR_PARALLEL_SIZE="${#GPU_NUMS[@]}"

CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_NUMS[*]}")"
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

vllm serve \
$MODEL_NAME \
--dtype auto \
--port 1314$PORT_SUFFIX \
--tensor-parallel-size $TENSOR_PARALLEL_SIZE \
--gpu-memory-utilization 0.9 \
--max_model_len 3000 \
--limit-mm-per-prompt '{"image": 2, "video": 2}' \
--trust-remote-code