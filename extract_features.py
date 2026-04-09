import os
import glob
import numpy as np
from PIL import Image
from src.model.hf_model import HFModelForPCA

def main():
    print("正在加载模型...")
    model = HFModelForPCA(model_path="Qwen/Qwen2.5-VL-7B-Instruct") 
    
    # 1. 加载图像路径
    img_dir = "/workspace/yp/MPO/datasets/classification/cub/images/031.Black_billed_Cuckoo"
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not img_paths:
        print("未在该目录下找到图片，请检查路径或图片格式。")
        return

    # 2. 定义 Prompt
    raw_question = "Given the image, answer the following question.Classify the content of the target image. Choices: ['black_billed_cuckoo', 'yellow_billed_cuckoo', 'mangrove_cuckoo'],"
    # mpo_optimized_prefix = "Please focus on the visual characteristics of the bird, such as its bill color and body markings, to make your classification. Your answer should directly indicate which species it is, using the exact name as listed in the choices. For example, if the bird has a bright yellow bill, respond with 'yellow_billed_cuckoo'. If you cannot determine the species based on the visual clues, please indicate that the species is unclear. " 
    mpo_optimized_prefix="You are a multimodal reasoning assistant developed by Alibaba Cloud. You can analyze images and answer questions about them accurately.You must independently analyze images to extract evidence chains of object features and spatial relations,employ step-by-step deduction that explicitly cites visual evidence ensuring robust decision support. Respond in the same language as the user's question."
    # 新增的前缀
    cot_prefix = "Let's think step by step based on the image. "
    qwen_prefix = "You are a helpful and harmless assistant. You\nare Qwen developed by Alibaba Cloud. You can analyze images and answer\nquestions about them accurately. Please respond in the same language as the user's\nquestion. "
    
    baseline_prompts = []
    mpo_prompts = []
    cot_prompts = []
    qwen_prompts = []
    
    image_token = "<|vision_start|><|image_pad|><|vision_end|>\n"
    
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((512, 512))
        
        baseline_prompts.append({"text": image_token + raw_question, "image": img})
        mpo_prompts.append({"text": image_token + mpo_optimized_prefix + raw_question, "image": img})
        # 组装新增方法的 prompt
        cot_prompts.append({"text": image_token + cot_prefix + raw_question, "image": img})
        qwen_prompts.append({"text": image_token + qwen_prefix + raw_question, "image": img})
        
    # 3. 提取特征
    print(f"开始提取 Baseline 特征 (共 {len(img_paths)} 个样本)...")
    baseline_features = model.extract_hidden_states(baseline_prompts)
    
    print("开始提取 MPO 特征...")
    mpo_features = model.extract_hidden_states(mpo_prompts)
    
    print("开始提取 CoT 特征...")
    cot_features = model.extract_hidden_states(cot_prompts)
    
    print("开始提取 Qwen 特征...")
    qwen_features = model.extract_hidden_states(qwen_prompts)
    
    # 4. 保存特征文件
    save_dir = "./logs/pca_features"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "Baseline_features.npy"), baseline_features)
    np.save(os.path.join(save_dir, "MPO_features.npy"), mpo_features)
    # 保存新增方法的特征
    np.save(os.path.join(save_dir, "CoT_features.npy"), cot_features)
    np.save(os.path.join(save_dir, "Qwen_features.npy"), qwen_features)
    
    print(f"特征提取完毕，已保存至 {save_dir}。现在可以运行 plot_pca.py 了。")

if __name__ == "__main__":
    main()