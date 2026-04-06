import torch
import numpy as np
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

class HFModelForPCA:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        self.model.eval()

    def extract_hidden_states(self, prompts_with_images):
        """
        传入格式化好的输入，返回降维前的特征向量
        """
        hidden_states_list = []
        for item in prompts_with_images:
            text = item.get("text", "")
            image = item.get("image", None) # 需为PIL Image对象
            
            inputs = self.processor(
                text=[text], 
                images=[image] if image else None, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                # 关键：开启 output_hidden_states
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # 提取最后一层隐藏状态，并在Token维度做平均池化: (1, seq_len, hidden_dim) -> (hidden_dim,)
            last_layer_hidden = outputs.hidden_states[-1]
            avg_pool_feature = last_layer_hidden.mean(dim=1).squeeze().cpu().numpy()
            hidden_states_list.append(avg_pool_feature)
            
        return np.array(hidden_states_list)