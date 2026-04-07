import re
from pathlib import Path
from .model import get_language_model
from .tasks import BaseTask
from .utils import check_mm_type
from .model.mmgenerator import MMGenerator
import os
# 导入新增的 RAG 模块
from .rag_utils import RAGModule

def get_image_feature_extraction_prompt(image_path):
    text_prompt = """Please analyze the image and extract ONLY the core biological entity and its visual features.

Task Requirements:
1. Entity: Identify the main species or object.
2. Features: List its distinctive colors and specific body parts.
3. STRICT EXCLUSION: You must completely ignore the environment. Do not describe the background, branches, leaves, weather, or sky.

Output Format:
Provide ONLY a comma-separated list of core entities and visual keywords. Do not write full sentences. Do not include conversational filler.

Example Output:
Yellow-billed Cuckoo, yellow beak, white underbelly, brown wings"""
    
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image", "image": image_path}
            ],
        },
    ]
    return prompt

### MPO: Cohesive Backpropagation
def get_multimodal_analysis_prompt(text_prompt, mm_prompt_path, example_prompt, modality="image"):
    before_image_prompt = f"""
You are a Prompt Failure Analysis Agent specialized in multimodal prompt optimization. Your task is to analyze the failure case of a Multimodal Large Language Model (MLLM) and identify the potential reasons in the prompt for the model's incorrect prediction. Based on the given input, output, and ground truth, analyze both the Text Prompt and the Image Prompt used in the task.

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Auxiliary Diagnostic Information(For your analysis ONLY, the MLLM did not see this):
- <Image_Features>: Textual extraction of the visual elements in the wrong example's image.
- <Auxiliary_Knowledge>: External domain knowledge retrieved to help you understand the core concepts.

### Prompts:
- Text Prompt : {text_prompt}
- Image Prompt : 
""".strip()

    after_image_prompt = f"""
### Wrong Examples:
""".strip()

    # === 修改处：增加要求优化器利用辅助知识的说明 ===
    after_example_prompt = f"""
### Output Format:
Text Prompt Analysis:
- Identify missing information, vague instructions, or ambiguous wording that could have misled the model.
- Explain how weaknesses in the Text Prompt may have contributed to the wrong output.
- Utilize the provided <Auxiliary_Knowledge> to understand domain-specific concepts, and suggest how to incorporate this key knowledge into the improved Text Prompt.

Image Prompt Analysis:
- If a image Prompt was used, analyze its effectiveness.
- Identify problems such as lack of clarity, poor composition, irrelevant details, or missing key features.
- If no image Prompt was used, suggest what kind of image (visual content, attributes, composition) would help correct the failure.
""".strip()

    mm_prompt_modality = check_mm_type(mm_prompt_path) if mm_prompt_path else "text"
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before_image_prompt},
                (
                    {"type": mm_prompt_modality, mm_prompt_modality: mm_prompt_path}
                    if mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_image_prompt},
                *example_prompt,
                {"type": "text", "text": after_example_prompt},
            ],
        },
    ]

    return prompt


### MPO Generation
def get_multimodal_generation_prompt(text_prompt, mm_prompt_path, example_prompt, analysis, modality="image"):
    before_image_prompt = f"""
You are a Prompt-Improvement Agent specializing in multimodal prompt optimization. Your task is to design improved prompts for both image generation and text instruction, aimed at enhancing the performance of Multimodal Large Language Model (MLLM).

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Auxiliary Diagnostic Information(For your analysis ONLY, the MLLM did not see this):
- <Image_Features>: Textual extraction of the visual elements in the wrong example's image.
- <Auxiliary_Knowledge>: External domain knowledge retrieved to help you understand the core concepts.

### Provided Material
- Text Prompt: {text_prompt}
- Image Prompt: 
""".strip()
    after_image_prompt = f"""
- Wrong Examples: 
""".strip()
    after_example_prompt = f"""    
- Failure Analysis: {analysis}
### Your Task
Your task is review the failure analysis carefully to understand the issues and create two improved prompts that directly address the issues in the failure analysis:
1. Image Generation Prompt
   - Write a detailed prompt for an image generator.
   - Enhance or redesign the reference image to resolve issues found in the analysis.
   - Ensure the image highlights critical visual features necessary for success.
    - If no reference image is provided, suggest an appropriate one based on the failure analysis.

2. Improved Text Prompt
   - Write a clear, concise, and unambiguous instruction for the MLLM.
   - Resolve ambiguities found in the failure analysis.
   - Elaborate how the reference image should be interpreted.
""".strip()
    output_format = """
### Output Format
<image_generation_prompt>{image_generation_prompt}</image_generation_prompt>
<improved_text_prompt>{improved_text_prompt}</improved_text_prompt>
""".strip()
    mm_prompt_modality = check_mm_type(mm_prompt_path) if mm_prompt_path else "text"
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before_image_prompt},
                (
                    {"type": mm_prompt_modality, mm_prompt_modality: mm_prompt_path}
                    if mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_image_prompt},
                *example_prompt,
                {"type": "text", "text": after_example_prompt},
                {"type": "text", "text": output_format},
            ],
        },
    ]

    return prompt


### MPO Edit
def get_multimodal_edit_prompt(text_prompt, mm_prompt_path, example_prompt, analysis, modality="image"):
    before_image_prompt = f"""
You are a Prompt-Improvement Agent specializing in multimodal prompt optimization, with a focus on prompt editing. Your task is to design improved prompts for both image editing and text instruction, aimed at enhancing the performance of Multimodal Large Language Model (MLLM).

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Auxiliary Diagnostic Information(For your analysis ONLY, the MLLM did not see this):
- <Image_Features>: Textual extraction of the visual elements in the wrong example's image.
- <Auxiliary_Knowledge>: External domain knowledge retrieved to help you understand the core concepts.

### Provided Material
- Text Prompt: {text_prompt}
- Image Prompt: 
""".strip()
    after_image_prompt = f"""
- Wrong Examples: 
""".strip()
    after_example_prompt = f"""
- Failure Analysis: {analysis}
### Your Task
Your task is review the failure analysis carefully to understand the issues and create two improved prompts that directly address the issues in the failure analysis:
1. Image Editing Prompt:
   - Write a precise and context-aware prompt instructing the image editor to modify the given reference image.
   - Specify which visual components (e.g., objects, colors, textures, lighting, perspective, composition) should be added, removed, or replaced based on the failure analysis.
   - Clearly identify any undesirable visual elements that led to the failure.
   - Guide the editor on how to retain key features, proportions, or stylistic elements that are critical to the intended outcome.

2. Improved Text Prompt
   - Write a clear, concise, and unambiguous instruction for the MLLM.
   - Resolve ambiguities found in the failure analysis.
   - Elaborate how the reference image should be interpreted.
""".strip()
    output_format = """
### Output Format
<image_edit_prompt>{image_edit_prompt}</image_edit_prompt>
<improved_text_prompt>{improved_text_prompt}</improved_text_prompt>
""".strip()

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before_image_prompt},
                (
                    {"type": "image", "image": mm_prompt_path}
                    if mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_image_prompt},
                *example_prompt,
                {"type": "text", "text": after_example_prompt},
                {"type": "text", "text": output_format},
            ],
        },
    ]
    return prompt


# MPO Mix
def get_multimodal_improvement_mix_prompt(parents, analyses, example_prompts, modality="image"):
    assert len(parents) == 2 and len(analyses) == 2 and len(example_prompts) == 2
    before_imageA_prompt = f"""
You are a Prompt-Improvement Agent specializing in multimodal prompt optimization, with a focus on cross-prompt fusion. Your task is to create improved, mixed prompts for both image prompt and text instruction, aimed at enhancing the performance of Multimodal Large Language Model (MLLM).

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Auxiliary Diagnostic Information(For your analysis ONLY, the MLLM did not see this):
- <Image_Features>: Textual extraction of the visual elements in the wrong example's image.
- <Auxiliary_Knowledge>: External domain knowledge retrieved to help you understand the core concepts.

### Provided Material
#### Prompt A
- Text Prompt A: {parents[0].instruction}
- Image Prompt A:
"""

    after_imageA_prompt = f"""
- Wrong Examples from Prompt A: 
""".strip()

    after_imageA_examples_prompt = f"""
- Failure Analysis for Prompt A: {analyses[0]}

#### Prompt B
- Text Prompt B: {parents[1].instruction}
- Image Prompt B: 
""".strip()
    after_imageB_prompt = f"""
- Wrong Examples from Prompt B: 
""".strip()
    after_imageB_examples_prompt = f"""
- Failure Analysis for Prompt B: {analyses[1]}

### Your Task
Your task is review the failure analysis carefully to understand the issues and create two improved prompts that directly address the issues in the failure analysis:
1. Image Mixing Prompt:
    - Write a guidance for the image generator to combine and improve both reference images.
    - Address visual issues identified in both failure analyses.
    - Guide the model to create a new hybrid image that merges key beneficial visual features from both references while mitigating their weaknesses.
    - Explicitly state which visual elements from each image should be retained, modified, or discarded to achieve task success.

2. Improved Text Prompt
   - Write a clear, concise, and unambiguous instruction for the MLLM.
   - Incorporate key visual or task-relevant features identified in both failure analysis.
   - Explain how the reference image should be used to assist the task.
""".strip()
    output_format = """
### Output Format
<image_mixing_prompt>{image_mixing_prompt}</image_mixing_prompt>
<mixed_text_prompt>{mixed_text_prompt}</mixed_text_prompt>
""".strip()

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before_imageA_prompt},
                (
                    {"type": "image", "image": parents[0].mm_prompt_path}
                    if parents[0].mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_imageA_prompt},
                *example_prompts[0],
                {"type": "text", "text": after_imageA_examples_prompt},
                (
                    {"type": "image", "image": parents[1].mm_prompt_path}
                    if parents[1].mm_prompt_path
                    else {"type": "text", "text": "<No image provided>"}
                ),
                {"type": "text", "text": after_imageB_prompt},
                *example_prompts[1],
                {"type": "text", "text": after_imageB_examples_prompt},
                {"type": "text", "text": output_format},
            ],
        },
    ]

    return prompt


class OptimizationModel:
    # 增加 log_dir 参数
    def __init__(self, optim_model_setting, mm_generator: MMGenerator, task: BaseTask, logger, log_dir):
        self.model = get_language_model(optim_model_setting["model_name"])(**optim_model_setting)
        self.mm_generator = mm_generator
        self.mm_generator_modality = self.mm_generator.target_modality
        self.task = task
        self.logger = logger
        
        # 拼接 RAG 日志文件的绝对路径
        rag_log_path = os.path.join(log_dir, "rag_retrieval_scores.jsonl")
        
        # === 初始化 RAG 模块时传入 log_file ===
        self.rag_module = RAGModule(
            index_path="/workspace/yp/MPO/datasets/rag_index/e5_Flat.index", 
            corpus_path="/workspace/yp/MPO/datasets/FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl",
            model_name="/workspace/yp/MPO/datasets/intfloat_e5-base-v2",
            log_file=rag_log_path  # 新增此行
        )

    def _clean_response(self, optim_response, tag_name):
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        matches = re.findall(pattern, optim_response, re.DOTALL)
        matches = [m.strip() for m in matches]
        return matches[0] if matches else "None"

    def mpo_failure_analysis(self, node, example_prompt):
        analysis_prompt = get_multimodal_analysis_prompt(
            node.instruction,
            node.mm_prompt_path,
            example_prompt,
            modality=self.mm_generator_modality,
        )
        analysis = self.model.generate(analysis_prompt)
        self.log_information(analysis_prompt, analysis)
        return analysis

    def mpo_optim_generation(self, node, model_responses_num):
        examples = node.get_wrong_examples(model_responses_num)

        # --- 新增：提取并记录当前用到的原始错图路径 ---
        wrong_image_paths = [self.task.get_mm_path(ex) for ex in examples]

        example_prompt = self.get_example_prompt(examples, is_response=True)
        analysis = self.mpo_failure_analysis(node, example_prompt)
        generate_prompt = get_multimodal_generation_prompt(
            text_prompt=node.instruction,
            mm_prompt_path=node.mm_prompt_path,
            example_prompt=example_prompt,
            analysis=analysis,
            modality=self.mm_generator_modality,
        )
        response = self.model.generate(generate_prompt)
        self.log_information(generate_prompt, response)

        improved_text_prompt = self._clean_response(response, "improved_text_prompt")
        mm_condition_prompt = self._clean_response(response, f"{self.mm_generator_modality}_generation_prompt")

        generated_mm_data = self.generate_mm(mm_condition_prompt, text_prompt=improved_text_prompt)

        # --- 新增：在日志中打印映射关系 ---
        self.logger.info(f"👉 [Mapping] Generated Reference Image: {generated_mm_data['mm_prompt_path']}")
        self.logger.info(f"👉 [Mapping] Based on Original Wrong Images: {wrong_image_paths}\n")

        return improved_text_prompt, generated_mm_data

    def generate_mm(self, mm_condition_prompt: str, text_prompt: str = None) -> dict:
        generated_mm_path = self.mm_generator(mm_condition_prompt, text_prompt=text_prompt)
        return {
            "mm_condition_prompt": mm_condition_prompt,
            "mm_prompt_path": generated_mm_path,
        }

    def mpo_optim_edit(self, node, model_responses_num):
        examples = node.get_wrong_examples(model_responses_num)

        wrong_image_paths = [self.task.get_mm_path(ex) for ex in examples] # 新增

        example_prompt = self.get_example_prompt(examples, is_response=True)
        analysis = self.mpo_failure_analysis(node, example_prompt)
        edit_prompt = get_multimodal_edit_prompt(
            text_prompt=node.instruction,
            mm_prompt_path=node.mm_prompt_path,
            example_prompt=example_prompt,
            analysis=analysis,
            modality=self.mm_generator_modality,
        )

        response = self.model.generate(edit_prompt)
        self.log_information(edit_prompt, response)

        improved_text_prompt = self._clean_response(response, "improved_text_prompt")
        mm_edit_prompt = self._clean_response(response, f"{self.mm_generator_modality}_edit_prompt")

        generated_mm_data = self.edit_mm(mm_edit_prompt, mm_prompt_path=node.mm_prompt_path, text_prompt=improved_text_prompt)
        # --- 新增日志 ---
        self.logger.info(f"👉 [Mapping] Edited Reference Image: {generated_mm_data['mm_prompt_path']}")
        self.logger.info(f"👉 [Mapping] Based on Original Wrong Images: {wrong_image_paths}\n")
        return improved_text_prompt, generated_mm_data

    def edit_mm(self, mm_edit_prompt: str, mm_prompt_path, text_prompt: str = None):
        assert self.mm_generator_modality in ["image", "molecule"]
        generated_mm_path = self.mm_generator(mm_edit_prompt, mm_prompt_path=mm_prompt_path, text_prompt=text_prompt)
        return {
            "mm_condition_prompt": mm_edit_prompt,
            "mm_prompt_path": generated_mm_path,
        }

    def mpo_optim_mix(self, parents, model_responses_num):
        analyses, example_prompts = [], []
        wrong_image_paths_all = [] # 新增

        for parent in parents:
            examples = parent.get_wrong_examples(model_responses_num)
            wrong_image_paths_all.extend([self.task.get_mm_path(ex) for ex in examples]) # 新增
            example_prompt = self.get_example_prompt(examples, is_response=True)
            example_prompts.append(example_prompt)
            analysis = self.mpo_failure_analysis(parent, example_prompt)
            analyses.append(analysis)

        mix_prompt = get_multimodal_improvement_mix_prompt(
            parents=parents, analyses=analyses, example_prompts=example_prompts, modality=self.mm_generator_modality
        )

        response = self.model.generate(mix_prompt)
        self.log_information(mix_prompt, response)

        improved_text_prompts = self._clean_response(response, "mixed_text_prompt")
        mm_mix_prompt = self._clean_response(response, f"{self.mm_generator_modality}_mixing_prompt")

        generated_mm_data = self.mix_mm(parents, mm_mix_prompt)

        # --- 新增日志 ---
        self.logger.info(f"👉 [Mapping] Mixed Reference Image: {generated_mm_data['mm_prompt_path']}")
        self.logger.info(f"👉 [Mapping] Based on Original Wrong Images: {wrong_image_paths_all}\n")

        return improved_text_prompts, generated_mm_data

    def mix_mm(self, parents, mm_mix_prompt):
        generated_mm_path = self.mm_generator.multimodal_mixing(
            parents=parents,
            mm_mix_prompt=mm_mix_prompt,
        )
        return {
            "mm_condition_prompt": mm_mix_prompt,
            "mm_prompt_path": generated_mm_path,
        }

    def log_information(self, generate_prompt, response: str) -> None:
        self.logger.info("=" * 80)
        total_prompt = ""
        for role_content in generate_prompt:
            if isinstance(role_content["content"], list):
                for item in role_content["content"]:
                    item_type = item.get("type")
                    if item_type == "text":
                        total_prompt += f'{item["text"]}\n'
                    elif item_type in {"image", "video"}:
                        abs_path = Path(item[item_type]).resolve()
                        total_prompt += f"{abs_path}\n"
                    elif item_type == "molecule":
                        total_prompt += f"{item['molecule']['smiles'][0]}\n"
            else:
                total_prompt += f'{role_content["role"]}\n{role_content["content"]}\n'

        self.logger.info(f"{total_prompt}\n{'-' * 80}\n{response}\n\n")

    def get_example_prompt(self, examples, is_response=True):
        example_prompt = []
        # 使用 enumerate 增加索引 i，方便在日志中区分样本
        for i, example in enumerate(examples):
            example_string = self._get_example_string(example, is_response)
            
            mm_path = self.task.get_mm_path(example)
            original_query = self.task.get_query(example)
            
            # --- 新增日志：标记当前处理的错误样本 ---
            self.logger.info(f"--- Processing Wrong Example {i+1} ---")
            self.logger.info(f"Image Path: {mm_path}")
            
            image_features_text = ""
            features = ""
            if mm_path:
                features = self.extract_image_features(mm_path)
                if features:
                    image_features_text = f"<Image_Features>\n{features}\n</Image_Features>\n"
                    # --- 新增日志：记录该样本提取的特征 ---
                    self.logger.info(f"[Example {i+1}] Extracted Features: {features}")
            
            # === RAG 知识召回及封装 ===
            # E5 模型要求在查询端添加 "query: " 前缀
            # 仅使用视觉特征进行召回，不再拼接 original_query
            rag_query = f"query: {features}" if features else ""
            
            # 如果没有提取到特征（rag_query 为空），则跳过召回以节省时间
            retrieved_knowledge = self.rag_module.retrieve(rag_query, top_k=5) if rag_query else ""

            knowledge_text = ""
            if retrieved_knowledge.strip():
                knowledge_text = f"<Auxiliary_Knowledge>\n{retrieved_knowledge}\n</Auxiliary_Knowledge>\n"
                # --- 新增日志：记录该样本召回的知识 ---
                self.logger.info(f"[Example {i+1}] Retrieved Knowledge: {retrieved_knowledge}")
            
            # 整合到 example prompt 中 (严格保持原样，不修改发给MLLM的内容)
            example_content = [
                {"type": "text", "text": f"<Example>\n"},
                {"type": "text", "text": f"--- Original Input Seen by MLLM ---\n<Query>\n{original_query}\n</Query>\n"},
                {"type": "text", "text": f"--- Auxiliary Info For Your Analysis (The MLLM did NOT see this) ---\n{image_features_text}{knowledge_text}"},
                {"type": "text", "text": f"--- Model Performance ---\n{example_string}\n</Example>\n"},
            ]
            example_prompt.extend(example_content)

        return example_prompt

    def _format_answer(self, example):
        answer = self.task.get_answer(example)
        if isinstance(answer, list):
            return ", ".join(map(str, answer))
        return str(answer)

    def _get_example_string(self, example, is_response=True):
        if is_response:
            example_string = f'Response: \n{example["response"]}\n\nModel answer: \n{example["model_answer"]}\n\nThe correct answer is : \n{self._format_answer(example)}'
        else:
            example_string = f"The Answer is \n{self._format_answer(example)}"
        return example_string
    
    def extract_image_features(self, image_path):
        if not image_path:
            return ""
        
        mm_type = check_mm_type(image_path)
        if mm_type != "image":
            return ""
        
        feature_prompt = get_image_feature_extraction_prompt(image_path)
        self.logger.info(f"Extracting image features with prompt:\n{feature_prompt}\n")
        description = self.model.generate(feature_prompt)
        return description