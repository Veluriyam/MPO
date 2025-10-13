import re
from pathlib import Path
from .model import get_language_model
from .tasks import BaseTask
from .utils import check_mm_type
from .model.mmgenerator import MMGenerator


### MPO: Cohesive Backpropagation
def get_multimodal_analysis_prompt(text_prompt, mm_prompt_path, example_prompt, modality="image"):
    before_image_prompt = f"""
You are a Prompt Failure Analysis Agent specialized in multimodal prompt optimization. Your task is to analyze the failure case of a Multimodal Large Language Model (MLLM) and identify the potential reasons in the prompt for the model's incorrect prediction. Based on the given input, output, and ground truth, analyze both the Text Prompt and the Image Prompt used in the task.

### Input Structure for MLLM:  
- Text Prompt: A task-specific textual instruction for the MLLM.
- Image Prompt: A reference image that supports task understanding.
- Input Query: The actual target instance (text, image, or both) on which the MLLM must generate an answer.

### Prompts:
- Text Prompt : {text_prompt}
- Image Prompt : 
""".strip()

    after_image_prompt = f"""
### Wrong Examples:
""".strip()

    after_example_prompt = f"""
### Output Format:
Text Prompt Analysis:
- Identify missing information, vague instructions, or ambiguous wording that could have misled the model.
- Explain how weaknesses in the Text Prompt may have contributed to the wrong output.
- Suggest specific improvements (e.g., clearer task definition, additional constraints, better examples) to help the model produce the correct answer.

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
    def __init__(self, optim_model_setting, mm_generator: MMGenerator, task: BaseTask, logger):
        self.model = get_language_model(optim_model_setting["model_name"])(**optim_model_setting)
        self.mm_generator = mm_generator
        self.mm_generator_modality = self.mm_generator.target_modality  # "image" or "video" or "molecule"
        self.task = task
        self.logger = logger

    def _clean_response(self, optim_response, tag_name):
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        matches = re.findall(pattern, optim_response, re.DOTALL)
        matches = [m.strip() for m in matches]
        return matches[0] if matches else "None"

    # MPO: Cohesive Backpropagation
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

    # MPO: Generation Operator
    def mpo_optim_generation(self, node, model_responses_num):
        examples = node.get_wrong_examples(model_responses_num)
        example_prompt = self.get_example_prompt(examples, is_response=True)

        # Failure Analysis
        analysis = self.mpo_failure_analysis(node, example_prompt)

        # Prompt Optimization
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

        # Generate MultiModal Data
        generated_mm_data = self.generate_mm(
            mm_condition_prompt,
            text_prompt=improved_text_prompt,
        )

        return improved_text_prompt, generated_mm_data

    def generate_mm(self, mm_condition_prompt: str, text_prompt: str = None) -> dict:

        generated_mm_path = self.mm_generator(mm_condition_prompt, text_prompt=text_prompt)

        return {
            "mm_condition_prompt": mm_condition_prompt,
            "mm_prompt_path": generated_mm_path,
        }

    # MPO: Edit Operator
    def mpo_optim_edit(self, node, model_responses_num):
        examples = node.get_wrong_examples(model_responses_num)
        example_prompt = self.get_example_prompt(examples, is_response=True)

        # Failure Analysis
        analysis = self.mpo_failure_analysis(node, example_prompt)

        # Prompt Optimization
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

        # Generate MultiModal Data
        generated_mm_data = self.edit_mm(
            mm_edit_prompt,
            mm_prompt_path=node.mm_prompt_path,
            text_prompt=improved_text_prompt,
        )

        return improved_text_prompt, generated_mm_data

    def edit_mm(self, mm_edit_prompt: str, mm_prompt_path, text_prompt: str = None):
        assert self.mm_generator_modality in ["image", "molecule"]
        generated_mm_path = self.mm_generator(mm_edit_prompt, mm_prompt_path=mm_prompt_path, text_prompt=text_prompt)

        return {
            "mm_condition_prompt": mm_edit_prompt,
            "mm_prompt_path": generated_mm_path,
        }

    # MPO: Mix Operator
    def mpo_optim_mix(self, parents, model_responses_num):
        analyses, example_prompts = [], []
        for parent in parents:
            examples = parent.get_wrong_examples(model_responses_num)
            example_prompt = self.get_example_prompt(examples, is_response=True)
            example_prompts.append(example_prompt)

            analysis = self.mpo_failure_analysis(parent, example_prompt)
            analyses.append(analysis)

        mix_prompt = get_multimodal_improvement_mix_prompt(
            parents=parents, analyses=analyses, example_prompts=example_prompts, modality=self.mm_generator_modality
        )

        response = self.model.generate(mix_prompt)

        self.log_information(mix_prompt, response)

        # Process the response
        improved_text_prompts = self._clean_response(response, "mixed_text_prompt")
        mm_mix_prompt = self._clean_response(response, f"{self.mm_generator_modality}_mixing_prompt")

        # Generate Multimodal Data
        generated_mm_data = self.mix_mm(parents, mm_mix_prompt)

        return improved_text_prompts, generated_mm_data

    def mix_mm(
        self,
        parents,
        mm_mix_prompt,
    ):
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
        for example in examples:
            example_string = self._get_example_string(example, is_response)
            example_content = [
                {"type": "text", "text": f"<Example>\n{self.task.get_query(example)}\n"},
                {"type": "text", "text": f"{example_string}\n</Example>\n"},
            ]
            example_prompt.extend(example_content)

        return example_prompt

    def _format_answer(self, example):
        answer = self.task.get_answer(example)
        if isinstance(answer, list):
            return ", ".join(map(str, answer))
        return str(answer)

    def _get_example_string(self, example, is_response=True):
        # Format example text content
        if is_response:
            example_string = f'Response: \n{example["response"]}\n\nModel answer: \n{example["model_answer"]}\n\nThe correct answer is : \n{self._format_answer(example)}'
        else:
            example_string = f"The Answer is \n{self._format_answer(example)}"
        return example_string
