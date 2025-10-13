import openai
from openai import OpenAI
from abc import ABC, abstractmethod
import base64
import os
from rich import print

MM_GENERATION_MODEL_CONFIG = {
    "gpt-image": {
        "class": "OpenAIImageGenerator",
        "full_name": "gpt-image-1",
        "target_modality": "image",
        "quality": "low",  # "medium",
        "response_format": None,
        "price": {"input": 5 / 1000000, "output": 40 / 1000000},
    },
    "gpt-image-medium": {
        "class": "OpenAIImageGenerator",
        "full_name": "gpt-image-1",
        "target_modality": "image",
        "quality": "medium",
        "response_format": None,
        "price": {"input": 5 / 1000000, "output": 40 / 1000000},
    },
}


class MMGenerator(ABC):
    def __init__(self, logger, **kwargs):
        self.logger = logger
        self.image_dir = os.path.abspath(os.path.join(logger.log_dir, "images"))
        self.target_modality = None  # "image" or "video" or "molecule"
        self.total_cost = 0

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    @abstractmethod
    def generate(self, prompt, **kwargs):
        """
        get text prompt and return multimodal path
        """
        pass

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


class OpenAIImageGenerator(MMGenerator):
    def __init__(
        self,
        mm_generator_model_name,
        openai_api_key,
        logger,
        image_size="1024x1024",
        **kwargs,
    ):
        super().__init__(logger=logger)
        self.client = OpenAI(api_key=openai_api_key)
        self.model_config = MM_GENERATION_MODEL_CONFIG[mm_generator_model_name]
        self.model_name = self.model_config["full_name"]
        self.target_modality = self.model_config["target_modality"]

        self.sampling_params = {
            "size": image_size,
            "quality": self.model_config["quality"],
        }
        if self.model_config["response_format"]:
            self.sampling_params["response_format"] = self.model_config["response_format"]

        self.price = self.model_config["price"]

        self.image_size = image_size
        self.total_cost = 0

    def save_b64_image(self, b64_string):
        """Save a base64 string as an image file"""
        if b64_string:
            # Get number of existing images
            existing_images = len([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
            image_path = os.path.join(self.image_dir, f"image_{existing_images+1}.jpg")

            with open(image_path, "wb") as f:
                f.write(base64.b64decode(b64_string))
            return image_path
        return None

    def generate(self, prompt, mm_prompt_path=None, **kwargs):
        try:
            if mm_prompt_path is not None:  # Edit
                response = self.client.images.edit(
                    model=self.model_name,
                    prompt=prompt,
                    image=[open(mm_prompt_path, "rb")],
                    **self.sampling_params,
                )
            else:
                response = self.client.images.generate(
                    model=self.model_name,
                    prompt=prompt,
                    **self.sampling_params,
                )

            b64_image = response.data[0].b64_json
            image_path = self.save_b64_image(b64_image)
            cost = self.calculate_cost(response)
            self.total_cost += cost
            self.logger.info(f"OPENAI IMAGE GENERATION COST: {cost:.4f} USD, TOTAL COST: {self.total_cost:.4f} USD")

            return image_path

        except openai.OpenAIError as e:
            self.logger.error(f"Error generating image: {e}")
            self.logger.info(f"Prompt: {prompt}")

            return None

    def multimodal_mixing(self, parents, mm_mix_prompt, **kwargs):
        try:
            response = self.client.images.edit(
                model=self.model_name,
                prompt=mm_mix_prompt,
                image=[open(parent.mm_prompt_path, "rb") for parent in parents if parent.mm_prompt_path is not None],
                **self.sampling_params,
            )

            b64_image = response.data[0].b64_json
            image_path = self.save_b64_image(b64_image)
            cost = self.calculate_cost(response)
            self.total_cost += cost
            self.logger.info(f"OPENAI IMAGE GENERATION COST: {cost:.4f} USD, TOTAL COST: {self.total_cost:.4f} USD")

            return image_path

        except openai.OpenAIError as e:
            self.logger.error(f"Error generating image: {e}")
            self.logger.info(f"Prompt: {mm_mix_prompt}")

            return None

    def calculate_cost(self, response):
        usage = response.usage
        if usage is None:
            return self.price["output"]
        else:
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cost = (input_tokens * self.price["input"]) + (output_tokens * self.price["output"])
            return cost
