import logging
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from tensorflow import keras
import numpy as np
from typing import Optional


def make_stable_diffusion_model(height: int, width: int) -> StableDiffusion:
	logging.debug(f"Creating stable diffusion model for images of dimension {width}x{height}")
	generator = StableDiffusion(img_height=height, img_width=width, jit_compile=False)	
	return generator


def run_generator(generator: StableDiffusion, prompt: str, steps: int, scale: float, temperature: int, batch_size: int,
									seed: int, negative_prompt: Optional[str], input_image: Optional[np.ndarray]) -> np.ndarray:
	
	logging.debug(f"Start running generation for prompt `{prompt}` with negative prompt `{negative_prompt}`")

	image = generator.generate(
			prompt,
			negative_prompt=negative_prompt,
			num_steps=steps,
			unconditional_guidance_scale=scale,
			temperature=temperature,
			input_image=input_image,
			batch_size=batch_size,
			seed=seed,
	)

	return image[0]
