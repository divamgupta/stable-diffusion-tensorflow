import numpy as np
import logging
import logging
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Optional


def save_image(image_data: np.ndarray, output_path: str, prompt: str, negative_prompt: Optional[str]):
	pnginfo = PngInfo()
	pnginfo.add_text('prompt', prompt)

	if negative_prompt:
		pnginfo.add_text('negative_prompt', negative_prompt)
	
	image = Image.fromarray(image_data)
	image.save(output_path, pnginfo=pnginfo)
	logging.info(f"saved at {output_path}")


def load_image(image_path: Optional[str], width: int, height: int) -> Optional[np.ndarray]:
	if image_path:
		image = Image.open(image_path)
		logging.debug(f"Loaded input image from {image_path}")
		image = image.resize((width, height))
		logging.debug(f"Resizing input image to {width}x{height}")
		return image
	else:
		logging.debug("No input image given")
		return None
