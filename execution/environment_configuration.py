import logging
import tensorflow as tf
from tensorflow.python import keras

def configure_keras(mixed_precision: bool):
	if mixed_precision:
		logging.info("Using mixed precision.")
		keras.mixed_precision.set_global_policy("mixed_float16")


def set_log_level(log_level: str):
	logging.basicConfig(level=log_level)


def get_gpus():
	gpu_devies = tf.config.list_physical_devices('GPU')
	for gpu_device in gpu_devies:
		logging.debug(f"Available GPU devices found:")
		logging.debug(f"{gpu_device}")
