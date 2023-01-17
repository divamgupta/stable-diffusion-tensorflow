PROMPT_HELP = "the prompt to render"

DEFAULT_OUTPUT = "output.png"
OUTPUT_HELP = "Path where to save the output image"

NEGATIVE_PROMPT_HELP = "the negative prompt to use (if any)"

DEFAULT_HEIGHT = 512
HEIGHT_HELP = "Image height, in pixels"

DEFAULT_WIDTH = 512
WIDTH_HELP = "Image width, in pixels"

DEFAULT_SCALE = 7.5
SCALE_HELP = "Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"

DEFAULT_STEPS = 50
STEPS_HELP = "Number of ddim sampling steps"

SEED_HELP = "Optionally specify a seed integer for reproducible results"

DEFAULT_MIXED_PRECISION = False
MIXED_PRECISION_HELP = "Enable mixed precision (fp16 computation)"

DEFAULT_TEMPERATURE = 1
TEMPERATURE_HELP = "Generator temperature"

DEFAULT_BATCH_SIZE = 1
BATCH_SIZE_HELP = "Batch size temperature"

INPUT_IMGE_IMAGE_HELP = "Path to input image"

DEFAULT_LOGLEVEL = "INFO"
LOGLEVEL_HELP = "Python Log level value"
AVAILABLE_LOGLEVELS = ["NOTSET", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
LOGLEVEL_ENV_VAR = "LOGLEVEL"
