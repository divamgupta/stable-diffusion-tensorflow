from execution.command_line_constants import *
from execution.generator_factory import make_stable_diffusion_model, run_generator
from execution.environment_configuration import configure_keras, set_log_level
from execution.persistence import save_image, load_image

import click


@click.command()
@click.option("--prompt", "-p", type=click.STRING, required=True, help=PROMPT_HELP)
@click.option("--output", "-o", type=click.STRING, default=DEFAULT_OUTPUT, help=OUTPUT_HELP)
@click.option("--negative-prompt", type=click.STRING, required=False, help=NEGATIVE_PROMPT_HELP)
@click.option("--height", "--H", "-H", type=click.INT, default=DEFAULT_HEIGHT, help=HEIGHT_HELP)
@click.option("--width", "--W", "-W", type=click.INT, default=DEFAULT_WIDTH, help=WIDTH_HELP)
@click.option("--scale", type=click.FLOAT, default=DEFAULT_SCALE, help=SCALE_HELP)
@click.option("--steps", type=click.INT, default=DEFAULT_STEPS, help=STEPS_HELP)
@click.option("--seed", type=click.INT, required=False, help=SEED_HELP)
@click.option("--mixed-precision", "--mp", type=click.BOOL, default=DEFAULT_MIXED_PRECISION, help=MIXED_PRECISION_HELP)
@click.option("--temperature", type=click.INT, default=DEFAULT_TEMPERATURE, help=TEMPERATURE_HELP)
@click.option("--batch-size", type=click.INT, default=DEFAULT_BATCH_SIZE, help=BATCH_SIZE_HELP)
@click.option("--input-image-path", "--input", "-i", type=click.STRING, required=False, help=INPUT_IMGE_IMAGE_HELP)
@click.option("--log-level", type=click.Choice(choices=AVAILABLE_LOGLEVELS, case_sensitive=False),
                             default=DEFAULT_LOGLEVEL, help=LOGLEVEL_HELP, envvar=LOGLEVEL_ENV_VAR)
def main(prompt: str, output: str, negative_prompt: str, height: int, width: int, scale: float, steps: int,
        seed: int, mixed_precision: bool, temperature: int, batch_size: int, log_level: str, input_image_path: str):
    set_log_level(log_level)
    configure_keras(mixed_precision)

    input_image = load_image(input_image_path, width, height)
    model = make_stable_diffusion_model(height, width)
    image = run_generator(model, prompt, steps, scale, temperature, batch_size, seed, negative_prompt, input_image)
    
    save_image(image, output, prompt, negative_prompt)

if __name__ == "__main__":
    main()
