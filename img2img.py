import argparse
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    required=True,
    help="the prompt to render",
)

parser.add_argument(
    "--negative-prompt",
    type=str,
    help="the negative prompt to use (if any)",
)

parser.add_argument(
    "--steps", 
    type=int, 
    default=50, 
    help="number of ddim sampling steps"
)

parser.add_argument(
    "--input",
    type=str,
    nargs="?",
    required=True,
    help="the input image filename",
)

parser.add_argument(
    "--output",
    type=str,
    nargs="?",
    default="img2img-out.jpeg",
    help="the output image filename",
)

args = parser.parse_args()

generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False,  # You can try True as well (different performance profile)
)

img = generator.generate(
    args.prompt,
    negative_prompt=args.negative_prompt,
    num_steps=args.steps,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
    input_image=args.input,
    input_image_strength=0.8
)
Image.fromarray(img[0]).save(args.output)
