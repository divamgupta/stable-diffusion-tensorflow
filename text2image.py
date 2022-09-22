from tensorflow import keras
from stable_diffusion_tf.stable_diffusion import Text2Image
import argparse
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render",
)

parser.add_argument(
    "--output",
    type=str,
    nargs="?",
    default="output.png",
    help="where to save the output image",
)

parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixels",
)

parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixels",
)

parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
    "--steps", type=int, default=50, help="number of ddim sampling steps"
)

parser.add_argument(
    "--seed",
    type=int,
    help="optionally specify a seed integer for reproducible results",
)

parser.add_argument(
    "--mp",
    default=False,
    action="store_true",
    help="Enable mixed precision (fp16 computation)",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="number of prompt/seed images to generate",
)

args = parser.parse_args()

if args.mp:
    print("Using mixed precision.")
    keras.mixed_precision.set_global_policy("mixed_float16")

prompts = [args.prompt for _ in range(args.batch_size)]
seeds = [args.seed for _ in range(args.batch_size)]

generator = Text2Image(img_height=args.H, img_width=args.W, jit_compile=False)
imgs = generator.generate(
    prompts,
    seeds,
    num_steps=args.steps,
    unconditional_guidance_scale=args.scale,
    temperature=1,
)
for i, img in enumerate(imgs):
    fname = f"{i:02d}.{args.output}"
    Image.fromarray(img).save(fname)
    print(f"saved at {fname}")
