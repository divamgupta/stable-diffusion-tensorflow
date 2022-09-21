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
    "--batch", type=int, default=1, help="number of images to generate"
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

args = parser.parse_args()

if args.mp:
    print("Using mixed precision.")
    keras.mixed_precision.set_global_policy("mixed_float16")

generator = Text2Image(img_height=args.H, img_width=args.W, jit_compile=False)
img = generator.generate(
    args.prompt,
    num_steps=args.steps,
    unconditional_guidance_scale=args.scale,
    temperature=1,
    batch_size=args.batch,
    seed=args.seed,
)

if(args.batch > 1):
    Image.fromarray(img[0]).save(args.output)
    print(f"saved at {args.output}")
else:
    split_filename = args.output.split(".")
    filename = split_filename[0:-1]
    extension = split_filename[-1]
    generate_filename = lambda x: f"{filename}-{x}.{extension}"
    for i in range(args.batch):
        filename = generate_filename(i + 1)
        Image.fromarray(img[i]).save(args.output)
    
    print(f"saved {args.batch} images as {generate_filename(f"1, {args.batch + 1}")}")

