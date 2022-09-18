from stable_diffusion_tf.stable_diffusion import Text2Image
import argparse
from PIL import Image


parser = argparse.ArgumentParser()

parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)

parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--steps",
    type=int,
    default=50,
    help="number of ddim sampling steps"
)
args = parser.parse_args()

generator = Text2Image(
    img_height=args.H, 
    img_width=args.W,
    batch_size=1,
    jit_compile=True)  # Also try jit_compile=False, might be faster on your hardware

img = generator.generate(
    args.prompt, 
	n_steps=args.steps, 
	unconditional_guidance_scale =args.scale, 
	temperature=1
)

Image.fromarray(img[0]).save("output.png")
print("saved at output.png")
