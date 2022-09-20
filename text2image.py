from tensorflow import keras
from stable_diffusion_tf.stable_diffusion import Text2Image
import argparse
from PIL import Image
import os
from os.path import exists, isdir;

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--prompt", type=str, nargs="?", default="post-apocalyptic wasteland, Thomas Kinkade, Colorful",
    help="The prompt used to generate images")
parser.add_argument("-o", "--output", type=str, nargs="?", default="output",
    help="The prefix for the image file - will be appended with a number based on the number of copies")
parser.add_argument("-H", "--height", type=int, default=512, help="Image height, in pixels")
parser.add_argument("-W", "--width", type=int, default=512, help="Image width, in pixels")
parser.add_argument("-s", "--scale", type=float, default=7.5,
    help="Guidance scale. How closely the final output adheres to the prompt. The lower the value, the less adherence to the prompt. "
         "Generally 0 to 15 or so but can go lower and higher. ")
parser.add_argument("-t", "--steps", type=int, default=50, help="The number of DDIM sampling steps")
parser.add_argument("-c", "--copies", type=int, default=1, help="The number of image copies to be created")
parser.add_argument("-r", "--seed", type=int, help="The seed value to use for repeating a previous image generation - optional")
parser.add_argument("--mp", default=False, action="store_true", help="Enable mixed precision (fp16 computation)")

args = parser.parse_args()
if args.mp:
    print("Using mixed precision.")
    keras.mixed_precision.set_global_policy("mixed_float16")

# Create output folder if it doens't exist
if not isdir('output'):
    os.mkdir('output')

generator = Text2Image(img_height=args.height, img_width=args.width, jit_compile=False)

images = generator.generate(
    args.prompt,
    num_steps=args.steps,
    unconditional_guidance_scale=args.scale,
    temperature=1,
    batch_size=args.copies,
    seed=args.seed,
)
# Save results
for img in images:
    i = 0
    fn = f'output/{args.output}_{i}.png'
    while exists(fn):
        i += 1
        fn = f'output/{args.output}_{i}.png'
    Image.fromarray(img).save(fn)
print(f'Completed')