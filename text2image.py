from stable_diffusion_tf.stable_diffusion import get_model, text2image
import argparse


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

text_encoder, diffusion_model, decoder = get_model(512, 512, download_weights=True)



img = text2image(args.prompt , 
	img_height=args.H, 
	img_width=args.W,  
	text_encoder=text_encoder, 
	diffusion_model=diffusion_model, 
	decoder=decoder,  
	batch_size=1, 
	n_steps=args.steps, 
	unconditional_guidance_scale =args.scale , 
	temperature = 1
)


from PIL import Image
Image.fromarray(img[0]).save("output.png")
print("saved at output.png")
