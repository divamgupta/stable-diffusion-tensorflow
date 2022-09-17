# Stable Diffusion in Tensorflow / Keras

This is a port of the stable diffusion in Tensorflow / Keras


## How to use

1) Using the command line 

2) Using the python interface

```
from stable_diffusion_tf.stable_diffusion import get_model, text2image
text_encoder, diffusion_model, decoder = get_model(512, 512, download_weights=True)

img = text2image("An astronaut riding a hourse" , 
	img_height=512, 
	img_width=512,  
	text_encoder=text_encoder, 
	diffusion_model=diffusion_model, 
	decoder=decoder
)

cv2.imwrite("/tmp/a.png" , img[0][... , ::-1])

```
