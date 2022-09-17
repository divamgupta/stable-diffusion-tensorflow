from stable_diffusion_tf.stable_diffusion import get_model, text2image

text_encoder, diffusion_model, decoder = get_model(512, 512)


diffusion_model.load_weights("/tmp/diffusion_model.h5")
decoder.load_weights("/tmp/decoder.h5")
text_encoder.load_weights("/tmp/text_encoder.h5")



img = text2image("A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed. anime, pixiv, uhd 8k cryengine, octane render" , 
	img_height=512, 
	img_width=512,  
	text_encoder=text_encoder, 
	diffusion_model=diffusion_model, 
	decoder=decoder,  
	batch_size=1, 
	n_steps=25, 
	unconditional_guidance_scale = 7.5 , 
	temperature = 1
)

import cv2
cv2.imwrite("/tmp/a.png" , img[0][... , ::-1])
