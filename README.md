# Stable Diffusion in TensorFlow / Keras

A Keras / Tensorflow implementation of Stable Diffusion.

This is a fork of [stable-diffusion-tensorflow](https://github.com/divamgupta/stable-diffusion-tensorflow)
created by @divamgupta. The weights were ported from the original implementation.


## Usage

1) Try it out with [this GPU Colab](https://colab.research.google.com/drive/1zVTa4mLeM_w44WaFwl7utTaa6JcaH1zK).

2) Using the command line :

```
python text2image.py --prompt="An astronaut riding a horse"
```

3) Using the python interface:

```
pip install git+https://github.com/fchollet/stable-diffusion-tensorflow
```

```python
from stable_diffusion_tf.stable_diffusion import Text2Image
from PIL import Image

generator = Text2Image( 
    img_height=512,
    img_width=512,
    jit_compile=False,
)
img = generator.generate(
    "An astronaut riding a horse",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
	batch_size=1,
)
Image.fromarray(img[0]).save("output.png")
```

## Example outputs 

The following outputs have been generated using the this implementation:

1) *A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed. anime, pixiv, uhd 8k cryengine, octane render*

![a](https://user-images.githubusercontent.com/1890549/190841598-3d0b9bd1-d679-4c8d-bd5e-b1e24397b5c8.png)


2) *Spider-Gwen Gwen-Stacy Skyscraper Pink White Pink-White Spiderman Photo-realistic 4K*

![a](https://user-images.githubusercontent.com/1890549/190841999-689c9c38-ece4-46a0-ad85-f459ec64c5b8.png)


3) *A vision of paradise, Unreal Engine*

![a](https://user-images.githubusercontent.com/1890549/190841886-239406ea-72cb-4570-8f4c-fcd074a7ad7f.png)


## References

1) https://github.com/CompVis/stable-diffusion
2) https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
