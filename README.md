# Stable Diffusion in TensorFlow / Keras

**Stable Diffusion** is a latent text-to-image diffusion model. This is an unofficial implementation in `TensorFlow 2 (Keras)`. The weights were ported from the [original](https://github.com/CompVis/stable-diffusion) implementation. Stable Diffusion model is now available in [KerasCV](https://github.com/keras-team/keras-cv).

## Quick Start

The easiest way to try it out is to use one of the Colab notebooks:


- [GPU Colab](https://colab.research.google.com/drive/1zVTa4mLeM_w44WaFwl7utTaa6JcaH1zK)
- [GPU Colab + Mixed Precision](https://colab.research.google.com/drive/15mQgITh3e9HQMNys0zR8JN4R2vp06d-N)
  - ~10s generation time per image (512x512) on default Colab GPU without drop in quality
    ([source](https://twitter.com/fchollet/status/1571954014845308928))
- [TPU Colab](https://colab.research.google.com/drive/17zQOm_2Iu6pcP8otT-v6rx0D-pKgfaLm).
  - Slower than GPU for single-image generation, faster for large batch of 8+ images
    ([source](https://twitter.com/fchollet/status/1572004717362028546)).
- [GPU Colab with Gradio](https://colab.research.google.com/drive/1ANTUur1MF9DKNd5-BTWhbWa7xUBfCWyI)
- [Multi-GPU Inference](https://colab.research.google.com/drive/1CdWmT9CNF_L2XjCERv8gX8cq-PgzT2qZ?usp=sharing) - with [KerasCV](https://github.com/keras-team/keras-cv) API.



## Requirements

It's recommened to create [conda](https://docs.conda.io/en/latest/) environment.

```
conda env create -n std_model
conda activate std_model
```

Next, clone the repo and install necessary packages from `requirements.txt` or the `requirements_m1.txt` file.

```
git clone https://github.com/divamgupta/stable-diffusion-tensorflow.git
pip install -r requirements.txt
```

## Usage

There are 2 ways to run the code.

1. Using the Python interface

If you installed the package, you can use it as follows:

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

2. Using command line interface as follows:

Assuming you have installed the required packages, 
you can generate images from a text prompt using:

```bash
python text2image.py --prompt="An astronaut riding a horse"

# specify output path
# python text2image.py --prompt="An astronaut riding a horse" --output="my_image.png"
```

Check out the `text2image.py` file for more options, including image size, number of steps, etc.

## Example outputs 

The following outputs have been generated using this implementation:

1) *A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed. anime, pixiv, uhd 8k cryengine, octane render*

![a](https://user-images.githubusercontent.com/1890549/190841598-3d0b9bd1-d679-4c8d-bd5e-b1e24397b5c8.png)


2) *Spider-Gwen Gwen-Stacy Skyscraper Pink White Pink-White Spiderman Photo-realistic 4K*

![a](https://user-images.githubusercontent.com/1890549/190841999-689c9c38-ece4-46a0-ad85-f459ec64c5b8.png)


3) *A vision of paradise, Unreal Engine*

![a](https://user-images.githubusercontent.com/1890549/190841886-239406ea-72cb-4570-8f4c-fcd074a7ad7f.png)


## References

1) https://github.com/CompVis/stable-diffusion
2) https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
