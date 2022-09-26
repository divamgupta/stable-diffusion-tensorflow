# Stable Diffusion in TensorFlow / Keras


![output](https://user-images.githubusercontent.com/17668390/192362513-7161bc59-aee8-4129-a8de-e5aef305f344.png)


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

1. Using the Python interface. If you installed the package, you can use it as follows:

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

```bash
python text2image.py --prompt="An astronaut riding a horse"

# specify output path
# python text2image.py --prompt="An astronaut riding a horse" --output="my_image.png"
```

Check out the `text2image.py` file for more options, including image size, number of steps, etc.

## Text-to-Image with Stable Diffusion

The following outputs have been generated using [KerasCV API](https://github.com/keras-team/keras-cv):

> "Ultra high definiton of an alien cat seahorse fursona, smooth edege, soften" \
  " autistic graphic designer, attractive fluffy".

![output](https://user-images.githubusercontent.com/17668390/192363741-c268b2d3-72b1-4ca9-b5e2-46f96a9365ae.png)


> "A epic and beautiful rococo werewolf drinking coffee, in a burning coffee shop. ultra-detailed."\
  " anime, pixiv, uhd 8k cryengine, octane render, smooth edege, soften."


![output (1)](https://user-images.githubusercontent.com/17668390/192364449-ed2cf988-bdb1-42d6-a544-96f7639e2928.png)

> "Ultra high definiton realistic alien city, ambient light, smooth edege, soften, parallel levitation world" \
  " over the pretty jungle highly detailed, vibrant, style ultra realistic, Sci-Fi "

![output (2)](https://user-images.githubusercontent.com/17668390/192364789-079eb6e8-f9a6-411f-b631-431d702f41e0.png)

> "3d illustration painting of under the sea steampunk planet, soften"

![output (3)](https://user-images.githubusercontent.com/17668390/192365102-eb53f6ca-2f58-4077-8225-0ca2e22e1b65.png)


## References

1) https://github.com/CompVis/stable-diffusion
2) https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
