# Stable Diffusion in TensorFlow / Keras

A Keras / Tensorflow implementation of Stable Diffusion.

The weights were ported from the original implementation.

## Colab Notebooks

The easiest way to try it out is to use one of the Colab notebooks:

- [GPU Colab](https://colab.research.google.com/drive/1zVTa4mLeM_w44WaFwl7utTaa6JcaH1zK)
- [GPU Colab Img2Img](https://colab.research.google.com/drive/1gol0M611zXP6Zpggfri-fG8JDdpMEpsI?usp=sharing)
- [GPU Colab Inpainting](https://colab.research.google.com/drive/1Bf-bNmAdtQhPcYNyC-guu0uTu9MYYfLu)
- [GPU Colab + Mixed Precision](https://colab.research.google.com/drive/15mQgITh3e9HQMNys0zR8JN4R2vp06d-N)
  - ~10s generation time per image (512x512) on default Colab GPU without drop in quality
    ([source](https://twitter.com/fchollet/status/1571954014845308928))
- [TPU Colab](https://colab.research.google.com/drive/17zQOm_2Iu6pcP8otT-v6rx0D-pKgfaLm).
  - Slower than GPU for single-image generation, faster for large batch of 8+ images
    ([source](https://twitter.com/fchollet/status/1572004717362028546)).
- [GPU Colab with Gradio](https://colab.research.google.com/drive/1ANTUur1MF9DKNd5-BTWhbWa7xUBfCWyI)

## Example outputs

The following outputs have been generated using this implementation:

1. _A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution._

![a](https://user-images.githubusercontent.com/44222184/194685370-e87970f7-dbf5-4d6d-a9d1-31594cdf751a.png)

## Installation

### Install as a python package

Install using pip with the git repo:

```bash
pip install git+https://github.com/divamgupta/stable-diffusion-tensorflow
```

### Installing using the repo

Download the repo, either by downloading the
[zip](https://github.com/divamgupta/stable-diffusion-tensorflow/archive/refs/heads/master.zip)
file or by cloning the repo with git:

```bash
git clone git@github.com:divamgupta/stable-diffusion-tensorflow.git
```

#### Using pip without a virtual environment

Install dependencies using the `requirements.txt` file or the `requirements_m1.txt` file,:

```bash
pip install -r requirements.txt
```

#### Using a virtual environment with _virtualenv_

1. Create your virtual environment for `python3`:

   ```bash
   python3 -m venv venv
   ```

2. Activate your virtualenv:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies using the `requirements.txt` file or the `requirements_m1.txt` file,:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Using the Python interface

If you installed the package, you can use it as follows:

```python
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image

generator = StableDiffusion(
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

# for image to image :
img = generator.generate(
    "A Halloween bedroom",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
    input_image="/path/to/img.png"
)

# for inpainting :
# mask should be single channel image with 2 dims [W, H]
img = generator.generate(
    "A Halloween bedroom",
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
    input_image="/path/to/img.png",
    input_mask="/path/to/mask.png"
)

Image.fromarray(img[0]).save("output.png")
```

### Using `text2image.py` from the git repo

Assuming you have installed the required packages,
you can generate images from a text prompt using:

```bash
python text2image.py --prompt="An astronaut riding a horse"
```

The generated image will be named `output.png` on the root of the repo.
If you want to use a different name, use the `--output` flag.

```bash
python text2image.py --prompt="An astronaut riding a horse" --output="my_image.png"
```

Check out the `text2image.py` file for more options, including image size, number of steps, etc.

## References

1. https://github.com/CompVis/stable-diffusion
2. https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py
