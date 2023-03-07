from setuptools import find_packages, setup
import platform, sys


with open('requirements.txt') as f: # default requirements
    install_requires = [i.strip() for i in f.read().splitlines()]

if platform.processor() == "arm" and sys.platform == "darwin": # check is mac chip m1
    with open('requirements_m1.txt') as f:
        install_requires = [i.strip() for i in f.read().splitlines()]

setup(
    name="stable_diffusion_tf",
    version="0.1",
    description="Stable Diffusion in Tensorflow / Keras",
    author="Divam Gupta",
    author_email="guptadivam@gmail.com",
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    install_requires=install_requires, # install requirements
    url="https://github.com/divamgupta/stable-diffusion-tensorflow",
    packages=find_packages(),
)
