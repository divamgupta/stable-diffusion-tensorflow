from setuptools import find_packages, setup

setup(
    name="stable_diffusion_tf",
    version="0.1",
    description="Stable Diffusion in Tensorflow / Keras",
    author="Divam Gupta",
    author_email="guptadivam@gmail.com",
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    url="https://github.com/divamgupta/stable-diffusion-tensorflow",
    packages=find_packages(),
    install_requires=[
        "tensorflow-gpu==2.10.0",
        "h5py==3.7.0",
        "Pillow==9.2.0",
        "tqdm==4.64.1",
        "ftfy==6.1.1",
        "regex==2022.9.13",
        "tensorflow-addons==0.17.1",

        "click~=8.1.3"
    ],
    entry_points = {
        'console_scripts': [
            'text2image=execution.main:main',
            'image2image=execution.main:main',
            'stable-diffusion=execution.main:main',
        ],
    }
)
