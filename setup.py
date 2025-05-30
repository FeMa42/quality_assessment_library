from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quality_assessment_library",
    version="0.1.0",
    author="Damian Boborzi",
    author_email="damian.boborzi@uni-a.de",
    description="A library for assessing the quality of 3D models, with a focus on 3D car models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FeMa42/quality_assessment_library",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
    "torch>=2.4",
    "torchvision>=0.19",
    "numpy>=1.26",
    "scipy>=1.14",
    "scikit-learn>=1.6",
    "Pillow>=10.4",
    "imageio>=2.34",
    "transformers>=4.45",
    "huggingface_hub>=0.30",
    "safetensors>=0.4", 
    "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33",
    "sentencepiece>=0.2.0",
    "open_clip_torch>=2.24",
    "onnxruntime==1.21.0",
    "pytorch_lightning>=2.2", 
    "torchmetrics>=1.7",
    "torchdata>=0.8", 
    "pytorch-fid>=0.3", 
    "protobuf>=4.25",
    "torch-fidelity>=0.3",
    "opencv-contrib-python>=4.10", 
    "rembg>=2.0",
    "tqdm>=4.66",
    "requests>=2.32", 
    "packaging>=24.1", 
    "sdata>=0.23",
    "image-reward>=1.5",
],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'car-quality-download-models=scripts.download_models:download_models',
        ],
    },
)
