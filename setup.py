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
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "Pillow>=8.0.0",
        "transformers>=4.20.0",
        "huggingface_hub>=0.10.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'car-quality-download-models=scripts.download_models:download_models',
        ],
    },
)
