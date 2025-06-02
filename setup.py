# setup.py
from setuptools import setup, find_packages

setup(
    name='medical_imaging',
    version='0.1',
    author='Tina Hou',
    author_email='xh363@cam.ac.uk',
    description='A package for PET-CT Reconstruction (Module1), MRI Denoising (Module2), and CT Segmentation & Radiomics (Module3).',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-image',
        'matplotlib',
        'tqdm',
        'nibabel',
        'pandas',
        'seaborn',
        'scikit-learn',
        'sphinx',
        'furo',
        'matplotlib',

    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
