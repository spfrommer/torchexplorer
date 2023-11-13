from setuptools import setup, find_namespace_packages

setup(
    name='torchexplorer',
    packages=find_namespace_packages(include=['torchexplorer.*', 'lib.*']),
    version='0.1',
    install_requires=[
        'click',
        'colored_traceback',
        'colorama',
        'plotly',
        'tqdm',
        'dacite',
        'pytest',
        'pygraphviz',
        'pyperclip',

        'torch',
        'torchvision',

        'jaxtyping',
        'beartype',
        'einops',
        'wandb',

        'numpy',
        'scikit-learn',
        'flask'
    ])
