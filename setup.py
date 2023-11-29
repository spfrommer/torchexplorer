from setuptools import setup, find_namespace_packages, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='torchexplorer',
    version='0.4.1',
    description="Interactively inspect pytorch modules during training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spfrommer/torchexplorer",
    author="Samuel Pfrommer",
    author_email="sam.pfrommer@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'torch',
        'pygraphviz',
        'wandb',
        'flask',
        'numpy',
        'click',
        'loguru',
    ],
    extras_require = {
        'dev': [
            'torchvision',
            'lightning',
            'scikit-learn',
            'pytest',
            'pyperclip',
            'mypy',
            'twine',
        ]
    }
)
