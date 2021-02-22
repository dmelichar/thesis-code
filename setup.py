import setuptools, platform

with open("README.md", "r", encoding="utf_8") as fh:
    long_description = fh.read()

install_requires = [
    "gym==0.18.0",
    "Keras==2.3.1",
    "numpy==1.18.5",
    "pandas==1.1.5",
    "tensorflow==1.15.5",
    "h5py==2.10.0",
    "cloudpickle==1.6.0",
    "Box2D==2.3.10",
    "seaborn==0.11.1",
    "scipy==1.5.3",
    "statsmodels==0.12.1",
    "typer==0.3.2",
]

setuptools.setup(
    name="safe-agents",
    version="0.1.0",
    description="Implementation of Safe Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/danielmelichar/thesis-code",
    author="Daniel Melichar",
    author_email="daniel@melichar.xyz",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=install_requires,
)
