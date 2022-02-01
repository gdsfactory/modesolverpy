from setuptools import setup

setup(
    name="modes",
    version="1.0.6",
    description="Photonic mode solver.",
    url="https://modes.readthedocs.io/en/latest/",
    author="Jean-Luc Tambasco",
    author_email="an.obscurity@gmail.com",
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "numpy",
        "opticalmaterialspy",
        "pytest",
        "scipy",
        "tqdm",
    ],
    packages=["modes"],
    include_package_data=True,
    zip_safe=False,
)
