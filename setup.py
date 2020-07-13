from setuptools import setup

setup(
    name="modes",
    version="1.0.2",
    description="Photonic mode solver.",
    url="https://github.com/jtambasco/modesolverpy",
    author="Jean-Luc Tambasco",
    author_email="an.obscurity@gmail.com",
    license="MIT",
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
