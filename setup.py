from setuptools import setup

setup(
    name="modes",
    version="0.4.4",
    description="Photonic mode solver.",
    url="https://github.com/jtambasco/modesolverpy",
    author="Jean-Luc Tambasco",
    author_email="an.obscurity@gmail.com",
    license="MIT",
    install_requires=["tqdm", "scipy", "numpy", "opticalmaterialspy", "matplotlib"],
    packages=["modes"],
    include_package_data=True,
    zip_safe=False,
)
