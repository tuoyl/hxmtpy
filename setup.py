import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hxmtpy",
    version="0.0.1",
    author="Youli Tuo",
    author_email="tuoyl@ihep.ac.cn",
    description="A python software package for HXMT data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuoyl/hxmtpy",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "astropy",
        "matplotlib"
    ],
)   
