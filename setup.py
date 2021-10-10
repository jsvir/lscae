import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lscae-jsvir",
    version="0.0.1",
    author="Jonathan Svirsky",
    author_email="js@alumni.technion.ac.il",
    description="Laplacian Score-regularized CAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jsvir/lscae",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=['torch',
                      'scikit-learn',
                      'omegaconf',
                      'scipy',
                      'matplotlib'],
    python_requires=">=3.7",
)
