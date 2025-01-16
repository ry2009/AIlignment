from setuptools import setup, find_packages

setup(
    name="ml_environment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "psutil",
        "pytest",
        "pytest-asyncio"
    ],
    python_requires=">=3.8",
) 