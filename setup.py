from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="retail-investor-sentiment-analysis",
    version="0.1.0",
    author="Srishti Agarwal",
    author_email="Srishtiaggarwal67@gmail.com",
    description="Analyze the investor sentiments",
    packages=find_packages(),
    install_requires=requirements,  
    python_requires=">=3.10",
)