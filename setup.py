from setuptools import setup, find_packages

print(f"Found packages: {find_packages()}")

setup(
    name="augmedical",
    packages=find_packages(),
    description="this package helps with the augmentation of medical data",
    version="0.1",
    url="",
    author="Philipp Sodmann",
    author_email="sodmann_p@ukw.de",
    keywords=["Image augmentation", "medical data"],
    include_package_data=True,
    package_data={},
)
