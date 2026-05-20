import setuptools


setuptools.setup(
    name="bgl2",
    version="0.1.0",
    description="bgl2 pipeline scheduler extension for Megatron-LM",
    packages=setuptools.find_namespace_packages(where="..", include=["bgl2", "bgl2.*"]),
    package_dir={"": ".."},
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
