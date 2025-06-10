from setuptools import setup

setup(
    name="eqtempclust",
    version="1.0.1",
    description="Temporal clustering analysis via occupation probability",
    classifiers=[
        "License :: GPL License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Statistical Seismology :: Temporal Clustering",
    ],
    url="https://github.com/ebeauce/eqtempclust",
    author="Eric Beauce",
    author_email="",
    license="GPL",
    py_modules=["eqtempclust"],
    install_requires=["numpy"],
    include_package_data=True,
    zip_safe=False,
)
