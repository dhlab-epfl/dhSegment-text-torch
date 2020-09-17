#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="dh_segment_text",
    version="0.1.0",
    license="GPL",
    url="https://github.com/dhlab-epfl/dhSegment-text",
    description="",
    packages=find_packages(),
    project_urls={
        "Source Code": "https://github.com/dhlab-epfl/dhSegment-text",
    },
    install_requires=[
        "dh_segment_torch @ git+https://github.com/dhlab-epfl/dhSegment-torch.git@master"
    ],
    test_require=["pytest"],
    zip_safe=False,
)
