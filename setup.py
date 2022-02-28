from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ITR',
    version='0.1',
    description='This package helps companies and financial institutions to assess the temperature alignment of current'
                'targets, commitments, and investment and lending portfolios.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ortec Finance',
    author_email='joris.cramwinckel@ortec-finance.com',
    packages=find_packages(),
    download_url = "https://pypi.org/project/ITR-Temperature-Alignment-Tool/",
    url="https://github.com/os-climate/ITR",
    project_urls={
        "Bug Tracker": "https://github.com/os-climate/ITR",
        "Documentation": 'https://github.com/os-climate/ITR',
        "Source Code": "https://github.com/os-climate/ITR",
    },
    keywords = ['Climate', 'ITR', 'Finance'],
    package_data={
        'SBTi': [],
    },
    include_package_data=True,
    install_requires=['pandas',
                      'xlrd',
                      'pydantic'],
    python_requires='>=3.8',
    extras_require={
        'dev': [
            'nose2',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering"

    ],
    test_suite='nose2.collector.collector',
)
