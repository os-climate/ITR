from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ITR',
    version='1.0.1',
    description='Assess the temperature alignment of current targets, commitments, and investment '
                'and lending portfolios.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ortec Finance',
    author_email='joris.cramwinckel@ortec-finance.com',
    packages=find_packages(),
    download_url="https://pypi.org/project/ITR-Temperature-Alignment-Tool/",
    url="https://github.com/os-climate/ITR",
    project_urls={
        "Bug Tracker": "https://github.com/os-climate/ITR",
        "Documentation": 'https://github.com/os-climate/ITR',
        "Source Code": "https://github.com/os-climate/ITR",
    },
    keywords=['Climate', 'ITR', 'Finance'],
    package_data={
        'ITR': [],
    },
    include_package_data=True,
    install_requires=[
                      'iam-units>=2021.11.12',
                      'openpyxl>=3.0.9',
                      'openscm-units>=0.5.0',
                      'pandas>=1.4.2',
                      'pint>=0.18',
                      'pint-pandas>=0.2',
                      'pip>=22.0.3',
                      'pydantic>=1.8.2',
                      'setuptools>=60.9.3',
                      'wheel>=0.36.2',
                      'xlrd>=2.0.1',
                      ],
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering"

    ],
    test_suite='nose2.collector.collector',
)
