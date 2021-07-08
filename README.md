# ITR
This Python module implements the ITR methodology.


## Getting started for Contributors:
if you use Anadonda environments:
```
conda env create -f environment.yml
conda activate itr_env
```

For virtual environments:
```
TODO:
```

## Development
For development purposes, install the SBTi package using the following command:
```bash
pip install -e .[dev]
``` 

If you want to work with notebooks from the examples folder pleasse register the kernel from your conda environment such it is avilable in Jupyter. Virtual environments will be available by default.

```
python -m ipykernel install --user --name=itr_env
```