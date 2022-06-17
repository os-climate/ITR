# ITR
This Python module implements the ITR methodology.


## Getting started for Contributors:
if you use Anaconda environments:
```
conda env create -f environment.yml
conda activate itr_env
```

For virtual environments:

```
python3 -m venv itr_env
source itr_env/bin/activate
pip install -r 'requirements.txt'
```

## Development
For development purposes, install the ITR package using the following command:
```bash
pip install -e .[dev]
``` 

If you want to work with notebooks from the examples folder please register the kernel from your conda environment such
it is avilable in Jupyter. Virtual environments will be available by default.

```
python -m ipykernel install --user --name=itr_env
```

## User Interface
![](ITR_demo.gif)