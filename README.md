# FROG
Face Regression for Occlusion Gauging


## Running example

This will trial the naive model on a sample of 300 images (Train/Test/Val)

```bash
pixi shell 
CUDA_VISIBLE_DEVICES=0 python src/naive_runner.py -pdd data -r False -s False
```


## Setup

Install [pixi](https://pixi.sh/latest/)

Setup environment & run example

```bash
pixi shell
pixi run python src/helloworld.py
```

Pre-commit formatting & linting
```bash
pixi shell
ruff check --fix
ruff format
```

