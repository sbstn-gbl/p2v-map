# Code for P2V-MAP

## Content

<!-- vim-markdown-toc GFM -->

  * [Background](#background)
  * [Repository content](#repository-content)
  * [System requirements](#system-requirements)
    * [Makefile targets](#makefile-targets)
    * [Step-by-step instructions](#step-by-step-instructions)
* [Visualization of training results](#visualization-of-training-results)
* [Contact](#contact)

<!-- vim-markdown-toc -->

## Background

The code in this repository implements P2V-MAP as proposed in

> Gabel S, Guhl D, and Klapper D (2019) P2V-MAP: Mapping Market Structure for Large Assortments. *Journal of Marketing Research*, 56(4), 557-580

Please cite the paper if you use this software. Please note that all code and data is
provided here as is and without any warranty or certification of any kind, express or
implied. The original code was written in TensorFlow.

Note that I built this repository for teaching, so the implementation emphasizes
simplicity and not code performance.

## Repository content

```
.
├── Makefile                  # run `make help` to see make targets
├── README.md                 # this readme file
├── requirements.txt          # virtualenv requirements file
├── p2vmap.ipynb              # notebook that runs P2V-MAP
├── p2vmap_config.yaml        # P2V-MAP config
├── p2vmap_lib.py             # P2V-MAP code library
├── data                      # sample data
└── source                    # sources, e.g., images for notebooks
```

## System requirements

- `Python3`
- `virtualenv`

Tested with `Python 3.8.2`, `virtualenv 20.4.7`, macOS 11.5.1.

### Makefile targets

```
$ make help
Make targets:
  build          create virtualenv and install packages
  build-lab      runs build and installs lab extensions
  freeze         persist installed packaged to requirements.txt
  clean          remove *.pyc files and __pycache__ directory
  distclean      remove virtual environment
  run            run JupyterLab
  runtb          run TensorBoard
Check the Makefile for more details
```

### Step-by-step instructions

1. Open a terminal and navigate to the path that you want to clone the repository to
1. Clone the repository
    ```
    $ git clone git@github.com:sbstn-gbl/p2v-map.git
    ```
1. Navigate to repository path, create virtual environment and install required modules with
    ```
    $ cd p2v-map && make build-lab
    ```
1. Start a notebook server, open `p2vmap.ipynb`, and run notebook
    ```
    $ make run
    ```

# Visualization of training results

The training and validation loss of the P2V model can be visualized using TensorBoard. The
Makefile target `make tb` launches a TensorBoard server that looks like this:

<p align="center">
<img src="/source/tensorboard.png" width="800">
</p>

The product embedding that is the result of the P2V model can be visualized with the
`DashboardP2V` dashboard class. The dashboard's method `plot_product_embedding` produces
the following embedding heat map using the `plotly` library:

<p align="center">
<img src="/source/embedding.png" width="800">
</p>

The product map that is the result of applying `t-SNE` to the product embedding can be
visualized with the `DashboardP2V` dashboard class in the `p2v` library. The dashboard's
method `plot_tsne_map` produces the following embedding scatter plot using the `plotly`
library:

<p align="center">
<img src="/source/map.png" width="800">
</p>

All Plotly results allow you to track the training progress over epochs and batches.

# Contact

https://sebastiangabel.com
