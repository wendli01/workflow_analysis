# Methods for automatic Machine-Learning Workflow Analysis


This repository accompanies the ECML PKDD 2021 paper [Methods for automatic Machine-Learning Workflow Analysis](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_473.pdf).

It contains all [code](ddagl) as well as [experimental setups](experiments) described in the paper including results as standalone `jupyter` notebooks.

## Citation

If you use the code, results or any other resources in this repository, please cite the following paper:

```bibtex
@inproceedings{wendlinger2021methods,
  title={Methods for Automatic Machine-Learning Workflow Analysis},
  author={Wendlinger, Lorenz and Berndl, Emanuel and Granitzer, Michael},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={52--67},
  year={2021},
  organization={Springer}
}
```

## Installation


Installation via the provided conda envirionment is encouraged.

> `conda env create -f ddagl.yml`

Installing the provided wheel via pip is also possible, but requires separate installation of cuda.

> `pip install ddagl-0.1-py3-none-any.whl`

To replicate the experiments, [`jupyter`](https://jupyter.org/install) needs to be installed as well, e.g. with


> `conda install -c conda-forge notebook`
> 
> or 
> 
> `pip install jupyterlab`

For comparison with state-of-the-art methods, implementations in [`karateclub`](https://karateclub.readthedocs.io/en/latest/notes/installation.html) are used for [ODDS representation learning](experiments/representation_learning.ipynb).
It can be installed via
> `pip install karateclub`

While datasets are downloaded automatically in the notebooks using them, [NAS-bench-101](https://github.com/google-research/nasbench/blob/master/setup.py) requires [tensorflow](https://www.tensorflow.org/install) to load and parse the original NAS-bench-101 files.

## Use


All models and transformers are implemented as `sklearn` Estimators.


```python
from ddagl import graph_level_nn, graph_feature_extraction
from sklearn.pipeline import make_pipeline
import networkx as nx
from typing import Sequence

# some graph data
X: Sequence[nx.DiGraph]
y: Sequence[float]

graph_reg = graph_level_nn.GraphLevelRegressor(nb_epochs=100, batch_size=100)
graph_reg_pipeline = make_pipeline(graph_feature_extraction.bidirected_transformer, 
                                   graph_feature_extraction.NodeLevelFeatureTransformer(), 
                                   graph_reg)

graph_reg_pipeline.fit(X, y)

graph_reg_pipeline.predict(X)

```


## Datasets

- [NASBench: A Neural Architecture Search Dataset and Benchmark](https://github.com/google-research/nasbench), introduced in [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635)

- [ONE DATA Data Sience workflows](https://zenodo.org/record/4633704), introduced in [Methods for automatic Machine-Learning Workflow Analysis](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_473.pdf)

