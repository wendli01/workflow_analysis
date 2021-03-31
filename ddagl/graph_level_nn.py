"""
Poling Graph-Convolutional Neural Networks (P-GCNs) for various (un)supervised tasks.
"""

import itertools
import os
from typing import Sequence, Tuple, Dict, List, Union, Optional

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch as th
from dgl import DGLGraph
from dgl.nn import TAGConv
from matplotlib import pyplot as plt
from scipy import sparse as sp
from scipy.sparse.compressed import _cs_matrix
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.metrics import r2_score, pairwise_distances, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.backends import cudnn
from torch.nn import Module

from .visualization import show_activations, show_training_history


class FunctionalsPoolingLayer(Module):
    """
        Pooling layer that uses predefined functionals.

        Parameters
        ----------

        pooling_ops:
            Pooling operations used to transform node-level into graph-level features. Must be torch
            functions that work on arrays of arbitrary size.  Use either this or `nb_quantiles`.
        mean_group_size:
            Minimum mean group size for vectorized pooling. If this is met, graphs are grouped by size to circumvent
            limitations of ragged volumes which improves runtime. Set to `None` for no vectorization.

    """

    def __init__(self,
                 pooling_ops: Optional[Union[Sequence[callable], int]] = (th.max, th.min, th.mean, th.std),
                 mean_group_size: Optional[float] = 3):
        super(FunctionalsPoolingLayer, self).__init__()
        self.pooling_ops = pooling_ops
        self.mean_group_size = mean_group_size

    def _pool_batch(self, x: List[th.Tensor]) -> th.Tensor:
        def _get_pooling_results(feats: th.Tensor):
            op_results = [op(feats, dim=0) for op in self.pooling_ops]
            op_results = [op_res[0] if isinstance(op_res, tuple) else op_res for op_res in op_results]
            return th.stack(op_results)

        return th.stack(list(map(_get_pooling_results, x)))

    def _pool_batch_vec(self, x: List[th.Tensor]) -> th.Tensor:
        def _get_pooling_results(feats: th.Tensor):
            op_results = [op(feats, dim=1) for op in self.pooling_ops]
            op_results = [op_res[0] if isinstance(op_res, tuple) else op_res for op_res in op_results]
            return th.stack(op_results).transpose(0, 1)

        group_indices = {}
        for ind, v in enumerate(x):
            size = v.shape[0]
            if size not in group_indices:
                group_indices[size] = []
            group_indices[size].append(ind)

        group_indices = group_indices.values()

        graph_groups = [th.stack([x[idx] for idx in indices]) for indices in group_indices]
        new_idx_dict = {new_idx: ind for ind, new_idx in enumerate(itertools.chain(*group_indices))}
        order = [new_idx_dict[i] for i in range(len(x))]
        return th.cat(list(map(_get_pooling_results, graph_groups)))[order]

    def forward(self, x: List[th.Tensor]):
        graph_sizes = [v.shape[0] for v in x]
        vectorize_pooling: bool = self.mean_group_size is not None and pd.value_counts(
            graph_sizes).mean() >= self.mean_group_size
        return self._pool_batch_vec(x) if vectorize_pooling else self._pool_batch(x)


def combine_loss_criteria(loss_funs: Sequence[callable], weights: Sequence[float]):
    def _combined_loss_criterion(y_true, y_pred):
        weighted_losses = [weight * loss_fun(y_true, y_pred) for weight, loss_fun in zip(weights, loss_funs)]
        return th.sum(th.stack(weighted_losses))

    weights = th.Tensor(weights / np.sum(weights))
    return _combined_loss_criterion


def correlation_loss(y_true, y_pred) -> th.Tensor:
    r"""
    Correlation loss as  :math:`1 - \rho_p(y, \hat{y})`.

    :param y_true: target variable
    :param y_pred: predicted variable
    """
    if y_true.shape[0] == 1 or th.var(y_true) == 0 or th.var(y_pred) == 0:
        return th.tensor(0).to(y_pred.data.device)
    vx = y_true - th.mean(y_true)
    vy = y_pred - th.mean(y_pred)
    corr = th.sum(vx * vy) / (th.sqrt(th.sum(vx ** 2)) * th.sqrt(th.sum(vy ** 2)))
    return 1 - corr


def ktau_loss(y_true, y_pred) -> th.Tensor:
    """
    Kendall Tau loss as  :math:`1 - KTau(y, \hat{y})`.

    :param y_true: target variable
    :param y_pred: predicted variable
    """
    if y_true.shape[0] == 1:
        return th.sum(y_true - y_true)
    true_pairs = th.combinations(y_true, 2).T
    pred_pairs = th.combinations(y_pred, 2).T
    ktau = th.mean(th.sign(true_pairs[0] - true_pairs[1]) * th.sign(pred_pairs[0] - pred_pairs[1]))
    return 1 - ktau


def hinge_ranking_loss(y_true, y_pred, margin: float = 0.1):
    """
    Wrapper for easier use of `th.nn.MarginRankingLoss`.
    :param y_true: target variable
    :param y_pred: prediction variable
    :param margin: margin
    :return:
    """
    true_pairs = th.cartesian_prod(y_true, y_true).T
    pred_pairs = th.cartesian_prod(y_pred, y_pred).T
    target = th.sign(pred_pairs[0] - pred_pairs[1])
    return th.nn.MarginRankingLoss(margin=margin)(true_pairs[0], true_pairs[1], target)


class ResGCN(Module):
    """
    Graph-Level Conv Net with pooling functions for aggregation.

    Parameters
    ----------

    layer_sizes:
        Sequence of depths (number of filters) for the GraphConv Layers. Consequently, this also
        determines the number of hidden layers.
    dropout:
        drop probability for dropout applied before pooling, set to `None` for no dropout.
    conv_cls:
        Convolution Module that is applied for feature extraction.
    output_size:
        Numer of outputs per graph - set to 0 or None for raw graph-level embeddings as output.
    conv_kwargs:
        Keyword arguments used for instantiation of each Conv layer.
    conv_agg:
        Use a 1D convolution that convolves the aggregates for each embedding dimension instead of flattening and a
        Dense layer. Through weight sharing, this layer learns common weights for the pooling results of each embedding
        dimension, and so caputres the relationship between those ops.
    skip_connections:
        Use skip connections where applicable (i.e. between layers with matching sizes).
    batch_norm:
        Use batchnorm layers after each convolutional layer to reduce covariate shift for improved convergence.
    softmax:
        Use a softmax layer to produce prediction probabilities.
    aggregation_batchnorm:
        Use a batchnorm layer after the aggregation, i.e. for scaling outputs of the pooling operations.
    dense_layer_sizes:
        Sequence of widths (number of neurons) for suprevised tasks.
    node_type_embedding_size:
        Size, i.e. number of dimensions, for learnable node-type embeddings. Requires features to be indices.
        Set to None to disable learnable embeddings.
    hybrid_output:
        Produce hybrid node-level output; uses a joint node- and graph-level representation by concatenation or some
        other function provided.
    hybrid_combination_fun:
        Function to combine node- and graph-level output in hybrid mode. Can any torch function that squeezes a
        provided dimension, e.g. mean, sum, std. Set to `None` for concatenation of the outputs.
    activation_layer_cls:
        Class of the activation layer used after every (but the last) graph convolution and every dense layer.

    """

    def __init__(self, num_features: int, layer_sizes: Tuple[int] = (128, 128, 128, 128, 128, 64),
                 pooling_layer: FunctionalsPoolingLayer = FunctionalsPoolingLayer(),
                 output_size: Union[Sequence[int], int, None] = 1, dropout: float = 0.05,
                 conv_cls: type = TAGConv, conv_kwargs=None, conv_agg: bool = False,
                 skip_connections: bool = True, batch_norm: bool = True, softmax: bool = False,
                 aggregation_batchnorm: bool = True, dense_layer_sizes: Optional[Tuple[int]] = (),
                 node_type_embedding_size: Optional[int] = None, global_node: bool = False,
                 hybrid_output: bool = False, hybrid_combination_fun: Optional[callable] = None,
                 activation_layer_cls=th.nn.ReLU):
        super(ResGCN, self).__init__()
        if conv_kwargs is None:
            conv_kwargs = {}
        self.conv_layers_ = th.nn.ModuleList()
        self.num_features = num_features
        self.dropout = dropout
        self.conv_agg = conv_agg
        self.layer_sizes = layer_sizes
        self.conv_cls = conv_cls
        self._layer_ind_mapping = {}
        self.skip_indices_ = {}
        self.skip_connections = skip_connections
        self.softmax = softmax
        self.dense_layer_sizes = dense_layer_sizes
        self.node_type_embedding_size = node_type_embedding_size
        self.pooling_layer = pooling_layer
        self.global_node = global_node or self.pooling_layer is None
        self.activation_layer_cls = activation_layer_cls
        self.hybrid_combination_fun, self.hybrid_output = hybrid_combination_fun, hybrid_output

        if dropout is not None and (dropout < 0 or dropout > 1):
            raise UserWarning("Choose None for no dropout or a float between 0 and 1.")

        if node_type_embedding_size is not None and node_type_embedding_size > 0:
            self.node_type_embeddings_ = th.nn.Embedding(num_embeddings=self.num_features,
                                                         embedding_dim=self.node_type_embedding_size)
            input_size = self.node_type_embedding_size
        else:
            input_size = self.num_features

        for layer_ind, size in enumerate(self.layer_sizes):
            is_last = layer_ind == len(self.layer_sizes) - 1

            if conv_cls in (dgl.nn.GATConv, dgl.nn.EdgeConv):
                conv_layer = conv_cls(input_size, size, **conv_kwargs)
            else:
                conv_layer = conv_cls(input_size, size, **conv_kwargs, bias=not batch_norm)
            self.conv_layers_.append(conv_layer)

            if batch_norm:
                self.conv_layers_.append(th.nn.BatchNorm1d(size))

            # no activation for last conv layer
            if not (is_last and output_size is None):
                self.conv_layers_.append(self.activation_layer_cls())

            actual_ind = len(self.conv_layers_) - 1
            self._layer_ind_mapping[layer_ind] = actual_ind

            if self.skip_connections and layer_ind >= 2 and size == self.layer_sizes[layer_ind - 2]:
                self.skip_indices_[actual_ind] = self._layer_ind_mapping[layer_ind - 2]
            input_size = size

        pooling_size = 1 if self.global_node else len(self.pooling_layer.pooling_ops)

        if self.hybrid_output:
            pooling_size = pooling_size + 1 if self.hybrid_combination_fun is None else 1

        self.aggregation_batchnorm_layer_ = th.nn.BatchNorm1d(pooling_size) if aggregation_batchnorm else None

        if not self.global_node:
            if conv_agg:
                self.agg_layer_ = th.nn.Conv1d(kernel_size=pooling_size, out_channels=1, in_channels=1,
                                               stride=pooling_size)
                self.agg_layer_.weight.data.fill_(1)
                self.agg_layer_.bias.data.fill_(0)
            else:
                self.agg_layer_ = th.nn.Linear(pooling_size * self.layer_sizes[-1], self.layer_sizes[-1])

        if self.dense_layer_sizes not in ((), None):
            self.dense_layers_ = th.nn.ModuleList()
            in_features = self.layer_sizes[-1]
            for layer_size in self.dense_layer_sizes:
                dense_layer = th.nn.Linear(in_features=in_features, out_features=layer_size)
                self.dense_layers_.append(
                    th.nn.Sequential(dense_layer, self.activation_layer_cls(), th.nn.LayerNorm(layer_size)))
                in_features = layer_size

            output_input_size = self.dense_layer_sizes[-1]
        else:
            output_input_size = self.layer_sizes[-1]

        if output_size in (None, 0):
            self.output_layer_ = None
        else:
            self.output_layer_ = th.nn.Linear(output_input_size, output_size)
            if self.softmax:
                self.output_layer_ = th.nn.Sequential(self.output_layer_, th.nn.Softmax(dim=0))

        if dropout is not None and 0 < dropout <= 1:
            self.conv_layers_.append(th.nn.Dropout(dropout))

    def _apply_output_layers(self, x, activations=None, layer_names=None, return_activations: bool = False):
        if not self.global_node:
            if self.aggregation_batchnorm_layer_ is not None:
                x = self.aggregation_batchnorm_layer_(x)
                if return_activations:
                    activations.append(x.squeeze().detach().cpu().numpy())
                    layer_names.append(type(self.aggregation_batchnorm_layer_).__name__)

            x = x.flatten(1).unsqueeze(1)
            x = self.agg_layer_(x)
            if return_activations:
                activations.append(x.squeeze().detach().cpu().numpy())
                layer_names.append('conv agg' if self.conv_agg else 'linear agg')

        if len(x.shape) > 2:
            x = th.flatten(x, 1)

        if self.dense_layer_sizes not in ((), None):
            for layer in self.dense_layers_:
                x = layer(x)
                if return_activations:
                    activations.append(x.squeeze().detach().cpu().numpy())
                    layer_names.append('Dense')

        if self.output_layer_ is None:
            return (activations, layer_names) if return_activations else x

        x = self.output_layer_(x)
        if return_activations:
            activations.append(x.detach().cpu().numpy())
            layer_names.append(type(self.output_layer_).__name__)
            return activations, layer_names
        return x

    def forward(self, g: DGLGraph, features: th.Tensor, nodes: Optional[Sequence[int]] = None,
                return_activations: bool = False):

        layer_outputs, activations, layer_names = {}, [], []
        if self.node_type_embedding_size is not None:
            node_types = th.nonzero(features).T[1]
            x = self.node_type_embeddings_.forward(node_types)
        else:
            x = features

        if return_activations:
            activations.append(x.detach().cpu().numpy())
            layer_names.append('Input')

        for ind, layer in enumerate(self.conv_layers_):
            if ind in self.skip_indices_:
                x = x + layer_outputs[self.skip_indices_[ind]]

            x = layer(g, x) if isinstance(layer, self.conv_cls) else layer(x)
            if ind in self.skip_indices_.values():
                layer_outputs[ind] = x

            if return_activations:
                activations.append(x.detach().cpu().numpy())
                layer_names.append(type(layer).__name__)

        del layer_outputs

        x = th.split(x, tuple(g.batch_num_nodes())) if len(g.batch_num_nodes()) > 1 else [x]

        if self.global_node:
            graph_level_x = th.stack([x_[-1] for x_ in x])
        else:
            graph_level_x = self.pooling_layer(x)

        if self.hybrid_output:
            if self.global_node:
                graph_level_x = th.unsqueeze(graph_level_x, 1)
            node_level_x = th.stack([x_[node_id] for node_id, x_ in zip(nodes, x)]).unsqueeze(1)
            if self.hybrid_combination_fun is None:
                graph_level_x = th.hstack([graph_level_x, node_level_x])
            else:
                graph_level_x = th.cat([graph_level_x, node_level_x], dim=1)
                graph_level_x = self.hybrid_combination_fun(graph_level_x, dim=1)

        if return_activations:
            activations.append(graph_level_x.squeeze().detach().cpu().numpy())
            layer_names.append(type(self.pooling_layer).__name__)

        return self._apply_output_layers(graph_level_x, activations=activations, layer_names=layer_names,
                                         return_activations=return_activations)


class ResGCNEstimator(BaseEstimator):
    """


    Parameters
    ----------
    device:
        Device to run model on. If set to `'cpu'`, the CPU will be used, `'cuda'` or GPU index for GPU.
    inference_batch_size:
        Batch size for inference. Should be set as high as device memory allows for efficiency.
        This also limits the maximum size of chunks fed into the model - which may change training results if
        batch norm is used!
    random_state:
        Determines batch split and shuffle as well as triple sampling. Pass an int or np.random.RandomState to make the
        randomness deterministic, leave at or set to `'None'` for pseudo random.
    global_node:
        Use a global node to get from node- to graph-level.

    Attributes
    ----------
    model_:
        ResGCN
    training_history_:
        List of Dictionaries for loss, valid and train scores calculated each epoch.
    """

    def __init__(self, device: Union[int, str] = 'cuda', inference_batch_size: int = 10000,
                 random_state: Union[int, np.random.RandomState, int] = None, global_node: bool = False):
        self.device: str = device
        self.inference_batch_size: int = inference_batch_size
        self.model_: Optional[ResGCN] = None
        self.training_history_ = []
        self.random_state, self.rng_ = random_state, None
        self.global_node: bool = global_node

    def _get_model_outputs(self, graphs, features, **kwargs):
        def _to_device_tensor(a: Union[np.ndarray, _cs_matrix]) -> th.Tensor:
            tensor = th.Tensor(a) if isinstance(a, np.ndarray) else th.sparse.Tensor(a.toarray())
            return tensor.to(self.device) if self.device not in ('cpu', None) else tensor

        graph_batch = dgl.batch(graphs).to(self.device)
        stack_ = np if isinstance(features[0], np.ndarray) else sp
        feature_batch = stack_.vstack(features) if len(np.shape(features[0])) > 1 else stack_.hstack(features)

        feature_batch = _to_device_tensor(feature_batch)
        return self.model_(graph_batch, feature_batch, **kwargs)

    def _get_batch_transform(self, X: Union[
        Sequence[Tuple[nx.DiGraph, np.ndarray]], Sequence[Tuple[nx.DiGraph, np.ndarray, int]]], **kwargs):
        if len(X[0]) == 2:
            graphs, features = self._preprocess_input(*np.transpose(X))
        else:
            graphs, features = self._preprocess_input(*np.transpose(X)[:2])
            kwargs['nodes'] = np.transpose(X)[-1]
        model_outputs = self._get_model_outputs(graphs, features, **kwargs)
        th.cuda.empty_cache()
        return model_outputs.detach().cpu().numpy() if isinstance(model_outputs, th.Tensor) else model_outputs

    def _get_transform(self, X: Union[
        Sequence[Tuple[nx.DiGraph, np.ndarray]], Sequence[Tuple[nx.DiGraph, np.ndarray, int]]]) -> np.ndarray:
        X = np.array(X, dtype=object)
        X_batches = np.array_split(X, int(np.ceil(np.shape(X)[0] / self.inference_batch_size)))
        with th.no_grad():
            return np.vstack(list(map(self._get_batch_transform, X_batches)))

    def get_activations(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], return_layer_names: bool = False,
                        layer_ind: int = None, ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        X = np.array(X, dtype=object)
        X_batches = np.array_split(X, int(np.ceil(np.shape(X)[0] / self.inference_batch_size)))
        activations, layer_names = [], []

        for X_batch in X_batches:
            batch_activations, batch_layer_names = self._get_batch_transform(X_batch, return_activations=True)
            if layer_ind is None:
                activations.append(batch_activations)
            else:
                activations.append(batch_activations[layer_ind])
            layer_names.append(batch_layer_names)

        if return_layer_names:
            return np.vstack(np.array(activations, dtype=object)), layer_names
        return np.vstack(np.array(activations, dtype=object))

    def show_activations(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], file_name: str = 'activations.gif',
                         interval: int = 500, cmap: str = 'RdBu', center: float = 0, notebook: bool = False,
                         **heatmap_kwargs):
        show_activations(*self.get_activations(X, return_layer_names=True), fn=file_name, interval=interval, cmap=cmap,
                         center=center, **heatmap_kwargs)
        if notebook:
            from IPython.display import HTML
            return HTML('<img src="{}"/>'.format(file_name))

    def show_training_history(self, figsize: Tuple[int, int] = (9, 9), dpi: int = 120, epoch_cutoff: float = .25,
                              **kwargs) -> plt.Figure:
        df = pd.DataFrame(self.training_history_)
        return show_training_history(df, figsize=figsize, dpi=dpi, epoch_cutoff=epoch_cutoff, **kwargs)

    def _preprocess_graph(self, g: Union[nx.DiGraph, nx.Graph]) -> DGLGraph:
        if isinstance(g, nx.DiGraph):
            edges = g.edges
        else:
            edges = [*g.edges, *np.transpose(g.to_undirected().edges)[::-1].T]

        nodes = list(g.nodes)
        src_ids, dst_ids = list(map(list, zip(*edges)))

        if self.global_node:
            global_node = np.max(nodes) + 1
            src_ids += nodes
            dst_ids += [global_node] * g.number_of_nodes()

        return dgl.graph((src_ids + nodes, dst_ids + nodes))

    @staticmethod
    def _add_global_node_feature(x: np.ndarray):
        new_row, new_column = np.zeros([1, x.shape[1]]), np.zeros([x.shape[0] + 1, 1])
        if isinstance(x, np.ndarray):
            return np.hstack([np.vstack([x, new_row]), new_column])
        return sp.hstack([sp.vstack([x, new_row]), new_column])

    def _preprocess_input(self, graphs, features) -> Tuple[Sequence[DGLGraph], Sequence[th.Tensor]]:
        # workaround for https://github.com/dmlc/dgl/issues/2416
        a = np.empty(len(graphs), dtype=DGLGraph)
        for i, graph in enumerate(map(self._preprocess_graph, graphs)):
            a[i] = graph
        if self.global_node:
            features = np.array(list(map(self._add_global_node_feature, features)), dtype=object)

        return a, features

    @staticmethod
    def _get_rng(random_state):
        if random_state is None:
            return np.random.RandomState()
        if type(random_state) is np.random.RandomState:
            return random_state
        return np.random.RandomState(random_state)

    def _set_random_state(self):
        self.rng_ = self._get_rng(self.random_state)
        if self.random_state is not None:
            th.manual_seed(self.random_state)
            cudnn.deterministic = True
            cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class SupervisedResGCN(ResGCNEstimator):
    """
    Graph Level Regressor. Takes graphs as inputs and performes regression on them. Node-level features are extracted
    from each graphs node attributes.

    Parameters
    ----------

    nb_epochs:
        number of epochs to train
    batch_size:
        Mini-Batch size. Increase for less, decrease for more stochastic behaviour in training.
    shuffle_batches:
        whether to suffle indices before batching for each epoch
    loss_criterion:
        loss criterion implementing `torch.nn.modules._WeightedLoss` that is used for gradient descent.
    optimizer_cls:
        torch.optim.optimizer class that is used for training
    lr:
        learning rate
    weight_decay:
        weight decay coefficient for the optimizer.
    model_kwargs:
        Keyword Arguments passed to the ResGCN for instantiating `self.model_`.
    verbose:
        whether to print debug information
    device:
        Device to run model on. If set to `'cpu'`, the CPU will be used, `'cuda'` or GPU index for GPU.
    scoring:
        Scoring function for validation scores and `score`.
    random_state:
        Determines batch split and shuffle as well as triple sampling. Pass an int or np.random.RandomState to make the
        randomness deterministic, leave at or set to `'None'` for pseudo random.
    lr_scheduler_cls:
        Class of the learning rate scheduler to use. Must be initialized after the model.
    lr_scheduler_kwargs:
        Init keyword arguments for learning rate scheduler.
    dense_layer_sizes:
        Sequence of widths (number of neurons) for suprevised tasks.
    warm_start:
        Set to `True` to continue training for on-line tasks - do not use for cross validation and the like.
    hybrid_output:
        Produce hybrid node-level output.

    Attributes
    ----------

    model_:
        ResGCN
    training_history_:
        List of Dictionaries for loss, valid and train scores calculated each epoch.
    """

    def __init__(self, nb_epochs=100, batch_size: int = 1000, shuffle_batches: bool = True,
                 loss_criterion=th.nn.MSELoss, optimizer_cls=th.optim.Adam, lr: float = 0.01,
                 weight_decay: float = 0.01, verbose: bool = False, device: Optional[int] = 'cuda',
                 scoring: callable = r2_score, random_state: Union[int, np.random.RandomState, int] = None,
                 lr_scheduler_cls=th.optim.lr_scheduler.ExponentialLR, lr_scheduler_kwargs=dict(gamma=0.95),
                 dense_layer_sizes=(64,), global_node: bool = False, inference_batch_size: int = 10000,
                 warm_start: bool = False, hybrid_output: bool = False, **model_kwargs):
        super(SupervisedResGCN, self).__init__(device=device, random_state=random_state,
                                               global_node=global_node, inference_batch_size=inference_batch_size)
        if model_kwargs is None:
            model_kwargs = {}
        self.optimizer_cls = optimizer_cls
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose
        self.model_: ResGCN
        self.optimizer_: optimizer_cls
        self.lr_scheduler_: th.optim.lr_scheduler
        self.training_history_: List[Dict[str, float]]
        self.loss_criterion = loss_criterion
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.model_kwargs = model_kwargs
        self.weight_decay = weight_decay
        self.scoring = scoring
        self.lr_scheduler_cls, self.lr_scheduler_kwargs = lr_scheduler_cls, lr_scheduler_kwargs
        self.dense_layer_sizes = dense_layer_sizes
        self.warm_start = warm_start
        self.hybrid_output = hybrid_output

    def _fit(self, X: Union[Sequence[Tuple[nx.DiGraph, np.ndarray]], Sequence[Tuple[nx.DiGraph, np.ndarray, int]]],
             y: Sequence, output_size: Optional[int] = 1):
        self._set_random_state()
        X, y = np.array(X, dtype=object), th.Tensor(y)
        if len(X[0]) == 3:
            X, node_indices = np.transpose(X)[:-1].T, np.array(np.transpose(X)[-1])
        else:
            node_indices = None
        graphs, features = self._preprocess_input(*np.array(X, dtype=object).T)

        if not self.warm_start or self.model_ is None:
            self.model_ = ResGCN(num_features=features[0].shape[-1], output_size=output_size,
                                 dense_layer_sizes=self.dense_layer_sizes, global_node=self.global_node,
                                 hybrid_output=self.hybrid_output, **self.model_kwargs)
            self.model_.to(self.device)
            self.training_history_ = []
            self.optimizer_ = self.optimizer_cls(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.lr_scheduler_ = self.lr_scheduler_cls(optimizer=self.optimizer_, **self.lr_scheduler_kwargs)

        for epoch in range(self.nb_epochs):
            epoch_results = self._train_epoch(optimizer=self.optimizer_, lr_scheduler=self.lr_scheduler_, epoch=epoch,
                                              graphs_train=graphs, features_train=features, y_train=y,
                                              nodes_train=node_indices)
            self.training_history_.append(epoch_results)
        th.cuda.empty_cache()
        return self

    def _train_batch(self, optimizer: th.optim.Optimizer, g_batch: th.Tensor, f_batch: th.Tensor, y_batch: th.Tensor,
                     n_batch: Union[None, th.Tensor] = None) -> Tuple[float, float]:
        def _get_chunk_output(indices: Sequence[int]) -> Tuple[float, float]:
            g_chunk, f_chunk, y_chunk = g_batch[indices], f_batch[indices], y_batch[indices]
            if n_batch is not None:
                y_pred = self._get_model_outputs(g_chunk, f_chunk, nodes=n_batch[indices]).squeeze(-1)
            else:
                y_pred = self._get_model_outputs(g_chunk, f_chunk).squeeze(-1)

            if len(y_pred.shape) > 1:
                y_chunk = y_chunk.long()

            chunk_loss = self.loss_criterion(y_pred, y_chunk.to(self.device))
            if len(y_pred.shape) > 1:
                chunk_train_score = self.scoring(y_chunk, th.argmax(y_pred, dim=1).detach().cpu())
            else:
                chunk_train_score = self.scoring(y_chunk, y_pred.detach().cpu())

            chunk_loss.backward()
            return float(chunk_loss.detach()), chunk_train_score

        optimizer.zero_grad()
        batch_indices = range(len(y_batch))
        if self.inference_batch_size < len(g_batch):
            chunk_losses, chunk_scores = [], []
            chunks = [batch_indices[i:i + self.inference_batch_size] for i in
                      range(0, len(batch_indices), self.inference_batch_size)]
            for chunk_indices in chunks:
                chunk_loss, chunk_score = _get_chunk_output(chunk_indices)
                chunk_losses.append(chunk_loss * len(chunk_indices))
                chunk_scores.append(chunk_score * len(chunk_indices))
            loss, train_score = np.sum(chunk_losses) / len(batch_indices), np.sum(chunk_scores) / len(batch_indices)
        else:
            loss, train_score = _get_chunk_output(batch_indices)

        optimizer.step()
        return loss, train_score

    def _train_epoch(self, optimizer, lr_scheduler: th.optim.lr_scheduler, epoch, graphs_train, features_train,
                     y_train, nodes_train=None) -> Dict[str, float]:
        indices = self.rng_.permutation(len(y_train)) if self.shuffle_batches else range(len(y_train))
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        loss_values, batch_train_scores = [], []
        self.model_.train()
        for batch_indices in batches:
            nodes_batch = nodes_train[batch_indices] if nodes_train is not None else None
            graphs, features, y = graphs_train[batch_indices], features_train[batch_indices], y_train[batch_indices]
            batch_loss, batch_train_score = self._train_batch(optimizer, g_batch=graphs, f_batch=features, y_batch=y,
                                                              n_batch=nodes_batch)
            loss_values.append(batch_loss)
            batch_train_scores.append(batch_train_score)

        epoch_results = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'], loss=np.mean(loss_values),
                             loss_std=np.std(loss_values), train_score=np.mean(batch_train_scores))

        if self.verbose:
            fields = [k + (' {:03d}' if type(v) is int else ' {:.5f}').format(v) for k, v in epoch_results.items()]
            print(' | '.join(fields))

        lr_scheduler.step()
        self.model_.eval()
        return epoch_results

    def predict(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]]) -> np.ndarray:
        return self._get_transform(X).squeeze()

    def score(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], y: Sequence) -> np.ndarray:
        return self.scoring(y, self.predict(X))

    def transform(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], y: Sequence = None,
                  layer_id: int = -2) -> np.ndarray:
        return np.vstack(self.get_activations(X, layer_ind=layer_id).squeeze())


class GraphLevelRegressor(SupervisedResGCN, RegressorMixin):
    def __init__(self, scoring=r2_score,
                 loss_criterion=combine_loss_criteria([th.nn.MSELoss(), hinge_ranking_loss], [.5, .5]), **model_kwargs):
        """
        Graph level Res-GCN regressor.

        Parameters
        ----------

        scoring:
            Scoring function for validation scores and `self.score()`.
        loss_criterion:
            loss criterion implementing `torch.nn.modules._WeightedLoss` that is used for gradient descent.
        model_kwargs:
            Keyword Arguments passed to the ResGCN for instantiating `self.model_`.

        Notes
        -----

        This Estimators inheritance structure makes it usuitable for cloning via `get_params()` and `set_params()`,
        e.g. in `sklearn.model_selection.cross_val_score()`.
        """
        super().__init__(scoring=scoring, loss_criterion=loss_criterion, **model_kwargs)

    def fit(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], y: Sequence):
        return self._fit(X, y)


class GraphLevelEmbedder(SupervisedResGCN, TransformerMixin):
    def __init__(self, **model_kwargs):
        """
        Graph level embedder using a randomly initialized Res-GCN.

        Parameters
        ----------

        model_kwargs:
            Keyword Arguments passed to the ResGCN for instantiating `self.model_`.

        Notes
        -----

        This Estimators inheritance structure makes it usuitable for cloning via `get_params()` and `set_params()`,
        e.g. in `sklearn.model_selection.cross_val_score()`.
        """
        super().__init__(dense_layer_sizes=None, nb_epochs=0, **model_kwargs)

    def fit(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], y):
        return self._fit(X, y, output_size=None)

    def transform(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], y: Sequence = None) -> np.ndarray:
        return self._get_transform(X)


class GraphLevelClassifier(SupervisedResGCN, ClassifierMixin):
    def __init__(self, scoring=lambda y, y_: f1_score(y, y_, average='micro'),
                 loss_criterion=th.nn.CrossEntropyLoss(), **model_kwargs):
        """
        Graph level Res-GCN classifier.

        Parameters
        ----------

        scoring:
            Scoring function for validation scores and `self.score()`.
        loss_criterion:
            loss criterion implementing `torch.nn.modules._WeightedLoss` that is used for gradient descent.
        model_kwargs:
            Keyword Arguments passed to the ResGCN for instantiating `self.model_`.

        Notes
        -----

        This Estimators inheritance structure makes it usuitable for cloning via `get_params()` and `set_params()`,
        e.g. in `sklearn.model_selection.cross_val_score()`.
        """
        super().__init__(scoring=scoring, loss_criterion=loss_criterion, **model_kwargs)
        self.label_encoder_ = LabelEncoder()
        self.classes_ = None

    def fit(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], y: Sequence):
        y = self.label_encoder_.fit_transform(np.array(y))
        self.classes_ = self.label_encoder_.classes_
        return self._fit(X, y, output_size=len(self.label_encoder_.classes_))

    def predict(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]]) -> np.ndarray:
        preds = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(np.argmax(preds, axis=1))

    def predict_proba(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]]) -> np.ndarray:
        return self._get_transform(X)


class NodeLevelClassifier(SupervisedResGCN, ClassifierMixin):
    def __init__(self, scoring=accuracy_score, loss_criterion=th.nn.CrossEntropyLoss(), loss_weighting: bool = False,
                 **model_kwargs):
        """
        Node level Res-GCN classifier.

        Parameters
        ----------

        scoring:
            Scoring function for validation scores and `self.score()`.
        loss_criterion:
            loss criterion implementing `torch.nn.modules._WeightedLoss` that is used for gradient descent.
        model_kwargs:
            Keyword Arguments passed to the ResGCN for instantiating `self.model_`.

        Notes
        -----

        This Estimators inheritance structure makes it usuitable for cloning via `get_params()` and `set_params()`,
        e.g. in `sklearn.model_selection.cross_val_score()`.
        """
        super().__init__(scoring=scoring, loss_criterion=loss_criterion, hybrid_output=True, **model_kwargs)
        self.label_encoder_ = LabelEncoder()
        self.loss_weighting = loss_weighting
        self.classes_ = None

    def fit(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray, int]], y: Sequence):
        y = self.label_encoder_.fit_transform(np.array(y))
        self.classes_ = self.label_encoder_.classes_

        if self.loss_weighting:
            class_weights = th.Tensor(1 / pd.value_counts(y, sort=False).sort_index().values).to(self.device)
            self.loss_criterion.weight = class_weights
        return self._fit(X, y, output_size=len(self.classes_))

    def predict(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray, int]]) -> np.ndarray:
        preds = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(np.argmax(preds, axis=1))

    def predict_proba(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray, int]]) -> np.ndarray:
        return self._get_transform(X)


class GroupedGraphEmbedder(ResGCNEstimator, TransformerMixin):
    """
    Grouped Graph Level Embedder. Takes groups of graphs as inputs embeds them. Node-level features are extracted
    from each graphs node attributes.

    Parameters
    ----------

    nb_epochs:
        number of epochs to train
    batch_size:
        Mini-Batch size. Increase for less, decrease for more stochastic behaviour in training.
    shuffle_batches:
        whether to suffle indices before batching for each epoch
    loss_criterion:
        loss criterion implementing `torch.nn.modules._WeightedLoss` that is used for gradient descent.
    negatives_from_batch:
        Whether to sample negatives from within a batch, reducing the number of forward passes by a third. Only
        recommended if batch_size is high enough to guarantee that there is a negative for each anchor in the batch.
    optimizer_cls:
        torch.optim.optimizer class that is used for training
    lr:
        learning rate
    lr_decay:
        decay factor for exponential learning rate annealing. Consequently, for `1`, no annealing is applied.
    weight_decay:
        weight decay coefficient for the optimizer.
    model_kwargs:
        Keyword Arguments passed to the ResGCN for instantiating `self.model_`.
    verbose:
        whether to print debug information
    device:
        Device to run model on. If set to `'cpu'`, the CPU will be used, `'cuda'` or GPU index for GPU.

    Attributes
    ----------

    model_:
        ResGCN
    training_history_:
        List of Dictionaries for loss, valid and train scores calculated each epoch.

    Notes
    -----

    This Estimators inheritance structure makes it usuitable for cloning via `get_params()` and `set_params()`,
    e.g. in `sklearn.model_selection.cross_val_score()`.
    """

    def __init__(self, nb_epochs=50, batch_size: int = 1000, shuffle_batches: bool = True,
                 loss_criterion=th.nn.TripletMarginLoss(), random_state: Union[int, np.random.RandomState, int] = None,
                 negatives_from_batch: bool = True, optimizer_cls=th.optim.Adam, lr: float = 0.01, lr_decay: float = .9,
                 weight_decay: float = 0.01, verbose: bool = False, device: Optional[int] = 'cuda',
                 inference_batch_size: int = 1000, **model_kwargs):

        super(GroupedGraphEmbedder, self).__init__(device=device, random_state=random_state,
                                                   inference_batch_size=inference_batch_size)
        if model_kwargs is None:
            model_kwargs = {}
        self.optimizer_cls = optimizer_cls
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.verbose = verbose
        self.model_: ResGCN
        self.training_history_: List[Dict[str, float]]
        self.loss_criterion = loss_criterion
        self.negatives_from_batch = negatives_from_batch
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.model_kwargs = model_kwargs
        self.weight_decay = weight_decay
        self.device = device

    def fit(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], groups: Sequence[Union[str, int]],
            inner_group_emb: np.ndarray = None):
        self._set_random_state()

        if type(groups[0]) is str:
            groups = LabelEncoder().fit_transform(groups)

        if pd.value_counts(groups).min() < 2:
            raise UserWarning(
                'All groups must have a minimum of 2 members to be used as positives for the triplet loss.')

        graphs_train, features_train = self._preprocess_input(*np.array(X, dtype=object).T)

        self.model_ = ResGCN(num_features=features_train[0].shape[-1], output_size=None, **self.model_kwargs)
        self.model_.to(self.device)
        self.training_history_ = []
        optimizer = self.optimizer_cls(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.lr_decay)

        for epoch in range(self.nb_epochs):
            epoch_results = self._train_epoch(optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch,
                                              graphs_train=graphs_train, features_train=features_train,
                                              groups_train=groups, inner_group_emb=inner_group_emb)
            self.training_history_.append(epoch_results)

        return self

    def _train_epoch(self, optimizer: th.optim, lr_scheduler: th.optim.lr_scheduler, epoch: int,
                     graphs_train: Sequence[DGLGraph], features_train: Sequence[th.Tensor], groups_train,
                     inner_group_emb=None) -> Dict[str, float]:
        def _get_triplet_loss(batch_indices: np.ndarray) -> th.Tensor:
            def _get_outputs(indices):
                graphs = graphs_train[indices]
                features = np.array(features_train)[indices]
                return self._get_model_outputs(graphs, features)

            def _get_negative_index(anchor_index, batch_indices=np.array(range(len(groups_train)))):
                anchor_group = groups_train[anchor_index]
                groups = groups_train[batch_indices]
                return self.rng_.choice(np.where(groups != anchor_group)[0])

            def _get_positive_index(anchor_index, eps=1e-5):
                indices = np.array(range(len(groups_train)))
                anchor_group = groups_train[anchor_index]
                is_pos = (groups_train == anchor_group) & (indices != anchor_index)
                pos_indices = np.where(is_pos)[0]
                if inner_group_emb is None:
                    return self.rng_.choice(pos_indices)
                dists = pairwise_distances([inner_group_emb[anchor_index]], inner_group_emb[pos_indices])[0]
                weights = 1 / (dists + eps)
                return self.rng_.choice(pos_indices, p=weights / np.sum(weights))

            anchor_outputs = _get_outputs(batch_indices)
            if self.negatives_from_batch:
                negative_indices = [_get_negative_index(i, batch_indices=batch_indices) for i in batch_indices]
                negative_outputs = anchor_outputs[negative_indices]
            else:
                negative_outputs = _get_outputs(list(map(_get_negative_index, batch_indices)))
            positive_outputs = _get_outputs(list(map(_get_positive_index, batch_indices)))

            return self.loss_criterion(anchor_outputs, positive_outputs, negative_outputs)

        def _train_batch(optimizer, batch_indices: np.ndarray) -> float:
            loss = _get_triplet_loss(batch_indices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        indices = self.rng_.permutation(len(graphs_train)) if self.shuffle_batches else range(len(graphs_train))
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        loss_values = []
        self.model_.train()
        for batch_indices in batches:
            batch_loss = _train_batch(optimizer, batch_indices)
            loss_values.append(batch_loss)

        epoch_results = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'], loss=np.mean(loss_values),
                             loss_std=np.std(loss_values))
        lr_scheduler.step()
        self.model_.eval()

        if self.verbose:
            print(' | '.join(
                [k + (' {:03d}' if type(v) is int else ' {:.5f}').format(v) for k, v in epoch_results.items()]))
        return epoch_results

    def transform(self, X: Sequence[Tuple[nx.DiGraph, np.ndarray]], *_) -> np.ndarray:
        return self._get_transform(X)
