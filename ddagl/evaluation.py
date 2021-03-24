"""
Evaluation functions and helpers for various network analysis tasks.
"""
import itertools
from typing import Sequence, Tuple, Union, Dict, Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import v_measure_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


def group_cluster_score(labels_true, X_emb, metric=v_measure_score, clusterer_cls=AgglomerativeClustering,
                        scaler=StandardScaler()):
    """
    Score how well a feature space is suited to clustering.

    :param labels_true: Ground truth instance classes.
    :param X_emb: Feature space, i.e. instance embeddings.
    :param metric: Metric to score clustering with
    :param clusterer_cls: Clusterer to be used, deterministic is recommended.
    :param scaler: Scaler to remove skewing through different feature scales.
    :return: Metric applied to clustering of scaled features.
    """
    clusterer = clusterer_cls(n_clusters=len(np.unique(labels_true)))
    pipeline = make_pipeline(scaler, clusterer)
    labels_pred = pipeline.fit_predict(X_emb)
    return metric(labels_true, labels_pred)


def group_cluster_scorer(transformer, X, labels_true, **kwargs):
    """
    Score how well a feature space is suited to clustering.

    :param transformer: Feature space transformer.
    :param X: Original features.
    :param labels_true: Ground truth instance classes.
    :param kwargs: keyword arguments passed to `group_cluster_score()`.
    :return: Metric applied to clustering of scaled features.
    """
    X_emb = transformer.transform(X)
    return group_cluster_score(labels_true, X_emb, **kwargs)


def triplet_ratio_score(groups, X_emb, scaler=StandardScaler()):
    """
    Calculate triplet ratio score, i.e. distance to positives divided by distance to negatives for each anchor.
    Positives are samples from within the anchors own group, negatives are from other groups.

    :param groups: Group for each instance.
    :param X_emb: Instance features.
    :param scaler: Scaler to remove skewing through different feature scales.
    :return: Mean of triplet ratio score for all samples.
    """

    def _get_score(anchor_index):
        anchor_group = groups[anchor_index]
        is_pos = (groups == anchor_group) & (indices != anchor_index)
        pos_indices = np.where(is_pos)[0]
        neg_indices = np.where(groups != anchor_group)[0]
        pos_dists, neg_dists = pw_dists[anchor_index][pos_indices], pw_dists[anchor_index][neg_indices]
        return np.mean(pos_dists) / np.mean(neg_dists)

    X_emb_scaled = scaler.fit_transform(X_emb)
    pw_dists = pairwise_distances(X_emb_scaled)
    indices = range(len(groups))
    return np.mean(list(map(_get_score, indices)))


def triplet_ratio_scorer(transformer, X, labels_true, **kwargs):
    """
    Calculate triplet ratio score for each sample.

    :param transformer: Feature space transformer.
    :param X: Original features.
    :param labels_true: Ground truth instance classes.
    :param kwargs: keyword arguments passed to `triplet_ratio_score()`.
    :return: Mean of triplet ratio score for all samples.
    """
    X_emb = transformer.transform(X)
    return triplet_ratio_score(labels_true, X_emb, **kwargs)


def top_n_scorer(clf, X, y, n: int = 5, weights=None):
    """
    Score top-n classification accuracy.

    :param clf: Classifier.
    :param X: Features.
    :param y: Labels.
    :param n: Number of predictions to be considered, i.e. the `n` classes predicted with the highest probability are
        used.
    :param weights: Weight of length `n` applied to the top-n predictions. Does not need to be normalized.
    :return: Top-n accuracy.
    """
    y_pred_probas = clf.predict_proba(X)
    return top_n_score(y_true=y, y_pred_probas=y_pred_probas, n=n, weights=weights, classes=clf.classes_)


def top_n_score(y_true, y_pred_probas, n: int = 5, weights=None, classes: Union[None, Dict[str, Any]] = None):
    """
    Score top-n classification accuracy.

    :param y_true: True labels.
    :param y_pred_probas: Predicted class probabilities.
    :param n: Number of predictions to be considered, i.e. the `n` classes predicted with the highest probability are
        used.
    :param weights: Weight of length `n` applied to the top-n predictions. Does not need to be normalized.
    :param classes: Dictionary of class names, necessary if classes are not just their indices.
    :return: Top-n accuracy.
    """

    top_n_preds = np.argsort(y_pred_probas, axis=1)[:, -n:]

    if classes is not None:
        top_n_preds = classes[top_n_preds]

    weights = np.ones(len(y_true)) if weights is None else weights / np.sum(weights) * len(y_true)

    successes = 0
    for class_t, class_preds, w in zip(y_true, top_n_preds, weights):
        if class_t in class_preds:
            successes += w
    return float(successes) / len(y_true)


def version_subsampling(graphs: pd.Series, subsampling_factor: int = 5, grouping_col: str = 'id',
                        version_col: str = 'version') -> pd.Series:
    """
    Subsample workflows according to their version so only every `subsampling_factor`th version is retained.
    The first sample is chosen so that the newest version is always retained.

    :param graphs: Instances to subsample.
    :param subsampling_factor: Subsampling factor.
    :param grouping_col: Column that instances are grouped by.
    :param version_col: Column that gives an instances version.
    :return: Subsampled instances.
    """

    def _every_nth_row(group: pd.DataFrame) -> pd.DataFrame:
        subsampled_versions = group[version_col].values[::subsampling_factor]
        return group[group[version_col].isin(subsampled_versions)].reset_index(drop=True)

    groups = graphs.apply(lambda g: g.graph[grouping_col])
    versions = graphs.apply(lambda g: g.graph[version_col])
    wf_df = pd.DataFrame({'graph': graphs, grouping_col: groups, version_col: versions})
    wf_groups = wf_df.sort_values(version_col, ascending=False).groupby(grouping_col)
    return pd.concat([_every_nth_row(group) for name, group in wf_groups]).reset_index(drop=True)['graph']


def prepare_for_component_refinement(graphs: Sequence[nx.DiGraph], attrs_to_delete: Tuple[str] = ('category', 'type'),
                                     label_attr: str = 'type', deleted_token='<UNK>', encode_node_ids: bool = True,
                                     encode_labels: bool = True,
                                     drop_attrs: Tuple = ()) -> Tuple[Sequence[nx.DiGraph], np.ndarray]:
    """
    Prepare graphs to be used as a training set for component refinement tasks.
    For each node, a graph missing that nodes information is created and that nodes information is set as a label.
    The removed information can encompass multiple node attributes, the label has to be singular.

    :param graphs: Original graphs.
    :param attrs_to_delete: Attributes to be removed from the component that is refined. Should encompass any
        node-level information that is not available for a node at test time.
    :param label_attr: Attribute to be used as a label for a node, i.e. the information about the node that is refined.
    :param deleted_token: Token that the deleted attributes are set to.
    :param encode_node_ids: Whether to encode node IDs as integers. Can save memory if node IDs are long strs.
    :param encode_labels: Whether to encode node labels as integers. Can save a little memory if labels are long strs.
    :param drop_attrs: Tuple of attributes that will be dropped from all nodes to reduce memory requirements.
    :return: Prepared graph for each node in the original graphs.
    """

    return prepare_for_component_suggestion(graphs=graphs, attrs_to_delete=attrs_to_delete, label_attr=label_attr,
                                            deleted_token=deleted_token, encode_node_ids=encode_node_ids,
                                            encode_labels=encode_labels, drop_attrs=drop_attrs,
                                            remove_descendants=False, min_nb_ancestors=0)


def prepare_for_component_suggestion(graphs: Sequence[nx.DiGraph], remove_descendants: bool = True,
                                     remove_node: bool = False, predict_node: bool = False,
                                     min_nb_ancestors: int = 5, attrs_to_delete: Tuple[str] = ('category', 'type'),
                                     label_attr: Union[str, None] = 'type', deleted_token='<UNK>',
                                     encode_node_ids: bool = True, encode_labels: bool = True,
                                     drop_attrs: Tuple = ()) -> Tuple[
    Sequence[nx.DiGraph], np.ndarray]:
    """
    Prepare graphs to be used as a training set for component suggestion tasks.
    For each node, a graph missing that nodes information is created. How much of a nodes information is removed can be
    configured, i.e. certain attributes, or the node itself.
    The removed attributes can be multiple, the label has to be singular.
    Furthermore, all of the nodes descendants are removed if so configured.

    Also produces labels for the new graphs, either a specified node attribute, the nodes position, both or None.


    :param graphs: Original graphs.
    :param remove_descendants: Whether to remove the descendants of the node to be suggested.
    :param remove_node: Whether to remove the node itself. Only useful if `predict_node` is True.
    :param predict_node: Whether to add the node itself, i.e. the nodes of its incoming edges, as a prediction target.
        If this is True and a `label_attr` is provided, a Tuple of (node_label, in_nodes) is provided.
    :param min_nb_ancestors: Minimum number of ancestors a node has to have for consideration. If this criterion is not
        met, no graph is created for the node.
    :param attrs_to_delete: Attributes to be removed from the component that is suggested. Should encompass any
        node-level information that is not available for a node at test time.
    :param label_attr: Attribute to be used as a label for a node, i.e. the information about the node that is refined.
        Set to `None` if no node attribute is desired as target, i.e. for only predicting node position.
    :param deleted_token: Token that the deleted attributes are set to.
    :param encode_node_ids: Whether to encode node IDs as integers. Can save memory if node IDs are long strs.
    :param encode_labels: Whether to encode node labels as integers. Can save a little memory if labels are long strs.
    :param drop_attrs: Tuple of attributes that will be dropped from all nodes to reduce memory requirements.
    :return: Prepared graph for each node in the original graphs, Labels for each exploded graph.
    """

    def _get_new_graph(graph: nx.DiGraph, node: str):
        if min_nb_ancestors > 0:
            node_ancestors = nx.ancestors(graph, node)
            if len(node_ancestors) < min_nb_ancestors:
                return

        graph_ = graph.copy()
        del graph

        if remove_descendants:
            node_descendants = nx.descendants(graph_, node)
            for node_descendant in node_descendants:
                graph_.remove_node(node_descendant)

        if remove_node:
            graph_.remove_node(node)
        else:
            for col in attrs_to_delete:
                graph_.nodes[node][col] = deleted_token

        if encode_node_ids:
            mapping = {n : i for i, n in enumerate(graph_.nodes)}
            graph_ = nx.convert_node_labels_to_integers(graph_)
            node = mapping[node]
        graph_.graph['node'] = node
        return graph_

    def _drop_attributes(graph: nx.DiGraph):
        for drop_attr in drop_attrs:
            drop_attr_dict = nx.get_node_attributes(graph, drop_attr)
            for node in graph.nodes():
                if node in drop_attr_dict:
                    del graph.nodes[node][drop_attr]
        return graph

    def _get_in_nodes(graph: nx.DiGraph, node: str):
        return [edge_def[0] for edge_def in graph.in_edges(node)]

    def _explode_graph(graph: nx.DiGraph) -> Tuple[Sequence[nx.DiGraph], Union[
        Sequence[str], Tuple[Sequence[None], Sequence[int]], Sequence[
            Tuple[Sequence[str], Sequence[Sequence[int]]]]]]:

        if drop_attrs:
            graph = _drop_attributes(graph.copy())

        new_graphs, node_labels = [], []

        if label_attr is not None:
            labels = nx.get_node_attributes(graph, label_attr).values()
            if len(labels) < graph.number_of_nodes():
                raise UserWarning('Label attribute <{}> is not present for some nodes!'.format(label_attr))
            if predict_node:
                labels = [(label, _get_in_nodes(graph, node)) for node, label in zip(graph.nodes, labels)]
        elif predict_node:
            labels = [_get_in_nodes(graph, node) for node in graph.nodes]
        else:
            labels = [None] * graph.number_of_nodes()

        for node, node_label in zip(graph.nodes, labels):
            new_graph = _get_new_graph(graph, node)
            if new_graph is not None:
                new_graphs.append(new_graph)
                node_labels.append(node_label)

        return new_graphs, node_labels

    if encode_labels:
        all_node_labels = np.unique(np.hstack([list(nx.get_node_attributes(g, label_attr).values()) for g in graphs]))
        label_encoder = LabelEncoder().fit(all_node_labels)
        label_dtype = np.uint8 if len(label_encoder.classes_) < np.iinfo(np.uint8).max else int
        for g in graphs:
            original_labels: Dict[str, str] = dict(nx.get_node_attributes(g, label_attr))
            new_labels = np.array(label_encoder.transform(list(original_labels.values())), dtype=label_dtype)
            nx.set_node_attributes(g, {n: l_ for n, l_ in zip(original_labels.keys(), new_labels)}, label_attr)

    graph_lists, label_lists = np.array(list(map(_explode_graph, graphs)), dtype=object).T
    return np.array(list(itertools.chain(*graph_lists)), dtype=object), np.array(list(itertools.chain(*label_lists)))
