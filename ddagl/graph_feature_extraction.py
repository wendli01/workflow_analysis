"""
Feature transformers and extractors for graph- or node-level network analysis.
"""
from typing import Sequence, Dict, List, Any, Tuple, Union

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from joblib import delayed
from pandas import SparseDtype
from pandas.core.dtypes.common import is_numeric_dtype
from scipy import sparse as sp
from scipy.sparse.compressed import _cs_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


def robust_dag_longest_path_length(g: nx.DiGraph) -> int:
    """
    Return DAG longest path lenght for DAGs, 0 otherwise.
    :param g: networkx DiGraph
    :return: longest path length
    """
    try:
        nx.find_cycle(g)
        return 0
    except nx.NetworkXNoCycle:
        return nx.dag_longest_path_length(g)


def longest_path_to(g: nx.DiGraph, u: str):
    """
    Return the length of the longest direct path to the specified node.

    :param g: graph.
    :param u: target node.
    :return: length of longest path.
    """
    shortest_paths_to = nx.shortest_path_length(g, target=u)
    return np.max(list(shortest_paths_to.values()))


def longest_path_from(g: nx.DiGraph, u: str):
    """
    Return the length of the longest direct path from the specified node.

    :param g: graph.
    :param u: source node.
    :return: length of longest path.
    """
    shortest_paths_from = nx.shortest_path_length(g, source=u)
    return np.max(list(shortest_paths_from.values()))


def num_ancestors(g: nx.DiGraph, u) -> int:
    return len(nx.ancestors(g, u))


def num_descendants(g: nx.DiGraph, u) -> int:
    return len(nx.descendants(g, u))


class GraphLevelFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Graph-level feature extractor that extracts node-level features and aggregates them for each graph and also
    extracts graph-level features.

    Parameters
    ----------

    type_attr:
        Name of the attribute containing node type information.
    agg_funs:
        Sequence of aggregation functions that are used to aggregate from node-level features to the graph-level.
    node_level_funs:
        Node-level feature extraction functions, expected to return a dictionary with one value per node.
    graph_level_funs:
        Graph-level feature extraction functions.
    n_jobs:
        Number of jobs to use for featue extraction.
    """

    def __init__(self, type_attr: str = 'type', agg_funs: Tuple[callable] = (np.min, np.mean, np.var, np.max),
                 node_level_funs: Tuple[callable] = (nx.in_degree_centrality, nx.out_degree_centrality,
                                                     nx.closeness_centrality, nx.edge_betweenness_centrality,
                                                     nx.average_neighbor_degree),
                 graph_level_funs: Tuple[callable] = (nx.number_of_nodes, nx.number_of_edges, nx.density,
                                                      robust_dag_longest_path_length),
                 n_jobs: int = 1):
        self.type_col = type_attr
        self.agg_funs = agg_funs
        self.node_level_funs = node_level_funs
        self.graph_level_funs = graph_level_funs
        self.n_jobs = n_jobs

    def fit(self, *_):
        """
        Fit Extractor.

        :return: self.
        """
        return self

    def transform(self, X: Sequence[Tuple[nx.DiGraph, int]]) -> pd.DataFrame:
        if self.n_jobs not in (None, 0):
            features = joblib.Parallel(n_jobs=self.n_jobs)(delayed(self._extract_features)(g) for g in X)
        else:
            features = list(map(self._extract_features, X))
        return pd.DataFrame(list(features)).fillna(0)

    def _extract_features(self, g: nx.DiGraph) -> Dict[str, float]:
        num_nodes = len(g.nodes())
        if num_nodes == 0:
            return dict(num_nodes=num_nodes, num_edges=0, num_types=0)

        node_types = list(nx.get_node_attributes(g, self.type_col).values())
        unique_types, counts = np.unique(node_types, return_counts=True)

        feature_dict = dict(num_types=len(unique_types), highest_type_count=np.max(counts))

        for node_level_fun in self.node_level_funs:
            res = list(node_level_fun(g).values())
            for agg_fun in self.agg_funs:
                feature_dict[node_level_fun.__name__ + '-' + agg_fun.__name__] = agg_fun(res)
        for graph_level_fun in self.graph_level_funs:
            feature_dict[graph_level_fun.__name__] = graph_level_fun(g)
        return feature_dict


class NodeLevelFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Node-level feature extractor that extracts node-level features and aggregates them for each graph and also
    extracts graph-level features.

    Parameters
    ----------

    type_attr:
        Name of the attribute containing node type information.
    node_level_funs:
        Node-level feature extraction functions, expected to return a dictionary with one value per node.
    graph_level_funs:
        Graph-level feature extraction functions.
    n_jobs:
        Number of jobs to use for featue extraction.
    """

    def __init__(self, type_attr: str = 'type',
                 node_level_funs: Tuple[callable] = (nx.in_degree_centrality, nx.out_degree_centrality,
                                                     nx.closeness_centrality, nx.harmonic_centrality, nx.pagerank,
                                                     longest_path_to, longest_path_from, num_ancestors,
                                                     num_descendants, nx.average_neighbor_degree,
                                                     nx.load_centrality, nx.katz_centrality, nx.betweenness_centrality),
                 graph_level_funs: Tuple[callable] = (nx.number_of_nodes, nx.number_of_edges, nx.density,
                                                      robust_dag_longest_path_length),
                 n_jobs: int = 1):
        self.type_col = type_attr
        self.node_level_funs = node_level_funs
        self.graph_level_funs = graph_level_funs
        self.n_jobs = n_jobs

    def fit(self, *_):
        """
        Fit Extractor.

        :return: self.
        """
        return self

    def transform(self, X: Sequence[Tuple[nx.DiGraph, int]]) -> pd.DataFrame:
        if self.n_jobs not in (None, 0):
            features = joblib.Parallel(n_jobs=self.n_jobs)(delayed(self._extract_features)(g, n) for (g, n) in X)
        else:
            features = list(map(self._extract_features, X))
        return pd.DataFrame(list(features)).fillna(0)

    def _extract_features(self, g: nx.DiGraph, n: int) -> Dict[str, float]:
        num_nodes = len(g.nodes())
        if num_nodes == 0:
            return dict(num_nodes=num_nodes, num_edges=0, num_types=0)

        node_types = list(nx.get_node_attributes(g, self.type_col).values())
        node_type_counts = pd.value_counts(node_types).to_dict()
        unique_types, counts = np.unique(node_types, return_counts=True)

        feature_dict = dict(num_types=len(unique_types), highest_type_count=np.max(counts),
                            **node_type_counts)

        for node_level_fun in self.node_level_funs:
            try:
                res = node_level_fun(g, u=n)
            except TypeError:
                res = node_level_fun(g)[n]
            feature_dict[node_level_fun.__name__] = res
        for graph_level_fun in self.graph_level_funs:
            feature_dict[graph_level_fun.__name__] = graph_level_fun(g)
        return feature_dict


class NodeTypeLevelFeatureExtractor(BaseEstimator, TransformerMixin):
    """
        Node-type-level feature extractor. Computes features and aggregates them for each node type feature.
        This is useful for extracting usage features for node types.
    """

    def __init__(self, type_col: str = 'type', category_col: str = 'category', agg_fun: callable = np.mean):
        self.type_col = type_col
        self.category_col = category_col
        self.agg_fun = agg_fun

    def fit(self, *_):
        """
        Fit Extractor.

        :return: self.
        """
        return self

    def transform(self, X: Sequence[nx.DiGraph], processor_features: pd.DataFrame = None):
        return self._extract_processor_features(X, processor_features)

    def _extract_processor_features(self, workflow_graphs: Sequence[nx.DiGraph],
                                    processor_features: pd.DataFrame = None, processor_id_col: str = 'processor'):
        def _get_type_proportions(g: nx.DiGraph):
            node_types = list(nx.get_node_attributes(g, 'type').values())
            unique_types, counts = np.unique(node_types, return_counts=True)
            num_nodes = g.number_of_nodes()
            return [{processor_id_col: t, 'processor_proportion': c / num_nodes, 'processor_num': c}
                    for t, c in zip(unique_types, counts)]

        def _get_type_prevalence(graphs: pd.Series):
            def _get_prevalence(g: nx.DiGraph) -> np.ndarray:
                return np.unique(list(nx.get_node_attributes(g, self.type_col).values()))

            unique_types, counts = np.unique(np.hstack(graphs.apply(_get_prevalence)), return_counts=True)
            return pd.Series(index=unique_types, data=counts / len(graphs), name='processor_prevalence')

        def _get_degrees(g: nx.DiGraph) -> Sequence[Dict[str, Any]]:
            node_types = nx.get_node_attributes(g, self.type_col)
            in_degs, out_degs = nx.in_degree_centrality(g), nx.out_degree_centrality(g)
            return [{processor_id_col: node_types[n], 'in_degree': in_degs[n], 'out_degree': out_degs[n]}
                    for n in g.nodes()]

        def _get_path_lengths(g: nx.DiGraph) -> Sequence[Dict[str, Any]]:
            node_types = nx.get_node_attributes(g, self.type_col)
            return [{processor_id_col: node_types[n], 'longest_path_to': longest_path_to(g, n),
                     'longest_path_from': longest_path_from(g, n)} for n in g.nodes()]

        def _get_ancestors_descendents(g: nx.DiGraph) -> Sequence[Dict[str, Any]]:
            node_types = nx.get_node_attributes(g, self.type_col)
            return [{processor_id_col: node_types[n], 'ancestors': len(nx.ancestors(g, n)),
                     'descendants': len(nx.descendants(g, n))} for n in g.nodes()]

        def _get_features(feature_fun):
            raw_features = pd.DataFrame.from_records(np.hstack(workflow_graphs.apply(feature_fun)))
            return raw_features

        def _aggregate_features(feature_list):
            def _deagg(df: pd.DataFrame):
                for col in df.columns:
                    if col != processor_id_col:
                        feature_cols.append(df.groupby(processor_id_col)[col].apply(list))

            if self.agg_fun is None:
                feature_cols = []
                for f in feature_list:
                    if type(f) is pd.Series:
                        feature_cols.append(f)
                    else:
                        _deagg(f)
                return feature_cols
            return [f if type(f) is pd.Series else f.groupby(processor_id_col).aggregate(self.agg_fun) for f in
                    features]

        workflow_graphs = pd.Series(workflow_graphs)

        depths = _get_features(_get_path_lengths)

        node_proportion = _get_features(_get_type_proportions)
        node_prevalence = _get_type_prevalence(workflow_graphs)

        degrees = _get_features(_get_degrees)

        ancestors_descendents = _get_features(_get_ancestors_descendents)

        features = [node_prevalence, node_proportion, degrees, depths, ancestors_descendents]
        feature_df = pd.concat(_aggregate_features(features), axis=1)

        if processor_features is None:
            return processor_features
        return feature_df.join(processor_features)


class NodeLevelFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms node attributes of a graph into a node-level feature matrix. Automatically adds dummies for all features.
    Can also remove singular features, split features that represent a matrix into columns, scale and add features from
    textual descriptions.

    Parameters
    ----------

    type_column:
        name of attribute that contains the node type.
    config_column:
        name of attribute that contains the node configuration dict.
    use_configs:
        whether or not to use the node configurations.
    drop_singular_features:
        whether or not to drop singular features (i.e. features that consist of only one non-nan value).
    scale:
        whether or not to use a standard scaler to scale the final config features.
    split_sequence_features:
        whether or not to split sequence features found in node configs into separate columns.
    description_column:
        Name of column containing node-level descriptions - e.g. corresponding to the node types.
    description_transformer:
        Transformer that generates a numerical feature representation from the descriptions.
    """

    def __init__(self, type_column: str = 'type', config_column: str = 'config', use_configs: bool = True,
                 drop_singular_features: bool = True, scale: bool = True, split_sequence_features: bool = True,
                 node_id_column: str = None, description_column: Union[str, None] = None,
                 description_transformer=TfidfVectorizer(stop_words='english', ngram_range=(3, 3), min_df=2)):
        self.type_column = type_column
        self.config_column = config_column
        self.type_encoder_ = None
        self.use_configs = use_configs
        self.drop_singular_features = drop_singular_features
        self.scale = scale
        self.split_sequence_features = split_sequence_features
        self.is_singular_, self.categorical_cols_, self.config_cols_ = None, [], None
        self.imputer_, self.scaler_ = None, None
        self.node_id_column = node_id_column
        self.description_column, self.description_transformer = description_column, description_transformer

    def fit(self, graphs: Sequence[nx.DiGraph], *_):
        """
        Fit Extractor.

        :param graphs: Sequence of graphs.
        :return: self.
        """

        def _are_singular_features(configs_df: pd.DataFrame):
            return {col: len(configs_df[col].dropna().astype(str).unique()) <= 1
                    for col in configs_df.columns}

        def _set_cat_features(g: nx.DiGraph):
            configs_df = pd.DataFrame(nx.get_node_attributes(g, self.config_column).values())
            for col in configs_df.columns:
                if not is_numeric_dtype(configs_df[col]) and col not in self.categorical_cols_:
                    self.categorical_cols_.append(col)

        all_types = np.unique(np.hstack([list(nx.get_node_attributes(g, self.type_column).values()) for g in graphs]))
        self.type_encoder_ = OneHotEncoder(handle_unknown='ignore', sparse=self.use_configs)
        self.type_encoder_.fit(np.array(sorted(all_types)).reshape(-1, 1))

        if self.use_configs:
            list(map(_set_cat_features, graphs))

            all_config_features = self._get_all_config_features(graphs)
            all_config_features = self._convert_config_features(all_config_features)

            if self.drop_singular_features:
                is_singular = _are_singular_features(all_config_features)
                self.config_cols_ = [col for col in all_config_features if not is_singular[col]]
                all_config_features = all_config_features[self.config_cols_]
            else:
                self.config_cols_ = all_config_features.columns.values

            self.imputer_ = SimpleImputer().fit(all_config_features.values)
            if self.scale:
                self.scaler_ = StandardScaler().fit(all_config_features.values)

        if self.description_column is not None:
            unique_descriptions = np.unique(np.hstack([np.unique(self._get_node_descriptions(g)) for g in graphs]))
            self.description_transformer.fit(unique_descriptions)

        return self

    def _convert_config_features(self, configs_df: pd.DataFrame) -> pd.DataFrame:
        def _convert_categorical(col: str):
            all_sequences = all([isinstance(v, Sequence) for v in configs_df[col]])
            if all_sequences and self.split_sequence_features:
                max_len = np.max(list(map(len, configs_df[col])))
                for pos in range(max_len):
                    values = [v[pos] if len(v) > pos else np.nan for v in configs_df[col]]
                    configs_df[col + '_{}'.format(pos)] = pd.arrays.SparseArray(values, fill_value=np.nan)
                return configs_df.drop(col, axis='columns')

            dummies = pd.get_dummies(configs_df[col].astype(str), prefix=col, sparse=True)
            dummies = dummies.astype({dummy_col: SparseDtype(dummies.dtypes[dummy_col], fill_value=np.nan)
                                      for dummy_col in dummies})
            return pd.concat([configs_df.drop(col, axis='columns'), dummies], 1)

        for col in configs_df.columns:
            if col in self.categorical_cols_:
                configs_df = _convert_categorical(col)
            else:
                configs_df[col + '_nan'] = configs_df[col].isna().astype(
                    SparseDtype(configs_df[col].dtype, fill_value=np.nan))
        if self.config_cols_ is not None:
            configs_df = configs_df[[c for c in self.config_cols_ if c in configs_df.columns]]
        return pd.DataFrame(configs_df)

    def _get_config_features(self, g: nx.DiGraph) -> Sequence[Dict[str, Dict]]:
        node_attributes = nx.get_node_attributes(g, self.config_column)
        return list(dict(node_attributes).values())

    def _get_all_config_features(self, graphs: Sequence[nx.DiGraph]):
        all_config_dicts = list(np.hstack(list(map(self._get_config_features, graphs))))
        index = list(np.hstack([g.nodes() for g in graphs]))
        return pd.DataFrame(index=index).from_dict(all_config_dicts, dtype=pd.SparseDtype(object))

    def _get_description_features(self, g: nx.DiGraph) -> _cs_matrix:
        node_descriptions = self._get_node_descriptions(g)
        return self.description_transformer.transform(node_descriptions)

    def _get_node_descriptions(self, g: nx.DiGraph) -> List[str]:
        return list(dict(nx.get_node_attributes(g, self.description_column)).values())

    @staticmethod
    def _get_node_types(g: nx.DiGraph) -> List:
        attributes = nx.get_node_attributes(g, 'type')
        return list(NodeLevelFeatureTransformer._fill_attribute_dict(attributes, g.nodes).values())

    @staticmethod
    def _fill_attribute_dict(attributes: Dict[int, Any], index: Sequence[int]) -> Dict[int, Any]:
        for ind in index:
            if ind not in attributes:
                attributes[ind] = np.nan
        return {ind: attributes[ind] for ind in sorted(index)}

    def transform(self, graphs: Sequence[nx.DiGraph]) -> Union[
        Sequence[Tuple[nx.DiGraph, _cs_matrix]], Sequence[Tuple[nx.DiGraph, _cs_matrix, int]]]:
        def _split_features() -> Sequence[_cs_matrix]:
            cum_nb_nodes = 0
            for g in graphs:
                yield all_features[cum_nb_nodes: cum_nb_nodes + g.number_of_nodes()]
                cum_nb_nodes += g.number_of_nodes()

        if self.type_encoder_ is None:
            raise UserWarning('Transformer needs to be fit first.')

        all_node_types = list(np.hstack(list(map(self._get_node_types, graphs))))
        all_node_type_features = self.type_encoder_.transform(np.array(all_node_types).reshape(-1, 1))

        if self.use_configs:
            all_config_features: pd.DataFrame = self._get_all_config_features(graphs)
            all_config_features = self._convert_config_features(all_config_features)
            all_config_features: np.ndarray = self.imputer_.transform(all_config_features.values)
            if self.scaler_ is not None:
                all_config_features = self.scaler_.transform(all_config_features)
            all_features = sp.hstack([all_node_type_features, all_config_features], format='csr')
        else:
            all_features = all_node_type_features

        if self.description_column is not None:
            all_description_features = sp.vstack(list(map(self._get_description_features, graphs)))
            all_features = sp.hstack([all_features, all_description_features], format='csr')

        if self.node_id_column is not None:
            node_ids: Sequence[int] = [g.graph[self.node_id_column] for g in graphs]
            return list(zip(graphs, list(_split_features()), node_ids))

        return list(zip(graphs, list(_split_features())))


class FeatureAdder(BaseEstimator, TransformerMixin):
    """
    Adds node-level features via provided extraction functions.

    Parameters
    ----------

    node_level_funs:
        Functions that extract node-level features, e.g. networkx centralitiy functions.
    scale:
        Whether to scale features to zero mean and unit variance across all nodes from all graphs, i.e. column-wise.

    """

    def __init__(self, node_level_funs: Sequence[callable] = (nx.in_degree_centrality,), scale: bool = True):
        self.node_level_funs = node_level_funs
        self.scale = scale
        self.means_, self.stds_ = {}, {}

    def _add_features(self, x) -> Tuple[nx.DiGraph, sp.coo_matrix]:
        g, features = x
        new_features = []
        for node_level_fun in self.node_level_funs:
            new_feature = list(node_level_fun(g).values())
            if self.scale:
                new_feature = (np.array(new_feature) - self.means_[node_level_fun]) / self.stds_[node_level_fun]
            new_features.append(new_feature)
        features = sp.hstack([features, sp.csr_matrix(new_features).T])
        return g, features

    def fit(self, X, *_):
        if self.scale:
            for node_level_fun in self.node_level_funs:
                features = np.hstack([list(node_level_fun(g).values()) for g, f in X])
                self.means_[node_level_fun] = np.mean(features)
                self.stds_[node_level_fun] = np.std(features)
        return self

    def transform(self, X, *_):
        return np.array(list(map(self._add_features, X)), dtype=object)


def _to_bidirected(g: nx.DiGraph) -> nx.DiGraph:
    return nx.disjoint_union(g, g.reverse(copy=False))


bidirected_transformer = FunctionTransformer(lambda graphs: np.array(list(map(_to_bidirected, graphs)), dtype=object),
                                             check_inverse=False)
