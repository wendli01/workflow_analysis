import importlib
import os
from urllib import request

import tensorflow as tf
import numpy as np
import json
import base64
import networkx as nx
from typing import Sequence, Dict, Union, Tuple, Any, List

PROTOBUF_MODULE_NAME = 'model_metrics_pb2'
PROTOBUF_MODULE_URL = 'https://raw.githubusercontent.com/google-research/nasbench/master/nasbench/lib/model_metrics_pb2.py'


def load_nasbench(dataset_file: str) -> Tuple[Dict[str, nx.DiGraph], List[Dict[str, Any]]]:
    def _get_protobuf_module():
        if not os.path.exists(PROTOBUF_MODULE_NAME + '.py'):
            request.urlretrieve(PROTOBUF_MODULE_URL, PROTOBUF_MODULE_NAME + '.py')
        return importlib.import_module(PROTOBUF_MODULE_NAME)

    def _delete_protobuf_module():
        os.remove(PROTOBUF_MODULE_NAME + '.py')

    def _create_graph(adj: np.ndarray, operations: Sequence[str]) -> nx.DiGraph:
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)

        type_dict = {node_id: op for node_id, op in zip(graph.nodes, operations)}
        nx.set_node_attributes(graph, type_dict, name='type')
        return graph

    def _create_evaluation_dict(raw_metrics, epochs: int) -> Dict[str, Union[float, int]]:
        metrics = model_metrics_pb2.ModelMetrics.FromString(
            base64.b64decode(raw_metrics))

        # Evaluation statistics at the end of training
        final_evaluation = metrics.evaluation_data[2]

        return dict(training_time=final_evaluation.training_time,
                    train_accuracy=final_evaluation.train_accuracy,
                    validation_accuracy=final_evaluation.validation_accuracy,
                    test_accuracy=final_evaluation.test_accuracy,
                    trainable_params=metrics.trainable_parameters,
                    epochs=epochs)

    model_metrics_pb2 = _get_protobuf_module()
    rows, graphs = [], {}

    try:
        for serialized_row in tf.python_io.tf_record_iterator(dataset_file):
            module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = (
                json.loads(serialized_row.decode('utf-8')))

            if module_hash not in graphs:
                dim = int(np.sqrt(len(raw_adjacency)))
                adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
                adjacency = np.reshape(adjacency, (dim, dim))
                operations = raw_operations.split(',')
                graphs[module_hash] = _create_graph(adjacency, operations)

            row = {**_create_evaluation_dict(raw_metrics, epochs), 'graph_id': module_hash}
            rows.append(row)
    finally:
        _delete_protobuf_module()

    return graphs, rows


def convert_nasbench(in_path: str, out_path: str, **kwargs):
    graphs, fitness_dicts = load_nasbench(in_path, **kwargs)
    data = dict(graphs={graph_id: nx.node_link_data(graph) for graph_id, graph in graphs.items()}, target=fitness_dicts)

    with open(out_path, 'w') as fp:
        json.dump(data, fp, indent=4)
