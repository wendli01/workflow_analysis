# Deep Directed Acyclical Graph Learning tools

This module provides tools for unsupervised and supervised learning on DAGs.

Be advised that while the models in `graph_level_nn` implement `sklearn` estimators, they are **not** fully compatible.
Specifically, their inheritance structure makes them **usuitable** for cloning via `get_params()` and `set_params()`, e.g.
in `sklearn.model_selection.cross_val_score()`.

---

It contains the following modules:

-`evaluation` <br/>
Evaluation functions and helpers for various network analysis tasks.

-`graph_level_feature_extraction` <br/>
Feature transformers and extractors for graph- or node-level network analysis.

-`graph_level_nn` <br/>
Poling Graph-Convolutional Neural Networks (P-GCNs) for various (un)supervised tasks.
P-GCN is optimized for high throughput even and especially with many small graphs by transforming input data into a
larger disconnected batch. Similar optimizations are applied wherever processing single graphs is necessary.

-`visualization` <br/>
Tools for visualizing DAGs, training histories and network activations.

---

For more information, please refer to the respective component documentation.