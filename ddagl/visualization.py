"""
Tools for visualizing DAGs, training histories and network activations.
"""
from typing import Tuple, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import networkx as nx
import matplotlib as mpl


def show_training_history(df: pd.DataFrame, figsize: Tuple[int, int] = (9, 9), dpi: int = 120,
                          epoch_cutoff: float = .33, mark_best_value: bool = True,
                          higher_is_better: bool = True) -> plt.Figure:
    def plot_scores(ax, start_epoch: int):
        ax.plot(df.epoch[start_epoch:], df.train_score[start_epoch:], label='train score')
        mark_best(ax, df['train_score'], higher_is_better=higher_is_better)
        if 'valid_score' in df:
            ax.plot(df.epoch[start_epoch:], df.valid_score[start_epoch:], label='valid score')
            mark_best(ax, df['valid_score'], higher_is_better=higher_is_better)
        ax.legend(), ax.grid(), plt.xlabel('epoch')
        ax.set_title('scores epochs {}+'.format(start_epoch) if start_epoch > 0 else 'scores')

    def mark_best(ax, series: pd.Series, higher_is_better, **scatter_kwargs):
        if not mark_best_value:
            return
        best_epoch = series.argmax() if higher_is_better else series.argmin()
        ax.scatter([best_epoch], series[best_epoch], marker=7 if higher_is_better else 6, **scatter_kwargs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=figsize, sharex='all', dpi=dpi)
    l: Line2D = ax1.plot(df.epoch, df.loss, label='loss')[0]
    ax1.fill_between(df.epoch, (df.loss - df.loss_std), (df.loss + df.loss_std), alpha=.25)
    mark_best(ax1, df['loss'], c=l.get_color(), higher_is_better=False)
    ax1.legend(), ax1.grid(), ax1.semilogy(), ax1.set_title('training loss')

    plot_scores(ax2, 0)
    plot_scores(ax3, start_epoch=int(round(epoch_cutoff * len(df))))
    plt.xlabel('epoch')
    plt.tight_layout()
    return fig


def show_activations(layer_activations: np.ndarray, layer_names_list=None, fn: str = 'activations.gif',
                     interval: int = 500, cmap: str = 'RdBu', center: float = 0, col_width: float = 4,
                     row_height: float = 2, ncols=None, **heatmap_kwargs):
    def _plot_activation(layer_activations: Sequence[np.ndarray], layer_names=None) -> Tuple[None, plt.Figure]:
        for ind, activations in enumerate(layer_activations):
            if len(activations.shape) == 1:
                activations = [activations]
                square = False
            else:
                square = True

            ax = np.hstack(axes)[ind]
            ax.cla()
            annot = bool(np.max(np.shape(activations)) <= 4)
            sns.heatmap(activations, center=center, cmap=cmap, square=square, cbar=False, vmin=vmin, vmax=vmax,
                        yticklabels=[], xticklabels=[], ax=ax, annot=annot, **heatmap_kwargs)
            shape_str = 'x'.join(list(map(str, np.shape(activations))))

            layer_name = layer_names[ind] if layer_names is not None else 'layer {}'.format(ind)
            ax.set_title(layer_name + ' ({})'.format(shape_str))
        return None, axes

    def update(i: int) -> Tuple[None, plt.Figure]:
        ret = _plot_activation(layer_activations[i],
                               layer_names_list[i]) if layer_names_list is not None else _plot_activation(
            layer_activations[i])
        plt.suptitle('sample {}'.format(i))
        return ret

    vmin = np.min([np.min(a) for a in np.hstack(layer_activations)])
    vmax = np.max([np.max(a) for a in np.hstack(layer_activations)])

    nb_layers = len(layer_activations[0])
    ncols = int(np.ceil(nb_layers ** .5)) if ncols is None else ncols
    nrows = int(np.ceil(nb_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(col_width * ncols, row_height * nrows), dpi=120)
    for ax in np.hstack(axes)[nb_layers:]:
        ax.axis('off')
    plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(layer_activations), interval=interval, blit=False)
    anim.save(fn, writer='imagemagick')
    plt.close()


def draw_workflow_graph(G: nx.DiGraph, cmap: str = 'Set3', base_node_size=500, dpi: int = 90,
                        layout: nx.drawing.layout = nx.circular_layout, type_col='type',
                        label_col='name', ax=None, show_legend=False):
    def get_colors(labels: Sequence, alpha=1.0):
        types = set(labels)
        color_values = [plt.get_cmap(cmap)(i / len(types)) for i, _ in enumerate(types)]
        type_colors = {k: (*v[:3], alpha) for k, v in zip(types, color_values)}

        return [type_colors[l] for l in labels]

    def create_legend(labels, loc=None, title=''):
        type_colors = get_colors(labels)
        legend_elements = [mpl.lines.Line2D([0], [0], color=c, label=l, linewidth=5)
                           for c, l in zip(type_colors, labels)]
        return ax.legend(handles=legend_elements, loc=loc, title=title)

    if G.number_of_nodes() < 1:
        return
    types = nx.get_node_attributes(G, type_col)
    labels = nx.get_node_attributes(G, label_col)
    node_size = [d * base_node_size for d in dict(G.degree).values()]

    pos = layout(G)
    size = 3 * G.number_of_nodes() ** .5
    if ax is None:
        fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)

    nx.draw_networkx_nodes(G, node_color=get_colors(labels.values()), pos=pos, ax=ax, node_size=node_size)
    nx.draw_networkx_edges(G, width=3, alpha=.5, pos=pos, ax=ax, arrowsize=base_node_size / 20,
                           node_size=node_size)
    nx.draw_networkx_labels(G, labels=labels, ax=ax, pos=pos)

    if show_legend:
        create_legend(labels=set(types.values()), title=type_col)
    ax.axis('off')
