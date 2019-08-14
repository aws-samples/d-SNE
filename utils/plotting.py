"""
Plotting helper functions
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn import manifold
import seaborn as sns
import numpy as np


def plot_2d_embeds(X, y, text=None, save_path=None):
    """
    Plot 2D embeddings
    :param X: Embeddings
    :param y: label
    :param text: text of plot, None by default
    :param save_path: save path, None by default
    :return:
    """
    f = plt.figure(figsize=(8, 8))

    classes = np.unique(y)
    n_class = len(classes)
    palette = np.array(sns.color_palette('hls', n_class))

    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X[:, 0], X[:, 1], lw=0, s=40, c=palette[y.astype(np.int)], alpha=0.3)
    if text is not None:
        ax.text(1, 1, text, ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    ax.axis('tight')

    for c in classes:
        x_txt, y_txt = np.mean(X[y == c, :], axis=0)
        txt = ax.text(x_txt, y_txt, str(c))

        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'),
                              PathEffects.Normal()])

    if save_path is not None:
        f.savefig(save_path)
    else:
        plt.show()

    plt.close()


# def plot_3d_embeds(X, y, text=None, save_path=None):
#     f = plt.figure(figsize=(8, 8))
#
#     classes = np.unique(y)
#     n_class = len(classes)
#
#     palette = np.array(sns.color_palette('hls', n_class))


def cal_tsne_embeds(X, y, n_components=2, text=None, save_path=None):
    """
    Plot using tSNE
    :param X: embedding
    :param y: label
    :param n_components: number of components
    :param text: text for plot
    :param save_path: save path
    :return:
    """
    X = X[: 500]
    y = y[: 500]

    tsne = manifold.TSNE(n_components=n_components)
    X_tsne = tsne.fit_transform(X, y)

    plot_2d_embeds(X_tsne, y, text, save_path)


def plot_embedding_src_tgt(Xs, ys, Xt, yt, text=None, save_path=None, names=None):
    """
    Plot source and target embeddings
    :param Xs: embeddings for source
    :param ys: labels for source
    :param Xt: embeddings for target
    :param yt: labels for target
    :param text: text of plot
    :param save_path: save path
    :return:
    """
    f = plt.figure(figsize=(8, 8))

    classes = np.unique(ys)
    n_class = len(classes)
    palette = np.array(sns.color_palette('hls', n_class))

    ax = plt.subplot(aspect='equal')
    ax.scatter(Xs[:, 0], Xs[:, 1], lw=0, s=80, c=palette[ys.astype(np.int)], marker='o', alpha=0.3)
    ax.scatter(Xt[:, 0], Xt[:, 1], lw=0, s=80, c=palette[yt.astype(np.int)], marker='*', alpha=0.7)

    if text is not None:
        ax.text(1, 1, text, ha='center', va='center', transform=ax.transAxes)

    ax.axis('off')
    ax.axis('tight')

    for c in classes:
        if names is None:
            c_name = str(c)
        else:
            c_name = names[str(c)]

        x_txt, y_txt = np.mean(Xs[ys == c, :], axis=0)
        txt = ax.text(x_txt, y_txt, c_name)

        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'),
                              PathEffects.Normal()])

        x_txt, y_txt = np.mean(Xt[yt == c, :], axis=0)

        txt = ax.text(x_txt, y_txt, c_name)

        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'),
                              PathEffects.Normal()])

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        f.savefig(save_path)
    else:
        plt.show()

    plt.close()


def cal_tsne_embeds_src_tgt(Xs, ys, Xt, yt, n_components=2, text=None, save_path=None, n_samples=1000, names=None):
    """
    Plot embedding for both source and target domain using tSNE
    :param Xs:
    :param ys:
    :param Xt:
    :param yt:
    :param n_components:
    :param text:
    :param save_path:
    :return:
    """
    Xs = Xs[: min(len(Xs), n_samples)]
    ys = ys[: min(len(ys), n_samples)]
    Xt = Xt[: min(len(Xt), n_samples)]
    yt = yt[: min(len(Xt), n_samples)]
    
    X = np.concatenate((Xs, Xt), axis=0)
    tsne = manifold.TSNE(n_components=n_components)
    X = tsne.fit_transform(X)
    Xs = X[: len(Xs)]
    Xt = X[len(Xs):]

    plot_embedding_src_tgt(Xs, ys, Xt, yt, text, save_path, names=names)
