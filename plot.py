import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter3d(x, y, z, title='scatter3d', init_view=(30, -45), display_ticks=True):
    """Create a scatterplot (y-up)"""
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, [.1, .1, .7, .8], auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.scatter3D(x, z, y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(init_view[0], init_view[1])
    if not display_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    plt.savefig(os.path.join('.', 'output', f'{title}.png'))


def plot_loss(losses):
    """Plot the loss history on a line plot"""
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    ax.plot(losses, label="chamfer loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    plt.savefig('./output/plot.png')