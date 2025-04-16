import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.serif'] = 'Helvetica'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['font.style'] = 'italic'
matplotlib.rcParams['text.usetex'] = True


def definition_plot_basic(
        rectangular_plot: bool = False,
        labels: list = None,
        x_limits: list = None,
        y_limits: list = None,
):
    """

    :param: x_limits
    :param: y_limits
    :param: labels
    :param: rectangular_plot: bool, whether to use a rectangular shape for the plot in case of a more extended side
    :return: basic definition of the axes for the plotting
    """

    fig = plt.figure(
        figsize=(7, 7) if not rectangular_plot else (8, 6),
        constrained_layout=True
    )
    gs = grd.GridSpec(1, 1, figure=fig)
    ax = plt.subplot(gs[0])

    ax.tick_params(axis="both", labelsize=16)
    ax.tick_params(which="major", direction="in", length=7, width=2)
    ax.tick_params(which="minor", direction="in", length=4, width=1)

    if labels is not None:
        ax.set_xlabel(labels[0], fontsize=16)
        ax.set_ylabel(labels[1], fontsize=16)

    # limits of the plot
    if x_limits is not None:
        ax.set_xlim(x_limits[0], x_limits[1])
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    return ax
