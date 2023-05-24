import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import numpy as np
import pandas as pd
import colormaps as cmaps

smarkers = {
    'avg': 'o',
    'max': '^'}

scolors = {
    'avg': 'tab:blue',
    'max': 'tab:red'}


def dot_plot(data, error=None, datalabels=None, ax=None,
             marker='.', markersize=10, **kwargs):
    colormap = mpl.cm.Dark2.colors
    colormap2 = cmaps.bold.colors
    if ax is None:
        ax = plt.gca()

    try:
        n = len(data)
    except ValueError:
        n = data.size

    y = np.arange(n)[::-1]

    l = ax.plot(data, y, marker=marker, linestyle='', markersize=markersize,
                markeredgewidth=0, **kwargs)

    if error is not None:
        lo = data - error
        hi = data + error

        l1 = ax.hlines(y, lo, hi, color=l[0].get_color())
        l.append(l1)

    ticks = ax.yaxis.set_ticks(range(n))
    text = ax.yaxis.set_ticklabels(datalabels[::-1])
    colors = colormap
    indices = list(reversed(list(range(len(ax.get_yticklabels())))))
    Glycolysis = indices[:10]
    PPP = indices[10:12]
    TCA = indices[12:22]
    AAA = indices[22:38]
    Bio = indices[38]

    for i, xtick in enumerate(ax.get_yticklabels()):
        if i in Glycolysis:
            xtick.set_color(colors[0])
        elif i in PPP:
            xtick.set_color(colors[1])
        elif i in TCA:
            xtick.set_color(colormap2[2])
        elif i in AAA:
            xtick.set_color(colors[3])
        elif i == Bio:
            xtick.set_color(colors[4])
        else:
            xtick.set_color('k')
    ax.set_ylim(-1, n)

    ax.tick_params(axis='y', which='major', right='on', left='on', color='0.8')
    ax.grid(axis='y', which='major', color='0.8', zorder=-10, linestyle='-', label='data')
    return l

if __name__ == '__main__':
    data_extra_raw = pd.read_excel('multi_scale_model/result/flux_rate/result.xlsx', sheet_name="no-sum")
    data_extra_raw.columns = ['Process Time'] + ['24 Hours'] * 8 + ['48 Hours'] * 8

    with plt.style.context('tableau-colorblind10'):
        colormap = mpl.cm.tab10.colors
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(1, 1, 1)
        dot_plot(data_extra_raw.iloc[2:-1, 8].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[4], label=r'Inner Cell (360 $\mu$m)', markersize=10)
        dot_plot(data_extra_raw.iloc[2:-1, 6].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[3], label=r'Inner Cell (240 $\mu$m)', marker='^', markersize=6)
        dot_plot(data_extra_raw.iloc[2:-1, 4].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[2], label=r'Inner Cell (120 $\mu$m)', marker='*', markersize=8)
        dot_plot(data_extra_raw.iloc[2:-1, 2].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[1], label=r'Inner Cell (60 $\mu$m)', marker='s', markersize=5)
        dot_plot(data_extra_raw.iloc[2:-1, 1].values, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[0], marker='v', markersize=6, label='Outer Cell')

        ax.axvline(x=2200, color='black', ls='--')
        outer_cell = data_extra_raw.iloc[2:-1, 1].values
        dot_plot(data_extra_raw.iloc[2:-1, 9].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 marker='v', markersize=8, color=colormap[0])
        dot_plot(data_extra_raw.iloc[2:-1, 10].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[1])
        dot_plot(data_extra_raw.iloc[2:-1, 12].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[2])
        dot_plot(data_extra_raw.iloc[2:-1, 14].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[3])
        dot_plot(data_extra_raw.iloc[2:-1, 16].values + 2500, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[4])
        ax.set_xticks(labels=[0, 1000, 2000, 0, 1000, 2000, 3000], ticks=[0, 1000, 2000, 2500, 3500, 4500, 5500],
                      fontsize=12)
        plt.title("24 Hours                                    48 Hours", fontsize=16)
        plt.xlabel(r"Flux Rates (nmol/$10^6$ cells/h)", fontsize=16)
        plt.legend(bbox_to_anchor=(1., 1.15), ncol=5, columnspacing=0.2, labelspacing=0.2, handletextpad=0.2)
        # plt.savefig("multi_scale_model/result/flux_rate/flux_comparison.svg",
        #             bbox_inches='tight')
        # plt.savefig("multi_scale_model/result/flux_rate/flux_comparison.pdf",
        #             bbox_inches='tight')
        plt.show()


    data_extra_raw = pd.read_excel('multi_scale_model/result/flux_rate/result.xlsx', sheet_name="Sheet1")
    data_extra_raw.columns = ['Process Time'] + ['24 Hours'] * 8 + ['48 Hours'] * 8
    with plt.style.context('seaborn-v0_8-colorblind'):
        colormap = mpl.cm.tab10.colors
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(1, 1, 1)
        outer_cell = data_extra_raw.iloc[2:-1, 1].values
        # dot_plot([1] * len(data_extra_raw.iloc[2:-1, 1].values), datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax, color=colormap[0], marker='*', markersize=14)
        dot_plot(data_extra_raw.iloc[2:-1, 8].values / outer_cell, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[4], label=r'Inner Cell (360 $\mu$m)', markersize=10)
        dot_plot(data_extra_raw.iloc[2:-1, 6].values / outer_cell, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[3], label=r'Inner Cell (240 $\mu$m)', marker='^', markersize=6)
        dot_plot(data_extra_raw.iloc[2:-1, 4].values / outer_cell, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[2], label=r'Inner Cell (120 $\mu$m)', marker='*', markersize=8)
        dot_plot(data_extra_raw.iloc[2:-1, 2].values / outer_cell, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax,
                 color=colormap[1], label=r'Inner Cell (60 $\mu$m)', marker='s', markersize=5)
        ax.axvline(x=1.6, color='black', ls='-')
        ax.axvline(x=1, color=colormap[0], ls='--', label='Outer Cell')
        ax.axvline(x=3, color=colormap[0], ls='--')
        outer_cell = data_extra_raw.iloc[2:-1, 9].values
        # dot_plot(data_extra_raw.iloc[2:-1, 9].values / outer_cell + 2, datalabels=data_extra_raw.iloc[2:-1, 0].values, ax=ax, color=colormap[0], marker='*', markersize=14)
        dot_plot(data_extra_raw.iloc[2:-1, 10].values / outer_cell + 2, datalabels=data_extra_raw.iloc[2:-1, 0].values,
                 ax=ax, color=colormap[1], marker='s', markersize=5)
        dot_plot(data_extra_raw.iloc[2:-1, 12].values / outer_cell + 2, datalabels=data_extra_raw.iloc[2:-1, 0].values,
                 ax=ax, color=colormap[2], marker='*', markersize=8)
        dot_plot(data_extra_raw.iloc[2:-1, 14].values / outer_cell + 2, datalabels=data_extra_raw.iloc[2:-1, 0].values,
                 ax=ax, color=colormap[3], marker='^', markersize=6)
        dot_plot(data_extra_raw.iloc[2:-1, 16].values / outer_cell + 2, datalabels=data_extra_raw.iloc[2:-1, 0].values,
                 ax=ax, color=colormap[4])
        ax.set_xticks(labels=[0, 50, 100, 150, 0, 50, 100], ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fontsize=12)
        plt.title("24 Hours                                    48 Hours", fontsize=16)
        plt.xlabel("Relative Flux Rates (%)", fontsize=16)
        plt.legend(bbox_to_anchor=(1., 1.12), ncol=5, columnspacing=0.2, labelspacing=0.5, handletextpad=0.)
        plt.savefig("multi_scale_model/result/flux_rate/relative_flux.svg",
                    bbox_inches='tight')
        plt.savefig("multi_scale_model/result/flux_rate/relative_flux.pdf",
                    bbox_inches='tight')
        plt.show()

