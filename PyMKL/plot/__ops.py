from typing import List, Union, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mcolors.to_rgb(c1))
    c2=np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((1-mix)*c1 + mix*c2)


def descriptors(descriptors: List[np.ndarray], groupby: int, figsize: Tuple[int,int] = (16,16),
                varnames: Union[np.ndarray, list] = None, dimnames: Union[np.ndarray, list] = None,
                same_scale: bool = True, return_axes: bool = False, grid: bool = True,
                color_from: Union[str, List[str]] = "blue", color_to: Union[str, List[str]] = "red",
                groupcolors: int = None, **kwargs):
    # Rest
    n_features = len(descriptors)
    n_dimensions = descriptors[0].shape[1]//groupby
    if groupcolors is None:
        groupcolors = groupby

    # Color range
    if isinstance(color_from, str):
        color_from = [color_from]*n_features
    if isinstance(color_to, str):
        color_to   = [color_to]*n_features
    assert (len(color_from) == n_features) and (len(color_from) == n_features), "The number of specified colors must match the number of descriptors"
    color_range = [
        [colorFader(mcolors.to_hex(c_from),mcolors.to_hex(c_to),v) for v in np.linspace(0,1,groupcolors)] 
        for (c_from,c_to) in zip(color_from,color_to)
    ]

    # Initialize figure
    fig,ax = plt.subplots(nrows=n_features,ncols=n_dimensions,figsize=figsize,**kwargs)
    if ax.ndim == 1:
        ax = ax[:,None]
    for i,des in enumerate(descriptors):
        for j,val in enumerate(des.T):
            ax[i,j//groupby].plot(val,color=color_range[i][j%groupcolors])
            ax[i,j//groupby].set_xlim([0,val.size-1])
        
    # Set figure options
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]-1) for j in range(ax.shape[1])]
    [ax[i,j].set_yticks([]) for i in range(ax.shape[0])   for j in range(1,ax.shape[1])]
    if varnames is not None:
        [ax[i,0].set_ylabel(f"{varnames[i]}") for i in range(ax.shape[0])]
    else:
        [ax[i,0].set_ylabel(f"Feat. {i+1}") for i in range(ax.shape[0])]
    if dimnames is not None:
        [ax[0,j].set_title(f"{dimnames[j]}") for j in range(ax.shape[1])]
    else:
        [ax[0,j].set_title(f"Dim. {j+1}") for j in range(ax.shape[1])]
    # Set figure ylims
    if same_scale and "sharey" not in kwargs:
        for i in range(ax.shape[0]):
            ylims = [0,0]
            for j in range(ax.shape[1]):
                ylim = ax[i,j].get_ylim()
                ylims[0] = min([ylims[0],ylim[0]])
                ylims[1] = max([ylims[1],ylim[1]])
            
            for j in range(ax.shape[1]):
                ax[i,j].set_ylim(ylims)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01)
    fig.align_ylabels(ax[:,0])

    if return_axes:
        return fig,ax



def array_descriptor(descriptor: List[np.ndarray], groupby: int, figsize: Tuple[int,int] = (16,16),
                varname: Union[np.ndarray, list] = None, variability: List = None, dimnames: Union[np.ndarray, list] = None,
                same_scale: bool = True, return_axes: bool = False, grid: bool = True,
                color_from: Union[str, List[str]] = "blue", color_to: Union[str, List[str]] = "red",
                groupcolors: int = None, **kwargs):
    """
        Use when we have mixed data. Some descriptors are from numerical variables and others from
        array variables and we need to analyse one by one
    """
    # Rest
    n_features = 1
    n_dimensions = descriptor.shape[1]//groupby
    if groupcolors is None:
        groupcolors = groupby

    # Color range
    if isinstance(color_from, str):
        color_from = [color_from]*n_features
    if isinstance(color_to, str):
        color_to = [color_to]*n_features
    assert (len(color_from) == n_features) and (len(color_from) == n_features), "The number of specified colors must match the number of descriptors"
    color_range = [
        [colorFader(mcolors.to_hex(c_from),mcolors.to_hex(c_to),v) for v in np.linspace(0,1,groupcolors)]
        for (c_from,c_to) in zip(color_from,color_to)
    ]

    minimum = np.min(descriptor)
    maximum = np.max(descriptor)

    # Initialize figure
    fig,ax = plt.subplots(nrows=n_features,ncols=n_dimensions,figsize=figsize)
    if ax.ndim == 1:
        ax = ax[:,None]
        ax = ax.flatten()

    for j, val in enumerate(descriptor.T):
        ax[j//groupby].plot(val, color=color_range[0][j%groupcolors], label=variability[j%groupcolors])
        ax[j//groupby].set_xlim([0,val.size-1])
        ax[j//groupby].legend()

    # Set figure options
    ax[0].set_ylabel(f"{varname}")

    if dimnames is not None:
        [ax[j].set_title(f"{dimnames[j]}") for j in range(ax.shape[0])]
    else:
        [ax[j].set_title(f"Dim. {j+1}") for j in range(ax.shape[0])]
    # Set figure ylims
    if same_scale and "sharey" not in kwargs:
        for i in range(ax.shape[0]):
            ylims = [minimum-0.1, maximum+0.1]
            ax[i].set_ylim(ylims)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01)
    fig.align_ylabels(ax[:])

    if return_axes:
        return fig, ax


def numerical_descriptor(descriptor, xticks, variability, varname,
                         dimensions, groupby, y_min, y_max):
    """
    Use when we have mixed data. Some descriptors are from numerical variables and others from
    array variables and we need to analyse one by one
    """
    minimum = y_min if y_min < np.min(descriptor) else np.min(descriptor)
    maximum = y_max if y_max > np.max(descriptor) else np.max(descriptor)
    init = 0
    n_dims = len(dimensions) if isinstance(dimensions, list) else dimensions

    f, axs = plt.subplots(nrows=1, ncols=n_dims, figsize=(20,5))
    for j in range(n_dims):
        descriptor_by_dim = descriptor[0][0][init:init + groupby]
        axs[j].plot(xticks, descriptor_by_dim, '*-', color='black', ms=10)
        axs[j].set_xticks(xticks)
        axs[j].set_xticklabels(variability, rotation=90)
        axs[j].set_ylim(minimum, maximum)
        if j == 0: axs[j].set_ylabel(varname)
        axs[j].set_title('Dim {}'.format(j + 1))
        init += groupby


def numerical_descriptors(descriptors, xticks, variability, varnames, dimensions, groupby):
    for i, varname in enumerate(varnames):
        feature_descriptors = descriptors[i][0, :]
        minimum = np.min(feature_descriptors)
        maximum = np.max(feature_descriptors)
        init = 0

        f, axs = plt.subplots(nrows=1, ncols=dimensions)
        for j in range(dimensions):
            feature_descriptors_by_dim = feature_descriptors[init:init + groupby]
            axs[j].plot(xticks, feature_descriptors_by_dim, '*-', color='black', ms=10)
            axs[j].set_xticks(xticks)
            axs[j].set_xticklabels(variability)
            axs[j].set_ylim(minimum - 0.5, maximum + 0.5)
            if j == 0: axs[j].set_ylabel(varname)
            axs[j].set_title('Dim {}'.format(j + 1))
            init += groupby


def path(descriptors: List[np.ndarray], figsize=(10,10), varnames: Union[np.ndarray, list] = None, dimnames: Union[np.ndarray, list] = None):
    # Rest
    n_features = len(descriptors)
    n_dimensions = descriptors[0].shape[1]

    # Initialize figure
    fig,ax = plt.subplots(nrows=n_features,ncols=n_dimensions,figsize=figsize)
    for i,des in enumerate(descriptors):
        for j,val in enumerate(des.T):
            col = 1/5*(j%5)
            ax[i,j].plot(val,color=[col,0,1-col])
        
    # Set figure options
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]-1) for j in range(ax.shape[1])]
    if varnames is not None:
        [ax[i,0].set_ylabel(f"{varnames[i]}") for i in range(ax.shape[0])]
    else:
        [ax[i,0].set_ylabel(f"Feat. {i+1}") for i in range(ax.shape[0])]
    if dimnames is not None:
        [ax[0,j].set_title(f"Point {dimnames[j]}") for j in range(ax.shape[1])]
    else:
        [ax[0,j].set_title(f"Point {j+1}") for j in range(ax.shape[1])]
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]) for j in range(ax.shape[1])]
    [ax[i,j].set_yticks([]) for i in range(ax.shape[0]) for j in range(ax.shape[1])]
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01,wspace=0.01)
    fig.align_ylabels(ax[:,0])


def plot_MKL_space(MKL_projection, color, d1=1, d2=0, cmap='coolwarm', color_edge='dimgray',
                   col_bar_label=None):
    plt.figure()
    im = plt.scatter(MKL_projection[:, d1], MKL_projection[:, d2],
                     marker='o', s=75, edgecolors=color_edge, c=color,
                     cmap=cmap)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.xlabel('MKL{}'.format(d1 + 1))
    plt.ylabel('MKL{}'.format(d2 + 1))
    plt.title('MKL space')
    if col_bar_label:
        plt.colorbar(im, label=col_bar_label)


def plot_multipleDims_MKL_space(F_data, ndims, color='skyblue', cmap = 'coolwarm', col_bar_label=None):
    n_plots = sum([i for i in range(0, ndims)])
    n_rows = ndims-1
    n_cols = ndims-1
    im= None
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*5, n_rows*5))
    # axs = axs.flatten()

    for i in range(n_rows):
        for j in range(1, ndims):
            if j > i:
                im = axs[i, j-1].scatter(F_data[:, j], F_data[:, i],
                                    marker='o', edgecolors='dimgray', c=color,
                                    cmap=cmap)
                axs[i, j-1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
                axs[i, j-1].set_xlabel('MKL{}'.format(j + 1))
                axs[i, j-1].set_ylabel('MKL{}'.format(i + 1))
    # Plots to remove
    for i in range(1, n_rows):
        for j in range(0, n_cols-1):
            if i>j:
                axs[i, j].remove()

    if col_bar_label:
        cbar_ax = fig.add_axes([0.15, 0.15, 0.2, 0.015])
        fig.colorbar(im, cax=cbar_ax, label=col_bar_label, orientation='horizontal')

    plt.tight_layout()
