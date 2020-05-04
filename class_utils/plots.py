#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .corr import corr, mask_corr_significance
from matplotlib.colors import LogNorm
from seaborn.matrix import _DendrogramPlotter
from .utils import numpy_crosstab
import pandas as pd

def error_histogram(Y_true, Y_predicted, Y_fit_scaling=None,
                    with_error=True, 
                    with_output=True, 
                    with_mae=True, with_mse=False, 
                    error_color='tab:red', output_color='tab:blue',
                    error_kwargs=dict(alpha=0.8), output_kwargs={},
                    mae_kwargs=dict(c='k', ls='--'),
                    mse_color=dict(c='k', ls='--'),
                    standardize_outputs=True, ax=None,
                    num_label_precision=3):
    """
    Plots the error and output histogram.

    Arguments:
        * Y_true: the array of desired outputs;
        * Y_predicted: the array of predicted outputs;
        * Y_fit_scaling: the array to be used to fit the output scaling: if
            None, Y_true is used instead.
    """

    if ax is None:
        ax = plt.gca()

    if Y_fit_scaling is None:
        Y_fit_scaling = Y_true

    if standardize_outputs:
        hist_preproc = StandardScaler()
        hist_preproc.fit(np.asarray(Y_true).reshape(-1, 1))
        Y_true = hist_preproc.transform(np.asarray(Y_true).reshape(-1, 1))
        Y_predicted = hist_preproc.transform(np.asarray(Y_predicted).reshape(-1, 1))

    error = Y_true - Y_predicted

    # we share the same x-axis but create an additional y-axis
    ax1 = ax
    ax2 = ax.twinx()

    if with_output:
        sns.distplot(Y_true, label="desired output", ax=ax1, color=output_color,
            hist_kws=output_kwargs)
        ax1.set_xlabel('value')
        ax1.set_ylabel('output frequency', color=output_color)
        ax1.tick_params(axis='y', labelcolor=output_color)

    if with_error:
        sns.distplot(error, label="error", color=error_color,
            ax=ax2, hist_kws=error_kwargs)
        ax2.set_ylabel('error frequency', color=error_color)
        ax2.tick_params(axis='y', labelcolor=error_color)

    if with_mae:
        mae = mean_absolute_error(Y_true, Y_predicted)
        plt.axvline(mae, **mae_kwargs)
        plt.annotate(
            "MAE = {}".format(
                np.array2string(np.asarray(mae), precision=num_label_precision)
            ),
            xy=(mae, 0.8),
            xycoords=('data', 'figure fraction'),
            textcoords='offset points', xytext=(5, 0),
            ha='left', va='bottom', color='k'
        )

    if with_mse:
        mse = mean_squared_error(Y_true, Y_predicted)
        plt.axvline(mse, **mse_kwargs)
        plt.annotate(
            "MSE = {}".format(
                np.array2string(np.asarray(mse), precision=num_label_precision)
            ),
            xy=(mse, 0.8),
            xycoords=('data', 'figure fraction'),
            textcoords='offset points', xytext=(5, 0),
            ha='left', va='bottom', color='k'
        )

    ax.grid(ls='--')

def corr_heatmap(data_frame, *args, p_bound=0.01, ax=None,
                 matplot_func=sns.heatmap, **kwargs):
    if ax is None:
        ax = plt.gca()

    default_kwargs = dict(center=0, square=True, linewidths=1)
    default_kwargs.update(**kwargs)
    kwargs = default_kwargs

    if p_bound is None:
        r = data_frame.corr()
        matplot_func(r, *args, ax=ax, **kwargs)
        ax.xaxis.set_tick_params(rotation=45)
        plt.setp(ax.get_xticklabels(),
            rotation_mode="anchor", horizontalalignment="right")
    else:
        r, p = corr(data_frame)
        mask_corr_significance(r, p, p_bound)
        matplot_func(r, *args, ax=ax, **kwargs)
        ax.xaxis.set_tick_params(rotation=45)
        plt.setp(ax.get_xticklabels(),
            rotation_mode="anchor", horizontalalignment="right")

    return r

class ColGrid:
    def __init__(self, data, x_cols, y_cols, col_wrap=4, height=3, aspect=4/3):
        self.data = data
        self.x_cols = x_cols if not isinstance(x_cols, str) else [x_cols]
        self.y_cols = y_cols if not isinstance(y_cols, str) else [y_cols]
        self.col_wrap = col_wrap
        self.height = height
        self.aspect = aspect

    def map(self, func, *args, **kwargs):
        height = self.height
        width = self.height * self.aspect
        num_plots = len(self.x_cols) * len(self.y_cols)
        num_rows = int(np.ceil(num_plots / self.col_wrap))

        fig, axes = plt.subplots(num_rows, self.col_wrap, squeeze=False)
        axes = np.ravel(axes)

        for iax, ((x_col, y_col), ax) in enumerate(
            zip(itertools.product(self.x_cols, self.y_cols), axes)
        ):
            plt.sca(ax)
            func(self.data[x_col], self.data[y_col], *args, **kwargs)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        for ax in axes[iax+1:]:
            ax.axis('off')

        fig.set_size_inches(self.col_wrap * width, num_rows * height)
        plt.tight_layout()

        return fig, axes

def infer_orient(x, y, orient=None):
    """Determine how the plot should be oriented based on the data."""
    orient = str(orient)

    def is_categorical(s):
        return pd.api.types.is_categorical_dtype(s)

    def is_not_numeric(s):
        try:
            np.asarray(s, dtype=np.float)
        except ValueError:
            return True
        return False

    no_numeric = "Neither the `x` nor `y` variable appears to be numeric."

    if orient.startswith("v"):
        return "v"
    elif orient.startswith("h"):
        return "h"
    elif x is None:
        return "v"
    elif y is None:
        return "h"
    elif is_categorical(y):
        if is_categorical(x):
            raise ValueError(no_numeric)
        else:
            return "h"
    elif is_not_numeric(y):
        if is_not_numeric(x):
            raise ValueError(no_numeric)
        else:
            return "h"
    else:
        return "v"

def sorted_order(func, by='median'):
    def wrapper(x=None, y=None, data=None, orient=None, *args, **kwargs):
        if not data is None:
            xx = data[x]
            yy = data[y]
        else:
            xx = x
            yy = y
        
        df = pd.concat([pd.Series(xx), pd.Series(yy)], axis=1)
        
        orient = infer_orient(xx, yy, orient)

        if orient == 'h':
            groupby_col = df.columns[1]
            other_col = df.columns[0]
        else:
            groupby_col = df.columns[0]
            other_col = df.columns[1]
        
        df_groups = df.groupby(groupby_col)[other_col]
        sort_method = getattr(df_groups, by)
        df_med = sort_method()
        order = df_med.sort_values().index.tolist()
                    
        return func(x, y, *args, data=df, order=order, orient=orient, **kwargs)
    
    return wrapper

def crosstab_plot(x, y, dropna=False, shownan=False, *args, **kwargs):
    tab = numpy_crosstab(y, x, dropna=dropna, shownan=shownan)
    return heatmap_plot(tab, *args, **kwargs)

def heatmap_plot(
    tab, *args, logscale=False, vmin=0, vmax=None,
    annot=None, cbar=True, row_cluster=False, col_cluster=False, show_dendrograms=False,
    metric='euclidean', method='average', row_linkage=None, col_linkage=None,
    **kwargs
):
    kwargs = dict(**kwargs)
    
    if not show_dendrograms and row_cluster:
        row_ind = _DendrogramPlotter(
            tab, axis=0, metric=metric,
            method=method, linkage=row_linkage,
            label=True, rotate=False
        ).reordered_ind

        tab.index = tab.index[row_ind]
        tab.values[:] = tab.values[row_ind, :]
        
    if not show_dendrograms and col_cluster:
        col_ind = _DendrogramPlotter(
            tab, axis=1, metric=metric,
            method=method, linkage=col_linkage,
            label=True, rotate=False
        ).reordered_ind
        
        tab.columns = tab.columns[col_ind]
        tab.values[:] = tab.values[:, col_ind]
   
    if logscale:
        vmin = max(vmin, 1)
        tab += 1
        norm = LogNorm()
        
        if annot == True:
            annot = tab.values - 1
        
        kwargs.update(norm=norm)
        
    kwargs.update(vmin=vmin, vmax=vmax, annot=annot, cbar=cbar)
    
    if show_dendrograms:
        kwargs.update(metric=metric, method=method,
                      row_linkage=row_linkage, col_linkage=col_linkage,
                      row_cluster=row_cluster, col_cluster=col_cluster)
        mat = sns.clustermap(tab, *args, **kwargs)
    else:
        mat = sns.heatmap(tab, *args, **kwargs)
        
    cax = mat.cax if hasattr(mat, 'cax') else mat.collections[-1].colorbar.ax
    
    if not cbar: # if colorbar is off, hide its axes
        cax.axis('off')
    elif logscale: # fix colorbar ticks for logscale plots
        ytick_labels = cax.get_yticklabels()
        for tl in ytick_labels:
            tl.set_text(tl.get_text() + "$-1$")
        cax.set_yticklabels(ytick_labels)

    return mat