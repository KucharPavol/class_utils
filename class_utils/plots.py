#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .utils import corr, mask_corr_significance

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
        sns.distplot(Y_true, label="desired output", ax=ax1, color=output_color)
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

def corr_heatmap(data_frame, p_bound=0.01, ax=None):
    if ax is None:
        ax = plt.gca()

    if p_bound is None:
        r = data_frame.corr()
        sns.heatmap(r, ax=ax, center=0, square=True, linewidths=1)
        ax.xaxis.set_tick_params(rotation=45)
        plt.setp(ax.get_xticklabels(),
            rotation_mode="anchor", horizontalalignment="right")
    else:
        r, p = corr(data_frame)
        mask_corr_significance(r, p, p_bound)
        sns.heatmap(r, ax=ax, center=0, square=True, linewidths=1)
        ax.xaxis.set_tick_params(rotation=45)
        plt.setp(ax.get_xticklabels(),
            rotation_mode="anchor", horizontalalignment="right")

