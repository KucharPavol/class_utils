#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.patches as patches
import matplotlib.colorbar as colorbar
from itertools import combinations
import math
import itertools
from numpy import ma
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .corr import corr, CorrType
from matplotlib.colors import LogNorm, PowerNorm
from seaborn.matrix import _DendrogramPlotter
from .utils import numpy_crosstab
import pandas as pd
import numbers

def error_histogram(Y_true, Y_predicted, Y_fit_scaling=None,
                    with_error=True, 
                    with_output=True, 
                    with_mae=True, with_mse=False, 
                    error_color='tab:red', output_color='tab:blue',
                    error_kwargs=dict(alpha=0.8), output_kwargs={},
                    mae_kwargs=dict(c='k', ls='--'),
                    mse_kwargs=dict(c='k', ls='--'),
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

def _zaric_wrap_custom(source_text, separator_chars, width=70, keep_separators=True):
    current_length = 0
    latest_separator = -1
    current_chunk_start = 0
    output = ""
    char_index = 0
    while char_index < len(source_text):
        if source_text[char_index] in separator_chars:
            latest_separator = char_index
        output += source_text[char_index]
        current_length += 1
        if current_length == width:
            if latest_separator >= current_chunk_start:
                # Valid earlier separator, cut there
                cutting_length = char_index - latest_separator
                if not keep_separators:
                    cutting_length += 1
                if cutting_length:
                    output = output[:-cutting_length]
                output += "\n"
                current_chunk_start = latest_separator + 1
                char_index = current_chunk_start
            else:
                # No separator found, hard cut
                output += "\n"
                current_chunk_start = char_index + 1
                latest_separator = current_chunk_start - 1
                char_index += 1
            current_length = 0
        else:
            char_index += 1
    return output

def _zaric_heatmap(y, x, color=None, cmap=None, palette='coolwarm', size=None,
            x_order=None, y_order=None, circular=None,
            ax=None, face_color=None, wrap_x=12, wrap_y=13, square=True,
            cbar=True, cax=None, cbar_kws=None, mask=None,
            color_norm=None, size_norm=None):
    if ax is None:
        ax = plt.gca()

    if color_norm is None:
        color_norm = PowerNorm(0.925)
    
    if not color_norm.scaled():
        vmax = max(np.abs(np.nanmin(color)), np.abs(np.nanmax(color)))
        vmin = -vmax
        color_norm.autoscale([vmin, vmax])

    if size is None:
        size = np.ones(len(x))

    if size_norm is None:
        size_norm = PowerNorm(0.5)
    size_norm.autoscale_None(size)

    if cbar_kws is None:
        cbar_kws = {}
    
    if square:
        ax.set_aspect('equal')
        plt.draw()

    if face_color is None:
        face_color = '#fdfdfd'

    if color is None:
        color = [1]*len(x)

    if circular is None:
        circular = [False]*len(x)

    if cmap is None:
        cmap = sns.color_palette(palette, as_cmap=True)
    
    def do_wrapping(label, length):
        return _zaric_wrap_custom(label, ["_", "-"], length)
    
    if x_order is None:
        x_names = [t for t in reversed(sorted(set([v for v in x])))]
    else:
        x_names = [t for t in x_order]
        
    # Wrap to help avoid overflow
    x_names = [do_wrapping(label, wrap_x) for label in x_names]

    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if y_order is None:
        y_names = [t for t in sorted(set([v for v in y]))]
    else:
        y_names = [t for t in y_order[::-1]]
        
    # Wrap to help avoid overflow
    y_names = [do_wrapping(label, wrap_y) for label in y_names]

    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    ax.tick_params(labelbottom='on', labeltop='on')
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=90,
        horizontalalignment='center', linespacing=0.8)
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num], linespacing=0.8)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor(face_color)
    delta_in_pix = ax.transData.transform((1, 1)) - ax.transData.transform((0, 0))

    index = 0
    patch_col = []
    patch_col_ind = []

    for cur_x, cur_y, use_circ in zip(x, y, circular):
        if (size[index] == 0 or
            np.isnan(color[index]) or
            (not mask is None and mask[index])
        ):
            index = index + 1
            continue

        wrapped_x_name = do_wrapping(cur_x, wrap_x)
        wrapped_y_name = do_wrapping(cur_y, wrap_y)
        before_coordinate = np.array(
            ax.transData.transform((x_to_num[wrapped_x_name]-0.5,
                                    y_to_num[wrapped_y_name]-0.5)))
        after_coordinate = np.array(
            ax.transData.transform((x_to_num[wrapped_x_name]+0.5,
                                    y_to_num[wrapped_y_name]+0.5)))
        before_pixels = np.round(before_coordinate, 0)
        after_pixels = np.round(after_coordinate, 0)
        desired_fraction = size_norm(size[index])

        delta_in_pix = after_pixels - before_pixels
        gap = np.round((1.0 - desired_fraction) * delta_in_pix / 2, 0)
        # make sure that non-zero sized markers don't disappear
        gap[np.where(delta_in_pix - gap*2 < 3)] -= 3

        start = before_pixels + gap
        ending = after_pixels - gap
        start[0] = start[0] + 1
        ending[1] = ending[1] - 1
        start_doc = ax.transData.inverted().transform(start)
        ending_doc = ax.transData.inverted().transform(ending)
        cur_size = ending_doc - start_doc

        if use_circ:
            cur_rect = patches.Circle(
                (start_doc[0] + cur_size[0] / 2,
                 start_doc[1] + cur_size[1] / 2),
                cur_size[1] / 2, antialiased=True)
        else:
            cur_rect = patches.Rectangle(
                (start_doc[0], start_doc[1]),
                cur_size[0], cur_size[1], antialiased=True)

        cur_rect.set_antialiased(True)
        patch_col.append(cur_rect)
        patch_col_ind.append(index)

        index = index + 1

    patch_col = PatchCollection(
        patch_col, array=color[patch_col_ind],
        norm=color_norm, cmap=cmap
    )
    ax.add_collection(patch_col)

    if cbar:
        plt.colorbar(patch_col, cax=cax, cmap=cmap, **cbar_kws)

def heatmap(df, corr_types=None, map_type='zaric', ax=None, face_color=None,
            annot=None, cbar=True, mask=None,
            row_cluster=False, row_cluster_metric='euclidean',
            row_cluster_method='average', row_cluster_linkage=None,
            col_cluster=False, col_cluster_metric='euclidean',
            col_cluster_method='average', col_cluster_linkage=None,
             **kwargs):
    """
    Arguments:
        map_type: One of 'zaric', 'standard', 'dendrograms'.
    """
    if map_type == 'dendrograms':
        if not ax is None:
            raise ValueError("Argument 'ax' is not supported for map_type == 'dendrograms'.")
    else:
        if ax is None:
            ax = plt.gca()

    if not mask is None:
        mask = np.asarray(mask)

    if not corr_types is None:
        corr_types = np.asarray(corr_types)

    if row_cluster and not map_type == 'dendrograms':
        row_ind = _DendrogramPlotter(
            df, axis=0, metric=row_cluster_metric,
            method=row_cluster_method, linkage=row_cluster_linkage,
            label=False, rotate=False
        ).reordered_ind

        df = df.reindex(df.index[row_ind])

        if not mask is None:
            mask = mask[row_ind, :]
        
        if not corr_types is None:
            corr_types = corr_types[row_ind, :]

    if col_cluster and not map_type == 'dendrograms':
        col_ind = _DendrogramPlotter(
            df, axis=1, metric=col_cluster_metric,
            method=col_cluster_method, linkage=col_cluster_linkage,
            label=False, rotate=False
        ).reordered_ind
        
        df = df.reindex(df.columns[col_ind], axis=1)
        
        if not mask is None:
            mask = mask[:, col_ind]

        if not corr_types is None:
            corr_types = corr_types[:, col_ind]

    if map_type == "zaric":
        l = np.asarray(list(itertools.product(df.index, df.columns)))
        x = l[:, 0]
        y = l[:, 1]
        v = df.values.reshape(-1)
        m = mask.reshape(-1) if not mask is None else None
        circ = np.zeros(len(x))
        if not corr_types is None:
            circ[np.where(corr_types.reshape(-1) == CorrType.num_vs_num)] = True

        default_kwargs = dict(
            color=v,
            size=np.abs(v),
            circular=circ
        )
        default_kwargs.update(**kwargs)
        kwargs = default_kwargs

        _zaric_heatmap(
            x, y,
            ax=ax,
            face_color=face_color,
            cbar=cbar,
            mask=m,
            x_order=df.columns,
            y_order=df.index,
            **kwargs
        )

        ax.set_xlabel(df.columns.name)
        ax.set_ylabel(df.index.name)

    elif map_type == 'standard' or map_type == 'dendrograms':
        if annot is None:
            annot = True

        if face_color is None:
            face_color = 'black'

        default_kwargs = dict(center=0, square=True, linewidths=1,
                              annot=annot)
        default_kwargs.update(**kwargs)
        kwargs = default_kwargs

        if map_type == 'dendrograms':
            del kwargs['square']
            sns.clustermap(df, cbar=cbar, mask=mask, **kwargs)
        else:
            sns.heatmap(df, ax=ax, cbar=cbar, mask=mask, **kwargs)
        
        if ax is None: ax = plt.gcf().axes[2]
        ax.set_facecolor(face_color)
        ax.xaxis.set_tick_params(rotation=45)
        plt.setp(ax.get_xticklabels(),
            rotation_mode="anchor", horizontalalignment="right")
    
    else:
        raise ValueError("Unknown map_type '{}'.".format(map_type))

def _mask_corr_significance(mask, p, p_bound):
    mask = np.asarray(mask)
    mask[np.where(mask >= p_bound)] = True

def _mask_diagonal(mask):
    mask = np.asarray(mask)
    np.fill_diagonal(mask, True)

def corr_heatmap(data_frame, categorical_inputs=None, numeric_inputs=None,
                 corr_method=None, nan_strategy='mask', nan_replace_value=0,
                 mask_diagonal=True, p_bound=None, ax=None, map_type='zaric',
                 annot=None, face_color=None, square=True, mask=None, **kwargs):

    r, p, ct = corr(data_frame, categorical_inputs=categorical_inputs,
                 numeric_inputs=numeric_inputs, corr_method=corr_method,
                 nan_strategy=nan_strategy, nan_replace_value=nan_replace_value,
                 return_corr_types=True)

    mask = np.zeros(r.shape) if mask is None else np.copy(mask)
    
    if not p_bound is None:
        _mask_corr_significance(mask, p, p_bound)

    if mask_diagonal:
        _mask_diagonal(mask)
    
    heatmap(r, corr_types=ct, map_type=map_type, ax=ax, face_color=face_color,
            annot=annot, square=square, mask=mask, **kwargs)
    return r

class ColGrid:
    def __init__(self, data, x_cols, y_cols=None, col_wrap=None,
                 height=3, aspect=4/3):
        self.data = data
        self.x_cols = x_cols if not isinstance(x_cols, str) else [x_cols]
        self.y_cols = y_cols if not (isinstance(y_cols, str) or y_cols is None) else [y_cols]

        if col_wrap is None:
            col_wrap = min(4, len(self.x_cols)*len(self.y_cols))

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

            if y_col is None:
                func(x=x_col, data=self.data, *args, **kwargs)
            else:
                func(x=x_col, y=y_col, data=self.data, *args, **kwargs)
            
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
             
def crosstab_plot(x, y, data=None, dropna=False, shownan=False, *args, **kwargs):
    if not data is None:
        x = data[x]
        y = data[y]
    tab = numpy_crosstab(y, x, dropna=dropna, shownan=shownan)
    heatmap(tab, *args, **kwargs)
    return tab

def _groupby_propplot(x, y):
    df = pd.concat([x, y], axis=1)
    propby = df.groupby(x.name)[y.name].mean()
    sns.barplot(propby.index, propby)

def proportion_plot(df, x_col, prop_cols, show_titles=True):
    scalar = False
    
    if isinstance(prop_cols, str):
        prop_cols = [prop_cols]
        scalar = True
    
    figs = []
    
    for prop_col in prop_cols:        
        dumm = pd.get_dummies(df[prop_col])
        dumm = dumm.rename(columns=lambda x: "{}={}".format(prop_col, x))
        
        dumm_df = pd.concat([df[x_col], dumm], axis=1)
        g = ColGrid(dumm_df, "Sex", dumm.columns)
        figs.append(g.map(_groupby_propplot))
        
    if scalar:
        return figs[0]
    else:
        return figs

def imscatter(x, y, images, ax=None, zoom=1,
              frame_cmap=None, frame_c=None,
              frame_linewidth=1, **kwargs):
    if ax is None:
        ax = plt.gca()
        
    if isinstance(frame_cmap, str):
        frame_cmap = plt.cm.get_cmap(frame_cmap)
    elif frame_cmap is None:
        frame_cmap = plt.cm.get_cmap('jet')
    
    if len(images) == 1:
        images = [images[0] for i in range(len(x))]
        
    if frame_c is None:
        frame_c = ['k' for i in range(len(x))]

    x, y = np.atleast_1d(x, y)
    artists = []
    
    for i, (x0, y0) in enumerate(zip(x, y)):
        fc = frame_c[i]
        if isinstance(fc, numbers.Number):
            fc = frame_cmap(fc)
      
        im = OffsetImage(images[i], zoom=zoom, **kwargs)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True,
                            bboxprops=dict(edgecolor=fc,
                                           linewidth=frame_linewidth))
        artists.append(ax.add_artist(ab))
        
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    
    return artists
