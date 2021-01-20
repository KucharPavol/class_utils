#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.patches as patches
from itertools import combinations
import math
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .corr import corr, _num_cat_select
from matplotlib.colors import LogNorm
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

def _zaric_heatmap(y, x, color=None, color_range=None,
            palette='coolwarm', size=None, size_range=[0, 1], marker='s',
            x_order=None, y_order=None, size_scale=None, circular=None,
            ax=None, face_color='#fdfdfd', wrap_x=12, wrap_y=13):

    CORRELATION_ERROR = 83572398457329.0
    CORRELATION_IDENTICAL = 1357239845732.0

    if color is None:
        color = [1]*len(x)

    if circular is None:
        circular = [False]*len(x)

    if palette is None:
        palette = []
        n_colors = 256
        for i in range(0,128):
            palette.append( (0.85, (0.85/128)*i, (0.85/128)*i ))
        for i in range(128,256):
            palette.append( (0.85 - 0.85*(i-128.0)/128.0, 0.85 - 0.85*(i-128.0)/128.0, 0.85 ))
    elif isinstance(palette, str):
        n_colors = 256
        palette = sns.color_palette(palette, n_colors=n_colors)
    else:
        n_colors = len(palette)

    if color_range is None:
        color_min, color_max = np.nanmin(color), np.nanmax(color)
        color_max = np.maximum(np.abs(color_min), np.abs(color_max))
        color_min = -color_max
    else:    
        color_min, color_max = color_range

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            # For now, return "max positive" correlation color
            if val == CORRELATION_IDENTICAL:
                return palette[(n_colors - 1)]
            if val == CORRELATION_ERROR:
                return palette[(n_colors - 1)]
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            # LOG IT
            val_position = math.pow(val_position, 0.925)
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if size is None:
        size = [1]*len(x)

    if size_range is None:
        size_min, size_max = min(size), max(size)
    else:
        size_min, size_max = size_range[0], size_range[1]

    if size_scale is None:
        size_scale = 500

    # Scale with num squares
    size_scale = size_scale / len(x)
    def value_to_size(val):
        if val == 0:
            return 0.0
        if val == abs(CORRELATION_IDENTICAL):
            return 1.0
        # TODO: Better/more intuitive display of correlation errors
        if val == abs(CORRELATION_ERROR):
            return 0.0
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.999 / (size_max - size_min) + 0.001 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            # LOG IT
            val_position = math.pow(val_position, 0.5)
            return val_position

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
        y_names = [t for t in y_order]
        
    # Wrap to help avoid overflow
    y_names = [do_wrapping(label, wrap_y) for label in y_names]

    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    ax.tick_params(labelbottom='on', labeltop='on')
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=90, horizontalalignment='center', linespacing=0.8)
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
    for cur_x, cur_y, use_circ in zip(x,y,circular):
        wrapped_x_name = do_wrapping(cur_x, wrap_x)
        wrapped_y_name = do_wrapping(cur_y, wrap_y)
        before_coordinate = np.array(ax.transData.transform((x_to_num[wrapped_x_name]-0.5, y_to_num[wrapped_y_name] -0.5)))
        after_coordinate = np.array(ax.transData.transform((x_to_num[wrapped_x_name]+0.5, y_to_num[wrapped_y_name] +0.5)))
        before_pixels = np.round(before_coordinate, 0)
        after_pixels = np.round(after_coordinate, 0)
        desired_fraction = value_to_size(size[index])
        if desired_fraction == 0.0:
            index = index + 1
            continue

        delta_in_pix = after_pixels - before_pixels
        gap = np.round((1.0 - desired_fraction) * delta_in_pix / 2, 0)
        start = before_pixels + gap[0]
        ending = after_pixels - gap[0]
        start[0] = start[0] + 1
        ending[1] = ending[1] - 1
        start_doc = ax.transData.inverted().transform(start)
        ending_doc = ax.transData.inverted().transform(ending)
        cur_size = ending_doc - start_doc

        if not np.isnan(color[index]):
            if use_circ:
                cur_rect = patches.Circle((start_doc[0] + cur_size[0] / 2, start_doc[1] + cur_size[1] / 2),
                                            cur_size[1] / 2, facecolor=value_to_color(color[index]),
                                            antialiased=True)
            else:
                cur_rect = patches.Rectangle((start_doc[0], start_doc[1]),
                                            cur_size[0], cur_size[1], facecolor=value_to_color(color[index]),
                                            antialiased=True)

            cur_rect.set_antialiased(True)
            ax.add_patch(cur_rect)

        index = index + 1

    # Add color legend on the right side of the plot
    if color_min < color_max:
        cax = ax.get_figure().add_axes([0.93, 0.1, 0.05, 0.8])

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars
        cax.set_ylim(color_min, color_max)

        bar_height = bar_y[1] - bar_y[0]
        cax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        cax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        cax.grid(False) # Hide grid
        cax.set_facecolor('white') # Make background white
        cax.set_xticks([]) # Remove horizontal ticks
        cax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        cax.yaxis.tick_right() # Show vertical ticks on the right

def _mask_corr_significance(r, p, p_bound):
    r.values[np.where(p.values >= p_bound)] = np.nan

def _mask_diagonal(r):
    np.fill_diagonal(r.values, np.nan)

def corr_heatmap(data_frame, categorical_inputs=None, numeric_inputs=None,
                 corr_method=None, nan_strategy='mask', nan_replace_value=0,
                 mask_diagonal=None, p_bound=None, ax=None,
                 map_type='zaric', annot=None, **kwargs):
    """
    Arguments:
        map_type: One of 'zaric', 'standard'.
    """
    if ax is None:
        ax = plt.gca()

    r, p = corr(data_frame, categorical_inputs=categorical_inputs,
                numeric_inputs=numeric_inputs, corr_method=corr_method,
                nan_strategy=nan_strategy, nan_replace_value=nan_replace_value)

    if not p_bound is None:
        _mask_corr_significance(r, p, p_bound)

    if map_type == "zaric":
        if mask_diagonal is None:
            mask_diagonal = True

        if mask_diagonal:
            _mask_diagonal(r)

        _, categorical_inputs, numeric_inputs = _num_cat_select(
            data_frame, categorical_inputs, numeric_inputs)

        x = []; y = []; v = []; circ = []
        for col1, content in r.items():
            for col2, val in content.items():
                x.append(col1)
                y.append(col2)
                v.append(val)
                circ.append(col1 in numeric_inputs and col2 in numeric_inputs)
 
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
            **kwargs
        )

    elif map_type == 'standard':
        if mask_diagonal is None:
            mask_diagonal = False

        if mask_diagonal:
            _mask_diagonal(r)

        if annot is None:
            annot = True

        default_kwargs = dict(center=0, square=True, linewidths=1,
                              annot=annot)
        default_kwargs.update(**kwargs)
        kwargs = default_kwargs
        sns.heatmap(r, ax=ax, **kwargs)
        ax.xaxis.set_tick_params(rotation=45)
        plt.setp(ax.get_xticklabels(),
            rotation_mode="anchor", horizontalalignment="right")
    
    else:
        raise ValueError("Unknown map_type '{}'.".format(map_type))

    return r

class ColGrid:
    def __init__(self, data, x_cols, y_cols, col_wrap=None, height=3, aspect=4/3):
        self.data = data
        self.x_cols = x_cols if not isinstance(x_cols, str) else [x_cols]
        self.y_cols = y_cols if not isinstance(y_cols, str) else [y_cols]

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
