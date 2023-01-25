#!/usr/bin/env python3
# -*- coding: utf-8 -*-
VERSION = "0.1"

try:
    from .plots import error_histogram, corr_heatmap, ColGrid, sorted_order
    from .plots import crosstab_plot, heatmap, proportion_plot
    from .plots import imscatter, smoothscatter
    from .utils import numpy_crosstab, split_col_by_type
    from .corr import corr, CorrType
except ModuleNotFoundError as err:
    print("Warning:", err)
except ImportError as err:
    print("Warning:", err)

try:
    from .image_utils import plot_bboxes, make_montage
except ModuleNotFoundError as err:
    print("Warning:", err)
except ImportError as err:
    print("Warning:", err)

try:
    from .plots import half_violinplot, RainCloud
except ModuleNotFoundError as err:
    print("Warning:", err)
except ImportError as err:
    print("Warning:", err)

try:
    from .sklearn import (
        make_ext_column_transformer, transformer_extensions,
        InvertibleColumnTransformer, TransformerExtensions,
        make_pd_column_transformer
    )
except ModuleNotFoundError as err:
    print("Warning:", err)
except ImportError as err:
    print("Warning:", err)

try:
    from .explain import Explainer
except ModuleNotFoundError:
    pass

try:
    from .graphs import show_tree
except ModuleNotFoundError:
    pass

try:
    from .tensorboard import tflog2pandas
except ModuleNotFoundError:
    pass

try:
    from .statsmodels import SMWrapper, SMLinearRegression, SMLogisticRegression
except ModuleNotFoundError:
    pass
