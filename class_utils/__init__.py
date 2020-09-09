#!/usr/bin/env python3
# -*- coding: utf-8 -*-
VERSION = "0.1"

try:
    from .plots import error_histogram, corr_heatmap, ColGrid, sorted_order
    from .plots import crosstab_plot, heatmap_plot, proportion_plot
    from .plots import imscatter
    from .utils import numpy_crosstab
except ModuleNotFoundError:
    pass

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
