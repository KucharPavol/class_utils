#!/usr/bin/env python3
# -*- coding: utf-8 -*-
VERSION = "0.1"

try:
    from .plots import error_histogram, corr_heatmap
except ModuleNotFoundError:
    pass

try:
    from .explain import show_tree, Explainer
except ModuleNotFoundError:
    pass

