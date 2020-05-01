#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.stats import pearsonr
from itertools import combinations
import pandas as pd
import numpy as np

def corr(df, corr_method=pearsonr):
    """
    A routine with interface similar to df.corr(),
     which also returns a matrix of p-values.
    """
    df_r = pd.DataFrame(np.ones([df.shape[1], df.shape[1]]), columns=df.columns, index=df.columns)
    df_p = pd.DataFrame(np.zeros([df.shape[1], df.shape[1]]), columns=df.columns, index=df.columns)
    
    for col1, col2 in combinations(df.columns, 2):
        # set up an index that will mask out NaNs and infinite numbers
        col1_mask = np.isfinite(df[col1])
        col2_mask = np.isfinite(df[col2])
        index = np.where(np.logical_and(col1_mask, col2_mask))
        # compute correlation on the masked columns
        r, p = corr_method(df[col1].values[index], df[col2].values[index])
        df_r.loc[col1, col2] = r
        df_r.loc[col2, col1] = r
        df_p.loc[col1, col2] = p
        df_p.loc[col2, col1] = p
    
    return df_r, df_p

def mask_corr_significance(r, p, p_bound):
    r.values[np.where(p.values >= p_bound)] = np.nan
