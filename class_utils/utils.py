import numpy as np
import pandas as pd
from ._from_sv import determine_feature_type, FeatureType

def split_col_by_type(df):
    """
    Splits columns into categorical, numeric, textual and other. Returns a tuple
    of four lists, each containing the names of the columns of the respective
    type.
    """

    categorical_inputs = []
    numeric_inputs = []
    textual_inputs = []
    other_inputs = []

    for col in df.columns:
        ft = determine_feature_type(df[col])
        if ft == FeatureType.TYPE_CAT or ft == FeatureType.TYPE_BOOL:
            categorical_inputs.append(col)
        elif ft == FeatureType.TYPE_NUM:
            numeric_inputs.append(col)
        elif ft == FeatureType.TYPE_TEXT:
            textual_inputs.append(col)
        else:
            other_inputs.append(col)

    return categorical_inputs, numeric_inputs, textual_inputs, other_inputs

def numpy_crosstab(x, y, dropna=False, shownan=False):
    """
    Gathers and crosstabulates different unique values from x and y, returning
    a dataframe with the count of their co-occurences.

    Arguments:
        x: A series object (e.g. a dataframe column).
        y: A series object (e.g. a dataframe column).
        dropna: Whether to drop entries where at least one of x and y
            is missing a value.
        shownan: Whether to include NaN entries in the crosstabulation
            or drop them before the dataframe is returned.
    """
    if dropna:
        ind = ~(x.isnull() | y.isnull())
        x = x[ind]
        y = y[ind]

    x_cats, x_enc = np.unique(x.astype('str'), return_inverse=True)
    y_cats, y_enc = np.unique(y.astype('str'), return_inverse=True)

    cm = np.zeros((len(x_cats), len(y_cats)), dtype=int)
    np.add.at(cm, (x_enc, y_enc), 1)

    cm_df = pd.DataFrame(cm, columns=y_cats, index=x_cats)
    cm_df.index.name = x.name
    cm_df.columns.name = y.name
    
    if not shownan:
        cm_df = cm_df.drop(['nan'], axis=0, errors='ignore')
        cm_df = cm_df.drop(['nan'], axis=1, errors='ignore')
    
    return cm_df
