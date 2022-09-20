import numpy as np
import pandas as pd
from collections import Counter
import math
import scipy.stats as ss

# These functions are adapted from the SweetViz package, which took them over
# from dython. SweetViz can be found at https://github.com/fbdesignpro/sweetviz

def convert(data, to):
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError(
            'cannot handle data conversion of type: {} to {}'.format(
                type(data), to))
    else:
        return converted

def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    **Returns:** float
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    """
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy

# IMPORTANT: look at the order of arguments y and x
def theils_u(y, x):
    """
    IMPORTANT: look at the order of arguments y and x
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-
    categorical association. This is the uncertainty of x given y: value is
    on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.
    This is an asymmetric coefficient: U(x,y) != U(y,x)
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    """
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def theils_sym_u(y, x):
    """
    Calculates the symmetric version of the Uncertainty coefficient.
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    """
    s_xy = conditional_entropy(x, y)
    s_yx = conditional_entropy(y, x)
    
    x_counter = Counter(x)
    x_total_occurrences = sum(x_counter.values())

    y_counter = Counter(y)
    y_total_occurrences = sum(y_counter.values())

    p_x = list(map(lambda n: n / x_total_occurrences, x_counter.values()))
    p_y = list(map(lambda n: n / y_total_occurrences, y_counter.values()))

    s_x = ss.entropy(p_x)
    s_y = ss.entropy(p_y)
    
    if s_x == 0:
        u_xy = 1
    else:
        u_xy = (s_x - s_xy) / s_x

    if s_y == 0:
        u_yx = 1
    else:
        u_yx = (s_y - s_yx) / s_y

    return (s_x * u_xy + s_y * u_yx) / (s_x + s_y)

def correlation_ratio(categories, measurements):
    """
    Calculates the Correlation Ratio (sometimes marked by the greek letter Eta)
    for categorical-continuous association.
    Answers the question - given a continuous value of a measurement, is it
    possible to know which category is it associated with?
    Value is in the range [0,1], where 0 means a category cannot be determined
    by a continuous measurement, and 1 means a category can be determined with
    absolute certainty.
    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    categories : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    measurements : list / NumPy ndarray / Pandas Series
        A sequence of continuous measurements
    """
    categories = convert(categories, 'array')
    measurements = convert(measurements, 'array')
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg),
                                      2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta

# The type detection code is from sweetviz.
from enum import Enum, unique
import configparser

config = configparser.ConfigParser()
config.read_string("""
[Type_Detection]
; Numeric columns will be considered CATEGORICAL if fewer than this many distinct
max_numeric_distinct_to_be_categorical = 10

; Text columns will be considered TEXT if more than this many distinct (CATEGORICAL otherwise)
max_text_distinct_to_be_categorical = 101

; Text columns will be considered TEXT if more than this fraction are distinct
max_text_fraction_distinct_to_be_categorical = 0.33
""")

def get_counts(series: pd.Series) -> dict:
    # The value_counts() function is used to get a Series containing counts of unique values.
    value_counts_with_nan = series.value_counts(dropna=False)

    # Fix for data with only a single value; reset_index was flipping the data returned
    if len(value_counts_with_nan) == 1:
        if pd.isna(value_counts_with_nan.index[0]):
            value_counts_without_nan = pd.Series()
        else:
            value_counts_without_nan = value_counts_with_nan
    else:
        value_counts_without_nan = (value_counts_with_nan.reset_index().dropna().set_index("index").iloc[:, 0])
    # print(value_counts_without_nan.index.dtype.name)

    # IGNORING NAN FOR NOW AS IT CAUSES ISSUES [FIX]
    # distinct_count_with_nan = value_counts_with_nan.count()

    distinct_count_without_nan = value_counts_without_nan.count()
    return {
        "value_counts_without_nan": value_counts_without_nan,
        "distinct_count_without_nan": distinct_count_without_nan,
        "num_rows_with_data": series.count(),
        "num_rows_total": len(series),
        # IGNORING NAN FOR NOW AS IT CAUSES ISSUES [FIX]:
        # "value_counts_with_nan": value_counts_with_nan,
        # "distinct_count_with_nan": distinct_count_with_nan,
    }

@unique
class FeatureType(Enum):
    TYPE_CAT = "CATEGORICAL"
    TYPE_BOOL = "BOOL"
    TYPE_NUM = "NUMERIC"
    TYPE_TEXT = "TEXT"
    TYPE_UNSUPPORTED = "UNSUPPORTED"
    TYPE_ALL_NAN = "ALL_NAN"
    TYPE_UNKNOWN = "UNKNOWN"
    TYPE_SKIPPED = "SKIPPED"
    def __str__(self):
        return "TYPE_" + str(self.value)

def is_boolean(series: pd.Series, counts: dict) -> bool:
    keys = counts["value_counts_without_nan"].keys()
    if pd.api.types.is_bool_dtype(keys):
        return True
    elif (
            1 <= counts["distinct_count_without_nan"] <= 2
            and pd.api.types.is_numeric_dtype(series)
            and series[~series.isnull()].between(0, 1).all()
    ):
        return True
    elif 1 <= counts["distinct_count_without_nan"] <= 4:
        unique_values = set([str(value).lower() for value in keys.values])
        accepted_combinations = [
            ["y", "n"],
            ["yes", "no"],
            ["true", "false"],
            ["t", "f"],
        ]

        if len(unique_values) == 2 and any(
                [unique_values == set(bools) for bools in
                 accepted_combinations]
        ):
            return True
    return False


def is_categorical(series: pd.Series, counts: dict) -> bool:
    keys = counts["value_counts_without_nan"].keys()
    # TODO: CHECK THIS CASE ACTUALLY WORKS
    if pd.api.types.is_categorical_dtype(keys):
        return True
    elif pd.api.types.is_numeric_dtype(series) and \
            counts["distinct_count_without_nan"] \
            <= config["Type_Detection"].getint("max_numeric_distinct_to_be_categorical"):
        return True
    else:
        if counts["num_rows_with_data"] == 0:
            return False
        num_distinct = counts["distinct_count_without_nan"]
        fraction_distinct = num_distinct / float(counts["num_rows_with_data"])
        if fraction_distinct \
             > config["Type_Detection"].getfloat("max_text_fraction_distinct_to_be_categorical"):
            return False
        if num_distinct <= config["Type_Detection"].getint("max_text_distinct_to_be_categorical"):
            return True
    return False


def is_numeric(series: pd.Series, counts: dict) -> bool:
    return pd.api.types.is_numeric_dtype(series) and \
           counts["distinct_count_without_nan"] \
           > config["Type_Detection"].getint("max_numeric_distinct_to_be_categorical")

# For coercion, might need more testing!
def could_be_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def determine_feature_type(
        series: pd.Series,
        counts: dict = None,
        must_be_this_type: FeatureType = FeatureType.TYPE_UNKNOWN,
        which_dataframe: str = "DF"
) -> object:
    # Replace infinite values with NaNs to avoid issues with histograms
    # TODO: INFINITE VALUE HANDLING/WARNING
    # series.replace(to_replace=[np.inf, np.NINF, np.PINF], value=np.nan,
    #                inplace=True)
    
    if counts is None:
        counts = get_counts(series)

    if counts["value_counts_without_nan"].index.inferred_type.startswith("mixed"):
        raise TypeError(f"\n\nColumn [{series.name}] has a 'mixed' inferred_type (as determined by Pandas).\n"
                        f"This is is not currently supported; column types should not contain mixed data.\n"
                        f"e.g. only floats or strings, but not a combination.\n\n"
                        f"POSSIBLE RESOLUTIONS:\n"
                        f"BEST -> Make sure series [{series.name}] only contains a certain type of data (numerical OR string).\n"
                        f"OR -> Convert series [{series.name}] to a string (if makes sense) so it will be picked up as CATEGORICAL or TEXT.\n"
                        f"     One way to do this is:\n"
                        f"     df['{series.name}'] = df['{series.name}'].astype(str)\n"
                        f"OR -> Convert series [{series.name}] to a numerical value (if makes sense):\n"
                        f"     One way to do this is:\n"
                        f"     df['{series.name}'] = pd.to_numeric(df['{series.name}'], errors='coerce')\n"
                        f"     # (errors='coerce' will transform string values to NaN, that can then be replaced if desired;"
                        f" consult Pandas manual pages for more details)\n"
                        )

    try:
        # TODO: must_be_this_type ENFORCING
        if counts["distinct_count_without_nan"] == 0:
            # Empty
            var_type = FeatureType.TYPE_ALL_NAN
            # var_type = FeatureType.TYPE_UNSUPPORTED
        elif is_boolean(series, counts):
            var_type = FeatureType.TYPE_BOOL
        elif is_numeric(series, counts):
            var_type = FeatureType.TYPE_NUM
        elif is_categorical(series, counts):
            var_type = FeatureType.TYPE_CAT
        else:
            var_type = FeatureType.TYPE_TEXT
    except TypeError:
        var_type = FeatureType.TYPE_UNSUPPORTED

    # COERCE: only supporting the following for now:
    # TEXT -> CAT
    # CAT/BOOL -> TEXT
    # CAT/BOOL -> NUM
    # NUM -> CAT
    # NUM -> TEXT
    if must_be_this_type != FeatureType.TYPE_UNKNOWN and \
                must_be_this_type != var_type and \
                must_be_this_type != FeatureType.TYPE_ALL_NAN and \
                var_type != FeatureType.TYPE_ALL_NAN:
        if var_type == FeatureType.TYPE_TEXT and must_be_this_type == FeatureType.TYPE_CAT:
            var_type = FeatureType.TYPE_CAT
        elif (var_type == FeatureType.TYPE_CAT or var_type == FeatureType.TYPE_BOOL ) and \
            must_be_this_type == FeatureType.TYPE_TEXT:
            var_type = FeatureType.TYPE_TEXT
        elif (var_type == FeatureType.TYPE_CAT or var_type == FeatureType.TYPE_BOOL) and \
             must_be_this_type == FeatureType.TYPE_NUM:
            # Trickiest: Coerce into numerical
            if could_be_numeric(series):
                var_type = FeatureType.TYPE_NUM
            else:
                raise TypeError(f"\n\nCannot force series '{series.name}' in {which_dataframe} to be converted from its {var_type} to\n"
                                f"DESIRED type {must_be_this_type}. Check documentation for the possible coercion possibilities.\n"
                                f"POSSIBLE RESOLUTIONS:\n"
                                f" -> Use the feat_cfg parameter (see docs on git) to force the column to be a specific type (may or may not help depending on the type)\n"
                                f" -> Modify the source data to be more explicitly of a single specific type\n"
                                f" -> This could also be caused by a feature type mismatch between source and compare dataframes:\n"
                                f"    In that case, make sure the source and compared dataframes are compatible.\n")
        elif var_type == FeatureType.TYPE_NUM and must_be_this_type == FeatureType.TYPE_CAT:
            var_type = FeatureType.TYPE_CAT
        elif var_type == FeatureType.TYPE_BOOL and must_be_this_type == FeatureType.TYPE_CAT:
            var_type = FeatureType.TYPE_CAT
        elif var_type == FeatureType.TYPE_NUM and must_be_this_type == FeatureType.TYPE_TEXT:
            var_type = FeatureType.TYPE_TEXT
        else:
            raise TypeError(f"\n\nCannot convert series '{series.name}' in {which_dataframe} from its {var_type}\n"
                            f"to the desired type {must_be_this_type}.\nCheck documentation for the possible coercion possibilities.\n"
                            f"POSSIBLE RESOLUTIONS:\n"
                            f" -> Use the feat_cfg parameter (see docs on git) to force the column to be a specific type (may or may not help depending on the type)\n"
                            f" -> Modify the source data to be more explicitly of a single specific type\n"
                            f" -> This could also be caused by a feature type mismatch between source and compare dataframes:\n"
                            f"    In that case, make sure the source and compared dataframes are compatible.\n")
    return var_type