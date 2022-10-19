from typing import Optional, Tuple, Callable, Union, Any
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import make_column_transformer
import pandas as pd
import numpy as np
import numbers

class BaseTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self,
        transformer,
    ):
        self.__dict__['transformer'] = transformer

    def __setattr__(self, key, value):
        if key in self.transformer.__dict__:
            self.transformer.__setattr__(key, value)
        else:
            self.__dict__[key] = value
    
    def __getattr__(self, name):
        return getattr(self.transformer, name)

    def __dir__(self):
        d = set(self.transformer.__dir__())
        d.update(set(super().__dir__()))
        return d

class TransformerExtensions(BaseTransformerWrapper):
    def __init__(self,
        transformer: TransformerMixin,
        inverse: Optional[Union[str, Callable, None]] = None, # 'identity', None or a callable
        feature_names_out=None # None, 'identity', or list of strings
    ):
        """A wrapper for sklearn transformers that adds missing interface
        methods such as inverse_transform and get_feature_names_out.

        Args:{__transformer_extensions_args__}
        """
        super().__init__(transformer=transformer)
        self.__dict__['inverse'] = inverse
        self.__dict__['feature_names_out'] = feature_names_out

    def inverse_transform(self, X):
        if self.inverse is None:
            return self.transformer.inverse_transform(X)
        if isinstance(self.inverse, str) and self.inverse == 'identity':
            return X
        elif callable(self.inverse):
            return self.inverse(X)            
        else:
            raise ValueError("inverse must be 'identity', None, or a callable")

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        if self.feature_names_out is None:
            return self.transformer.get_feature_names_out(input_features)
        elif isinstance(self.feature_names_out, str) and self.feature_names_out == 'identity':
            return input_features
        elif callable(self.inverse):
            return self.feature_names_out(input_features)
        else:
            raise ValueError("feature_names_out must be 'identity', None, or a callable")

__transformer_extensions_common_args__ = """
            transformer (TransformerMixin): An sklearn transformer that is to
                be extended;"""

__transformer_extensions_inverse_args__ = """
            inverse (Union[str, Callable, None], optional): If None, the
                inverse_transform of the underlying transformer will be used.
                If a string that reads 'indentity', inverse_transform will
                return its input without modification; if a callable, it
                will be used as the inverse_transform."""

__transformer_extensions_name_args__ = """
            feature_names_out (Union[str, List, None], optional): If None,
                get_feature_names_out of the underlying transformer will be
                used. If a string that reads 'identity', get_feature_names_out
                will return its input without modification; if a callable,
                it will be used as the get_feature_names_out."""

__transformer_extensions_args__ = (
    __transformer_extensions_common_args__ +
    __transformer_extensions_inverse_args__ +
    __transformer_extensions_name_args__
)

TransformerExtensions.__init__.__doc__ = (
    TransformerExtensions.__init__.__doc__.format(
        __transformer_extensions_args__=__transformer_extensions_args__
    )
)

class InvertibleColumnTransformer(BaseTransformerWrapper):
    def __init__(self,
        column_transformer: TransformerMixin,
        inverse_dropped: Optional[str] = 'nan',
        return_dataframe: Optional[bool] = False,
    ):
        """A wrapper that adds inverse_transform to a ColumnTransformer.
            
        Args:{__invertible_column_transformer_args__}
        """
            
        super().__init__(transformer=column_transformer)
        self.__dict__['inverse_dropped'] = inverse_dropped
        self.__dict__['return_dataframe'] = return_dataframe
    
    def transform(self, *args, **kwargs):
        res = self.transformer.transform(*args, **kwargs)

        if self.return_dataframe and not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res, columns=self.get_feature_names_out())

        return res

    def fit_transform(self, *args, **kwargs):
        res = self.transformer.fit_transform(*args, **kwargs)

        if self.return_dataframe and not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res, columns=self.get_feature_names_out())
        
        return res
    
    def inverse_transform(self, X):
        if hasattr(self.transformer, 'feature_names_in_'):
            feature_names = self.feature_names_in_
            cname2index = {c:i for i,c in enumerate(feature_names)} 
        else:
            feature_names = None
            cname2index = range(self.transformer.n_features_in_)

        invs = [None for i in range(len(cname2index))]

        # handle all except the remainder
        for tname, t, tcols in self.transformer.transformers_[:-1]:
            X_sel = X[:, self.transformer.output_indices_[tname]]
            if X_sel.size == 0: continue
            X_sel_inv = t.inverse_transform(X_sel)

            for cname, col in zip(tcols, range(X_sel_inv.shape[1])):
                if isinstance(cname, numbers.Integral):
                    cnum = cname
                else:
                    cnum = cname2index[cname]

                invs[cnum] = X_sel_inv[:, col]
            
        # handle the remainder
        tname, t, tcols = self.transformer.transformers_[-1]
        assert(tname == 'remainder')
            
        if t == 'drop':
            if self.inverse_dropped == 'ignore':
                tcols = {
                    cname if isinstance(cname, numbers.Integral)
                          else cname2index[cname]
                    for cname in tcols
                }

                tmp_invs = []
                tmp_feature_names = [] if not feature_names is None else None
                for i, inv in enumerate(invs):
                    if not i in tcols:
                        tmp_invs.append(inv)
                        if not feature_names is None:
                            tmp_feature_names.append(feature_names[i])

                invs = tmp_invs
                feature_names = tmp_feature_names

            elif self.inverse_dropped == 'nan':
                for cname in tcols:
                    if isinstance(cname, numbers.Integral):
                        cnum = cname
                    else:
                        cnum = cname2index[cname]

                    invs[cnum] = np.full(X.shape[0], np.nan)
            else:
                raise ValueError("inverse_dropped must be 'ignore' or 'nan'")
        elif t == 'passthrough':
            X_sel = X[:, self.transformer.output_indices_[tname]]
            print(X_sel.shape)

            for cname, xcol in zip(tcols, range(X_sel.shape[1])):
                if isinstance(cname, numbers.Integral):
                    cnum = cname
                else:
                    cnum = cname2index[cname]

                invs[cnum] = X_sel[:, xcol]
        else:
            raise ValueError("remainder must be 'drop' or 'passthrough'")

        invs = np.asarray(invs).transpose()
        if not feature_names is None:
            return pd.DataFrame(invs, columns=feature_names)

        return invs

__invertible_column_transformer_args__ = """
            column_transformer (TransformerMixin): A ColumnTransformer;
            inverse_dropped (str, optional): If 'ignore', the inverse_transform
                will ignore dropped remainder columns. If 'nan', dropped
                remainder columns will be filled with NaNs."""

InvertibleColumnTransformer.__init__.__doc__ = (
    InvertibleColumnTransformer.__init__.__doc__.format(
        __invertible_column_transformer_args__=__invertible_column_transformer_args__
    )
)

def make_ext_column_transformer(
    *args, inverse_dropped='nan', return_dataframe=False, **kwargs
):
    """Creates an InvertibleColumnTransformer.

    Args:{__invertible_column_transformer_args__}
    """
    return InvertibleColumnTransformer(
        make_column_transformer(*args, **kwargs),
        inverse_dropped=inverse_dropped,
        return_dataframe=return_dataframe
    )

make_ext_column_transformer.__doc__ = (
    make_ext_column_transformer.__doc__.format(
        __invertible_column_transformer_args__=__invertible_column_transformer_args__
    )
)

def make_pd_column_transformer(
    *args, inverse_dropped='ignore', return_dataframe=True,
    verbose_feature_names_out=False, **kwargs
):
    """Creates an InvertibleColumnTransformer with the following defaults:
        * return_dataframe=True;
        * inverse_dropped='nan';
        * verbose_feature_names_out=False;

    Args:{__invertible_column_transformer_args__}
    """
    return InvertibleColumnTransformer(
        make_column_transformer(
            *args,
            verbose_feature_names_out=verbose_feature_names_out,
            **kwargs
        ),
        inverse_dropped=inverse_dropped,
        return_dataframe=return_dataframe
    )

make_pd_column_transformer.__doc__ = (
    make_pd_column_transformer.__doc__.format(
        __invertible_column_transformer_args__=__invertible_column_transformer_args__
    )
)

make_invertible_column_transformer = make_ext_column_transformer

def inverse(transformer, inverse='identity', inverse_dropped='nan'):
    """
    Wraps the transformer in a wrapper that adds inverse_transform
    to its interface.

    Args:{__transformer_extensions_common_args__}{__transformer_extensions_inverse_args__}
    """
    if isinstance(transformer, TransformerExtensions):
        transformer.inverse = inverse
        return transformer

    return TransformerExtensions(
        transformer,
        inverse=inverse,
        inverse_dropped=inverse_dropped
    )

inverse.__doc__ = inverse.__doc__.format(
    __transformer_extensions_common_args__=__transformer_extensions_common_args__,
    __transformer_extensions_inverse_args__=__transformer_extensions_inverse_args__
)

def feature_names(transformer, feature_names_out='identity'):
    """
    Wraps the transformer in a wrapper that adds feature_names_out.

    Args:{__transformer_extensions_common_args__}{__transformer_extensions_name_args__}
    """
    if isinstance(transformer, TransformerExtensions):
        transformer.feature_names_out = feature_names_out
        return transformer
    
    return TransformerExtensions(
        transformer,
        feature_names_out=feature_names_out
    )

feature_names.__doc__ = feature_names.__doc__.format(
    __transformer_extensions_common_args__=__transformer_extensions_common_args__,
    __transformer_extensions_name_args__=__transformer_extensions_name_args__
)

def transformer_extensions(
    transformer,
    inverse='identity',
    feature_names_out='identity'
):
    """Wraps the transformer in a TransformerExtensions wrapper.

    Args:{__transformer_extensions_args__}
    """
    if isinstance(transformer, TransformerExtensions):
        transformer.inverse = inverse
        transformer.feature_names_out = feature_names_out
        return transformer

    return TransformerExtensions(
        transformer,
        inverse=inverse,
        feature_names_out=feature_names_out
    )

transformer_extensions.__doc__ = transformer_extensions.__doc__.format(
    __transformer_extensions_args__=__transformer_extensions_args__
)
