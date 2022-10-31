import statsmodels.api as sm
from statsmodels.iolib.summary import forg
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.sm_results = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant='add')
        
        sm_model = self.model_class(y, X)
        self.sm_results = sm_model.fit()

        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant='add')

        return self.sm_results.predict(X)

    def __getattr__(self, name):
        if self.sm_results is None:
            raise AttributeError("Model not yet fitted")
        return getattr(self.sm_results, name)

    def __dir__(self):
        return dir(self.sm_results)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return repr(self.sm_results)

    def __str__(self):
        return str(self.sm_results)

class SMLinearRegression(SMWrapper):
    def __init__(self, fit_intercept=True):
        super().__init__(sm.OLS, fit_intercept=fit_intercept)

class SMLogisticRegression(SMWrapper):
    def __init__(self, fit_intercept=True):
        super().__init__(sm.Logit, fit_intercept=fit_intercept)

    # define a summary() method that also returns odds ratios and their 95% CI
    def summary(self, return_or=True):
        if self.sm_results is None:
            raise AttributeError("Model not yet fitted")
            
        sm_summary = self.sm_results.summary()

        if return_or:
            odds_ratios = np.exp(self.params)
            odds_ratios_ci = np.exp(self.conf_int())
            odds_ratios_ci['OR'] = odds_ratios
            odds_ratios_ci.columns = ['2.5%', '97.5%', 'OR']

            data = odds_ratios_ci[['OR', '2.5%', '97.5%']].values
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    data[i, j] = forg(data[i, j], 3)

            odds_ratios_ci_tab = sm.iolib.table.SimpleTable(
                np.hstack([np.transpose([list(odds_ratios_ci.index)]), data]),
                headers=['', 'OR', '[0.025', '0.975]'],
                title="Odds Ratios",
            )
            sm_summary.tables.append(odds_ratios_ci_tab)

        return sm_summary
