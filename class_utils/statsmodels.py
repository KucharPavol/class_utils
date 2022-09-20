import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.sm_model = None
        self.sm_results = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant='add')
        
        self.sm_model = self.model_class(y, X)
        self.sm_results = self.sm_model.fit()

        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant='add')

        return self.sm_results.predict(X)

class SMLinearRegression(SMWrapper):
    def __init__(self, fit_intercept=True):
        super().__init__(sm.OLS, fit_intercept=fit_intercept)
