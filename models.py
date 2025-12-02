import pandas as pd
import lightgbm as lgb
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ngboost import NGBRegressor
from ngboost.distns import Poisson
from ngboost.scores import CRPScore

class Model:
    """Abstract base model with fit/predict interface."""

    def __init__(self):
        pass

    def fit(self, x_train, x_valid, y_train, y_valid):
        """Fit the model on training data (optionally using validation)."""
        pass

    def predict(self, x):
        """Predict target values for features ``x``."""
        pass
    
    
class LGBM(Model):
    """LightGBM Poisson regression model for point forecasts."""

    def __init__(self):
        super().__init__()
        self.model = None

        self.params = {
            'metric': 'rmse',
            'objective': 'poisson',
            'seed': 200,
            'force_row_wise': True,
            'learning_rate': 0.08,
            'lambda': 0.1,
            'num_leaves': 63,
            'sub_row': 0.7,
            'bagging_freq': 1,
            'colsample_bytree': 0.7,
            'num_threads': 8,
        }

    def fit(self, x_train, x_valid, y_train, y_valid):
        """Train LightGBM with Poisson loss and early stopping."""
        train = lgb.Dataset(data=x_train, label=y_train)
        valid = lgb.Dataset(data=x_valid, label=y_valid)

        # Use callbacks for early stopping / logging to be compatible with different lightgbm versions
        self.model = lgb.train(
            self.params,
            train,
            num_boost_round=2000,
            valid_sets=[valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=100),
            ],
        )

        lgb.plot_importance(self.model, importance_type="gain", precision=0, height=0.5, figsize=(6, 10))

        return self

    def predict(self, x):
        """Return point forecasts E[Y|x] from the fitted model."""
        return self.model.predict(x)
    

class LinearMixedModel(Model):
    """Linear mixed-effects model using statsmodels (not used for M5).

    This implementation is mainly for experimentation; it does not scale to
    the full M5 dataset and is therefore not used in the main pipeline.
    """
    
    def __init__(self):
        super().__init__()
        
    def fit(self, x_train, x_valid, y_train, y_valid):
        categorical_features = [
            'store_id',
            'dept_id',
            'cat_id',
            'item_id',
            'state_id',
            'event_name_1',
            'event_name_2',
        ]
        
        random_effect_features = [
            'store_id',
            'dept_id',
        ]
        
        train = pd.concat([x_train, y_train], axis=1)
        
        train['random_effect'] = train['store_id'].astype(str) + '_' + train['dept_id'].astype(str)
        
        formula = "demand ~ " + " + ".join(list(set(x_train.columns) - set(categorical_features))) + " + C(" + ") + C(".join(list(set(categorical_features) - set(random_effect_features))) + ")"
        
        print(formula)
        
        self.model = smf.mixedlm(
            formula,
            data=train,
            groups=train["random_effect"]
        ).fit(method='lbfgs')
        
        return self.model
    
    def predict(self, x):
        return self.model.predict(x)


class DistributionalLGBM(Model):
    """LightGBM model with a Negative Binomial predictive distribution.

    Trains a Poisson LightGBM model for the mean and estimates a global
    dispersion ``alpha`` to form a Negative Binomial distribution, from
    which moments and quantiles can be derived.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.alpha = None
        
        self.params = {
            'metric': 'rmse',
            'objective': 'poisson',
            'seed': 200,
            'force_row_wise': True,
            'learning_rate': 0.08,
            'lambda': 0.1,
            'num_leaves': 63,
            'sub_row': 0.7,
            'bagging_freq': 1,
            'colsample_bytree': 0.7,
            'num_threads': 8,
        }

    def fit(self, x_train, x_valid, y_train, y_valid):
        """Fit LightGBM on training data and estimate dispersion ``alpha``."""
        train = lgb.Dataset(data=x_train, label=y_train)
        valid = lgb.Dataset(data=x_valid, label=y_valid)

        self.model = lgb.train(
            self.params,
            train,
            num_boost_round=2000,
            valid_sets=[valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=100),
            ],
        )

        # Estimate global dispersion alpha via Negative Binomial MLE
        mu_train = self.model.predict(x_train)
        y = y_train.values.astype(float)

        # Avoid degenerate means
        mu_train = np.clip(mu_train, 1e-6, None)

        def neg_loglik(alpha):
            # alpha > 0 (dispersion). Use a simple scalar parameter.
            if alpha <= 0:
                return np.inf
            var = mu_train + alpha * mu_train ** 2
            p = mu_train / var
            r = mu_train ** 2 / (var - mu_train)
            # Numerical guards
            p = np.clip(p, 1e-12, 1 - 1e-12)
            r = np.clip(r, 1e-6, None)
            # Negative Binomial log pmf (up to additive constant):
            # log Gamma(y + r) - log Gamma(r) - log Gamma(y+1)
            #   + r * log(1-p) + y * log(p)
            from scipy.special import gammaln

            return -np.sum(
                gammaln(y + r)
                - gammaln(r)
                - gammaln(y + 1)
                + r * np.log(1 - p)
                + y * np.log(p)
            )

        # Simple line search on alpha over a log-grid
        alphas = np.logspace(-3, 1, 20)
        losses = [neg_loglik(a) for a in alphas]
        self.alpha = float(alphas[int(np.argmin(losses))])

        return self

    def predict(self, x):
        """Return the conditional mean E[Y|x]."""
        return self.model.predict(x)

    def _nbinom_params(self, mu):
        """Construct a SciPy ``nbinom`` distribution from mean and dispersion."""
        from scipy.stats import nbinom

        mu = np.asarray(mu, dtype=float)
        mu = np.clip(mu, 1e-6, None)
        alpha = self.alpha if self.alpha is not None else 0.0

        if alpha <= 0:
            # Fallback to Poisson-like variance (approx via large-n binomial)
            # Use a large r so that variance ~ mu.
            r = np.full_like(mu, 1e6)
            p = mu / (mu + r)
            return nbinom(r, 1 - p)

        var = mu + alpha * mu ** 2
        p = mu / var
        r = mu ** 2 / (var - mu)

        # Numerical guards
        p = np.clip(p, 1e-12, 1 - 1e-12)
        r = np.clip(r, 1e-6, None)

        from scipy.stats import nbinom
        return nbinom(r, 1 - p)

    def predict_quantile(self, x, q):
        """Return the q-th quantile of the predictive distribution."""
        mu = self.predict(x)
        dist = self._nbinom_params(mu)
        return dist.ppf(q)


class NGBoostPoisson(Model):
    """NGBoost regression model with a Poisson predictive distribution.

    Learns p(y | x) as a Poisson distribution and exposes mean and
    quantile forecasts.
    """

    def __init__(self):
        super().__init__()
        
        self.model = NGBRegressor(
            Dist=Poisson,
            natural_gradient=True,
            n_estimators=50,
            learning_rate=0.1,
            minibatch_frac=0.2, 
            col_sample=1.0,
            verbose=True,
            tol=1e-3,
            validation_fraction=0.1,
        )

    def fit(self, x_train, x_valid, y_train, y_valid):
        """Fit NGBoost on training data using Poisson likelihood."""
        self.model.fit(x_train, y_train, X_val=x_valid, Y_val=y_valid, early_stopping_rounds=10)
        return self

    def predict(self, x):
        """Return the conditional mean E[Y|x]."""
        X = x.values if hasattr(x, 'values') else x
        # NGBoost predict returns mean by default for Dist=Poisson
        return self.model.predict(X)

    def predict_quantile(self, x, q: float):
        """Return the q-th quantile of the predictive distribution."""
        X = x.values if hasattr(x, 'values') else x
        dist = self.model.pred_dist(X)
        return dist.ppf(q)