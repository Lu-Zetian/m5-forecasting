import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Hyperparameters controlling time-based feature construction and forecast horizon.
LAGS = [7, 28]          # how many days back to look for lag features
WINDOWS = [7, 28]       # window sizes (in days) for rolling mean features
FIRST = 1942            # first day index to predict
LENGTH = 28             # number of forecast days

def prep_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode calendar features for modeling.

    Drops unused string columns, converts ``d`` from labels ("d_1942") to
    integers, ordinal-encodes event names, and downcasts selected columns
    to ``int8``.
    """
    df = df.drop(["date", "weekday", "event_type_1", "event_type_2"], axis=1)
    df = df.assign(d = df.d.str[2:].astype(int))
    to_ordinal = ["event_name_1", "event_name_2"] 
    df[to_ordinal] = df[to_ordinal].fillna("1")
    df[to_ordinal] = OrdinalEncoder(dtype="int").fit_transform(df[to_ordinal]) + 1
    to_int8 = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"] + to_ordinal
    df[to_int8] = df[to_int8].astype("int8")
    
    return df

def add_demand_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-mean demand features to long-format sales data.

    For each ``id`` and each lag in ``LAGS``, creates ``lag_t{lag}`` and
    ``rolling_mean_lag{lag}_w{w}`` for all windows in ``WINDOWS``.
    """
    for lag in LAGS:
        df[f'lag_t{lag}'] = df.groupby('id')['demand'].transform(lambda x: x.shift(lag)).astype("float32")
        for w in WINDOWS:
            df[f'rolling_mean_lag{lag}_w{w}'] = df.groupby('id')[f'lag_t{lag}'].transform(lambda x: x.rolling(w).mean()).astype("float32")

    return df

def build_eval_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lag/rolling features for recursive one-step-ahead forecasting.

    Takes a recent-history window, keeps the last row per ``id`` and
    adds lag and rolling-mean features based on ``LAGS`` and ``WINDOWS``.
    """
    out = df.groupby('id', sort=False, observed=False).last()
    for lag in LAGS:
        out[f'lag_t{lag}'] = df.groupby('id', sort=False, observed=False)['demand'].nth(-lag-1).astype("float32")
        for w in WINDOWS:
            temp = df.groupby('id', sort=False, observed=False)[['id', 'demand']].nth(list(range(-lag-w, -lag)))
            out[f'rolling_mean_lag{lag}_w{w}'] = temp.groupby('id', sort=False, observed=False).mean().astype("float32")

    return out.reset_index()


def downcast(df):
    """Downcast numeric and object columns to smaller dtypes where possible."""
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i,t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df  


def prepare_training_data(sales: pd.DataFrame, calendar: pd.DataFrame, selling_prices: pd.DataFrame, drop_d: int = 1000):
    """Prepare train/validation and test sets for forecasting models.

    Converts wide sales data to long format, engineers lag/rolling
    features, merges calendar and price data, encodes categoricals and
    returns ``x_train``, ``x_valid``, ``y_train``, ``y_valid``, ``test``
    and the list of feature column names.
    """
    # 1. Kick out old dates from the wide sales table
    df = sales.drop(["d_" + str(i+1) for i in range(drop_d)], axis=1)

    # 2. Reshape to long format and standardize IDs
    df = df.assign(id=df.id.str.replace("_evaluation", ""))
    df = df.reindex(columns=df.columns.tolist() + ["d_" + str(FIRST + i) for i in range(LENGTH)])
    df = df.melt(id_vars=["id", "item_id", "store_id", "state_id", "dept_id", "cat_id"], var_name='d', value_name='demand')
    df = df.assign(d=df.d.str[2:].astype("int64"), demand=df.demand.astype("float32"))
    
    # 3. Add lag and rolling demand features
    df = add_demand_lag_features(df)
    
    # 4. Remove rows with insufficient history (NaNs in lag/rolling features)
    df = df[df.d > (drop_d + max(LAGS) + max(WINDOWS))]
 
    # 5. Join calendar features
    df = df.merge(calendar, how="left", on="d")
    
    # 6. Join selling price features
    df = df.merge(selling_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])
    df = df.drop(["wm_yr_wk"], axis=1)
    median_price = df['sell_price'].median()
    df['sell_price'] = df['sell_price'].fillna(median_price)
    
    # 7. Ordinal encoding of remaining categorical identifiers
    for v in ["item_id", "store_id", "state_id", "dept_id", "cat_id"]:
        df[v] = OrdinalEncoder(dtype="int").fit_transform(df[[v]]).astype("int16") + 1
    
    # Determine list of covariates (all non-ID, non-target columns)
    features = list(set(df.columns) - {'id', 'd', 'demand'})
            
    # 8. Split into test (for recursive prediction) and train/valid sets
    test = df[df.d >= FIRST - max(LAGS) - max(WINDOWS) - 28]
    df = df[df.d < FIRST]

    x_train, x_valid, y_train, y_valid = train_test_split(df[features], df["demand"], test_size=0.1, shuffle=True, random_state=54)

    return x_train, x_valid, y_train, y_valid, test, features


def forecast_point_horizon(model, test: pd.DataFrame, features):
    """Recursive 28-day point forecasts for all series.

    For each day in the forecast horizon, recomputes features from a
    moving window via ``demand_features_eval`` and fills ``demand`` with
    model predictions.
    """
    # Recursive prediction
    pred = test.copy()
    for i, day in enumerate(np.arange(FIRST, FIRST + LENGTH)):
        test_day = build_eval_features(pred[(pred.d <= day) & (pred.d >= day - max(LAGS) - max(WINDOWS))])
        pred.loc[pred.d==day, "demand"] = model.predict(test_day[features])
    
    return pred

    
def forecast_quantile_horizon(model, test: pd.DataFrame, features, q_levels):
    """Recursive 28-day forecasting for multiple quantile levels.

    For each ``q`` in ``q_levels`` calls ``model.predict_quantile``
    inside the same recursive scheme as ``pred_all`` and returns a dict
    mapping ``q`` to a prediction DataFrame.
    """
    pred_per_q = {}
    for q in q_levels:
        pred = test.copy()
        for day in np.arange(FIRST, FIRST + LENGTH):
            window = pred[(pred.d <= day) & (pred.d >= day - max(LAGS) - max(WINDOWS))]
            test_day = build_eval_features(window)
            pred.loc[pred.d == day, 'demand'] = model.predict_quantile(test_day[features], q)
        pred_per_q[q] = pred
    return pred_per_q


def save_accuracy_submission(pred: pd.DataFrame, cols_template: pd.DataFrame, filepath: str = "submission.csv") -> bool:
    """Convert long-format predictions to the official accuracy file layout.

    Adds validation/evaluation suffixes to ``id``, maps ``d`` to F1-F28,
    pivots to wide format using ``cols_template`` and writes ``filepath``.
    """

    # Prepare for reshaping
    pred = pred.assign(
        id=pred.id + "_" + np.where(pred.d < FIRST, "validation", "evaluation"),
        F="F" + (pred.d - FIRST + LENGTH + 1 - LENGTH * (pred.d >= FIRST)).astype("str"),
    )

    # Reshape to submission format
    submission = pred.pivot(index="id", columns="F", values="demand").reset_index()[cols_template.columns].fillna(1)

    # Export to CSV
    submission.to_csv(filepath, index=False)

    return True


def save_uncertainty_submission(pred_per_q, cols_template: pd.DataFrame, filepath: str = "submission_uncertainty.csv") -> bool:
    """Convert multi-quantile forecasts into the M5 uncertainty submission.

    Stacks per-quantile long predictions, builds validation/evaluation ids
    with embedded quantile, pivots to F1-F28 columns and writes ``filepath``.
    """

    # Build one long table over all quantiles with: base_id, quantile_str, d, demand
    records = []
    for q, pred_q in pred_per_q.items():
        tmp = pred_q.copy()
        # Base id (no validation/evaluation suffix) and quantile as string
        base_id = tmp['id'].str.replace("_evaluation", "", regex=False)
        base_id = base_id.str.replace("_validation", "", regex=False)
        q_str = tmp['quantile'].iloc[0] if 'quantile' in tmp.columns else q
        q_str = (np.format_float_positional(q_str, trim='-')
                 if isinstance(q_str, float) else str(q_str))
        df_long = pd.DataFrame({
            'base_id': base_id,
            'quantile': q_str,
            'd': tmp['d'].astype(int),
            'demand': tmp['demand'].astype(float)
        })
        records.append(df_long)

    long = pd.concat(records, axis=0, ignore_index=True)

    # Map d -> F label and split into validation / evaluation parts
    # Validation horizon: d < FIRST  -> ids end with _validation
    # Evaluation horizon: d >= FIRST -> ids end with _evaluation
    long = long.assign(
        F="F" + (long['d'] - FIRST + LENGTH + 1 - LENGTH * (long['d'] >= FIRST)).astype(str)
    )

    val = long[long['d'] < FIRST].copy()
    eval_ = long[long['d'] >= FIRST].copy()

    val['id'] = val['base_id'] + "_" + val['quantile'].astype(str) + "_validation"
    eval_['id'] = eval_['base_id'] + "_" + eval_['quantile'].astype(str) + "_evaluation"

    # Combine back and pivot to wide format: one row per (id), F1..F28 as columns
    long_all = pd.concat([val, eval_], axis=0, ignore_index=True)
    wide = long_all.pivot_table(index='id', columns='F', values='demand').reset_index()

    # Align columns to template; fill missing F* with 1.0 (safe baseline)
    target_cols = cols_template.columns
    for c in target_cols:
        if c not in wide.columns:
            wide[c] = 1.0
    wide = wide[target_cols]

    f_cols = [c for c in target_cols if c.startswith('F')]
    wide[f_cols] = wide[f_cols].fillna(1.0)

    wide.to_csv(filepath, index=False)
    return True