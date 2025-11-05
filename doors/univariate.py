import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def get_univariate_pvalues(
    data: pd.DataFrame,
    feats: Iterable[str],
    target_name: str,
    model_type: str = "linreg",
) -> pd.DataFrame:
    """
    Run univariate regressions of `target_name` on each feature in `feats`.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the features and target column.
    feats : Iterable[str]
        Iterable of column names in `data` to test (one model per feature).
    target_name : str
        Name of the target column. For `model_type='logreg'` the values will be cast
          to int
        (should be binary). For `model_type='linreg'` the values will be cast to float.
    model_type : {'logreg', 'linreg'}, optional
        Which univariate model to fit. 'logreg' fits statsmodels.Logit
        (default 'linreg' uses OLS).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by `feats` with columns:
        - 'Coefficient': estimated coefficient for the feature (intercept excluded).
        - 'p-value': raw p-value for the feature coefficient.
        - 'adj_p-value': FDR (Benjamini–Hochberg) adjusted p-values.
        - 'Adj_R2': adjusted R² (only populated for linreg).
    """
    p_values: list[float] = []
    coefficients: list[float] = []
    adj_r2: list[float] = []

    model_type = model_type.lower()
    if model_type not in {"logreg", "linreg"}:
        raise ValueError("model_type must be 'logreg' or 'linreg'")

    for column in feats:
        # prepare single-feature design matrix (with intercept)
        X = sm.add_constant(data[column].fillna(0).astype(float))
        if model_type == "logreg":
            y = data[target_name].astype(int)
        elif model_type == "linreg":
            y = data[target_name].astype(float)
        else:
            raise ValueError("Model type: {model_type} Not implemented")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if model_type == "logreg":
                    model = sm.Logit(y, X).fit(disp=False)
                else:
                    model = sm.OLS(y, X).fit()

            coef = model.params.get(column, np.nan)
            pval = model.pvalues.get(column, 1.0)

            # compute R^2 only for OLS
            if model_type == "logreg":
                adj_r2.append(np.nan)
            else:
                adj_r2_val = getattr(model, "rsquared_adj", np.nan)
                adj_r2.append(adj_r2_val)

            coefficients.append(coef)
            p_values.append(pval)

        except (np.linalg.LinAlgError, PerfectSeparationError) as exc:
            print(f"{column} skipped due to {type(exc).__name__}")
            coefficients.append(np.nan)
            p_values.append(1.0)
            adj_r2.append(np.nan)
        except Exception as exc:
            # catch other unexpected errors but continue processing remaining features
            print(
                f"{column} skipped due to unexpected error: {type(exc).__name__}: {exc}"
            )
            coefficients.append(np.nan)
            p_values.append(1.0)
            adj_r2.append(np.nan)

    # Assemble results
    results = pd.DataFrame(
        {
            "Coefficient": coefficients,
            "p-value": p_values,
            "Adj_R2": adj_r2,
        },
        index=list(feats),
    )

    # FDR correction (Benjamini-Hochberg)
    _, adj_pvalues = fdrcorrection(results["p-value"].fillna(1.0))
    results["adj_p-value"] = adj_pvalues

    # Sort by raw p-value (most significant first)
    results = results.sort_values("p-value", ascending=True)
    return results
