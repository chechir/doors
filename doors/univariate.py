import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def get_univariate_pvalues(
    data: pd.DataFrame, feats: list[str], target_name: str
) -> pd.DataFrame:
    p_values = []
    coefficients = []
    pseudo_r2 = []

    for column in feats:
        X = sm.add_constant(data[column].fillna(0))  # intercept + single feature
        y = data[target_name].astype(int)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.Logit(y, X).fit(disp=False)

            coef = model.params[column]
            pval = model.pvalues[column]

            # McFadden's pseudo-R^2
            llf = model.llf  # log-likelihood of fitted model
            llnull = model.llnull  # log-likelihood of intercept-only model
            r2_mcfadden = 1.0 - (llf / llnull)

            coefficients.append(coef)
            p_values.append(pval)
            pseudo_r2.append(r2_mcfadden)

        except (np.linalg.LinAlgError, PerfectSeparationError) as exc:
            print(f"{column} skipped due to {type(exc).__name__}")
            coefficients.append(np.nan)
            p_values.append(1.0)
            pseudo_r2.append(np.nan)

    # Assemble results
    logit_results = pd.DataFrame(
        {"Coefficient": coefficients, "p-value": p_values, "McFadden_R2": pseudo_r2},
        index=feats,
    )

    # FDR correction
    _, adj_pvalues = fdrcorrection(logit_results["p-value"].fillna(1.0))
    logit_results["adj_p-value"] = adj_pvalues

    # Sort by adjusted p-value (most significant first)
    logit_results = logit_results.sort_values("adj_p-value", ascending=True)
    return logit_results
