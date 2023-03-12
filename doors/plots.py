import numpy as np
import pandas as pd
from doors.np import rolling_mean
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from sklearn import linear_model, preprocessing


def add_best_fit_curve(x, y, degree, fit_intercept, **kwargs):
    mm = make_polynomial_mm(x=x, degree=degree)
    model = linear_model.LinearRegression(n_jobs=-1, fit_intercept=fit_intercept)
    model.fit(X=mm, y=y)
    predictions = model.predict(X=mm)

    ix = np.argsort(x)
    label = "Best Fit Degree {}".format(degree)
    plt.plot(x[ix], predictions[ix], "--", label=label, **kwargs)


def plot_best_fit(x, y, best_fit_degrees, fit_intercept):
    if not isinstance(best_fit_degrees, list):
        best_fit_degrees = [best_fit_degrees]
    for degree in best_fit_degrees:
        add_best_fit_curve(x, y, degree, fit_intercept)


def plot_rolling_mean(x, window, **matplotlib_kwargs):
    rolling_means = rolling_mean(x, window)
    plt.plot(rolling_means, **matplotlib_kwargs)


def make_polynomial_mm(x, degree):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    poly = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X=x)


def plot_venn2_primary_secondary(elements_by_group, venn_values, ax):
    """ Plot venn diagram of 2 groups
    it needs the elements_by_group (all elements that belong to a group,
    so if "element1" apears in A and B, it will apear in (A, b), (A, ), and (B, )
    venn_values: same as interactions but only shows elements once
    so if "element1" apears in A and B, it will apear only in (A, b)
    """
    A = elements_by_group[("primary",)]
    B = elements_by_group[("backup",)]
    v = venn2([A, B], ["backup", "primary"], ax=ax)

    v.get_label_by_id("11").set_text(
        "\n".join(np.array(list(venn_values[("backup", "primary")])).astype(str))
    )
    v.get_label_by_id("10").set_text(
        "\n".join(np.array(list(venn_values[("primary",)])).astype(str))
    )
    v.get_label_by_id("01").set_text(
        "\n".join(np.array(list(venn_values[("backup",)])).astype(str))
    )
    plt.show()

    return v


def get_correlations_for_col(df: pd.DataFrame, col: str, top=20) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)[col]
    corr = pd.DataFrame(
        {
            "abs": corr.abs(),
            "corr": corr,
        }
    )
    return corr
