import numpy as np
import pandas as pd

from doors.eda import get_correlations_for_col


def test_get_correlations_for_col():
    data = pd.DataFrame(
        {
            "target": [1, 2, 3],
            "feat": [2, 3, 9],
            "feat2": [4, 3, 3],
        }
    )
    res = get_correlations_for_col(data, col="target", method="spearman")
    assert res.shape[0] == 3
    assert np.allclose(res["abs"], [1, 1, 0.866025])
    assert np.allclose(res["corr"], [1, 1, -0.866025])
