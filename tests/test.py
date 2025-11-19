import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal

from your_module_name import bootstrap_forward_columns  # update module name


def test_bootstrap_adds_columns_correctly():
    df = pd.DataFrame({
        "term_yr": [1, 2],
        "spot_rate_pa": [0.05, 0.06]
    })

    result = bootstrap_forward_columns(df, inplace=False)

    assert "discount_factor" in result.columns
    assert "fwd_rate_bootstrapped" in result.columns


def test_discount_factor_calculation():
    df = pd.DataFrame({
        "term_yr": [1],
        "spot_rate_pa": [0.05]
    })

    result = bootstrap_forward_columns(df, inplace=False)
    expected_df = pd.DataFrame({
        "term_yr": [1],
        "spot_rate_pa": [0.05],
        "discount_factor": [1/(1.05)],
        "fwd_rate_bootstrapped": [0.05]
    })

    pd.testing.assert_frame_equal(result, expected_df, atol=1e-10)


def test_forward_rate_bootstrap_correctness():
    df = pd.DataFrame({
        "term_yr": [1, 2],
        "spot_rate_pa": [0.05, 0.06]
    })

    result = bootstrap_forward_columns(df)

    # Expected forward rate calculation
    D0 = 1 / (1 + 0.05) ** 1
    D1 = 1 / (1 + 0.06) ** 2
    expected_fwd = (D0 / D1) ** (1 / (2 - 1)) - 1

    assert abs(result.loc[1, "fwd_rate_bootstrapped"] - expected_fwd) < 1e-10


def test_unsorted_input_is_sorted():
    df = pd.DataFrame({
        "term_yr": [3, 1, 2],
        "spot_rate_pa": [0.07, 0.05, 0.06]
    })

    result = bootstrap_forward_columns(df)
    
    assert list(result["term_yr"]) == [1, 2, 3]


def test_inplace_modification():
    df = pd.DataFrame({
        "term_yr": [1, 2],
        "spot_rate_pa": [0.05, 0.06]
    })

    returned = bootstrap_forward_columns(df, inplace=True)

    # Should modify original df
    assert "discount_factor" in df.columns
    assert returned is df


def test_single_row_forward_rate_defaults_to_spot():
    df = pd.DataFrame({
        "term_yr": [1],
        "spot_rate_pa": [0.05]
    })

    result = bootstrap_forward_columns(df)

    assert result.loc[0, "fwd_rate_bootstrapped"] == 0.05
