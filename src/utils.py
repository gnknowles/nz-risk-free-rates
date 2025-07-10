import pandas as pd
import numpy as np
import scipy

from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

class ModellingFunctions:
    
    @staticmethod
    def bootstrap_forward_columns(
        df: pd.DataFrame,
        term_col: str = "term_yr",
        spot_col: str = "spot_rate_pa",
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Adds discount factor and bootstrapped forward rate columns to an existing DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with term and spot rate columns.
            term_col (str): Name of the column containing term in years.
            spot_col (str): Name of the column containing annual spot rates as decimals.
            inplace (bool): If True, modifies the original DataFrame; otherwise returns a copy.

        Returns:
            pd.DataFrame: Same as input with added:
                - 'discount_factor'
                - 'fwd_rate_bootstrapped'
        """
        if not inplace:
            df = df.copy()

        df = df.sort_values(by=term_col).reset_index(drop=True)

        # Compute discount factor
        df["discount_factor"] = 1 / (1 + df[spot_col]) ** df[term_col]

        # Compute bootstrapped forward rates
        fwd_rates = []
        for i in range(1, len(df)):
            t0 = df.loc[i - 1, term_col]
            t1 = df.loc[i, term_col]
            D0 = df.loc[i - 1, "discount_factor"]
            D1 = df.loc[i, "discount_factor"]
            fwd = (D0 / D1) ** (1 / (t1 - t0)) - 1
            fwd_rates.append((t1, fwd))

        fwd_map = dict(fwd_rates)
        df["fwd_rate_bootstrapped"] = df[term_col].map(fwd_map)

        # Fill missing (usually the first row)
        df["fwd_rate_bootstrapped"].fillna(df[spot_col], inplace=True)

        return df
    

    @staticmethod
    def fit_cubic_forward_curve_error(
        knot_fwd_rates: np.ndarray,
        knots: np.ndarray,
        df: pd.DataFrame,
        term_col: str = "term_yr",
        spot_col: str = "spot_rate_pa",
        weight_col: str = "weight"
    ) -> float:
        """
        Calculates the weighted squared error between modeled and market spot rates,
        using a forward rate cubic spline defined by knot values.

        Args:
            knot_fwd_rates (np.ndarray): Array of forward rates at specified knots.
            knots (np.ndarray): Knot positions (must match length of knot_fwd_rates).
            df (pd.DataFrame): Market data with term, spot rate, and weight columns.
            term_col (str): Column name for term in years.
            spot_col (str): Column name for market spot rates (as decimals).
            weight_col (str): Column name for market weights.

        Returns:
            float: Weighted squared error between modeled and market spot rates.
        """
        # Ensure knot_fwd_rates uses market spot rate at time zero
        knot_fwd_rates = knot_fwd_rates.copy()
        knot_fwd_rates[0] = df[spot_col].iloc[0]

        # Build forward curve spline
        spline = CubicSpline(knots, knot_fwd_rates, bc_type='natural', extrapolate=True)

        # Evaluate forward rates and compute discount factors
        times = df[term_col].values
        fwd_vals = spline(times)

        # Numerical integration of forward curve
        integral = cumulative_trapezoid(fwd_vals, times, initial=0)
        discount_factors = np.exp(-integral)

        # Spot rate from discount factor
        spot_model = -np.log(discount_factors) / times

        # Weighted squared error
        spot_market = df[spot_col].values
        weights = df[weight_col].values
        error = np.sum(weights * (spot_model - spot_market) ** 2)

        return error

    @staticmethod
    def optimize_forward_curve_spline(
        df: pd.DataFrame,
        knots: np.ndarray,
        error_func,
        term_col: str = "term_yr",
        spot_col: str = "spot_rate_pa",
        weight_col: str = "weight",
        bounds: tuple = (0.0001, 0.1),
        method: str = "L-BFGS-B",
        options: dict = None
    ) -> tuple[CubicSpline, np.ndarray, object]:
        """
        Fit a forward curve using cubic spline optimization.

        Args:
            df (pd.DataFrame): DataFrame with term, spot, and weight columns.
            knots (np.ndarray): Knot positions (years).
            error_func (function): Function to compute error (like fit_forward_curve_error).
            term_col (str): Name of term column.
            spot_col (str): Name of spot rate column.
            weight_col (str): Name of weight column.
            bounds (tuple): (min, max) bounds for each forward rate at the knots.
            method (str): Optimization method for `scipy.optimize.minimize` (e.g. "L-BFGS-B", "TNC").
            options (dict): Optional dictionary of solver options (e.g., {'maxiter': 100}).

        Returns:
            tuple:
                - CubicSpline: fitted spline object
                - np.ndarray: optimized forward rates at knots
                - OptimizeResult: raw result object from `scipy.optimize.minimize`
        """
        # Initial guess: interpolate spot curve onto knots
        init_guess = np.interp(knots, df[term_col], df[spot_col])

        # Set up bounds list
        bounds_list = [bounds] * len(knots)

        # Minimize the curve-fitting error
        res = minimize(
            fun=error_func,
            x0=init_guess,
            args=(knots, df, term_col, spot_col, weight_col),
            method=method,
            bounds=bounds_list,
            options=options or {}
        )

        # Build final forward curve spline from optimized knot rates
        spline = CubicSpline(knots, res.x, bc_type="natural", extrapolate=True)

        return spline, res.x, res


    @staticmethod
    def bridge_forward_curve_to_longterm(
        df_curve: pd.DataFrame,
        term_col: str = "term_yr",
        fwd_col: str = "fwd_rate_cubic",
        long_term_rate: float = 0.048,
        max_slope: float = 0.0005,
        extension_freq: float = 0.25,
        min_extension_years: float = 10.0
    ) -> pd.DataFrame:
        """
        Extends a forward curve using a capped linear slope to reach a long-term rate (bridging).

        Args:
            df_curve (pd.DataFrame): DataFrame with existing forward curve data.
            term_col (str): Name of the column for term in years.
            fwd_col (str): Name of the column for forward rate values.
            long_term_rate (float): Long-run forward rate assumption (e.g. 0.048 = 4.8%).
            max_slope (float): Max allowed absolute slope (e.g. 0.0005 = 0.05% per year).
            extension_freq (float): Frequency for additional terms (e.g. 0.25 = quarterly).
            min_extension_years (float): Minimum length of extension in years.

        Returns:
            pd.DataFrame: Extended curve with term_yr and fwd_rate_extended.
        """
        df_curve = df_curve.sort_values(term_col).copy()
        T_last = df_curve[term_col].max()
        last_fwd_rate = df_curve[df_curve[term_col] <= T_last][fwd_col].iloc[-1]

        # Determine how many years are needed to converge to long-term rate
        required_years = abs(long_term_rate - last_fwd_rate) / max_slope
        T_extension = max(min_extension_years, np.ceil(required_years * 4) / 4)
        T_max = T_last + T_extension

        # Compute capped slope
        raw_slope = (long_term_rate - last_fwd_rate) / T_extension
        slope = min(max(raw_slope, -max_slope), max_slope)

        # Build extension terms and rates
        extension_terms = np.arange(T_last + extension_freq, T_max + extension_freq, extension_freq)
        extension_fwd_rates = last_fwd_rate + slope * (extension_terms - T_last)

        # Combine original and extension
        base_df = df_curve[[term_col, fwd_col]].rename(columns={fwd_col: "fwd_rate_extended"})
        extension_df = pd.DataFrame({
            term_col: extension_terms,
            "fwd_rate_extended": extension_fwd_rates
        })

        df_extended = pd.concat([base_df, extension_df], ignore_index=True).reset_index(drop=True)
        return df_extended



