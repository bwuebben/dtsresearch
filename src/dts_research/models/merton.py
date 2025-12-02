"""
Merton model lambda calculations based on theoretical tables from Wuebben (2025).

These adjustment factors correct for:
1. Maturity effects (lambda_T): bonds of different maturities have different sensitivities
2. Credit quality effects (lambda_s): bonds at different spread levels have different sensitivities
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Union


class MertonLambdaCalculator:
    """
    Calculate Merton-based adjustment factors for bond spread sensitivities.

    Based on theoretical predictions from Wuebben (2025).
    """

    # Maturity adjustment factors: lambda_T(T; s_ref=5y)
    # Rows: spread levels (bps), Columns: maturities (years)
    LAMBDA_T_TABLE = pd.DataFrame({
        'spread_bps': [50, 100, 200, 300, 500, 1000, 2000],
        '1y': [3.62, 3.27, 2.78, 2.40, 1.91, 1.26, 1.08],
        '3y': [1.47, 1.42, 1.36, 1.30, 1.25, 1.12, 1.05],
        '5y': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        '7y': [0.79, 0.80, 0.82, 0.84, 0.86, 0.91, 0.97],
        '10y': [0.61, 0.64, 0.67, 0.70, 0.73, 0.81, 0.93]
    })

    # Credit quality adjustment factors: lambda_s(s; s_ref=100bps)
    LAMBDA_S_TABLE = pd.DataFrame({
        'spread_bps': [50, 100, 200, 300, 500, 1000, 2000],
        'exact_merton': [1.145, 1.000, 0.847, 0.746, 0.635, 0.468, 0.299],
        'power_law': [1.189, 1.000, 0.841, 0.760, 0.669, 0.562, 0.473]
    })

    def __init__(self, use_power_law: bool = False):
        """
        Initialize lambda calculator.

        Args:
            use_power_law: If True, use power law approximation lambda_s = (s/100)^-0.25
                          If False, use exact Merton values with interpolation
        """
        self.use_power_law = use_power_law
        self._setup_interpolators()

    def _setup_interpolators(self):
        """Set up interpolation functions for lambda tables."""
        # Lambda_T interpolators for each maturity
        self.lambda_T_interpolators = {}
        for maturity in ['1y', '3y', '5y', '7y', '10y']:
            self.lambda_T_interpolators[maturity] = interp1d(
                self.LAMBDA_T_TABLE['spread_bps'],
                self.LAMBDA_T_TABLE[maturity],
                kind='linear',
                bounds_error=False,
                fill_value=(self.LAMBDA_T_TABLE[maturity].iloc[0],
                           self.LAMBDA_T_TABLE[maturity].iloc[-1])
            )

        # Lambda_s interpolator
        lambda_s_col = 'power_law' if self.use_power_law else 'exact_merton'
        self.lambda_s_interpolator = interp1d(
            self.LAMBDA_S_TABLE['spread_bps'],
            self.LAMBDA_S_TABLE[lambda_s_col],
            kind='linear',
            bounds_error=False,
            fill_value=(self.LAMBDA_S_TABLE[lambda_s_col].iloc[0],
                       self.LAMBDA_S_TABLE[lambda_s_col].iloc[-1])
        )

    def lambda_T(
        self,
        maturity_years: Union[float, np.ndarray],
        spread_bps: Union[float, np.ndarray],
        reference_maturity: float = 5.0
    ) -> Union[float, np.ndarray]:
        """
        Calculate maturity adjustment factor lambda_T.

        Adjusts for the fact that bonds of different maturities from the same issuer
        have different spread sensitivities.

        Args:
            maturity_years: Time to maturity in years
            spread_bps: Spread level in basis points
            reference_maturity: Reference maturity (default 5 years)

        Returns:
            Adjustment factor (>1 means more sensitive than 5y reference,
                              <1 means less sensitive)
        """
        scalar_input = np.isscalar(maturity_years) and np.isscalar(spread_bps)

        maturity_years = np.atleast_1d(maturity_years)
        spread_bps = np.atleast_1d(spread_bps)

        # Ensure same length
        if len(maturity_years) == 1:
            maturity_years = np.repeat(maturity_years, len(spread_bps))
        if len(spread_bps) == 1:
            spread_bps = np.repeat(spread_bps, len(maturity_years))

        result = np.zeros(len(maturity_years))

        # Define maturity breakpoints for interpolation
        mat_breakpoints = [1, 3, 5, 7, 10]

        for i, (mat, spread) in enumerate(zip(maturity_years, spread_bps)):
            if mat == reference_maturity:
                result[i] = 1.0
            elif mat < 1:
                # Extrapolate: assume very short bonds follow 1y pattern
                result[i] = float(self.lambda_T_interpolators['1y'](spread))
            elif mat > 10:
                # Extrapolate: assume very long bonds follow 10y pattern
                result[i] = float(self.lambda_T_interpolators['10y'](spread))
            else:
                # Interpolate between two adjacent maturities
                # Find surrounding maturity points
                idx_upper = next(idx for idx, m in enumerate(mat_breakpoints) if m >= mat)
                if mat_breakpoints[idx_upper] == mat:
                    # Exact match
                    mat_key = f'{int(mat)}y'
                    result[i] = float(self.lambda_T_interpolators[mat_key](spread))
                else:
                    # Interpolate between two maturities
                    idx_lower = idx_upper - 1
                    mat_lower = mat_breakpoints[idx_lower]
                    mat_upper = mat_breakpoints[idx_upper]

                    mat_key_lower = f'{int(mat_lower)}y'
                    mat_key_upper = f'{int(mat_upper)}y'

                    lambda_lower = float(self.lambda_T_interpolators[mat_key_lower](spread))
                    lambda_upper = float(self.lambda_T_interpolators[mat_key_upper](spread))

                    # Linear interpolation between maturities
                    weight = (mat - mat_lower) / (mat_upper - mat_lower)
                    result[i] = lambda_lower + weight * (lambda_upper - lambda_lower)

        return result[0] if scalar_input else result

    def lambda_s(
        self,
        spread_bps: Union[float, np.ndarray],
        reference_spread: float = 100.0
    ) -> Union[float, np.ndarray]:
        """
        Calculate credit quality adjustment factor lambda_s.

        Adjusts for the fact that bonds at different spread levels have
        different spread sensitivities.

        Args:
            spread_bps: Spread level in basis points
            reference_spread: Reference spread level (default 100 bps)

        Returns:
            Adjustment factor (>1 means more sensitive than 100bps reference,
                              <1 means less sensitive)
        """
        scalar_input = np.isscalar(spread_bps)
        spread_bps = np.atleast_1d(spread_bps)

        if self.use_power_law:
            # Power law approximation: lambda_s = (s/100)^-0.25
            result = np.power(spread_bps / reference_spread, -0.25)
        else:
            # Use interpolated exact Merton values
            result = self.lambda_s_interpolator(spread_bps)

        return float(result[0]) if scalar_input else result

    def lambda_combined(
        self,
        maturity_years: Union[float, np.ndarray],
        spread_bps: Union[float, np.ndarray],
        reference_maturity: float = 5.0,
        reference_spread: float = 100.0
    ) -> Union[float, np.ndarray]:
        """
        Calculate combined adjustment factor: lambda = lambda_T * lambda_s.

        This is the total adjustment relative to a reference bond with
        5-year maturity and 100 bps spread.

        Args:
            maturity_years: Time to maturity in years
            spread_bps: Spread level in basis points
            reference_maturity: Reference maturity (default 5 years)
            reference_spread: Reference spread (default 100 bps)

        Returns:
            Combined adjustment factor
        """
        lambda_T_val = self.lambda_T(maturity_years, spread_bps, reference_maturity)
        lambda_s_val = self.lambda_s(spread_bps, reference_spread)

        return lambda_T_val * lambda_s_val

    def classify_regime(
        self,
        spread_bps: float,
        maturity_range: float
    ) -> str:
        """
        Classify bond into one of five theoretical regimes.

        Args:
            spread_bps: Average spread level
            maturity_range: Range of maturities in portfolio (max - min years)

        Returns:
            Regime classification string
        """
        if spread_bps < 300:
            if maturity_range < 2:
                return "Regime 1: IG narrow maturity"
            else:
                return "Regime 2: IG wide maturity (PRIMARY FAILURE MODE)"
        elif spread_bps < 1000:
            if maturity_range < 2:
                return "Regime 3: HY narrow maturity"
            else:
                return "Regime 4: HY wide maturity"
        else:
            return "Regime 5: Distressed"

    def get_regime_description(self, regime: str) -> str:
        """Get detailed description of regime characteristics."""
        descriptions = {
            "Regime 1: IG narrow maturity": (
                "Standard DTS works reasonably. Maturity effects not applicable. "
                "Credit quality variation causes 20-35% deviation—acceptable."
            ),
            "Regime 2: IG wide maturity (PRIMARY FAILURE MODE)": (
                "Cross-maturity lambda ratios of 3-6x create 300-500% deviations. "
                "This is where DTS models systematically fail."
            ),
            "Regime 3: HY narrow maturity": (
                "Cross-maturity effects reduced to 50-160% but still substantial. "
                "Same-maturity credit quality variation increases to 40-73%."
            ),
            "Regime 4: HY wide maturity": (
                "Both maturity and credit quality effects large. "
                "Comprehensive adjustments required."
            ),
            "Regime 5: Distressed": (
                "Proportionality paradoxically improves. "
                "Both cross-maturity and same-maturity deviations decline."
            )
        }
        return descriptions.get(regime, "Unknown regime")


# Convenience function for quick calculations
def calculate_merton_lambda(
    maturity_years: Union[float, np.ndarray],
    spread_bps: Union[float, np.ndarray],
    use_power_law: bool = False
) -> Union[float, np.ndarray]:
    """
    Quick calculation of combined Merton lambda.

    Args:
        maturity_years: Time to maturity in years
        spread_bps: Spread level in basis points
        use_power_law: Use power law approximation for lambda_s

    Returns:
        Combined adjustment factor lambda = lambda_T * lambda_s
    """
    calc = MertonLambdaCalculator(use_power_law=use_power_law)
    return calc.lambda_combined(maturity_years, spread_bps)
