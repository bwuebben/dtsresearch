"""
Stage 0: Within-Issuer Analysis

Implements the second component of evolved Stage 0:
- Uses same issuer, different maturities to isolate maturity effects
- Issuer-week fixed effects absorb common credit shocks
- Tests if cross-maturity spread CHANGES match Merton predictions
- Inverse-variance weighted pooling for meta-analysis

Based on Specification 0.2 / Equation 4.4 from the paper:
    Δs_{ij,t} / s_{ij,t-1} = α_{i,t} + β · λ_{ij,t}^Merton + ε_{ij,t}

where the issuer-week fixed effect α_{i,t} absorbs the common firm value shock,
and identification comes from cross-maturity variation within issuer-weeks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import statsmodels.api as sm
from scipy import stats

from ..data.issuer_identification import add_issuer_identification
from ..data.filters import apply_within_issuer_filters
from ..data.preprocessing import compute_spread_changes
from ..models.merton import MertonLambdaCalculator, calculate_merton_lambda
from ..utils.statistical_tests import (
    clustered_standard_errors,
    inverse_variance_weighted_pooling
)
from ..config import (
    MIN_BONDS_PER_ISSUER_WEEK,
    MIN_MATURITY_DISPERSION_YEARS,
    PULL_TO_PAR_EXCLUSION_YEARS
)


class WithinIssuerAnalysis:
    """
    Within-issuer analysis for Stage 0.

    For each issuer-week with ≥3 bonds spanning ≥2 years:
    1. Compute percentage spread CHANGES for each bond
    2. Compute Merton-predicted elasticity λ^Merton for each bond
    3. Run regression: Δs/s = α_{i,t} + β · λ^Merton + ε
    4. Pool β estimates using inverse-variance weighting
    5. Test H0: β = 1 (Merton predicts correctly)
    """

    def __init__(self):
        """Initialize within-issuer analysis."""
        self.min_bonds = MIN_BONDS_PER_ISSUER_WEEK
        self.min_dispersion = MIN_MATURITY_DISPERSION_YEARS
        self.pull_to_par = PULL_TO_PAR_EXCLUSION_YEARS
        self.merton_calc = MertonLambdaCalculator()

    def run_within_issuer_analysis(
        self,
        bond_data: pd.DataFrame,
        universe: str = 'IG',
        verbose: bool = False
    ) -> Dict:
        """
        Run complete within-issuer analysis.

        Args:
            bond_data: DataFrame with bond observations
            universe: 'IG' or 'HY'
            verbose: Print filter summary

        Returns:
            Dictionary with results:
            - issuer_week_estimates: DataFrame with β per issuer-week
            - pooled_estimate: Inverse-variance weighted pooled β
            - hypothesis_test: Test for β = 1 (Merton prediction)
            - diagnostics: Quality checks
        """
        # Step 1: Add issuer identification (ultimate_parent_id + seniority)
        bond_data = add_issuer_identification(
            bond_data,
            parent_id_col='ultimate_parent_id',
            seniority_col='seniority'
        )

        # Step 2: Filter to universe
        if universe == 'IG':
            bond_data = bond_data[bond_data['rating'].isin(['AAA', 'AA', 'A', 'BBB'])].copy()
        elif universe == 'HY':
            bond_data = bond_data[bond_data['rating'].isin(['BB', 'B', 'CCC'])].copy()
        else:
            raise ValueError("universe must be 'IG' or 'HY'")

        if len(bond_data) == 0:
            return self._empty_results(universe, "No bonds in universe")

        # Step 3: Compute spread changes
        bond_data = self._compute_spread_changes(bond_data)

        if len(bond_data) == 0:
            return self._empty_results(universe, "No data after computing spread changes")

        # Step 4: Compute Merton-predicted elasticity for each bond
        bond_data = self._compute_merton_lambda(bond_data)

        # Step 5: Apply within-issuer filters
        bond_data, filter_stats = apply_within_issuer_filters(
            bond_data,
            maturity_column='time_to_maturity',
            spread_change_column='spread_change',
            verbose=verbose
        )

        if len(bond_data) == 0:
            return self._empty_results(universe, "No data after filtering")

        # Step 6: Run issuer-week regressions
        issuer_week_estimates = self._run_issuer_week_regressions(bond_data)

        if len(issuer_week_estimates) == 0:
            return self._empty_results(universe, "No issuer-week estimates obtained")

        # Step 7: Pool estimates using inverse-variance weighting
        pooled = self._pool_estimates(issuer_week_estimates)

        # Step 8: Hypothesis test for β = 1 (not just β > 0)
        hypothesis_test = self._test_merton_prediction(pooled)

        # Step 9: Diagnostics
        diagnostics = self._compute_diagnostics(
            bond_data, issuer_week_estimates, filter_stats
        )

        return {
            'universe': universe,
            'issuer_week_estimates': issuer_week_estimates,
            'pooled_estimate': pooled,
            'hypothesis_test': hypothesis_test,
            'diagnostics': diagnostics,
            'filter_stats': filter_stats
        }

    def _compute_spread_changes(self, bond_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percentage spread changes for each bond.

        Uses centralized preprocessing from dts_research.data.preprocessing.

        Args:
            bond_data: DataFrame with 'oas', 'cusip' (or bond id), 'date'

        Returns:
            DataFrame with 'spread_change' and 'oas_lag' columns
        """
        bond_id_col = 'cusip' if 'cusip' in bond_data.columns else 'bond_id'

        # Use centralized spread change calculation
        bond_data = compute_spread_changes(
            bond_data,
            bond_id_col=bond_id_col,
            max_change_pct=1.0  # ±100% outlier filter
        )

        # Add legacy column name for backward compatibility
        bond_data['spread_change'] = bond_data['oas_pct_change']

        return bond_data

    def _compute_merton_lambda(self, bond_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Merton-predicted elasticity for each bond based on its
        lagged spread level and time to maturity.

        Args:
            bond_data: DataFrame with 'oas_lag' and 'time_to_maturity'

        Returns:
            DataFrame with 'lambda_merton' column
        """
        # Use lagged spread (spread at t-1) for computing lambda
        bond_data['lambda_merton'] = calculate_merton_lambda(
            bond_data['time_to_maturity'].values,
            bond_data['oas_lag'].values  # Spread in bps
        )

        return bond_data

    def _run_issuer_week_regressions(
        self,
        bond_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run within-issuer-week regressions.

        For each issuer-week:
            Δs_{ij,t} / s_{ij,t-1} = α_{i,t} + β · λ_{ij,t}^Merton + ε_{ij,t}

        The constant (α) absorbs the common firm value shock, and β
        captures whether cross-maturity spread changes match Merton predictions.

        Args:
            bond_data: Filtered DataFrame with spread_change and lambda_merton

        Returns:
            DataFrame with columns:
            - issuer_id, date, beta, beta_se, beta_tstat, beta_pvalue,
              n_bonds, r_squared, maturity_range
        """
        results = []

        for (issuer_id, date), group in bond_data.groupby(['issuer_id', 'date']):
            if len(group) < self.min_bonds:
                continue

            # Check maturity dispersion
            maturity_range = group['time_to_maturity'].max() - group['time_to_maturity'].min()
            if maturity_range < self.min_dispersion:
                continue

            # Check that we have variation in lambda_merton
            lambda_range = group['lambda_merton'].max() - group['lambda_merton'].min()
            if lambda_range < 0.1:  # Need sufficient variation for identification
                continue

            # Prepare data
            # Dependent variable: percentage spread change
            y = group['spread_change'].values

            # Independent variable: Merton-predicted elasticity
            X = group['lambda_merton'].values
            X_const = sm.add_constant(X)

            # Run OLS
            try:
                model = sm.OLS(y, X_const).fit()

                results.append({
                    'issuer_id': issuer_id,
                    'date': date,
                    'beta': model.params[1],  # Coefficient on lambda_merton
                    'beta_se': model.bse[1],
                    'beta_tstat': model.tvalues[1],
                    'beta_pvalue': model.pvalues[1],
                    'alpha': model.params[0],  # Absorbs common shock
                    'n_bonds': len(group),
                    'r_squared': model.rsquared,
                    'maturity_range': maturity_range,
                    'lambda_range': lambda_range,
                    'mean_spread': group['oas_lag'].mean(),
                    'spread_range': (group['oas_lag'].min(), group['oas_lag'].max())
                })

            except Exception:
                # Skip if regression fails
                continue

        if len(results) == 0:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def _pool_estimates(
        self,
        issuer_week_estimates: pd.DataFrame
    ) -> Dict:
        """
        Pool issuer-week estimates using inverse-variance weighting.

        Args:
            issuer_week_estimates: DataFrame with β estimates

        Returns:
            Dictionary with pooled estimate and inference
        """
        estimates = issuer_week_estimates['beta'].values
        variances = issuer_week_estimates['beta_se'].values ** 2

        pooled = inverse_variance_weighted_pooling(estimates, variances)

        # Rename for clarity
        pooled['pooled_beta'] = pooled.pop('pooled_estimate', np.nan)
        pooled['pooled_beta_se'] = pooled.pop('pooled_se', np.nan)

        return pooled

    def _test_merton_prediction(self, pooled: Dict) -> Dict:
        """
        Test Merton prediction: H0: β = 1 (Merton is correct)

        If β = 1, Merton's predicted elasticity ratios are exactly right.
        If β < 1, Merton over-predicts cross-maturity dispersion.
        If β > 1, Merton under-predicts cross-maturity dispersion.

        Args:
            pooled: Pooled estimate dictionary

        Returns:
            Dictionary with hypothesis test results
        """
        beta_pooled = pooled.get('pooled_beta', np.nan)
        beta_se = pooled.get('pooled_beta_se', np.nan)

        if np.isnan(beta_pooled) or np.isnan(beta_se) or beta_se == 0:
            return {
                'test': 'H0: β = 1 (Merton prediction)',
                't_statistic': np.nan,
                'p_value_beta_equals_1': np.nan,
                'p_value_beta_positive': np.nan,
                'reject_beta_equals_1': False,
                'beta_in_range': False,
                'interpretation': 'Insufficient data'
            }

        # Test 1: H0: β = 1 (two-sided)
        t_stat_eq_1 = (beta_pooled - 1.0) / beta_se
        p_value_eq_1 = 2 * (1 - stats.norm.cdf(np.abs(t_stat_eq_1)))

        # Test 2: H0: β ≤ 0 vs H1: β > 0 (sanity check)
        t_stat_pos = beta_pooled / beta_se
        p_value_pos = 1 - stats.norm.cdf(t_stat_pos)

        # Check if β is in acceptable range [0.9, 1.1]
        beta_in_range = 0.9 <= beta_pooled <= 1.1

        return {
            'test': 'H0: β = 1 (Merton prediction)',
            'beta_pooled': beta_pooled,
            'beta_se': beta_se,
            't_statistic_beta_eq_1': t_stat_eq_1,
            'p_value_beta_equals_1': p_value_eq_1,
            't_statistic_beta_pos': t_stat_pos,
            'p_value_beta_positive': p_value_pos,
            'reject_beta_equals_1': p_value_eq_1 < 0.05,
            'beta_in_range_0_9_1_1': beta_in_range,
            'merton_validates': beta_in_range and p_value_eq_1 > 0.10,
            'interpretation': self._interpret_merton_test(beta_pooled, beta_se, p_value_eq_1)
        }

    def _interpret_merton_test(
        self,
        beta_pooled: float,
        beta_se: float,
        p_value: float
    ) -> str:
        """Interpret Merton hypothesis test."""
        if np.isnan(beta_pooled) or np.isnan(p_value):
            return "Insufficient data for test"

        ci_lower = beta_pooled - 1.96 * beta_se
        ci_upper = beta_pooled + 1.96 * beta_se

        if 0.9 <= beta_pooled <= 1.1 and p_value > 0.10:
            return (
                f"β = {beta_pooled:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]) "
                f"is consistent with Merton (p={p_value:.3f} for H0: β=1)"
            )
        elif beta_pooled < 0.9:
            return (
                f"β = {beta_pooled:.3f} < 1: Merton OVER-predicts cross-maturity dispersion "
                f"(p={p_value:.4f} for H0: β=1)"
            )
        elif beta_pooled > 1.1:
            return (
                f"β = {beta_pooled:.3f} > 1: Merton UNDER-predicts cross-maturity dispersion "
                f"(p={p_value:.4f} for H0: β=1)"
            )
        else:
            return (
                f"β = {beta_pooled:.3f} is in range [0.9, 1.1] but significantly different from 1 "
                f"(p={p_value:.4f}) - marginal fit"
            )

    def _compute_diagnostics(
        self,
        bond_data: pd.DataFrame,
        issuer_week_estimates: pd.DataFrame,
        filter_stats: Dict
    ) -> Dict:
        """
        Compute diagnostic statistics.

        Args:
            bond_data: Filtered bond data
            issuer_week_estimates: Issuer-week estimates
            filter_stats: Filter statistics

        Returns:
            Dictionary with diagnostics
        """
        n_iw_total = bond_data.groupby(['issuer_id', 'date']).ngroups if len(bond_data) > 0 else 0

        return {
            'n_bonds_after_filter': len(bond_data),
            'n_unique_issuers': bond_data['issuer_id'].nunique() if len(bond_data) > 0 else 0,
            'n_unique_weeks': bond_data['date'].nunique() if len(bond_data) > 0 else 0,
            'n_issuer_weeks_total': n_iw_total,
            'n_issuer_weeks_with_estimate': len(issuer_week_estimates),
            'pct_issuer_weeks_with_estimate': 100.0 * len(issuer_week_estimates) / n_iw_total if n_iw_total > 0 else 0,
            'mean_bonds_per_issuer_week': issuer_week_estimates['n_bonds'].mean() if len(issuer_week_estimates) > 0 else np.nan,
            'mean_maturity_range': issuer_week_estimates['maturity_range'].mean() if len(issuer_week_estimates) > 0 else np.nan,
            'mean_r_squared': issuer_week_estimates['r_squared'].mean() if len(issuer_week_estimates) > 0 else np.nan,
            # β distribution
            'median_beta': issuer_week_estimates['beta'].median() if len(issuer_week_estimates) > 0 else np.nan,
            'mean_beta': issuer_week_estimates['beta'].mean() if len(issuer_week_estimates) > 0 else np.nan,
            'std_beta': issuer_week_estimates['beta'].std() if len(issuer_week_estimates) > 0 else np.nan,
            'pct_beta_in_0_8_1_2': 100.0 * ((issuer_week_estimates['beta'] >= 0.8) & (issuer_week_estimates['beta'] <= 1.2)).mean() if len(issuer_week_estimates) > 0 else 0,
            'pct_beta_positive': 100.0 * (issuer_week_estimates['beta'] > 0).mean() if len(issuer_week_estimates) > 0 else 0,
            'beta_range': (issuer_week_estimates['beta'].min(), issuer_week_estimates['beta'].max()) if len(issuer_week_estimates) > 0 else (np.nan, np.nan)
        }

    def _empty_results(self, universe: str, reason: str) -> Dict:
        """Return empty results structure with warning."""
        return {
            'universe': universe,
            'issuer_week_estimates': pd.DataFrame(),
            'pooled_estimate': {
                'pooled_beta': np.nan,
                'pooled_beta_se': np.nan,
                'n_estimates': 0,
                'warning': reason
            },
            'hypothesis_test': {
                'merton_validates': False,
                'interpretation': reason
            },
            'diagnostics': {
                'n_issuer_weeks_with_estimate': 0,
                'warning': reason
            }
        }

    def compare_ig_hy(
        self,
        ig_results: Dict,
        hy_results: Dict
    ) -> Dict:
        """
        Compare within-issuer results between IG and HY.

        Args:
            ig_results: Results from run_within_issuer_analysis(universe='IG')
            hy_results: Results from run_within_issuer_analysis(universe='HY')

        Returns:
            Dictionary with comparison statistics
        """
        ig_beta = ig_results['pooled_estimate'].get('pooled_beta', np.nan)
        hy_beta = hy_results['pooled_estimate'].get('pooled_beta', np.nan)

        ig_se = ig_results['pooled_estimate'].get('pooled_beta_se', np.nan)
        hy_se = hy_results['pooled_estimate'].get('pooled_beta_se', np.nan)

        ig_validates = ig_results['hypothesis_test'].get('merton_validates', False)
        hy_validates = hy_results['hypothesis_test'].get('merton_validates', False)

        # Test for difference in β
        if not np.isnan(ig_beta) and not np.isnan(hy_beta) and not np.isnan(ig_se) and not np.isnan(hy_se):
            diff = hy_beta - ig_beta
            se_diff = np.sqrt(ig_se**2 + hy_se**2)
            t_stat = diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        else:
            diff = np.nan
            se_diff = np.nan
            t_stat = np.nan
            p_value = np.nan

        return {
            'ig_beta': ig_beta,
            'hy_beta': hy_beta,
            'ig_validates_merton': ig_validates,
            'hy_validates_merton': hy_validates,
            'both_validate': ig_validates and hy_validates,
            'difference': diff,
            'difference_se': se_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05 if not np.isnan(p_value) else False,
            'interpretation': self._interpret_ig_hy_comparison(ig_beta, hy_beta, ig_validates, hy_validates)
        }

    def _interpret_ig_hy_comparison(
        self,
        ig_beta: float,
        hy_beta: float,
        ig_validates: bool,
        hy_validates: bool
    ) -> str:
        """Interpret IG vs HY comparison."""
        if np.isnan(ig_beta) or np.isnan(hy_beta):
            return "Insufficient data for comparison"

        if ig_validates and hy_validates:
            return f"Merton validated in both IG (β={ig_beta:.2f}) and HY (β={hy_beta:.2f})"
        elif ig_validates:
            return f"Merton validated in IG (β={ig_beta:.2f}) but NOT in HY (β={hy_beta:.2f})"
        elif hy_validates:
            return f"Merton validated in HY (β={hy_beta:.2f}) but NOT in IG (β={ig_beta:.2f})"
        else:
            return f"Merton NOT validated in either universe (IG β={ig_beta:.2f}, HY β={hy_beta:.2f})"


def run_within_issuer_analysis_both_universes(
    bond_data: pd.DataFrame,
    verbose: bool = False
) -> Dict:
    """
    Convenience function to run within-issuer analysis for both IG and HY.

    Args:
        bond_data: DataFrame with bond observations
        verbose: Print filter summaries

    Returns:
        Dictionary with results for both universes plus comparison
    """
    analyzer = WithinIssuerAnalysis()

    # Run for IG
    ig_results = analyzer.run_within_issuer_analysis(bond_data, universe='IG', verbose=verbose)

    # Run for HY
    hy_results = analyzer.run_within_issuer_analysis(bond_data, universe='HY', verbose=verbose)

    # Compare
    comparison = analyzer.compare_ig_hy(ig_results, hy_results)

    return {
        'IG': ig_results,
        'HY': hy_results,
        'comparison': comparison
    }
