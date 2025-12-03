"""
Stage 0: Within-Issuer Analysis

Implements the second component of evolved Stage 0:
- Uses same issuer, different maturities to isolate maturity effects
- Issuer-week fixed effects control for common credit shocks
- Inverse-variance weighted pooling for meta-analysis
- Tests for consistency with Merton model predictions

Based on Specification 0.2 from the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import statsmodels.api as sm
from scipy import stats

from ..data.issuer_identification import add_issuer_identification
from ..data.filters import apply_within_issuer_filters
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
    1. Run regression: ln(s_i) = α_issuer-week + λ·T_i + ε_i
    2. Extract issuer-week specific λ estimates
    3. Pool estimates using inverse-variance weighting
    4. Test for positive λ (Merton prediction)
    """

    def __init__(self):
        """Initialize within-issuer analysis."""
        self.min_bonds = MIN_BONDS_PER_ISSUER_WEEK
        self.min_dispersion = MIN_MATURITY_DISPERSION_YEARS
        self.pull_to_par = PULL_TO_PAR_EXCLUSION_YEARS

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
            - issuer_week_estimates: DataFrame with λ per issuer-week
            - pooled_estimate: Inverse-variance weighted pooled λ
            - hypothesis_test: Test for λ > 0 (Merton prediction)
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

        # Step 3: Apply within-issuer filters
        bond_data, filter_stats = apply_within_issuer_filters(
            bond_data,
            maturity_column='time_to_maturity',
            spread_change_column=None,  # Not using spread changes here
            verbose=verbose
        )

        if len(bond_data) == 0:
            return self._empty_results(universe, "No data after filtering")

        # Step 4: Run issuer-week regressions
        issuer_week_estimates = self._run_issuer_week_regressions(bond_data)

        if len(issuer_week_estimates) == 0:
            return self._empty_results(universe, "No issuer-week estimates obtained")

        # Step 5: Pool estimates using inverse-variance weighting
        pooled = self._pool_estimates(issuer_week_estimates)

        # Step 6: Hypothesis test for λ > 0
        hypothesis_test = self._test_merton_prediction(pooled)

        # Step 7: Diagnostics
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

    def _run_issuer_week_regressions(
        self,
        bond_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run within-issuer-week regressions.

        For each issuer-week: ln(s_i) = α + λ·T_i + ε_i

        Args:
            bond_data: Filtered DataFrame

        Returns:
            DataFrame with columns:
            - issuer_id, date, lambda, lambda_se, lambda_tstat, lambda_pvalue,
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

            # Prepare data
            y = np.log(group['oas'].values)
            X = group['time_to_maturity'].values
            X_const = sm.add_constant(X)

            # Run OLS
            try:
                model = sm.OLS(y, X_const).fit()

                results.append({
                    'issuer_id': issuer_id,
                    'date': date,
                    'lambda': model.params[1],
                    'lambda_se': model.bse[1],
                    'lambda_tstat': model.tvalues[1],
                    'lambda_pvalue': model.pvalues[1],
                    'alpha': model.params[0],
                    'n_bonds': len(group),
                    'r_squared': model.rsquared,
                    'maturity_range': maturity_range,
                    'spread_range': (group['oas'].min(), group['oas'].max())
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
            issuer_week_estimates: DataFrame with λ estimates

        Returns:
            Dictionary with pooled estimate and inference
        """
        estimates = issuer_week_estimates['lambda'].values
        variances = issuer_week_estimates['lambda_se'].values ** 2

        pooled = inverse_variance_weighted_pooling(estimates, variances)

        return pooled

    def _test_merton_prediction(self, pooled: Dict) -> Dict:
        """
        Test Merton prediction: H0: λ ≤ 0 vs H1: λ > 0

        Args:
            pooled: Pooled estimate dictionary

        Returns:
            Dictionary with hypothesis test results
        """
        lambda_pooled = pooled.get('pooled_estimate', np.nan)
        lambda_se = pooled.get('pooled_se', np.nan)

        if np.isnan(lambda_pooled) or np.isnan(lambda_se):
            return {
                'test': 'λ > 0 (Merton prediction)',
                't_statistic': np.nan,
                'p_value': np.nan,
                'reject_null': False,
                'interpretation': 'Insufficient data'
            }

        # One-sided t-test
        t_stat = lambda_pooled / lambda_se
        p_value = 1 - stats.norm.cdf(t_stat)  # Right-tail test

        return {
            'test': 'H0: λ ≤ 0 vs H1: λ > 0',
            't_statistic': t_stat,
            'p_value': p_value,
            'reject_null': p_value < 0.05,
            'interpretation': self._interpret_merton_test(lambda_pooled, p_value)
        }

    def _interpret_merton_test(self, lambda_pooled: float, p_value: float) -> str:
        """Interpret Merton hypothesis test."""
        if np.isnan(lambda_pooled) or np.isnan(p_value):
            return "Insufficient data for test"

        if p_value < 0.05:
            return f"λ = {lambda_pooled:.4f} is significantly positive (p={p_value:.4f}) - consistent with Merton"
        else:
            return f"λ = {lambda_pooled:.4f} not significantly positive (p={p_value:.4f}) - inconsistent with Merton"

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
        return {
            'n_bonds_after_filter': len(bond_data),
            'n_unique_issuers': bond_data['issuer_id'].nunique(),
            'n_unique_weeks': bond_data['date'].nunique(),
            'n_issuer_weeks_total': bond_data.groupby(['issuer_id', 'date']).ngroups,
            'n_issuer_weeks_with_estimate': len(issuer_week_estimates),
            'pct_issuer_weeks_with_estimate': 100.0 * len(issuer_week_estimates) / bond_data.groupby(['issuer_id', 'date']).ngroups if bond_data.groupby(['issuer_id', 'date']).ngroups > 0 else 0,
            'mean_bonds_per_issuer_week': issuer_week_estimates['n_bonds'].mean() if len(issuer_week_estimates) > 0 else np.nan,
            'mean_maturity_range': issuer_week_estimates['maturity_range'].mean() if len(issuer_week_estimates) > 0 else np.nan,
            'pct_significant_positive': 100.0 * ((issuer_week_estimates['lambda'] > 0) & (issuer_week_estimates['lambda_pvalue'] < 0.05)).mean() if len(issuer_week_estimates) > 0 else 0,
            'pct_significant_negative': 100.0 * ((issuer_week_estimates['lambda'] < 0) & (issuer_week_estimates['lambda_pvalue'] < 0.05)).mean() if len(issuer_week_estimates) > 0 else 0,
            'lambda_range': (issuer_week_estimates['lambda'].min(), issuer_week_estimates['lambda'].max()) if len(issuer_week_estimates) > 0 else (np.nan, np.nan)
        }

    def _empty_results(self, universe: str, reason: str) -> Dict:
        """Return empty results structure with warning."""
        return {
            'universe': universe,
            'issuer_week_estimates': pd.DataFrame(),
            'pooled_estimate': {
                'pooled_estimate': np.nan,
                'pooled_se': np.nan,
                'n_estimates': 0,
                'warning': reason
            },
            'hypothesis_test': {
                'reject_null': False,
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
        ig_lambda = ig_results['pooled_estimate'].get('pooled_estimate', np.nan)
        hy_lambda = hy_results['pooled_estimate'].get('pooled_estimate', np.nan)

        ig_se = ig_results['pooled_estimate'].get('pooled_se', np.nan)
        hy_se = hy_results['pooled_estimate'].get('pooled_se', np.nan)

        # Test for difference
        if not np.isnan(ig_lambda) and not np.isnan(hy_lambda) and not np.isnan(ig_se) and not np.isnan(hy_se):
            diff = hy_lambda - ig_lambda
            se_diff = np.sqrt(ig_se**2 + hy_se**2)
            t_stat = diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        else:
            diff = np.nan
            se_diff = np.nan
            t_stat = np.nan
            p_value = np.nan

        return {
            'ig_lambda': ig_lambda,
            'hy_lambda': hy_lambda,
            'difference': diff,
            'difference_se': se_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05 if not np.isnan(p_value) else False,
            'interpretation': self._interpret_ig_hy_comparison(ig_lambda, hy_lambda, p_value)
        }

    def _interpret_ig_hy_comparison(
        self,
        ig_lambda: float,
        hy_lambda: float,
        p_value: float
    ) -> str:
        """Interpret IG vs HY comparison."""
        if np.isnan(ig_lambda) or np.isnan(hy_lambda) or np.isnan(p_value):
            return "Insufficient data for comparison"

        if p_value >= 0.05:
            return "No significant difference between IG and HY λ"

        if hy_lambda > ig_lambda:
            return f"HY has significantly higher λ (p={p_value:.4f})"
        else:
            return f"IG has significantly higher λ (p={p_value:.4f})"


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
