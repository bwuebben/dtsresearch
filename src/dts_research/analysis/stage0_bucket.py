"""
Stage 0: Bucket-Level Analysis

Implements the first component of evolved Stage 0:
- Cross-sectional regression of spreads on maturity using 72 buckets
- Tests for monotonicity of λ across maturity buckets
- Separate analysis for IG and HY universes
- Used as input to decision framework

Based on Specification 0.1 from the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import statsmodels.api as sm
from scipy import stats

from ..data.bucket_definitions import BucketDefinitions, classify_bonds_into_buckets
from ..utils.statistical_tests import (
    clustered_standard_errors,
    test_monotonicity
)
from ..config import MIN_OBSERVATIONS_PER_BUCKET


class BucketLevelAnalysis:
    """
    Bucket-level cross-sectional analysis for Stage 0.

    For each universe (IG/HY):
    1. Create 72 buckets (6 rating × 6 maturity × 2 sector groups)
    2. Compute representative characteristics (s̄, T̄) per bucket
    3. Run cross-sectional regression: ln(s̄) = α + λ·T̄ + ε
    4. Test monotonicity of λ across maturity buckets
    """

    def __init__(self):
        """Initialize bucket-level analysis."""
        self.bucket_classifier = BucketDefinitions()
        self.min_observations = MIN_OBSERVATIONS_PER_BUCKET

    def run_bucket_analysis(
        self,
        bond_data: pd.DataFrame,
        universe: str = 'IG'
    ) -> Dict:
        """
        Run complete bucket-level analysis for a universe.

        Args:
            bond_data: DataFrame with bond observations
            universe: 'IG' or 'HY'

        Returns:
            Dictionary with results:
            - bucket_characteristics: DataFrame with (s̄, T̄) per bucket
            - regression_results: Dict with λ estimate, SE, p-value
            - monotonicity_test: Dict with Spearman correlation results
            - diagnostics: Quality checks
        """
        # Step 1: Filter to universe
        if universe == 'IG':
            bond_data = bond_data[bond_data['rating'].isin(['AAA', 'AA', 'A', 'BBB'])].copy()
        elif universe == 'HY':
            bond_data = bond_data[bond_data['rating'].isin(['BB', 'B', 'CCC'])].copy()
        else:
            raise ValueError("universe must be 'IG' or 'HY'")

        if len(bond_data) == 0:
            return self._empty_results(universe, "No bonds in universe")

        # Step 2: Classify bonds into buckets
        bond_data, bucket_chars = classify_bonds_into_buckets(
            bond_data,
            rating_column='rating',
            maturity_column='time_to_maturity',
            sector_column='sector',
            compute_characteristics=True
        )

        if bucket_chars is None or len(bucket_chars) == 0:
            return self._empty_results(universe, "No valid buckets created")

        # Step 3: Run cross-sectional regression
        regression_results = self._run_cross_sectional_regression(bucket_chars)

        # Step 4: Test monotonicity across maturity buckets
        monotonicity_test = self._test_maturity_monotonicity(bucket_chars)

        # Step 5: Diagnostics
        diagnostics = self._compute_diagnostics(bucket_chars, bond_data)

        return {
            'universe': universe,
            'bucket_characteristics': bucket_chars,
            'regression_results': regression_results,
            'monotonicity_test': monotonicity_test,
            'diagnostics': diagnostics
        }

    def _run_cross_sectional_regression(
        self,
        bucket_chars: pd.DataFrame
    ) -> Dict:
        """
        Run cross-sectional regression: ln(s̄) = α + λ·T̄ + ε

        Args:
            bucket_chars: DataFrame with bucket characteristics (s_bar, T_bar)

        Returns:
            Dictionary with regression results
        """
        # Prepare data
        df = bucket_chars[['s_bar', 'T_bar', 'n_observations']].dropna()

        if len(df) < 3:
            return {
                'lambda': np.nan,
                'lambda_se': np.nan,
                'lambda_tstat': np.nan,
                'lambda_pvalue': np.nan,
                'alpha': np.nan,
                'r_squared': np.nan,
                'n_buckets': len(df),
                'warning': 'Insufficient buckets for regression'
            }

        # Dependent variable: ln(spread)
        y = np.log(df['s_bar'].values)

        # Independent variable: maturity
        X = df['T_bar'].values
        X_const = sm.add_constant(X)

        # Weights: number of observations per bucket (for precision weighting)
        weights = df['n_observations'].values

        # WLS regression
        try:
            model = sm.WLS(y, X_const, weights=weights).fit()

            return {
                'lambda': model.params[1],  # Coefficient on maturity
                'lambda_se': model.bse[1],
                'lambda_tstat': model.tvalues[1],
                'lambda_pvalue': model.pvalues[1],
                'alpha': model.params[0],  # Intercept
                'alpha_se': model.bse[0],
                'r_squared': model.rsquared,
                'r_squared_adj': model.rsquared_adj,
                'n_buckets': len(df),
                'total_observations': int(df['n_observations'].sum())
            }

        except Exception as e:
            return {
                'lambda': np.nan,
                'lambda_se': np.nan,
                'lambda_tstat': np.nan,
                'lambda_pvalue': np.nan,
                'alpha': np.nan,
                'r_squared': np.nan,
                'n_buckets': len(df),
                'error': str(e)
            }

    def _test_maturity_monotonicity(
        self,
        bucket_chars: pd.DataFrame
    ) -> Dict:
        """
        Test monotonicity of λ across maturity buckets.

        For each rating-sector group, compute λ for each maturity bucket
        and test if λ increases monotonically with maturity.

        Args:
            bucket_chars: DataFrame with bucket characteristics

        Returns:
            Dictionary with monotonicity test results
        """
        # Group by rating and sector
        results = []

        for (rating, sector_group), group in bucket_chars.groupby(['rating_bucket', 'sector_group']):
            if len(group) < 3:
                continue

            # Sort by maturity
            group = group.sort_values('T_bar')

            # Compute implied λ for each maturity bucket
            # λ_implied = ln(s) / T  (rough approximation)
            group['lambda_implied'] = np.log(group['s_bar']) / group['T_bar']

            # Test monotonicity using Spearman correlation
            monotonicity = test_monotonicity(
                group['T_bar'].values,
                group['lambda_implied'].values,
                method='spearman'
            )

            results.append({
                'rating_bucket': rating,
                'sector_group': sector_group,
                'n_buckets': len(group),
                'spearman_rho': monotonicity['correlation'],
                'p_value': monotonicity['p_value'],
                'is_monotonic': monotonicity['p_value'] < 0.10 and monotonicity['correlation'] > 0
            })

        if len(results) == 0:
            return {
                'overall_monotonic': False,
                'pct_monotonic_groups': 0.0,
                'details': []
            }

        results_df = pd.DataFrame(results)

        return {
            'overall_monotonic': results_df['is_monotonic'].mean() > 0.7,  # 70% threshold
            'pct_monotonic_groups': 100.0 * results_df['is_monotonic'].mean(),
            'n_groups_tested': len(results_df),
            'details': results_df.to_dict('records')
        }

    def _compute_diagnostics(
        self,
        bucket_chars: pd.DataFrame,
        bond_data: pd.DataFrame
    ) -> Dict:
        """
        Compute diagnostic statistics for bucket analysis.

        Args:
            bucket_chars: DataFrame with bucket characteristics
            bond_data: Original bond data

        Returns:
            Dictionary with diagnostics
        """
        return {
            'n_buckets_populated': len(bucket_chars),
            'n_buckets_expected': 72,  # 6 rating × 6 maturity × 2 sectors
            'pct_coverage': 100.0 * len(bucket_chars) / 72,
            'total_bonds': len(bond_data),
            'total_observations': int(bucket_chars['n_observations'].sum()),
            'mean_obs_per_bucket': bucket_chars['n_observations'].mean(),
            'median_obs_per_bucket': bucket_chars['n_observations'].median(),
            'min_obs_per_bucket': int(bucket_chars['n_observations'].min()),
            'max_obs_per_bucket': int(bucket_chars['n_observations'].max()),
            'spread_range': (bucket_chars['s_bar'].min(), bucket_chars['s_bar'].max()),
            'maturity_range': (bucket_chars['T_bar'].min(), bucket_chars['T_bar'].max())
        }

    def _empty_results(self, universe: str, reason: str) -> Dict:
        """Return empty results structure with warning."""
        return {
            'universe': universe,
            'bucket_characteristics': pd.DataFrame(),
            'regression_results': {
                'lambda': np.nan,
                'lambda_se': np.nan,
                'lambda_pvalue': np.nan,
                'warning': reason
            },
            'monotonicity_test': {
                'overall_monotonic': False,
                'pct_monotonic_groups': 0.0
            },
            'diagnostics': {
                'n_buckets_populated': 0,
                'warning': reason
            }
        }

    def compare_ig_hy(
        self,
        ig_results: Dict,
        hy_results: Dict
    ) -> Dict:
        """
        Compare bucket-level results between IG and HY.

        Args:
            ig_results: Results from run_bucket_analysis(universe='IG')
            hy_results: Results from run_bucket_analysis(universe='HY')

        Returns:
            Dictionary with comparison statistics
        """
        ig_lambda = ig_results['regression_results'].get('lambda', np.nan)
        hy_lambda = hy_results['regression_results'].get('lambda', np.nan)

        ig_se = ig_results['regression_results'].get('lambda_se', np.nan)
        hy_se = hy_results['regression_results'].get('lambda_se', np.nan)

        # Test for difference (two-sample t-test approximation)
        if not np.isnan(ig_lambda) and not np.isnan(hy_lambda) and not np.isnan(ig_se) and not np.isnan(hy_se):
            diff = hy_lambda - ig_lambda
            se_diff = np.sqrt(ig_se**2 + hy_se**2)
            t_stat = diff / se_diff
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=100))  # Conservative df
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
            return f"HY has significantly higher λ (p={p_value:.4f}) - stronger maturity effect"
        else:
            return f"IG has significantly higher λ (p={p_value:.4f}) - stronger maturity effect"


def run_bucket_analysis_both_universes(bond_data: pd.DataFrame) -> Dict:
    """
    Convenience function to run bucket analysis for both IG and HY.

    Args:
        bond_data: DataFrame with bond observations

    Returns:
        Dictionary with results for both universes plus comparison
    """
    analyzer = BucketLevelAnalysis()

    # Run for IG
    ig_results = analyzer.run_bucket_analysis(bond_data, universe='IG')

    # Run for HY
    hy_results = analyzer.run_bucket_analysis(bond_data, universe='HY')

    # Compare
    comparison = analyzer.compare_ig_hy(ig_results, hy_results)

    return {
        'IG': ig_results,
        'HY': hy_results,
        'comparison': comparison
    }
