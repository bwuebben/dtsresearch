"""
Stage 0: Bucket-Level Analysis

Implements the first component of evolved Stage 0:
- Time-series regression of spread CHANGES on index-level DTS factor
- Separate regression for each of 72 buckets
- Compare empirical β to theoretical λ^Merton for each bucket
- Tests for monotonicity of β across maturity buckets

Based on Specification 0.1 / Equation 4.1 from the paper:
    y_{i,t} = α^(k) + β^(k) · f_{DTS,t}^(U) + ε_{i,t}^(k)

where y_{i,t} is the percentage spread change for bond i in week t,
and f_{DTS,t}^(U) is the index-level percentage spread change.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import statsmodels.api as sm
from scipy import stats

from ..data.bucket_definitions import BucketDefinitions, classify_bonds_into_buckets
from ..data.preprocessing import compute_spread_changes, compute_index_dts_factor
from ..models.merton import MertonLambdaCalculator, calculate_merton_lambda
from ..utils.statistical_tests import (
    clustered_standard_errors,
    test_monotonicity
)
from ..config import MIN_OBSERVATIONS_PER_BUCKET


class BucketLevelAnalysis:
    """
    Bucket-level time-series analysis for Stage 0.

    For each universe (IG/HY):
    1. Create 72 buckets (6 rating × 6 maturity × 2 sector groups)
    2. Compute percentage spread changes for each bond-week
    3. Compute index-level DTS factor (universe percentage spread change)
    4. Run time-series regression per bucket: y_{i,t} = α + β · f_{DTS,t} + ε
    5. Compare empirical β^(k) to theoretical λ^Merton for each bucket
    6. Test monotonicity of β across maturity buckets
    """

    def __init__(self):
        """Initialize bucket-level analysis."""
        self.bucket_classifier = BucketDefinitions()
        self.min_observations = MIN_OBSERVATIONS_PER_BUCKET
        self.merton_calc = MertonLambdaCalculator()

    def run_bucket_analysis(
        self,
        bond_data: pd.DataFrame,
        universe: str = 'IG'
    ) -> Dict:
        """
        Run complete bucket-level analysis for a universe.

        Args:
            bond_data: DataFrame with bond observations (must include 'date' for time-series)
            universe: 'IG' or 'HY'

        Returns:
            Dictionary with results:
            - bucket_results: DataFrame with β^(k), λ^Merton, ratio per bucket
            - summary_statistics: Overall statistics on β/λ ratios
            - monotonicity_test: Dict with test for β decreasing with maturity
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

        # Step 2: Compute spread changes and index-level DTS factor
        bond_data = self._compute_spread_changes(bond_data)
        bond_data, index_factor = self._compute_index_dts_factor(bond_data)

        if bond_data is None or len(bond_data) == 0:
            return self._empty_results(universe, "Insufficient data for spread changes")

        # Step 3: Classify bonds into buckets
        bond_data, bucket_chars = classify_bonds_into_buckets(
            bond_data,
            rating_column='rating',
            maturity_column='time_to_maturity',
            sector_column='sector',
            compute_characteristics=True
        )

        if bucket_chars is None or len(bucket_chars) == 0:
            return self._empty_results(universe, "No valid buckets created")

        # Step 4: Run time-series regression for each bucket
        bucket_results = self._run_bucket_regressions(bond_data, index_factor, bucket_chars)

        if len(bucket_results) == 0:
            return self._empty_results(universe, "No bucket regressions succeeded")

        # Step 5: Compare empirical β to theoretical λ^Merton
        bucket_results = self._compare_beta_to_merton(bucket_results)

        # Step 6: Summary statistics on β/λ ratios
        summary_stats = self._compute_summary_statistics(bucket_results)

        # Step 7: Test monotonicity of β across maturity buckets
        monotonicity_test = self._test_maturity_monotonicity(bucket_results)

        # Step 8: Diagnostics
        diagnostics = self._compute_diagnostics(bucket_results, bond_data)

        return {
            'universe': universe,
            'bucket_results': bucket_results,
            'summary_statistics': summary_stats,
            'monotonicity_test': monotonicity_test,
            'diagnostics': diagnostics
        }

    def _compute_spread_changes(self, bond_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percentage spread changes for each bond.

        Uses centralized preprocessing from dts_research.data.preprocessing.

        Args:
            bond_data: DataFrame with 'oas', 'cusip' (or bond id), 'date'

        Returns:
            DataFrame with 'spread_change' column (Δs/s_{t-1})
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

    def _compute_index_dts_factor(
        self,
        bond_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute index-level DTS factor: weighted average percentage spread change.

        Uses centralized preprocessing from dts_research.data.preprocessing.

        Args:
            bond_data: DataFrame with spread changes

        Returns:
            Tuple of (bond_data with f_dts column, index_factor DataFrame)
        """
        # Use centralized index factor calculation
        bond_data, index_factor_full = compute_index_dts_factor(
            bond_data,
            output_col='oas_index_pct_change'
        )

        # Add legacy column name for backward compatibility
        bond_data['f_dts'] = bond_data['oas_index_pct_change']

        # Create index_factor DataFrame with legacy format
        index_factor = bond_data.groupby('date').agg(
            f_dts=('spread_change', 'mean'),
            n_bonds=('spread_change', 'count')
        ).reset_index()

        return bond_data, index_factor

    def _run_bucket_regressions(
        self,
        bond_data: pd.DataFrame,
        index_factor: pd.DataFrame,
        bucket_chars: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run time-series regression for each bucket:
            y_{i,t} = α^(k) + β^(k) · f_{DTS,t} + ε_{i,t}

        Args:
            bond_data: DataFrame with spread changes and bucket assignments
            index_factor: DataFrame with index-level DTS factor
            bucket_chars: DataFrame with bucket characteristics

        Returns:
            DataFrame with regression results per bucket
        """
        results = []

        for bucket_id in bond_data['bucket_id'].unique():
            bucket_data = bond_data[bond_data['bucket_id'] == bucket_id]

            if len(bucket_data) < self.min_observations:
                continue

            # Get bucket characteristics for Merton comparison
            bucket_info = bucket_chars[bucket_chars['bucket_id'] == bucket_id]
            if len(bucket_info) == 0:
                continue

            median_spread = bucket_info['s_bar'].values[0]
            median_maturity = bucket_info['T_bar'].values[0]

            # Dependent variable: percentage spread change
            y = bucket_data['spread_change'].values

            # Independent variable: index-level DTS factor
            X = bucket_data['f_dts'].values
            X_const = sm.add_constant(X)

            try:
                # OLS with clustered standard errors by week
                model = sm.OLS(y, X_const).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': bucket_data['date'].values}
                )

                results.append({
                    'bucket_id': bucket_id,
                    'rating_bucket': bucket_info['rating_bucket'].values[0] if 'rating_bucket' in bucket_info.columns else None,
                    'maturity_bucket': bucket_info['maturity_bucket'].values[0] if 'maturity_bucket' in bucket_info.columns else None,
                    'sector_group': bucket_info['sector_group'].values[0] if 'sector_group' in bucket_info.columns else None,
                    'median_spread': median_spread,
                    'median_maturity': median_maturity,
                    'beta': model.params[1],
                    'beta_se': model.bse[1],
                    'beta_tstat': model.tvalues[1],
                    'beta_pvalue': model.pvalues[1],
                    'alpha': model.params[0],
                    'r_squared': model.rsquared,
                    'n_obs': len(bucket_data),
                    'n_weeks': bucket_data['date'].nunique()
                })

            except Exception as e:
                # Skip buckets where regression fails
                continue

        return pd.DataFrame(results)

    def _compare_beta_to_merton(self, bucket_results: pd.DataFrame) -> pd.DataFrame:
        """
        Compare empirical β^(k) to theoretical λ^Merton for each bucket.

        Args:
            bucket_results: DataFrame with regression results

        Returns:
            DataFrame with lambda_merton and beta_lambda_ratio columns
        """
        if len(bucket_results) == 0:
            return bucket_results

        # Compute Merton-predicted lambda for each bucket
        bucket_results['lambda_merton'] = calculate_merton_lambda(
            bucket_results['median_maturity'].values,
            bucket_results['median_spread'].values  # Already in bps typically
        )

        # Compute ratio of empirical beta to theoretical lambda
        bucket_results['beta_lambda_ratio'] = (
            bucket_results['beta'] / bucket_results['lambda_merton']
        )

        # Test if ratio is significantly different from 1
        bucket_results['ratio_diff_from_1'] = bucket_results['beta_lambda_ratio'] - 1.0

        return bucket_results

    def _compute_summary_statistics(self, bucket_results: pd.DataFrame) -> Dict:
        """
        Compute summary statistics on β/λ ratios across buckets.

        Args:
            bucket_results: DataFrame with bucket regression results

        Returns:
            Dictionary with summary statistics
        """
        if len(bucket_results) == 0:
            return {
                'median_beta_lambda_ratio': np.nan,
                'mean_beta_lambda_ratio': np.nan,
                'pct_within_10pct': 0,
                'pct_within_20pct': 0
            }

        ratios = bucket_results['beta_lambda_ratio'].dropna()

        return {
            'median_beta_lambda_ratio': ratios.median(),
            'mean_beta_lambda_ratio': ratios.mean(),
            'std_beta_lambda_ratio': ratios.std(),
            'pct_within_10pct': 100.0 * ((ratios >= 0.9) & (ratios <= 1.1)).mean(),
            'pct_within_20pct': 100.0 * ((ratios >= 0.8) & (ratios <= 1.2)).mean(),
            'n_buckets': len(ratios),
            'median_beta': bucket_results['beta'].median(),
            'median_lambda_merton': bucket_results['lambda_merton'].median()
        }

    def _test_maturity_monotonicity(
        self,
        bucket_results: pd.DataFrame
    ) -> Dict:
        """
        Test monotonicity of β across maturity buckets.

        Merton predicts that short-maturity buckets have HIGHER β (more sensitive)
        than long-maturity buckets. So β should DECREASE with maturity.

        Args:
            bucket_results: DataFrame with bucket regression results

        Returns:
            Dictionary with monotonicity test results
        """
        if len(bucket_results) == 0 or 'rating_bucket' not in bucket_results.columns:
            return {
                'overall_monotonic': False,
                'pct_monotonic_groups': 0.0,
                'details': []
            }

        # Group by rating and sector
        results = []

        group_cols = ['rating_bucket']
        if 'sector_group' in bucket_results.columns:
            group_cols.append('sector_group')

        for group_key, group in bucket_results.groupby(group_cols):
            if len(group) < 3:
                continue

            # Sort by maturity
            group = group.sort_values('median_maturity')

            # Test monotonicity: β should DECREASE with maturity (negative correlation)
            monotonicity = test_monotonicity(
                group['median_maturity'].values,
                group['beta'].values,
                method='spearman'
            )

            is_monotonic_decreasing = (
                monotonicity['p_value'] < 0.10 and
                monotonicity['correlation'] < 0  # Negative = decreasing
            )

            result_dict = {
                'n_buckets': len(group),
                'spearman_rho': monotonicity['correlation'],
                'p_value': monotonicity['p_value'],
                'is_monotonic_decreasing': is_monotonic_decreasing
            }

            # Add group identifiers
            if isinstance(group_key, tuple):
                result_dict['rating_bucket'] = group_key[0]
                if len(group_key) > 1:
                    result_dict['sector_group'] = group_key[1]
            else:
                result_dict['rating_bucket'] = group_key

            results.append(result_dict)

        if len(results) == 0:
            return {
                'overall_monotonic': False,
                'pct_monotonic_groups': 0.0,
                'details': []
            }

        results_df = pd.DataFrame(results)

        return {
            'overall_monotonic': results_df['is_monotonic_decreasing'].mean() > 0.7,
            'pct_monotonic_groups': 100.0 * results_df['is_monotonic_decreasing'].mean(),
            'n_groups_tested': len(results_df),
            'details': results_df.to_dict('records'),
            'interpretation': (
                "β decreases with maturity as Merton predicts"
                if results_df['is_monotonic_decreasing'].mean() > 0.7
                else "β does NOT consistently decrease with maturity"
            )
        }

    def _compute_diagnostics(
        self,
        bucket_results: pd.DataFrame,
        bond_data: pd.DataFrame
    ) -> Dict:
        """
        Compute diagnostic statistics for bucket analysis.

        Args:
            bucket_results: DataFrame with bucket regression results
            bond_data: Bond data with spread changes

        Returns:
            Dictionary with diagnostics
        """
        return {
            'n_buckets_with_regression': len(bucket_results),
            'n_buckets_expected': 72,  # 6 rating × 6 maturity × 2 sectors
            'pct_coverage': 100.0 * len(bucket_results) / 72,
            'total_bond_weeks': len(bond_data),
            'n_unique_bonds': bond_data['cusip'].nunique() if 'cusip' in bond_data.columns else bond_data['bond_id'].nunique() if 'bond_id' in bond_data.columns else np.nan,
            'n_unique_weeks': bond_data['date'].nunique(),
            'total_observations': int(bucket_results['n_obs'].sum()) if 'n_obs' in bucket_results.columns else 0,
            'mean_obs_per_bucket': bucket_results['n_obs'].mean() if 'n_obs' in bucket_results.columns else np.nan,
            'median_obs_per_bucket': bucket_results['n_obs'].median() if 'n_obs' in bucket_results.columns else np.nan,
            'min_obs_per_bucket': int(bucket_results['n_obs'].min()) if 'n_obs' in bucket_results.columns else 0,
            'mean_r_squared': bucket_results['r_squared'].mean() if 'r_squared' in bucket_results.columns else np.nan,
            'spread_range': (bucket_results['median_spread'].min(), bucket_results['median_spread'].max()) if 'median_spread' in bucket_results.columns else (np.nan, np.nan),
            'maturity_range': (bucket_results['median_maturity'].min(), bucket_results['median_maturity'].max()) if 'median_maturity' in bucket_results.columns else (np.nan, np.nan)
        }

    def _empty_results(self, universe: str, reason: str) -> Dict:
        """Return empty results structure with warning."""
        return {
            'universe': universe,
            'bucket_results': pd.DataFrame(),
            'summary_statistics': {
                'median_beta_lambda_ratio': np.nan,
                'mean_beta_lambda_ratio': np.nan,
                'pct_within_10pct': 0,
                'pct_within_20pct': 0,
                'warning': reason
            },
            'monotonicity_test': {
                'overall_monotonic': False,
                'pct_monotonic_groups': 0.0
            },
            'diagnostics': {
                'n_buckets_with_regression': 0,
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
        ig_ratio = ig_results['summary_statistics'].get('median_beta_lambda_ratio', np.nan)
        hy_ratio = hy_results['summary_statistics'].get('median_beta_lambda_ratio', np.nan)

        ig_pct_fit = ig_results['summary_statistics'].get('pct_within_20pct', 0)
        hy_pct_fit = hy_results['summary_statistics'].get('pct_within_20pct', 0)

        ig_monotonic = ig_results['monotonicity_test'].get('overall_monotonic', False)
        hy_monotonic = hy_results['monotonicity_test'].get('overall_monotonic', False)

        return {
            'ig_median_beta_lambda_ratio': ig_ratio,
            'hy_median_beta_lambda_ratio': hy_ratio,
            'ig_pct_within_20pct': ig_pct_fit,
            'hy_pct_within_20pct': hy_pct_fit,
            'ig_monotonic': ig_monotonic,
            'hy_monotonic': hy_monotonic,
            'both_ratios_near_1': (
                not np.isnan(ig_ratio) and not np.isnan(hy_ratio) and
                0.8 <= ig_ratio <= 1.2 and 0.8 <= hy_ratio <= 1.2
            ),
            'interpretation': self._interpret_ig_hy_comparison(ig_ratio, hy_ratio, ig_pct_fit, hy_pct_fit)
        }

    def _interpret_ig_hy_comparison(
        self,
        ig_ratio: float,
        hy_ratio: float,
        ig_pct: float,
        hy_pct: float
    ) -> str:
        """Interpret IG vs HY comparison."""
        if np.isnan(ig_ratio) or np.isnan(hy_ratio):
            return "Insufficient data for comparison"

        ig_good = 0.8 <= ig_ratio <= 1.2 and ig_pct >= 60
        hy_good = 0.8 <= hy_ratio <= 1.2 and hy_pct >= 60

        if ig_good and hy_good:
            return "Merton predictions hold well in both IG and HY"
        elif ig_good:
            return f"Merton works for IG (ratio={ig_ratio:.2f}) but not HY (ratio={hy_ratio:.2f})"
        elif hy_good:
            return f"Merton works for HY (ratio={hy_ratio:.2f}) but not IG (ratio={ig_ratio:.2f})"
        else:
            return f"Merton predictions do not hold well in either universe (IG={ig_ratio:.2f}, HY={hy_ratio:.2f})"


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
