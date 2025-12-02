"""
Stage 0: Raw Validation Using Bucket-Level Analysis

Implements pooled regression analysis to test Merton predictions
before any complex modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from scipy import stats


class Stage0Analysis:
    """
    Conducts Stage 0 bucket-level validation of Merton predictions.

    Approach:
    1. Define buckets by rating, maturity, sector
    2. Run pooled regression within each bucket: y = alpha + beta * f_DTS + error
    3. Compare empirical beta to theoretical lambda from Merton
    """

    def __init__(self):
        self.bucket_results = None

    def prepare_regression_data(
        self,
        bond_data: pd.DataFrame,
        index_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare data for regression analysis.

        Computes percentage spread changes for bonds and index.

        Args:
            bond_data: DataFrame with bond-level data including bucket_id
            index_data: DataFrame with index-level OAS data

        Returns:
            DataFrame with regression-ready data
        """
        # Sort by bond and date
        bond_data = bond_data.sort_values(['bond_id', 'date']).copy()

        # Compute percentage spread changes for bonds
        bond_data['oas_pct_change'] = bond_data.groupby('bond_id')['oas'].pct_change()

        # Merge with index data
        df = bond_data.merge(
            index_data[['date', 'oas']],
            on='date',
            how='inner',
            suffixes=('', '_index')
        )

        # Compute percentage spread changes for index
        df = df.sort_values('date')
        df['oas_index_pct_change'] = df['oas_index'].pct_change()

        # Drop first week (NaN values)
        df = df.dropna(subset=['oas_pct_change', 'oas_index_pct_change'])

        return df

    def run_bucket_regression(
        self,
        df: pd.DataFrame,
        bucket_id: str
    ) -> Dict:
        """
        Run pooled OLS regression for a single bucket.

        Regression: y_i,t = alpha + beta * f_DTS,t + epsilon_i,t

        Args:
            df: Regression-ready dataframe
            bucket_id: Bucket identifier

        Returns:
            Dictionary with regression results
        """
        bucket_data = df[df['bucket_id'] == bucket_id].copy()

        if len(bucket_data) < 30:  # Minimum sample size
            return None

        # Prepare regression variables
        y = bucket_data['oas_pct_change'].values
        X = bucket_data['oas_index_pct_change'].values
        X = sm.add_constant(X)

        # Get date for clustering
        dates = bucket_data['date'].values

        try:
            # Run OLS with clustered standard errors
            model = sm.OLS(y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': dates})

            beta = results.params[1]
            se_beta = results.bse[1]
            t_stat = results.tvalues[1]
            p_value = results.pvalues[1]
            r_squared = results.rsquared
            n_obs = len(y)

            return {
                'bucket_id': bucket_id,
                'beta': beta,
                'se_beta': se_beta,
                't_stat': t_stat,
                'p_value': p_value,
                'alpha': results.params[0],
                'r_squared': r_squared,
                'n_observations': n_obs,
                'n_weeks': len(np.unique(dates))
            }

        except Exception as e:
            print(f"Regression failed for bucket {bucket_id}: {str(e)}")
            return None

    def run_all_bucket_regressions(
        self,
        df: pd.DataFrame,
        bucket_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run regressions for all buckets.

        Args:
            df: Regression-ready dataframe
            bucket_stats: Bucket statistics with Merton lambdas

        Returns:
            DataFrame with regression results for all buckets
        """
        results = []

        for bucket_id in df['bucket_id'].unique():
            result = self.run_bucket_regression(df, bucket_id)
            if result is not None:
                results.append(result)

        results_df = pd.DataFrame(results)

        # Merge with bucket stats to get Merton lambda
        results_df = results_df.merge(
            bucket_stats[[
                'bucket_id', 'rating_bucket', 'maturity_bucket', 'sector',
                'lambda_merton', 'lambda_T', 'lambda_s',
                'median_maturity', 'median_spread', 'is_ig'
            ]],
            on='bucket_id',
            how='left'
        )

        # Calculate ratio and deviation
        results_df['beta_lambda_ratio'] = results_df['beta'] / results_df['lambda_merton']
        results_df['deviation'] = results_df['beta'] - results_df['lambda_merton']

        self.bucket_results = results_df
        return results_df

    def test_level_hypothesis(
        self,
        results_df: pd.DataFrame,
        bucket_id: str
    ) -> Dict:
        """
        Test H0: beta = lambda_merton for a specific bucket.

        Args:
            results_df: Bucket regression results
            bucket_id: Bucket to test

        Returns:
            Dictionary with test results
        """
        row = results_df[results_df['bucket_id'] == bucket_id].iloc[0]

        beta = row['beta']
        se_beta = row['se_beta']
        lambda_merton = row['lambda_merton']

        # t-test: H0: beta = lambda_merton
        t_stat = (beta - lambda_merton) / se_beta
        df_val = row['n_observations'] - 2  # OLS degrees of freedom
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))

        reject_null = p_value < 0.05

        return {
            'bucket_id': bucket_id,
            'beta': beta,
            'lambda_merton': lambda_merton,
            'se_beta': se_beta,
            't_stat_level_test': t_stat,
            'p_value_level_test': p_value,
            'reject_null': reject_null,
            'interpretation': 'Reject H0: beta ≠ lambda' if reject_null else 'Fail to reject: beta ≈ lambda'
        }

    def aggregate_level_test(self, results_df: pd.DataFrame) -> Dict:
        """
        Test whether average deviation from Merton is zero across all buckets.

        Args:
            results_df: Bucket regression results

        Returns:
            Dictionary with aggregate test results
        """
        mean_deviation = results_df['deviation'].mean()
        median_ratio = results_df['beta_lambda_ratio'].median()

        # Bootstrap standard error (simple version)
        n_boot = 1000
        boot_means = []
        for _ in range(n_boot):
            sample = results_df['deviation'].sample(n=len(results_df), replace=True)
            boot_means.append(sample.mean())

        se_mean_deviation = np.std(boot_means)
        t_stat = mean_deviation / se_mean_deviation
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        # Count how many buckets in acceptable range [0.8, 1.2]
        in_range = ((results_df['beta_lambda_ratio'] >= 0.8) &
                   (results_df['beta_lambda_ratio'] <= 1.2)).sum()
        pct_in_range = 100 * in_range / len(results_df)

        return {
            'mean_deviation': mean_deviation,
            'median_ratio': median_ratio,
            'se_mean_deviation': se_mean_deviation,
            't_stat': t_stat,
            'p_value': p_value,
            'n_buckets': len(results_df),
            'buckets_in_range_0.8_1.2': in_range,
            'pct_in_range': pct_in_range
        }

    def test_cross_maturity_pattern(
        self,
        results_df: pd.DataFrame,
        rating: str,
        sector: str
    ) -> Dict:
        """
        Test whether cross-maturity pattern matches Merton prediction.

        Merton prediction: shorter maturity → higher beta

        Args:
            results_df: Bucket regression results
            rating: Rating bucket (e.g., 'BBB')
            sector: Sector name

        Returns:
            Dictionary with test results
        """
        # Filter to specified rating and sector
        mask = (
            (results_df['rating_bucket'] == rating) &
            (results_df['sector'] == sector)
        )
        subset = results_df[mask].copy()

        if len(subset) < 3:
            return {
                'rating': rating,
                'sector': sector,
                'error': 'Insufficient data points'
            }

        # Define maturity ordering
        maturity_order = {
            '1-2y': 1, '2-3y': 2, '3-5y': 3,
            '5-7y': 4, '7-10y': 5, '10y+': 6
        }
        subset['maturity_order'] = subset['maturity_bucket'].map(maturity_order)
        subset = subset.sort_values('maturity_order')

        # Spearman correlation between maturity and beta (should be negative)
        rho_beta, p_beta = stats.spearmanr(
            subset['maturity_order'],
            subset['beta']
        )

        # Spearman correlation between empirical beta and theoretical lambda
        rho_theory, p_theory = stats.spearmanr(
            subset['lambda_merton'],
            subset['beta']
        )

        # Test monotonicity: is beta declining with maturity?
        betas = subset['beta'].values
        is_monotonic = all(betas[i] >= betas[i+1] for i in range(len(betas)-1))

        return {
            'rating': rating,
            'sector': sector,
            'n_maturity_buckets': len(subset),
            'spearman_rho_maturity_beta': rho_beta,
            'p_value_maturity': p_beta,
            'spearman_rho_theory_empirical': rho_theory,
            'p_value_theory': p_theory,
            'is_monotonic': is_monotonic,
            'interpretation': (
                'Cross-maturity pattern confirmed' if (rho_beta < 0 and p_beta < 0.05)
                else 'Cross-maturity pattern NOT confirmed'
            )
        }

    def test_regime_pattern(self, results_df: pd.DataFrame) -> Dict:
        """
        Test whether cross-maturity dispersion declines as spreads widen.

        Regime prediction: IG (narrow spreads) should have high dispersion,
        HY should have moderate, distressed should converge to beta ≈ 1.

        Args:
            results_df: Bucket regression results

        Returns:
            Dictionary with regime test results
        """
        # Group by rating and sector, compute cross-maturity std dev
        dispersion = results_df.groupby(['rating_bucket', 'sector']).agg({
            'beta': 'std',
            'median_spread': 'mean',
            'is_ig': 'first'
        }).reset_index()

        dispersion.columns = ['rating_bucket', 'sector', 'beta_std', 'avg_spread', 'is_ig']

        # Correlation between spread level and dispersion (should be negative)
        rho, p_value = stats.spearmanr(
            dispersion['avg_spread'],
            dispersion['beta_std']
        )

        # Compare IG vs HY dispersion
        ig_dispersion = dispersion[dispersion['is_ig']]['beta_std'].mean()
        hy_dispersion = dispersion[~dispersion['is_ig']]['beta_std'].mean()

        return {
            'spearman_rho_spread_dispersion': rho,
            'p_value': p_value,
            'ig_avg_dispersion': ig_dispersion,
            'hy_avg_dispersion': hy_dispersion,
            'dispersion_ratio_ig_hy': ig_dispersion / hy_dispersion if hy_dispersion > 0 else np.nan,
            'interpretation': (
                'Regime pattern confirmed: IG has higher dispersion than HY'
                if ig_dispersion > hy_dispersion else
                'Regime pattern NOT confirmed'
            )
        }

    def identify_outliers(
        self,
        results_df: pd.DataFrame,
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Identify buckets where beta/lambda ratio is far from 1.

        Args:
            results_df: Bucket regression results
            threshold: Ratio threshold (default: outside [1/1.5, 1.5])

        Returns:
            DataFrame of outlier buckets
        """
        lower_bound = 1.0 / threshold
        upper_bound = threshold

        outliers = results_df[
            (results_df['beta_lambda_ratio'] < lower_bound) |
            (results_df['beta_lambda_ratio'] > upper_bound)
        ].copy()

        outliers = outliers.sort_values('beta_lambda_ratio', ascending=False)

        return outliers[[
            'bucket_id', 'rating_bucket', 'maturity_bucket', 'sector',
            'beta', 'lambda_merton', 'beta_lambda_ratio',
            'median_spread', 'median_maturity', 'n_observations'
        ]]

    def generate_decision_recommendation(self, results_df: pd.DataFrame) -> str:
        """
        Generate Stage 0 decision recommendation based on results.

        Args:
            results_df: Bucket regression results

        Returns:
            String with recommendation
        """
        agg_test = self.aggregate_level_test(results_df)
        median_ratio = agg_test['median_ratio']
        pct_in_range = agg_test['pct_in_range']

        if 0.8 <= median_ratio <= 1.2 and pct_in_range > 70:
            return (
                "✓ GOOD BASELINE: Merton provides good baseline. Proceed with "
                "theory-constrained specifications in Stages A-C. Use lambda_merton "
                "as starting point."
            )
        elif median_ratio > 1.2 or median_ratio < 0.8:
            return (
                "⚠ SYSTEMATIC BIAS: Merton has right structure but wrong scale. "
                "Proceed to calibrated Merton in Stage B (estimate scaling factor)."
            )
        elif pct_in_range < 50:
            return (
                "⚠ HIGH DISPERSION: Merton captures average but misses heterogeneity. "
                "Proceed with both theory-constrained and unrestricted tracks in parallel."
            )
        else:
            return (
                "⚠ MIXED RESULTS: Requires detailed regime analysis. Check if Merton "
                "works in IG but fails in HY."
            )
