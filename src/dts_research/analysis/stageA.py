"""
Stage A: Establish Cross-Sectional Variation

Establishes that DTS betas differ across bonds BEFORE testing whether Merton explains why.

Two main specifications:
A.1: Bucket-level betas (discrete characteristics)
A.2: Continuous characteristics (two-step procedure)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from scipy import stats
from itertools import combinations
import warnings

# Suppress specific warnings from rolling window regressions with small samples
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')


class StageAAnalysis:
    """
    Implements Stage A: Establish Cross-Sectional Variation.

    Critical objective: Document that betas differ BEFORE explaining why.
    """

    def __init__(self):
        self.spec_a1_results = None
        self.spec_a2_results = None
        self.bond_specific_betas = None

    # =========================================================================
    # Specification A.1: Bucket-Level Betas
    # =========================================================================

    def run_specification_a1(
        self,
        df: pd.DataFrame,
        bucket_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Specification A.1: Bucket-level beta estimation.

        For each bucket k, estimate: y_i,t = α^(k) + β^(k) f_DTS,t + ε

        Args:
            df: Regression-ready dataframe (from Stage 0 prep)
            bucket_stats: Bucket statistics

        Returns:
            DataFrame with beta estimates per bucket
        """
        results = []

        for bucket_id in df['bucket_id'].unique():
            bucket_data = df[df['bucket_id'] == bucket_id].copy()

            if len(bucket_data) < 30:
                continue

            # Prepare regression
            y = bucket_data['oas_pct_change'].values
            X = bucket_data['oas_index_pct_change'].values
            X = sm.add_constant(X)

            dates = bucket_data['date'].values

            try:
                # OLS with clustered SE
                model = sm.OLS(y, X)
                reg_results = model.fit(cov_type='cluster', cov_kwds={'groups': dates})

                beta = reg_results.params[1]
                se_beta = reg_results.bse[1]
                t_stat = reg_results.tvalues[1]
                p_value = reg_results.pvalues[1]

                results.append({
                    'bucket_id': bucket_id,
                    'beta': beta,
                    'se_beta': se_beta,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'alpha': reg_results.params[0],
                    'se_alpha': reg_results.bse[0],
                    'r_squared': reg_results.rsquared,
                    'n_observations': len(y),
                    'n_weeks': len(np.unique(dates))
                })

            except Exception as e:
                print(f"Regression failed for bucket {bucket_id}: {str(e)}")
                continue

        results_df = pd.DataFrame(results)

        # Merge with bucket characteristics
        results_df = results_df.merge(
            bucket_stats[[
                'bucket_id', 'rating_bucket', 'maturity_bucket', 'sector',
                'median_maturity', 'median_spread', 'is_ig'
            ]],
            on='bucket_id',
            how='left'
        )

        self.spec_a1_results = results_df
        return results_df

    def test_beta_equality_overall(self, results_df: pd.DataFrame) -> Dict:
        """
        F-test for H0: all β^(k) equal across all buckets.

        This is the CRITICAL test: if we fail to reject, standard DTS is adequate.

        Args:
            results_df: Bucket-level results from A.1

        Returns:
            Dictionary with F-test results
        """
        # Extract betas and standard errors
        betas = results_df['beta'].values
        se_betas = results_df['se_beta'].values
        n_buckets = len(betas)

        # Weighted mean (inverse variance weighting)
        weights = 1 / (se_betas ** 2)
        weighted_mean = np.sum(betas * weights) / np.sum(weights)

        # Chi-squared test statistic (Wald test)
        chi2_stat = np.sum(weights * (betas - weighted_mean) ** 2)
        df = n_buckets - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

        # Convert to F-statistic (approximate)
        f_stat = chi2_stat / df

        return {
            'test': 'Overall beta equality',
            'h0': 'All β^(k) are equal',
            'n_buckets': n_buckets,
            'weighted_mean_beta': weighted_mean,
            'chi2_statistic': chi2_stat,
            'f_statistic': f_stat,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'reject_h0': p_value < 0.10,  # Use 10% threshold per paper
            'interpretation': (
                'Significant variation exists - proceed to Stage B'
                if p_value < 0.10 else
                'No significant variation - standard DTS adequate'
            )
        }

    def test_beta_equality_by_dimension(
        self,
        results_df: pd.DataFrame,
        dimension: str,
        fixed_values: Optional[Dict] = None
    ) -> Dict:
        """
        F-test for beta equality across one dimension, holding others fixed.

        Args:
            results_df: Bucket-level results
            dimension: 'maturity', 'rating', or 'sector'
            fixed_values: Dict of {dimension: value} to hold constant
                         e.g., {'rating': 'BBB', 'sector': 'Industrial'}

        Returns:
            Dictionary with F-test results
        """
        # Filter to fixed values
        subset = results_df.copy()
        if fixed_values:
            for dim, val in fixed_values.items():
                col_name = f'{dim}_bucket' if dim in ['rating', 'maturity'] else dim
                subset = subset[subset[col_name] == val]

        if len(subset) < 2:
            return {
                'test': f'Beta equality across {dimension}',
                'error': 'Insufficient data',
                'n_groups': len(subset)
            }

        # Group by dimension
        dim_col = f'{dimension}_bucket' if dimension in ['rating', 'maturity'] else dimension
        groups = subset.groupby(dim_col)

        if len(groups) < 2:
            return {
                'test': f'Beta equality across {dimension}',
                'error': 'Only one group',
                'n_groups': len(groups)
            }

        # Extract betas and SEs by group
        group_stats = []
        for name, group in groups:
            group_stats.append({
                'name': name,
                'beta': group['beta'].mean(),
                'se_beta': np.sqrt(np.mean(group['se_beta'] ** 2)),  # Pooled SE
                'n': len(group)
            })

        group_df = pd.DataFrame(group_stats)

        # Weighted mean
        weights = 1 / (group_df['se_beta'] ** 2)
        weighted_mean = np.sum(group_df['beta'] * weights) / np.sum(weights)

        # Chi-squared test
        chi2_stat = np.sum(weights * (group_df['beta'] - weighted_mean) ** 2)
        df = len(group_df) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        f_stat = chi2_stat / df

        # Range
        beta_range = group_df['beta'].max() - group_df['beta'].min()

        return {
            'test': f'Beta equality across {dimension}',
            'h0': f'All betas equal across {dimension} levels',
            'dimension': dimension,
            'fixed_values': fixed_values,
            'n_groups': len(group_df),
            'weighted_mean_beta': weighted_mean,
            'beta_range': beta_range,
            'group_betas': group_df.to_dict('records'),
            'chi2_statistic': chi2_stat,
            'f_statistic': f_stat,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'reject_h0': p_value < 0.10,
            'interpretation': (
                f'Significant variation across {dimension}'
                if p_value < 0.10 else
                f'No significant variation across {dimension}'
            )
        }

    def run_all_dimension_tests(self, results_df: pd.DataFrame) -> List[Dict]:
        """
        Run F-tests across all dimensions and key combinations.

        Tests:
        1. Overall equality
        2. Across maturities (holding rating/sector constant)
        3. Across ratings (holding maturity/sector constant)
        4. Across sectors (holding rating/maturity constant)

        Returns:
            List of test result dictionaries
        """
        tests = []

        # Test 1: Overall
        tests.append(self.test_beta_equality_overall(results_df))

        # Test 2: Across maturities for key rating/sector combos
        for rating in ['BBB', 'A', 'BB']:
            for sector in ['Industrial', 'Financial']:
                mask = (
                    (results_df['rating_bucket'] == rating) &
                    (results_df['sector'] == sector)
                )
                if mask.sum() >= 2:
                    test = self.test_beta_equality_by_dimension(
                        results_df,
                        'maturity',
                        fixed_values={'rating': rating, 'sector': sector}
                    )
                    tests.append(test)

        # Test 3: Across ratings for key maturity/sector combos
        for maturity in ['3-5y', '5-7y', '7-10y']:
            for sector in ['Industrial', 'Financial']:
                mask = (
                    (results_df['maturity_bucket'] == maturity) &
                    (results_df['sector'] == sector)
                )
                if mask.sum() >= 2:
                    test = self.test_beta_equality_by_dimension(
                        results_df,
                        'rating',
                        fixed_values={'maturity': maturity, 'sector': sector}
                    )
                    tests.append(test)

        # Test 4: Across sectors
        test = self.test_beta_equality_by_dimension(results_df, 'sector')
        tests.append(test)

        return tests

    # =========================================================================
    # Specification A.2: Continuous Characteristics
    # =========================================================================

    def estimate_bond_specific_betas(
        self,
        bond_data: pd.DataFrame,
        index_data: pd.DataFrame,
        window_weeks: int = 104  # 2 years of weekly data
    ) -> pd.DataFrame:
        """
        Step 1 of A.2: Estimate bond-specific betas using rolling windows.

        For each bond i, estimate β_i using 2-year rolling windows.

        Args:
            bond_data: Bond-level data
            index_data: Index-level data
            window_weeks: Window size in weeks (default 104 = 2 years)

        Returns:
            DataFrame with bond-specific betas at each window midpoint
        """
        # Merge bond and index data
        df = bond_data.merge(
            index_data[['date', 'oas']],
            on='date',
            how='inner',
            suffixes=('', '_index')
        )

        # Sort
        df = df.sort_values(['bond_id', 'date'])

        # Compute percentage changes
        df['oas_pct_change'] = df.groupby('bond_id')['oas'].pct_change()
        df['oas_index_pct_change'] = df.groupby('bond_id')['oas_index'].transform(
            lambda x: x.pct_change()
        )

        df = df.dropna(subset=['oas_pct_change', 'oas_index_pct_change'])

        # Rolling window estimation
        bond_betas = []

        for bond_id in df['bond_id'].unique():
            bond_df = df[df['bond_id'] == bond_id].sort_values('date')

            if len(bond_df) < window_weeks:
                continue

            # Rolling windows
            for i in range(len(bond_df) - window_weeks + 1):
                window_data = bond_df.iloc[i:i+window_weeks]

                y = window_data['oas_pct_change'].values
                X = window_data['oas_index_pct_change'].values
                X = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X)
                    results = model.fit()

                    # Window midpoint
                    midpoint_idx = i + window_weeks // 2
                    midpoint_row = bond_df.iloc[midpoint_idx]

                    bond_betas.append({
                        'bond_id': bond_id,
                        'window_midpoint': midpoint_row['date'],
                        'beta': results.params[1],
                        'se_beta': results.bse[1],
                        't_stat': results.tvalues[1],
                        'r_squared': results.rsquared,
                        'maturity': midpoint_row['time_to_maturity'],
                        'spread': midpoint_row['oas'],
                        'rating': midpoint_row['rating'],
                        'sector': midpoint_row['sector']
                    })

                except:
                    continue

        bond_betas_df = pd.DataFrame(bond_betas)
        self.bond_specific_betas = bond_betas_df
        return bond_betas_df

    def run_specification_a2(
        self,
        bond_betas_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Step 2 of A.2: Cross-sectional regression of betas on characteristics.

        Regression: β_hat_i,τ = γ0 + γ_M M + γ_s s + γ_M² M² + γ_Ms M·s + u

        Args:
            bond_betas_df: Bond-specific betas (from Step 1)

        Returns:
            Dictionary with regression results
        """
        if bond_betas_df is None:
            bond_betas_df = self.bond_specific_betas

        if bond_betas_df is None or len(bond_betas_df) == 0:
            return {'error': 'No bond-specific betas available'}

        # Prepare regression variables
        df = bond_betas_df.dropna(subset=['beta', 'maturity', 'spread']).copy()

        # Create polynomial and interaction terms
        df['maturity_sq'] = df['maturity'] ** 2
        df['spread_sq'] = df['spread'] ** 2
        df['maturity_spread'] = df['maturity'] * df['spread']

        # Dependent variable
        y = df['beta'].values

        # Independent variables
        X = df[['maturity', 'spread', 'maturity_sq', 'maturity_spread']].values
        X = sm.add_constant(X)

        # Cluster by bond
        clusters = df['bond_id'].values

        try:
            # OLS with clustered SE
            model = sm.OLS(y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

            return {
                'gamma_0': results.params[0],
                'gamma_M': results.params[1],
                'gamma_s': results.params[2],
                'gamma_M2': results.params[3],
                'gamma_Ms': results.params[4],
                'se_gamma_0': results.bse[0],
                'se_gamma_M': results.bse[1],
                'se_gamma_s': results.bse[2],
                'se_gamma_M2': results.bse[3],
                'se_gamma_Ms': results.bse[4],
                't_gamma_M': results.tvalues[1],
                't_gamma_s': results.tvalues[2],
                't_gamma_M2': results.tvalues[3],
                't_gamma_Ms': results.tvalues[4],
                'p_gamma_M': results.pvalues[1],
                'p_gamma_s': results.pvalues[2],
                'p_gamma_M2': results.pvalues[3],
                'p_gamma_Ms': results.pvalues[4],
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'n_observations': len(y),
                'n_bonds': df['bond_id'].nunique()
            }

        except Exception as e:
            return {'error': f'Regression failed: {str(e)}'}

    def run_specification_a2_by_regime(
        self,
        bond_betas_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Run Specification A.2 separately for IG and HY.

        Args:
            bond_betas_df: Bond-specific betas

        Returns:
            Dictionary with results for IG and HY
        """
        if bond_betas_df is None:
            bond_betas_df = self.bond_specific_betas

        # Split by spread regime
        ig_df = bond_betas_df[bond_betas_df['spread'] < 300].copy()
        hy_df = bond_betas_df[bond_betas_df['spread'] >= 300].copy()

        results = {
            'ig': self.run_specification_a2(ig_df) if len(ig_df) > 0 else {'error': 'No IG data'},
            'hy': self.run_specification_a2(hy_df) if len(hy_df) > 0 else {'error': 'No HY data'},
            'combined': self.run_specification_a2(bond_betas_df)
        }

        self.spec_a2_results = results
        return results

    # =========================================================================
    # Economic Significance
    # =========================================================================

    def compute_economic_significance(self, results_df: pd.DataFrame) -> Dict:
        """
        Assess economic significance of beta variation.

        Args:
            results_df: Bucket-level results

        Returns:
            Dictionary with economic significance metrics
        """
        betas = results_df['beta'].values

        return {
            'min_beta': betas.min(),
            'max_beta': betas.max(),
            'range': betas.max() - betas.min(),
            'iqr': np.percentile(betas, 75) - np.percentile(betas, 25),
            'std': betas.std(),
            'cv': betas.std() / betas.mean() if betas.mean() != 0 else np.nan,
            'ratio_max_min': betas.max() / betas.min() if betas.min() > 0 else np.nan,
            'p10': np.percentile(betas, 10),
            'p25': np.percentile(betas, 25),
            'p50': np.percentile(betas, 50),
            'p75': np.percentile(betas, 75),
            'p90': np.percentile(betas, 90),
            'interpretation': (
                f'Betas range from {betas.min():.2f} to {betas.max():.2f}, '
                f'a {betas.max()/betas.min():.1f}x variation. '
                f'IQR = {np.percentile(betas, 75) - np.percentile(betas, 25):.2f}'
            )
        }

    def compare_ig_vs_hy_variation(self, results_df: pd.DataFrame) -> Dict:
        """
        Compare variation in IG vs HY (Regime 2 prediction).

        Theory predicts IG should show more variation than HY.

        Args:
            results_df: Bucket-level results

        Returns:
            Dictionary with comparison results
        """
        ig_betas = results_df[results_df['is_ig']]['beta'].values
        hy_betas = results_df[~results_df['is_ig']]['beta'].values

        if len(ig_betas) == 0 or len(hy_betas) == 0:
            return {'error': 'Insufficient data for comparison'}

        # Levene's test for equality of variances
        levene_stat, levene_p = stats.levene(ig_betas, hy_betas)

        return {
            'ig_std': ig_betas.std(),
            'hy_std': hy_betas.std(),
            'ig_range': ig_betas.max() - ig_betas.min(),
            'hy_range': hy_betas.max() - hy_betas.min(),
            'std_ratio_ig_hy': ig_betas.std() / hy_betas.std(),
            'levene_statistic': levene_stat,
            'levene_p_value': levene_p,
            'ig_has_more_variation': ig_betas.std() > hy_betas.std(),
            'interpretation': (
                'IG shows more variation than HY (confirms Regime 2 prediction)'
                if ig_betas.std() > hy_betas.std() else
                'HY shows more variation than IG (contradicts theory)'
            )
        }

    def generate_stage_a_decision(
        self,
        overall_test: Dict,
        econ_significance: Dict,
        a2_results: Dict
    ) -> str:
        """
        Generate Stage A decision recommendation.

        Args:
            overall_test: Overall F-test results
            econ_significance: Economic significance metrics
            a2_results: Specification A.2 results

        Returns:
            Decision string
        """
        p_value = overall_test['p_value']
        r2 = a2_results.get('combined', {}).get('r_squared', 0)

        if p_value > 0.10 and r2 < 0.05:
            return (
                "❌ STOP HERE - NO SIGNIFICANT VARIATION\n"
                f"F-test p-value = {p_value:.3f} (> 0.10)\n"
                f"R² = {r2:.3f} (< 0.05)\n"
                "Standard DTS is adequate. No need for adjustments.\n"
                "Report this as primary finding."
            )
        elif p_value < 0.01 and r2 > 0.15:
            return (
                "✓ STRONG VARIATION - PROCEED TO STAGE B\n"
                f"F-test p-value < 0.001 (highly significant)\n"
                f"R² = {r2:.3f} (> 0.15)\n"
                f"Beta range: {econ_significance['range']:.2f} "
                f"({econ_significance['ratio_max_min']:.1f}x variation)\n"
                "Systematic variation exists. Proceed to Stage B to test if Merton explains it."
            )
        elif 0.01 <= p_value < 0.10:
            return (
                "⚠ MARGINAL VARIATION - PROCEED WITH CAUTION\n"
                f"F-test p-value = {p_value:.3f} (0.01 < p < 0.10)\n"
                f"R² = {r2:.3f}\n"
                "Some evidence of variation. Proceed to Stage B but theory may suffice."
            )
        else:
            return (
                "✓ PROCEED TO STAGE B\n"
                "Variation exists but magnitude uncertain. "
                "Stage B will determine if Merton provides adequate explanation."
            )
