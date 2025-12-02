"""
Reporting module for Stage 0 deliverables.

Generates tables and written summaries as specified in the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class Stage0Reporter:
    """
    Generates Stage 0 deliverables:
    - Table 0.1: Bucket-level results
    - Table 0.2: Cross-maturity pattern tests
    - Written summary
    """

    def __init__(self, output_dir: str = './output/reports'):
        self.output_dir = output_dir

    def create_table_01_bucket_results(
        self,
        results_df: pd.DataFrame,
        top_n: int = 30
    ) -> pd.DataFrame:
        """
        Table 0.1: Bucket-level results for key buckets.

        Columns:
        - Rating x Maturity combination
        - β (empirical)
        - λ (theoretical Merton)
        - Ratio (β/λ)
        - t-stat for H0: β=λ
        - Sample size

        Args:
            results_df: Bucket regression results
            top_n: Number of top buckets by sample size to include

        Returns:
            Formatted DataFrame
        """
        # Select key columns
        table = results_df[[
            'rating_bucket', 'maturity_bucket', 'sector',
            'beta', 'lambda_merton', 'beta_lambda_ratio',
            'se_beta', 'n_observations', 'median_spread', 'is_ig'
        ]].copy()

        # Calculate t-stat for H0: beta = lambda
        table['t_stat_level'] = (table['beta'] - table['lambda_merton']) / table['se_beta']

        # Sort by sample size and take top N
        table = table.sort_values('n_observations', ascending=False).head(top_n)

        # Format for display
        table['Rating/Maturity'] = table['rating_bucket'] + ' ' + table['maturity_bucket']

        display_table = pd.DataFrame({
            'Bucket': table['Rating/Maturity'],
            'Sector': table['sector'],
            'β (Empirical)': table['beta'].round(3),
            'λ (Merton)': table['lambda_merton'].round(3),
            'Ratio (β/λ)': table['beta_lambda_ratio'].round(3),
            't-stat': table['t_stat_level'].round(2),
            'Sample Size': table['n_observations'].astype(int),
            'Median Spread': table['median_spread'].round(0).astype(int),
            'Type': table['is_ig'].map({True: 'IG', False: 'HY'})
        })

        # Highlight outliers (ratio outside [0.8, 1.2])
        display_table['Flag'] = display_table['Ratio (β/λ)'].apply(
            lambda x: '***' if x < 0.8 or x > 1.2 else ''
        )

        return display_table

    def create_table_02_cross_maturity(
        self,
        results_df: pd.DataFrame,
        rating_buckets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Table 0.2: Cross-maturity pattern tests.

        For each rating class, shows β across maturity buckets.

        Args:
            results_df: Bucket regression results
            rating_buckets: List of rating buckets to include

        Returns:
            Formatted DataFrame
        """
        if rating_buckets is None:
            rating_buckets = ['AAA/AA', 'A', 'BBB', 'BB', 'B']

        # Aggregate across sectors for each rating/maturity
        summary = results_df.groupby(['rating_bucket', 'maturity_bucket']).agg({
            'beta': 'mean',
            'lambda_merton': 'mean',
            'n_observations': 'sum'
        }).reset_index()

        # Pivot to show maturities as columns
        beta_pivot = summary.pivot_table(
            index='rating_bucket',
            columns='maturity_bucket',
            values='beta'
        )

        lambda_pivot = summary.pivot_table(
            index='rating_bucket',
            columns='maturity_bucket',
            values='lambda_merton'
        )

        # Order maturity columns
        maturity_order = ['1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10y+']
        maturity_cols = [c for c in maturity_order if c in beta_pivot.columns]

        beta_pivot = beta_pivot[maturity_cols]
        lambda_pivot = lambda_pivot[maturity_cols]

        # Filter to requested ratings
        rating_rows = [r for r in rating_buckets if r in beta_pivot.index]
        beta_pivot = beta_pivot.loc[rating_rows]
        lambda_pivot = lambda_pivot.loc[rating_rows]

        # Combine into single table
        table = pd.DataFrame()
        for rating in rating_rows:
            # Empirical row
            row_beta = beta_pivot.loc[rating].to_frame().T
            row_beta.index = [f'{rating} (β)']
            table = pd.concat([table, row_beta])

            # Theoretical row
            row_lambda = lambda_pivot.loc[rating].to_frame().T
            row_lambda.index = [f'{rating} (λ)']
            table = pd.concat([table, row_lambda])

        table = table.round(3)
        return table

    def generate_written_summary(
        self,
        results_df: pd.DataFrame,
        aggregate_test: Dict,
        cross_maturity_tests: List[Dict],
        regime_test: Dict,
        outliers: pd.DataFrame
    ) -> str:
        """
        Generate written summary (2-3 pages) as specified in the paper.

        Args:
            results_df: Bucket regression results
            aggregate_test: Results from aggregate level test
            cross_maturity_tests: List of cross-maturity test results
            regime_test: Results from regime test
            outliers: DataFrame of outlier buckets

        Returns:
            Formatted string with summary
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary = f"""
{'='*80}
STAGE 0: RAW VALIDATION USING BUCKET-LEVEL ANALYSIS
Generated: {timestamp}
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}

Total buckets analyzed: {len(results_df)}
IG buckets: {results_df['is_ig'].sum()}
HY buckets: {(~results_df['is_ig']).sum()}
Total bond-week observations: {results_df['n_observations'].sum():,.0f}

Median β/λ ratio: {aggregate_test['median_ratio']:.3f}
Mean deviation (β - λ): {aggregate_test['mean_deviation']:.3f}
Buckets in acceptable range [0.8, 1.2]: {aggregate_test['pct_in_range']:.1f}%

DECISION: {self._get_decision(aggregate_test['median_ratio'], aggregate_test['pct_in_range'])}

{'='*80}
1. DOES MERTON PREDICT BUCKET-LEVEL SENSITIVITIES?
{'='*80}

Aggregate Level Test:
  Mean deviation from Merton: {aggregate_test['mean_deviation']:.4f}
  Standard error: {aggregate_test['se_mean_deviation']:.4f}
  t-statistic: {aggregate_test['t_stat']:.2f}
  p-value: {aggregate_test['p_value']:.4f}

Distribution of β/λ ratios:
  Min:  {results_df['beta_lambda_ratio'].min():.3f}
  Q1:   {results_df['beta_lambda_ratio'].quantile(0.25):.3f}
  Median: {results_df['beta_lambda_ratio'].median():.3f}
  Q3:   {results_df['beta_lambda_ratio'].quantile(0.75):.3f}
  Max:  {results_df['beta_lambda_ratio'].max():.3f}

Interpretation:
{self._interpret_level_test(aggregate_test)}

{'='*80}
2. IS CROSS-MATURITY PATTERN CORRECT?
{'='*80}

Theory predicts: Short maturity bonds should have higher β than long maturity bonds.

"""
        # Add cross-maturity test results
        for test in cross_maturity_tests:
            if 'error' not in test:
                summary += f"""
{test['rating']} - {test['sector']}:
  Maturity buckets tested: {test['n_maturity_buckets']}
  Spearman ρ (maturity vs β): {test['spearman_rho_maturity_beta']:.3f} (p={test['p_value_maturity']:.3f})
  Is monotonic declining: {test['is_monotonic']}
  Interpretation: {test['interpretation']}
"""

        summary += f"""
{'='*80}
3. DOES PATTERN DIFFER BY SPREAD LEVEL?
{'='*80}

Regime Pattern Test:
  IG average cross-maturity dispersion: {regime_test['ig_avg_dispersion']:.3f}
  HY average cross-maturity dispersion: {regime_test['hy_avg_dispersion']:.3f}
  Dispersion ratio (IG/HY): {regime_test['dispersion_ratio_ig_hy']:.2f}

  Spearman ρ (spread vs dispersion): {regime_test['spearman_rho_spread_dispersion']:.3f}
  p-value: {regime_test['p_value']:.3f}

Interpretation: {regime_test['interpretation']}

Theory predicts that cross-maturity dispersion should be:
  - HIGH in IG (spreads < 300 bps): Large maturity effects dominate
  - MODERATE in HY (300-1000 bps): Maturity effects diminish
  - LOW in distressed (> 1000 bps): Convergence toward β ≈ 1

Results {'CONFIRM' if regime_test['ig_avg_dispersion'] > regime_test['hy_avg_dispersion'] else 'DO NOT CONFIRM'} this prediction.

{'='*80}
4. WHERE DO LARGEST DEVIATIONS OCCUR?
{'='*80}

Top outliers (β/λ ratio furthest from 1.0):
"""
        # Add outlier details
        for idx, row in outliers.head(10).iterrows():
            summary += f"""
  {row['bucket_id']}:
    Empirical β: {row['beta']:.3f}, Theoretical λ: {row['lambda_merton']:.3f}
    Ratio: {row['beta_lambda_ratio']:.3f}
    Spread: {row['median_spread']:.0f} bps, Maturity: {row['median_maturity']:.1f}y
    Sample size: {row['n_observations']:,.0f}
"""

        summary += f"""
{'='*80}
5. PRACTICAL IMPLICATION
{'='*80}

Based on the aggregate results:
  - Median ratio: {aggregate_test['median_ratio']:.3f}
  - % in range [0.8, 1.2]: {aggregate_test['pct_in_range']:.1f}%

{self._get_practical_implication(aggregate_test['median_ratio'], aggregate_test['pct_in_range'])}

{'='*80}
NEXT STEPS
{'='*80}

Proceed to Stage A to establish cross-sectional variation using issuer-week
fixed effects methodology.

{'='*80}
"""
        return summary

    def _get_decision(self, median_ratio: float, pct_in_range: float) -> str:
        """Get decision recommendation."""
        if 0.8 <= median_ratio <= 1.2 and pct_in_range > 70:
            return "✓ Merton provides good baseline"
        elif median_ratio > 1.2 or median_ratio < 0.8:
            return "⚠ Systematic bias detected"
        else:
            return "⚠ High dispersion - mixed results"

    def _interpret_level_test(self, aggregate_test: Dict) -> str:
        """Interpret aggregate level test results."""
        if aggregate_test['p_value'] > 0.05:
            return (
                f"  We FAIL TO REJECT the null hypothesis that average deviation is zero "
                f"(p={aggregate_test['p_value']:.3f}). The Merton model provides unbiased "
                f"predictions on average across buckets."
            )
        else:
            direction = "overestimates" if aggregate_test['mean_deviation'] < 0 else "underestimates"
            return (
                f"  We REJECT the null hypothesis (p={aggregate_test['p_value']:.3f}). "
                f"The Merton model systematically {direction} empirical sensitivities "
                f"by {abs(aggregate_test['mean_deviation']):.3f} on average."
            )

    def _get_practical_implication(self, median_ratio: float, pct_in_range: float) -> str:
        """Get practical implication text."""
        if 0.8 <= median_ratio <= 1.2 and pct_in_range > 70:
            return """
CAN USE MERTON AS BASELINE:
  - Use λ^Merton tables as starting point for Stages A-C
  - Estimate small calibration adjustments empirically
  - Merton structure provides strong prior for hierarchical models
  - Production systems can rely on theoretical adjustments with confidence
"""
        elif median_ratio > 1.2 or median_ratio < 0.8:
            return """
SYSTEMATIC BIAS REQUIRES CALIBRATION:
  - Merton has correct functional form but wrong scale
  - Estimate global scaling factor in Stage B
  - Use calibrated Merton: β = α × λ^Merton where α is data-driven
  - Still exploit Merton's cross-sectional structure
"""
        else:
            return """
HIGH HETEROGENEITY - PARALLEL TRACKS:
  - Merton captures average but misses individual variation
  - Run both theory-constrained AND fully empirical specifications
  - Use theory-based as regularization prior
  - Investigate where and why Merton fails (regime analysis)
"""

    def save_all_reports(
        self,
        results_df: pd.DataFrame,
        aggregate_test: Dict,
        cross_maturity_tests: List[Dict],
        regime_test: Dict,
        outliers: pd.DataFrame,
        prefix: str = 'stage0'
    ):
        """
        Save all Stage 0 reports to files.

        Args:
            results_df: Bucket regression results
            aggregate_test: Aggregate test results
            cross_maturity_tests: Cross-maturity test results
            regime_test: Regime test results
            outliers: Outlier buckets
            prefix: Filename prefix
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Table 0.1
        table1 = self.create_table_01_bucket_results(results_df)
        table1.to_csv(f'{self.output_dir}/{prefix}_table01_bucket_results.csv', index=False)

        # Table 0.2
        table2 = self.create_table_02_cross_maturity(results_df)
        table2.to_csv(f'{self.output_dir}/{prefix}_table02_cross_maturity.csv')

        # Written summary
        summary = self.generate_written_summary(
            results_df, aggregate_test, cross_maturity_tests, regime_test, outliers
        )
        with open(f'{self.output_dir}/{prefix}_summary.txt', 'w') as f:
            f.write(summary)

        # Full results
        results_df.to_csv(f'{self.output_dir}/{prefix}_full_results.csv', index=False)

        print(f"Reports saved to {self.output_dir}/")
