"""
Reporting module for Stage A deliverables.

Generates tables and written summaries as specified in the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class StageAReporter:
    """
    Generates Stage A deliverables:
    - Table A.1: Bucket-level beta estimates
    - Table A.2: Tests of beta equality
    - Table A.3: Continuous characteristic regression
    - Written summary
    """

    def __init__(self, output_dir: str = './output/reports'):
        self.output_dir = output_dir

    def create_table_a1_bucket_betas(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Table A.1: Bucket-level β^(k) estimates.

        Pivot table showing betas by rating × maturity, with standard errors.

        Args:
            results_df: Bucket-level results from Spec A.1

        Returns:
            Formatted DataFrame
        """
        # Create pivot table of betas
        beta_pivot = results_df.pivot_table(
            index='rating_bucket',
            columns='maturity_bucket',
            values='beta',
            aggfunc='mean'
        )

        # Create pivot table of standard errors
        se_pivot = results_df.pivot_table(
            index='rating_bucket',
            columns='maturity_bucket',
            values='se_beta',
            aggfunc='mean'
        )

        # Create pivot table of sample sizes
        n_pivot = results_df.pivot_table(
            index='rating_bucket',
            columns='maturity_bucket',
            values='n_observations',
            aggfunc='sum'
        )

        # Order columns and rows
        maturity_order = ['1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10y+']
        rating_order = ['AAA/AA', 'A', 'BBB', 'BB', 'B', 'CCC']

        beta_pivot = beta_pivot.reindex(
            index=[r for r in rating_order if r in beta_pivot.index],
            columns=[m for m in maturity_order if m in beta_pivot.columns]
        )

        se_pivot = se_pivot.reindex(
            index=[r for r in rating_order if r in se_pivot.index],
            columns=[m for m in maturity_order if m in se_pivot.columns]
        )

        n_pivot = n_pivot.reindex(
            index=[r for r in rating_order if r in n_pivot.index],
            columns=[m for m in maturity_order if m in n_pivot.columns]
        )

        # Format as: β (se) [n]
        formatted = beta_pivot.copy()
        for idx in formatted.index:
            for col in formatted.columns:
                beta = beta_pivot.loc[idx, col]
                se = se_pivot.loc[idx, col]
                n = n_pivot.loc[idx, col]

                if pd.notna(beta):
                    formatted.loc[idx, col] = f"{beta:.3f} ({se:.3f}) [n={int(n):,}]"
                else:
                    formatted.loc[idx, col] = "—"

        return formatted

    def create_table_a2_equality_tests(
        self,
        test_results: List[Dict]
    ) -> pd.DataFrame:
        """
        Table A.2: Tests of beta equality.

        Args:
            test_results: List of F-test results

        Returns:
            Formatted DataFrame
        """
        rows = []

        for test in test_results:
            if 'error' in test:
                continue

            test_type = test.get('test', 'Unknown')
            dimension = test.get('dimension', 'Overall')
            fixed = test.get('fixed_values', {})

            # Format test description
            if dimension == 'Overall':
                description = 'Overall (all buckets)'
            else:
                fixed_str = ', '.join([f'{k}={v}' for k, v in fixed.items()])
                description = f'Across {dimension}' + (f' ({fixed_str})' if fixed else '')

            rows.append({
                'Test': description,
                'H0': test.get('h0', ''),
                'N Groups': test.get('n_groups', test.get('n_buckets', '')),
                'F-statistic': f"{test.get('f_statistic', np.nan):.2f}",
                'df': test.get('degrees_of_freedom', ''),
                'p-value': f"{test.get('p_value', np.nan):.4f}",
                'Reject H0?': 'Yes' if test.get('reject_h0', False) else 'No',
                'Interpretation': test.get('interpretation', '')
            })

        return pd.DataFrame(rows)

    def create_table_a3_continuous_spec(
        self,
        a2_results: Dict
    ) -> pd.DataFrame:
        """
        Table A.3: Continuous characteristic regression (Specification A.2).

        Args:
            a2_results: Results from Specification A.2

        Returns:
            Formatted DataFrame
        """
        rows = []

        for regime in ['combined', 'ig', 'hy']:
            results = a2_results.get(regime, {})

            if 'error' in results:
                continue

            regime_label = {
                'combined': 'Combined (All)',
                'ig': 'Investment Grade',
                'hy': 'High Yield'
            }[regime]

            # Extract coefficients
            coeffs = {
                'Intercept (γ₀)': ('gamma_0', 'se_gamma_0', 'p_gamma_0'),
                'Maturity (γ_M)': ('gamma_M', 'se_gamma_M', 'p_gamma_M'),
                'Spread (γ_s)': ('gamma_s', 'se_gamma_s', 'p_gamma_s'),
                'Maturity² (γ_M²)': ('gamma_M2', 'se_gamma_M2', 'p_gamma_M2'),
                'Maturity×Spread (γ_Ms)': ('gamma_Ms', 'se_gamma_Ms', 'p_gamma_Ms'),
            }

            for coeff_name, (coeff_key, se_key, p_key) in coeffs.items():
                coeff_val = results.get(coeff_key, np.nan)
                se_val = results.get(se_key, np.nan)
                p_val = results.get(p_key if p_key != 'p_gamma_0' else None, np.nan)

                # Significance stars
                stars = ''
                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < 0.05:
                    stars = '*'

                rows.append({
                    'Regime': regime_label,
                    'Coefficient': coeff_name,
                    'Estimate': f"{coeff_val:.4f}{stars}",
                    'Std Error': f"{se_val:.4f}",
                    'p-value': f"{p_val:.4f}" if pd.notna(p_val) else "—"
                })

            # Add R² row
            r2 = results.get('r_squared', np.nan)
            adj_r2 = results.get('adj_r_squared', np.nan)
            n_obs = results.get('n_observations', 0)

            rows.append({
                'Regime': regime_label,
                'Coefficient': 'R²',
                'Estimate': f"{r2:.4f}",
                'Std Error': f"Adj R² = {adj_r2:.4f}",
                'p-value': f"N = {n_obs:,}"
            })

        return pd.DataFrame(rows)

    def generate_written_summary(
        self,
        results_df: pd.DataFrame,
        test_results: List[Dict],
        a2_results: Dict,
        econ_significance: Dict,
        ig_hy_comparison: Dict,
        decision: str
    ) -> str:
        """
        Generate written summary (2 pages) as specified in the paper.

        Args:
            results_df: Bucket-level results
            test_results: F-test results
            a2_results: Specification A.2 results
            econ_significance: Economic significance metrics
            ig_hy_comparison: IG vs HY comparison
            decision: Decision recommendation

        Returns:
            Formatted string with summary
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract key test
        overall_test = next((t for t in test_results if 'Overall' in t.get('test', '')), {})

        summary = f"""
{'='*80}
STAGE A: ESTABLISH CROSS-SECTIONAL VARIATION
Generated: {timestamp}
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}

Total buckets analyzed: {len(results_df)}
IG buckets: {results_df['is_ig'].sum()}
HY buckets: {(~results_df['is_ig']).sum()}

CRITICAL QUESTION: Do DTS betas differ significantly across bonds?

Overall F-test: F({overall_test.get('degrees_of_freedom', 'N/A')}) = {overall_test.get('f_statistic', np.nan):.2f}, p = {overall_test.get('p_value', np.nan):.4f}
Specification A.2 R²: {a2_results.get('combined', {}).get('r_squared', np.nan):.3f}

DECISION:
{decision}

{'='*80}
1. IS VARIATION STATISTICALLY SIGNIFICANT?
{'='*80}

Overall Test (H0: All betas equal):
  F-statistic: {overall_test.get('f_statistic', np.nan):.2f}
  Degrees of freedom: {overall_test.get('degrees_of_freedom', 'N/A')}
  p-value: {overall_test.get('p_value', np.nan):.4f}
  Reject H0: {'YES' if overall_test.get('reject_h0', False) else 'NO'}

Interpretation:
{overall_test.get('interpretation', 'No test available')}

Dimension-Specific Tests:
"""

        # Add key dimension tests
        for test in test_results[1:6]:  # First few after overall
            if 'error' not in test:
                summary += f"""
  {test.get('test', 'Unknown')}:
    F = {test.get('f_statistic', np.nan):.2f}, p = {test.get('p_value', np.nan):.4f}
    {test.get('interpretation', '')}
"""

        summary += f"""
{'='*80}
2. IS VARIATION ECONOMICALLY MEANINGFUL?
{'='*80}

Beta Range: {econ_significance['min_beta']:.2f} to {econ_significance['max_beta']:.2f}
  Difference: {econ_significance['range']:.2f}
  Ratio (max/min): {econ_significance['ratio_max_min']:.2f}x

Distribution:
  10th percentile: {econ_significance['p10']:.2f}
  25th percentile: {econ_significance['p25']:.2f}
  Median: {econ_significance['p50']:.2f}
  75th percentile: {econ_significance['p75']:.2f}
  90th percentile: {econ_significance['p90']:.2f}

Interquartile Range: {econ_significance['iqr']:.2f}
Standard Deviation: {econ_significance['std']:.2f}
Coefficient of Variation: {econ_significance['cv']:.2f}

Interpretation:
{econ_significance['interpretation']}

Economic Significance Assessment:
"""
        if econ_significance['ratio_max_min'] > 2.0:
            summary += "  ✓ HIGHLY SIGNIFICANT: More than 2x variation across buckets\n"
        elif econ_significance['ratio_max_min'] > 1.5:
            summary += "  ⚠ MODERATE: 1.5-2x variation\n"
        else:
            summary += "  ✗ MODEST: Less than 1.5x variation\n"

        summary += f"""
{'='*80}
3. WHAT CHARACTERISTICS DRIVE VARIATION?
{'='*80}

Specification A.2: Continuous Characteristic Regression

Combined Sample (All Bonds):
  R² = {a2_results.get('combined', {}).get('r_squared', np.nan):.3f}
  Adj R² = {a2_results.get('combined', {}).get('adj_r_squared', np.nan):.3f}
  N = {a2_results.get('combined', {}).get('n_observations', 0):,}

Key Coefficients:
  Maturity (γ_M): {a2_results.get('combined', {}).get('gamma_M', np.nan):.4f} (p={a2_results.get('combined', {}).get('p_gamma_M', np.nan):.4f})
  Spread (γ_s): {a2_results.get('combined', {}).get('gamma_s', np.nan):.6f} (p={a2_results.get('combined', {}).get('p_gamma_s', np.nan):.4f})
  Maturity² (γ_M²): {a2_results.get('combined', {}).get('gamma_M2', np.nan):.4f} (p={a2_results.get('combined', {}).get('p_gamma_M2', np.nan):.4f})
  Maturity×Spread (γ_Ms): {a2_results.get('combined', {}).get('gamma_Ms', np.nan):.6f} (p={a2_results.get('combined', {}).get('p_gamma_Ms', np.nan):.4f})

Interpretation:
"""
        # Interpret coefficients
        gamma_M = a2_results.get('combined', {}).get('gamma_M', 0)
        gamma_s = a2_results.get('combined', {}).get('gamma_s', 0)

        if gamma_M < 0:
            summary += "  - Shorter maturity bonds have HIGHER betas (negative γ_M)\n"
        else:
            summary += "  - Longer maturity bonds have HIGHER betas (positive γ_M)\n"

        if gamma_s < 0:
            summary += "  - Lower spread bonds have HIGHER betas (negative γ_s)\n"
        else:
            summary += "  - Higher spread bonds have HIGHER betas (positive γ_s)\n"

        summary += f"""
{'='*80}
4. DOES IG SHOW MORE VARIATION THAN HY?
{'='*80}

Theory Prediction (Regime 2): IG should exhibit more variation than HY
  (IG narrow spreads → large maturity effects; HY wide spreads → convergence)

Observed:
  IG standard deviation: {ig_hy_comparison.get('ig_std', np.nan):.3f}
  HY standard deviation: {ig_hy_comparison.get('hy_std', np.nan):.3f}
  Ratio (IG/HY): {ig_hy_comparison.get('std_ratio_ig_hy', np.nan):.2f}

  IG range: {ig_hy_comparison.get('ig_range', np.nan):.3f}
  HY range: {ig_hy_comparison.get('hy_range', np.nan):.3f}

Levene's Test for Equality of Variances:
  Statistic: {ig_hy_comparison.get('levene_statistic', np.nan):.2f}
  p-value: {ig_hy_comparison.get('levene_p_value', np.nan):.4f}

Interpretation:
{ig_hy_comparison.get('interpretation', 'Insufficient data')}

Result: {'✓ CONFIRMS theory' if ig_hy_comparison.get('ig_has_more_variation', False) else '✗ CONTRADICTS theory'}

{'='*80}
5. RECOMMENDATION
{'='*80}

{decision}

{'='*80}
NEXT STEPS
{'='*80}
"""

        if 'STOP' in decision:
            summary += """
Standard DTS is adequate. No further analysis needed.

Report findings:
- Tested for cross-sectional variation in DTS betas
- Found no statistically or economically significant differences
- Standard DTS model (β = 1 for all bonds) is appropriate
- No need for maturity or credit quality adjustments
"""
        else:
            summary += """
Proceed to Stage B: Test whether Merton explains the variation

Stage B will determine:
- Does Merton's structural model predict observed beta patterns?
- Is theory-constrained specification adequate or need unrestricted?
- Can we use simple lambda tables or need full empirical estimation?

This establishes the foundation for production risk systems.
"""

        summary += """
{'='*80}
"""
        return summary

    def save_all_reports(
        self,
        results_df: pd.DataFrame,
        test_results: List[Dict],
        a2_results: Dict,
        econ_significance: Dict,
        ig_hy_comparison: Dict,
        decision: str,
        prefix: str = 'stageA'
    ):
        """
        Save all Stage A reports to files.

        Args:
            results_df: Bucket-level results
            test_results: F-test results
            a2_results: Specification A.2 results
            econ_significance: Economic significance metrics
            ig_hy_comparison: IG vs HY comparison
            decision: Decision recommendation
            prefix: Filename prefix
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Table A.1
        table1 = self.create_table_a1_bucket_betas(results_df)
        table1.to_csv(f'{self.output_dir}/{prefix}_table_a1_bucket_betas.csv')

        # Table A.2
        table2 = self.create_table_a2_equality_tests(test_results)
        table2.to_csv(f'{self.output_dir}/{prefix}_table_a2_equality_tests.csv', index=False)

        # Table A.3
        table3 = self.create_table_a3_continuous_spec(a2_results)
        table3.to_csv(f'{self.output_dir}/{prefix}_table_a3_continuous_spec.csv', index=False)

        # Written summary
        summary = self.generate_written_summary(
            results_df, test_results, a2_results,
            econ_significance, ig_hy_comparison, decision
        )
        with open(f'{self.output_dir}/{prefix}_summary.txt', 'w') as f:
            f.write(summary)

        # Full results
        results_df.to_csv(f'{self.output_dir}/{prefix}_full_results.csv', index=False)

        print(f"Stage A reports saved to {self.output_dir}/")
