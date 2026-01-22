"""
Stage 0: Reporting Module (Evolved Version)

Creates tables for Stage 0 analysis testing β ≈ 1 (Merton prediction):
1-2. Bucket regression results (IG, HY) - β vs λ^Merton
3-4. β/λ ratio summary statistics (IG, HY)
5-6. Monotonicity test results (IG, HY) - β decreasing with maturity
7-8. Within-issuer summary statistics (IG, HY)
9-10. Within-issuer pooled β estimates and H0: β=1 test (IG, HY)
11-12. Sector regression results (IG, HY)
13-14. Sector hypothesis tests (IG, HY)
15-16. Decision path summary (IG, HY)
17. Cross-universe comparison and recommendations

Based on reporting requirements from the evolved Stage 0 analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


class Stage0Reporting:
    """
    Reporting utilities for evolved Stage 0 analysis.

    Key difference from original: focuses on testing β ≈ 1 (Merton prediction)
    rather than just λ > 0.
    """

    def __init__(self, output_dir: str = "output/stage0_tables"):
        """
        Initialize reporting module.

        Args:
            output_dir: Directory to save tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tables = {}

    def generate_all_tables(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict,
        sector_results_ig: Dict,
        sector_results_hy: Dict,
        synthesis_ig: Dict,
        synthesis_hy: Dict,
        comparison: Dict
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all Stage 0 tables.

        Args:
            Results dictionaries from all analyses

        Returns:
            Dictionary mapping table names to DataFrames
        """
        print("Generating Stage 0 tables...")

        # Tables 1-2: Bucket regression results
        self.tables['table1_bucket_regression_ig'] = self._table_bucket_regression(
            bucket_results_ig, 'IG'
        )
        self.tables['table2_bucket_regression_hy'] = self._table_bucket_regression(
            bucket_results_hy, 'HY'
        )
        print("  [1-2/17] Bucket regression results")

        # Tables 3-4: β/λ ratio summary
        self.tables['table3_beta_lambda_summary_ig'] = self._table_beta_lambda_summary(
            bucket_results_ig, 'IG'
        )
        self.tables['table4_beta_lambda_summary_hy'] = self._table_beta_lambda_summary(
            bucket_results_hy, 'HY'
        )
        print("  [3-4/17] β/λ ratio summary")

        # Tables 5-6: Monotonicity tests
        self.tables['table5_monotonicity_ig'] = self._table_monotonicity(
            bucket_results_ig, 'IG'
        )
        self.tables['table6_monotonicity_hy'] = self._table_monotonicity(
            bucket_results_hy, 'HY'
        )
        print("  [5-6/17] Monotonicity tests")

        # Tables 7-8: Within-issuer summary
        self.tables['table7_within_issuer_summary_ig'] = self._table_within_issuer_summary(
            within_issuer_results_ig, 'IG'
        )
        self.tables['table8_within_issuer_summary_hy'] = self._table_within_issuer_summary(
            within_issuer_results_hy, 'HY'
        )
        print("  [7-8/17] Within-issuer summary")

        # Tables 9-10: Within-issuer pooled estimates
        self.tables['table9_within_issuer_pooled_ig'] = self._table_within_issuer_pooled(
            within_issuer_results_ig, 'IG'
        )
        self.tables['table10_within_issuer_pooled_hy'] = self._table_within_issuer_pooled(
            within_issuer_results_hy, 'HY'
        )
        print("  [9-10/17] Within-issuer pooled estimates")

        # Tables 11-12: Sector regression
        self.tables['table11_sector_regression_ig'] = self._table_sector_regression(
            sector_results_ig, 'IG'
        )
        self.tables['table12_sector_regression_hy'] = self._table_sector_regression(
            sector_results_hy, 'HY'
        )
        print("  [11-12/17] Sector regression results")

        # Tables 13-14: Sector hypothesis tests
        self.tables['table13_sector_tests_ig'] = self._table_sector_tests(
            sector_results_ig, 'IG'
        )
        self.tables['table14_sector_tests_hy'] = self._table_sector_tests(
            sector_results_hy, 'HY'
        )
        print("  [13-14/17] Sector hypothesis tests")

        # Tables 15-16: Decision paths
        self.tables['table15_decision_path_ig'] = self._table_decision_path(
            synthesis_ig, 'IG'
        )
        self.tables['table16_decision_path_hy'] = self._table_decision_path(
            synthesis_hy, 'HY'
        )
        print("  [15-16/17] Decision paths")

        # Table 17: Cross-universe comparison
        self.tables['table17_comparison'] = self._table_comparison(
            bucket_results_ig, bucket_results_hy,
            within_issuer_results_ig, within_issuer_results_hy,
            sector_results_ig, sector_results_hy,
            synthesis_ig, synthesis_hy,
            comparison
        )
        print("  [17/17] Cross-universe comparison")

        # Save all tables
        self._save_all_tables()

        return self.tables

    def _table_bucket_regression(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 1-2: Bucket regression results - β vs λ^Merton by bucket."""
        bucket_df = results.get('bucket_results', pd.DataFrame())

        if len(bucket_df) == 0:
            return pd.DataFrame({'Message': [f'No data for {universe}']})

        # Select and rename columns for display
        display_cols = ['rating_bucket', 'maturity_bucket', 'median_spread', 'median_maturity',
                       'beta', 'beta_se', 'beta_pvalue', 'lambda_merton', 'beta_lambda_ratio',
                       'r_squared', 'n_obs']

        available_cols = [c for c in display_cols if c in bucket_df.columns]
        summary = bucket_df[available_cols].copy()

        # Rename for clarity
        rename_map = {
            'rating_bucket': 'Rating',
            'maturity_bucket': 'Maturity Bucket',
            'median_spread': 'Spread (bps)',
            'median_maturity': 'TTM (yrs)',
            'beta': 'β (Empirical)',
            'beta_se': 'β SE',
            'beta_pvalue': 'β p-value',
            'lambda_merton': 'λ^Merton',
            'beta_lambda_ratio': 'β/λ Ratio',
            'r_squared': 'R²',
            'n_obs': 'N Obs'
        }
        summary = summary.rename(columns={k: v for k, v in rename_map.items() if k in summary.columns})
        summary = summary.round(4)

        return summary

    def _table_beta_lambda_summary(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 3-4: Summary statistics of β/λ^Merton ratios."""
        summary = results.get('summary_statistics', {})

        if 'median_beta_lambda_ratio' not in summary:
            return pd.DataFrame({'Message': [f'No summary statistics for {universe}']})

        data = {
            'Statistic': [
                'Median β/λ Ratio',
                'Mean β/λ Ratio',
                'Std Dev β/λ Ratio',
                '% Buckets within ±10%',
                '% Buckets within ±20%',
                'N Buckets',
                'Median Empirical β',
                'Median Theoretical λ^Merton',
                '',
                'Interpretation'
            ],
            'Value': [
                f"{summary.get('median_beta_lambda_ratio', np.nan):.4f}",
                f"{summary.get('mean_beta_lambda_ratio', np.nan):.4f}",
                f"{summary.get('std_beta_lambda_ratio', np.nan):.4f}",
                f"{summary.get('pct_within_10pct', 0):.1f}%",
                f"{summary.get('pct_within_20pct', 0):.1f}%",
                f"{summary.get('n_buckets', 0)}",
                f"{summary.get('median_beta', np.nan):.4f}",
                f"{summary.get('median_lambda_merton', np.nan):.4f}",
                '',
                self._interpret_beta_lambda_ratio(summary.get('median_beta_lambda_ratio', np.nan))
            ]
        }

        return pd.DataFrame(data)

    def _interpret_beta_lambda_ratio(self, ratio: float) -> str:
        """Interpret β/λ ratio."""
        if np.isnan(ratio):
            return "Insufficient data"
        if 0.9 <= ratio <= 1.1:
            return f"β/λ = {ratio:.2f} ≈ 1: Merton prediction validated"
        elif ratio < 0.9:
            return f"β/λ = {ratio:.2f} < 1: Merton OVER-predicts sensitivity"
        else:
            return f"β/λ = {ratio:.2f} > 1: Merton UNDER-predicts sensitivity"

    def _table_monotonicity(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 5-6: Monotonicity test results - β should decrease with maturity."""
        monotonicity = results.get('monotonicity_test', {})
        details = monotonicity.get('details', [])

        if len(details) == 0:
            return pd.DataFrame({'Message': [f'No monotonicity results for {universe}']})

        df = pd.DataFrame(details)

        # Select relevant columns
        cols = ['rating_bucket', 'n_buckets', 'spearman_rho', 'p_value', 'is_monotonic_decreasing']
        if 'sector_group' in df.columns:
            cols.insert(1, 'sector_group')

        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].copy()

        # Rename columns
        rename_map = {
            'rating_bucket': 'Rating',
            'sector_group': 'Sector',
            'n_buckets': 'N Buckets',
            'spearman_rho': 'Spearman ρ',
            'p_value': 'p-value',
            'is_monotonic_decreasing': 'β Decreases?'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if 'β Decreases?' in df.columns:
            df['β Decreases?'] = df['β Decreases?'].apply(lambda x: 'Yes' if x else 'No')

        df = df.round(4)

        # Add summary row
        summary_row = pd.DataFrame({
            df.columns[0]: ['OVERALL'],
            **{col: ['-'] for col in df.columns[1:-1]},
            df.columns[-1]: [f"{monotonicity.get('pct_monotonic_groups', 0):.0f}% groups"]
        })

        df = pd.concat([df, summary_row], ignore_index=True)

        # Add interpretation
        interp_row = pd.DataFrame({
            df.columns[0]: ['Interpretation'],
            **{col: [''] for col in df.columns[1:-1]},
            df.columns[-1]: [monotonicity.get('interpretation', 'N/A')]
        })

        return pd.concat([df, interp_row], ignore_index=True)

    def _table_within_issuer_summary(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 7-8: Within-issuer summary statistics."""
        diagnostics = results.get('diagnostics', {})

        data = {
            'Metric': [
                'Bonds after filtering',
                'Unique issuers',
                'Unique weeks',
                'Total issuer-weeks',
                'Issuer-weeks with estimate',
                '% with estimate',
                'Mean bonds per issuer-week',
                'Mean maturity range (years)',
                'Mean R² (within-issuer)',
                '',
                'β Distribution:',
                '  Median β',
                '  Mean β',
                '  Std Dev β',
                '  % β in [0.8, 1.2]',
                '  % β positive'
            ],
            'Value': [
                f"{diagnostics.get('n_bonds_after_filter', 0):,}",
                f"{diagnostics.get('n_unique_issuers', 0):,}",
                f"{diagnostics.get('n_unique_weeks', 0):,}",
                f"{diagnostics.get('n_issuer_weeks_total', 0):,}",
                f"{diagnostics.get('n_issuer_weeks_with_estimate', 0):,}",
                f"{diagnostics.get('pct_issuer_weeks_with_estimate', 0):.1f}%",
                f"{diagnostics.get('mean_bonds_per_issuer_week', np.nan):.2f}",
                f"{diagnostics.get('mean_maturity_range', np.nan):.2f}",
                f"{diagnostics.get('mean_r_squared', np.nan):.4f}",
                '',
                '',
                f"{diagnostics.get('median_beta', np.nan):.4f}",
                f"{diagnostics.get('mean_beta', np.nan):.4f}",
                f"{diagnostics.get('std_beta', np.nan):.4f}",
                f"{diagnostics.get('pct_beta_in_0_8_1_2', 0):.1f}%",
                f"{diagnostics.get('pct_beta_positive', 0):.1f}%"
            ]
        }

        return pd.DataFrame(data)

    def _table_within_issuer_pooled(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 9-10: Within-issuer pooled estimates and hypothesis tests."""
        pooled = results.get('pooled_estimate', {})
        hypothesis = results.get('hypothesis_test', {})

        data = {
            'Statistic': [
                'Pooled β (inverse-variance weighted)',
                'Standard Error',
                '95% CI Lower',
                '95% CI Upper',
                'N Estimates',
                '',
                'Test 1: H0: β = 1 (Merton prediction)',
                '  t-statistic',
                '  p-value (two-sided)',
                '  Reject H0 (β ≠ 1)?',
                '',
                'Test 2: H0: β ≤ 0 (sanity check)',
                '  t-statistic',
                '  p-value (one-sided)',
                '',
                'β in [0.9, 1.1]?',
                'Merton Validated?',
                '',
                'Interpretation'
            ],
            'Value': [
                f"{pooled.get('pooled_beta', np.nan):.6f}",
                f"{pooled.get('pooled_beta_se', np.nan):.6f}",
                f"{pooled.get('ci_lower', np.nan):.6f}",
                f"{pooled.get('ci_upper', np.nan):.6f}",
                f"{pooled.get('n_estimates', 0)}",
                '',
                '',
                f"{hypothesis.get('t_statistic_beta_eq_1', np.nan):.4f}",
                f"{hypothesis.get('p_value_beta_equals_1', np.nan):.4f}",
                'Yes' if hypothesis.get('reject_beta_equals_1', False) else 'No',
                '',
                '',
                f"{hypothesis.get('t_statistic_beta_pos', np.nan):.4f}",
                f"{hypothesis.get('p_value_beta_positive', np.nan):.4f}",
                '',
                'Yes' if hypothesis.get('beta_in_range_0_9_1_1', False) else 'No',
                'Yes' if hypothesis.get('merton_validates', False) else 'No',
                '',
                hypothesis.get('interpretation', 'N/A')
            ]
        }

        return pd.DataFrame(data)

    def _table_sector_regression(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 11-12: Sector regression results."""
        base_reg = results.get('base_regression', {})
        sector_reg = results.get('sector_regression', {})

        if 'beta_0' not in sector_reg and 'beta_0' not in base_reg:
            return pd.DataFrame({'Message': [f'No sector regression for {universe}']})

        data = {
            'Coefficient': [
                'Base Regression (no sector interactions):',
                '  β₀ (Merton-scaled DTS)',
                '',
                'Sector Regression (with interactions):',
                '  β₀ (Base - Industrial)',
                '  β_Financial (Financial deviation)',
                '  β_Utility (Utility deviation)',
                '  β_Energy (Energy deviation)',
                '',
                'Total Sector Sensitivities:',
                '  Industrial (β₀)',
                '  Financial (β₀ + β_fin)',
                '  Utility (β₀ + β_util)',
                '  Energy (β₀ + β_energy)',
                '',
                'Model Statistics:',
                '  R²',
                '  N Observations',
                '  N Clusters'
            ],
            'Estimate': [
                '',
                f"{base_reg.get('beta_0', np.nan):.4f}",
                '',
                '',
                f"{sector_reg.get('beta_0', np.nan):.4f}",
                f"{sector_reg.get('beta_financial', np.nan):.4f}",
                f"{sector_reg.get('beta_utility', np.nan):.4f}",
                f"{sector_reg.get('beta_energy', np.nan):.4f}",
                '',
                '',
                f"{sector_reg.get('sensitivity_industrial', np.nan):.4f}",
                f"{sector_reg.get('sensitivity_financial', np.nan):.4f}",
                f"{sector_reg.get('sensitivity_utility', np.nan):.4f}",
                f"{sector_reg.get('sensitivity_energy', np.nan):.4f}",
                '',
                '',
                f"{sector_reg.get('r_squared', np.nan):.4f}",
                f"{sector_reg.get('n_obs', 0):,}",
                f"{sector_reg.get('n_clusters', 0):,}"
            ],
            'Std Error': [
                '',
                f"{base_reg.get('beta_0_se', np.nan):.4f}",
                '',
                '',
                f"{sector_reg.get('beta_0_se', np.nan):.4f}",
                f"{sector_reg.get('beta_financial_se', np.nan):.4f}",
                f"{sector_reg.get('beta_utility_se', np.nan):.4f}",
                f"{sector_reg.get('beta_energy_se', np.nan):.4f}",
                '',
                '',
                '-',
                '-',
                '-',
                '-',
                '',
                '',
                '-',
                '-',
                '-'
            ],
            'p-value': [
                '',
                f"{base_reg.get('beta_0_pvalue', np.nan):.4f}",
                '',
                '',
                f"{sector_reg.get('beta_0_pvalue', np.nan):.4f}",
                f"{sector_reg.get('beta_financial_pvalue', np.nan):.4f}",
                f"{sector_reg.get('beta_utility_pvalue', np.nan):.4f}",
                f"{sector_reg.get('beta_energy_pvalue', np.nan):.4f}",
                '',
                '',
                '-',
                '-',
                '-',
                '-',
                '',
                '',
                '-',
                '-',
                '-'
            ]
        }

        return pd.DataFrame(data)

    def _table_sector_tests(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 13-14: Sector hypothesis tests."""
        joint = results.get('joint_test', {})
        sector_tests = results.get('sector_tests', {})

        data = []

        # Joint test
        data.append({
            'Test': 'Joint F-test',
            'Hypothesis': joint.get('test', 'H0: β_fin = β_util = β_energy = 0'),
            'Statistic': f"F = {joint.get('f_statistic', np.nan):.4f}",
            'p-value': f"{joint.get('p_value', np.nan):.4f}",
            'Reject H0?': 'Yes' if joint.get('sectors_differ', False) else 'No',
            'Interpretation': joint.get('interpretation', 'N/A')
        })

        # Financial test
        if 'financial_test' in sector_tests:
            fin = sector_tests['financial_test']
            data.append({
                'Test': 'Financial sector',
                'Hypothesis': fin.get('hypothesis', 'H0: β_fin ≤ 0 vs H1: β_fin > 0'),
                'Statistic': f"t = {fin.get('t_statistic', np.nan):.4f}",
                'p-value': f"{fin.get('p_value', np.nan):.4f}",
                'Reject H0?': 'Yes' if fin.get('reject_null', False) else 'No',
                'Interpretation': fin.get('interpretation', 'N/A')[:60] + '...' if len(fin.get('interpretation', '')) > 60 else fin.get('interpretation', 'N/A')
            })

        # Utility test
        if 'utility_test' in sector_tests:
            util = sector_tests['utility_test']
            data.append({
                'Test': 'Utility sector',
                'Hypothesis': util.get('hypothesis', 'H0: β_util ≥ 0 vs H1: β_util < 0'),
                'Statistic': f"t = {util.get('t_statistic', np.nan):.4f}",
                'p-value': f"{util.get('p_value', np.nan):.4f}",
                'Reject H0?': 'Yes' if util.get('reject_null', False) else 'No',
                'Interpretation': util.get('interpretation', 'N/A')[:60] + '...' if len(util.get('interpretation', '')) > 60 else util.get('interpretation', 'N/A')
            })

        # Energy test
        if 'energy_test' in sector_tests:
            energy = sector_tests['energy_test']
            data.append({
                'Test': 'Energy sector',
                'Hypothesis': energy.get('hypothesis', 'H0: β_energy = 0'),
                'Statistic': f"t = {energy.get('t_statistic', np.nan):.4f}",
                'p-value': f"{energy.get('p_value', np.nan):.4f}",
                'Reject H0?': 'Yes' if energy.get('reject_null', False) else 'No',
                'Interpretation': energy.get('interpretation', 'N/A')[:60] + '...' if len(energy.get('interpretation', '')) > 60 else energy.get('interpretation', 'N/A')
            })

        # Summary
        if 'summary' in sector_tests:
            summary = sector_tests['summary']
            data.append({
                'Test': 'SUMMARY',
                'Hypothesis': '',
                'Statistic': '',
                'p-value': '',
                'Reject H0?': '',
                'Interpretation': 'Sector adjustments needed' if summary.get('need_sector_adjustment', False) else 'No sector adjustments needed'
            })

        return pd.DataFrame(data)

    def _table_decision_path(self, synthesis: Dict, universe: str) -> pd.DataFrame:
        """Table 15-16: Decision path summary."""
        path = synthesis.get('decision_path', 0)
        path_name = synthesis.get('path_name', 'Unknown')
        rationale = synthesis.get('rationale', 'N/A')
        criteria = synthesis.get('decision_criteria', {})
        key_stats = synthesis.get('key_statistics', {})
        recommendations = synthesis.get('recommendations', {})

        data = {
            'Item': [
                'DECISION',
                'Path Number',
                'Path Name',
                'Rationale',
                '',
                'KEY STATISTICS (β should be ≈ 1):',
                '  Bucket median β/λ ratio',
                '  Within-issuer pooled β',
                '  Sector base β₀',
                '',
                'DECISION CRITERIA:',
                '  β ≈ 1 at bucket level?',
                '  β ≈ 1 at within-issuer level?',
                '  Monotonic (β ↓ with maturity)?',
                '  Sectors matter?',
                '  Consistent across analyses?',
                '  Theory validated?',
                '',
                'RECOMMENDATIONS:',
                '  Stage A',
                '  Stage B',
                '  Stage C',
                '  Stage D',
                '  Stage E'
            ],
            'Value': [
                '',
                f"Path {path}",
                path_name,
                rationale[:80] + '...' if len(rationale) > 80 else rationale,
                '',
                '',
                f"{key_stats.get('bucket_median_ratio', np.nan):.3f}",
                f"{key_stats.get('within_beta', np.nan):.3f}",
                f"{key_stats.get('base_beta', np.nan):.3f}",
                '',
                '',
                'Yes' if criteria.get('bucket_beta_near_1', False) else 'No',
                'Yes' if criteria.get('within_beta_near_1', False) else 'No',
                'Yes' if criteria.get('monotonic', False) else 'No',
                'Yes' if criteria.get('sectors_matter', False) else 'No',
                'Yes' if criteria.get('consistent', False) else 'No',
                'Yes' if criteria.get('theory_validated', False) else 'No',
                '',
                '',
                recommendations.get('stage_A', 'N/A')[:60] + '...' if len(recommendations.get('stage_A', '')) > 60 else recommendations.get('stage_A', 'N/A'),
                recommendations.get('stage_B', 'N/A')[:60] + '...' if len(recommendations.get('stage_B', '')) > 60 else recommendations.get('stage_B', 'N/A'),
                recommendations.get('stage_C', 'N/A')[:60] + '...' if len(recommendations.get('stage_C', '')) > 60 else recommendations.get('stage_C', 'N/A'),
                recommendations.get('stage_D', 'N/A')[:60] + '...' if len(recommendations.get('stage_D', '')) > 60 else recommendations.get('stage_D', 'N/A'),
                recommendations.get('stage_E', 'N/A')[:60] + '...' if len(recommendations.get('stage_E', '')) > 60 else recommendations.get('stage_E', 'N/A')
            ]
        }

        return pd.DataFrame(data)

    def _table_comparison(
        self,
        bucket_results_ig: Dict, bucket_results_hy: Dict,
        within_issuer_results_ig: Dict, within_issuer_results_hy: Dict,
        sector_results_ig: Dict, sector_results_hy: Dict,
        synthesis_ig: Dict, synthesis_hy: Dict,
        comparison: Dict
    ) -> pd.DataFrame:
        """Table 17: Cross-universe comparison."""
        data = []

        # β estimates across analyses (should be ≈ 1)
        data.append({
            'Category': 'β ESTIMATES (should be ≈ 1)',
            'IG': '',
            'HY': '',
            'Same Path?': '',
            'Notes': ''
        })

        ig_bucket_ratio = bucket_results_ig.get('summary_statistics', {}).get('median_beta_lambda_ratio', np.nan)
        hy_bucket_ratio = bucket_results_hy.get('summary_statistics', {}).get('median_beta_lambda_ratio', np.nan)
        data.append({
            'Category': '  Bucket β/λ ratio',
            'IG': f"{ig_bucket_ratio:.3f}",
            'HY': f"{hy_bucket_ratio:.3f}",
            'Same Path?': '-',
            'Notes': '✓' if 0.8 <= ig_bucket_ratio <= 1.2 and 0.8 <= hy_bucket_ratio <= 1.2 else '✗'
        })

        ig_within_beta = within_issuer_results_ig.get('pooled_estimate', {}).get('pooled_beta', np.nan)
        hy_within_beta = within_issuer_results_hy.get('pooled_estimate', {}).get('pooled_beta', np.nan)
        data.append({
            'Category': '  Within-Issuer β',
            'IG': f"{ig_within_beta:.3f}",
            'HY': f"{hy_within_beta:.3f}",
            'Same Path?': '-',
            'Notes': '✓' if 0.8 <= ig_within_beta <= 1.2 and 0.8 <= hy_within_beta <= 1.2 else '✗'
        })

        ig_base_beta = sector_results_ig.get('base_regression', {}).get('beta_0', np.nan)
        hy_base_beta = sector_results_hy.get('base_regression', {}).get('beta_0', np.nan)
        data.append({
            'Category': '  Sector base β₀',
            'IG': f"{ig_base_beta:.3f}",
            'HY': f"{hy_base_beta:.3f}",
            'Same Path?': '-',
            'Notes': '✓' if 0.8 <= ig_base_beta <= 1.2 and 0.8 <= hy_base_beta <= 1.2 else '✗'
        })

        # Blank row
        data.append({
            'Category': '',
            'IG': '',
            'HY': '',
            'Same Path?': '',
            'Notes': ''
        })

        # Decision paths
        ig_path = synthesis_ig.get('decision_path', 0)
        hy_path = synthesis_hy.get('decision_path', 0)

        data.append({
            'Category': 'DECISION PATHS',
            'IG': f"Path {ig_path}: {synthesis_ig.get('path_name', 'Unknown')[:30]}",
            'HY': f"Path {hy_path}: {synthesis_hy.get('path_name', 'Unknown')[:30]}",
            'Same Path?': 'Yes' if ig_path == hy_path else 'No',
            'Notes': ''
        })

        # Blank row
        data.append({
            'Category': '',
            'IG': '',
            'HY': '',
            'Same Path?': '',
            'Notes': ''
        })

        # Unified approach
        data.append({
            'Category': 'UNIFIED APPROACH',
            'IG': comparison.get('unified_approach', 'N/A'),
            'HY': '',
            'Same Path?': '',
            'Notes': ''
        })

        # Interpretation
        data.append({
            'Category': 'INTERPRETATION',
            'IG': comparison.get('interpretation', 'N/A')[:60] + '...' if len(comparison.get('interpretation', '')) > 60 else comparison.get('interpretation', 'N/A'),
            'HY': '',
            'Same Path?': '',
            'Notes': ''
        })

        return pd.DataFrame(data)

    def _save_all_tables(self):
        """Save all tables to CSV and create summary document."""
        for name, df in self.tables.items():
            df.to_csv(self.output_dir / f'{name}.csv', index=False)

        # Create summary text file
        summary_path = self.output_dir / 'TABLES_SUMMARY.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 0 REPORTING SUMMARY (EVOLVED VERSION)\n")
            f.write("Testing β ≈ 1 (Merton Prediction)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total tables: {len(self.tables)}\n\n")

            for name in sorted(self.tables.keys()):
                f.write(f"\n{name}:\n")
                f.write("-" * 80 + "\n")
                f.write(self.tables[name].to_string(index=False))
                f.write("\n")

        print(f"\nAll tables saved to: {self.output_dir}")
        print(f"Summary document: {summary_path}")


def generate_stage0_report(
    bucket_results_ig: Dict,
    bucket_results_hy: Dict,
    within_issuer_results_ig: Dict,
    within_issuer_results_hy: Dict,
    sector_results_ig: Dict,
    sector_results_hy: Dict,
    synthesis_ig: Dict,
    synthesis_hy: Dict,
    comparison: Dict,
    output_dir: str = "output/stage0_tables"
) -> Stage0Reporting:
    """
    Convenience function to generate all Stage 0 tables.

    Returns:
        Stage0Reporting instance with all tables
    """
    reporter = Stage0Reporting(output_dir=output_dir)

    reporter.generate_all_tables(
        bucket_results_ig, bucket_results_hy,
        within_issuer_results_ig, within_issuer_results_hy,
        sector_results_ig, sector_results_hy,
        synthesis_ig, synthesis_hy,
        comparison
    )

    return reporter
