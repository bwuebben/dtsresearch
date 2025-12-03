"""
Stage 0: Reporting Module

Creates 17 tables for Stage 0 analysis:
1-2. Bucket characteristics summary (IG, HY)
3-4. Bucket regression results (IG, HY)
5-6. Monotonicity test results (IG, HY)
7-8. Within-issuer summary statistics (IG, HY)
9-10. Within-issuer pooled estimates (IG, HY)
11-12. Sector regression results (IG, HY)
13-14. Sector hypothesis tests (IG, HY)
15-16. Decision path summary (IG, HY)
17. Cross-universe comparison and recommendations

Based on reporting requirements from the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


class Stage0Reporting:
    """
    Reporting utilities for Stage 0 analysis.
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
        Generate all 17 Stage 0 tables.

        Args:
            Results dictionaries from all analyses

        Returns:
            Dictionary mapping table names to DataFrames
        """
        print("Generating Stage 0 tables...")

        # Tables 1-2: Bucket characteristics
        self.tables['table1_bucket_chars_ig'] = self._table_bucket_characteristics(
            bucket_results_ig, 'IG'
        )
        self.tables['table2_bucket_chars_hy'] = self._table_bucket_characteristics(
            bucket_results_hy, 'HY'
        )
        print("  [1-2/17] Bucket characteristics")

        # Tables 3-4: Bucket regression
        self.tables['table3_bucket_regression_ig'] = self._table_bucket_regression(
            bucket_results_ig, 'IG'
        )
        self.tables['table4_bucket_regression_hy'] = self._table_bucket_regression(
            bucket_results_hy, 'HY'
        )
        print("  [3-4/17] Bucket regression results")

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

    def _table_bucket_characteristics(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 1-2: Bucket characteristics summary."""
        bucket_chars = results.get('bucket_characteristics', pd.DataFrame())

        if len(bucket_chars) == 0:
            return pd.DataFrame({'Message': [f'No data for {universe}']})

        summary = bucket_chars.groupby(['rating_bucket', 'maturity_bucket']).agg({
            'n_observations': 'sum',
            's_bar': 'mean',
            'T_bar': 'mean'
        }).reset_index()

        summary.columns = ['Rating', 'Maturity', 'N Observations', 'Avg Spread (bps)', 'Avg TTM (yrs)']
        summary = summary.round(2)

        return summary

    def _table_bucket_regression(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 3-4: Bucket regression results."""
        reg = results.get('regression_results', {})

        if 'lambda' not in reg:
            return pd.DataFrame({'Message': [f'No regression results for {universe}']})

        data = {
            'Statistic': ['λ (Maturity effect)', 'α (Intercept)', 'R²', 'R² (adjusted)',
                         'N Buckets', 'Total Observations'],
            'Estimate': [
                f"{reg.get('lambda', np.nan):.6f}",
                f"{reg.get('alpha', np.nan):.4f}",
                f"{reg.get('r_squared', np.nan):.4f}",
                f"{reg.get('r_squared_adj', np.nan):.4f}",
                f"{reg.get('n_buckets', 0)}",
                f"{reg.get('total_observations', 0)}"
            ],
            'Std Error': [
                f"{reg.get('lambda_se', np.nan):.6f}",
                f"{reg.get('alpha_se', np.nan):.4f}",
                '-',
                '-',
                '-',
                '-'
            ],
            't-statistic': [
                f"{reg.get('lambda_tstat', np.nan):.4f}",
                '-',
                '-',
                '-',
                '-',
                '-'
            ],
            'p-value': [
                f"{reg.get('lambda_pvalue', np.nan):.4f}",
                '-',
                '-',
                '-',
                '-',
                '-'
            ]
        }

        return pd.DataFrame(data)

    def _table_monotonicity(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 5-6: Monotonicity test results."""
        monotonicity = results.get('monotonicity_test', {})
        details = monotonicity.get('details', [])

        if len(details) == 0:
            return pd.DataFrame({'Message': [f'No monotonicity results for {universe}']})

        df = pd.DataFrame(details)
        df = df[['rating_bucket', 'sector_group', 'n_buckets', 'spearman_rho', 'p_value', 'is_monotonic']]
        df.columns = ['Rating', 'Sector Group', 'N Buckets', 'Spearman ρ', 'p-value', 'Monotonic?']
        df['Monotonic?'] = df['Monotonic?'].apply(lambda x: 'Yes' if x else 'No')
        df = df.round(4)

        # Add summary row
        summary_row = pd.DataFrame({
            'Rating': ['OVERALL'],
            'Sector Group': ['-'],
            'N Buckets': [df['N Buckets'].sum()],
            'Spearman ρ': ['-'],
            'p-value': ['-'],
            'Monotonic?': [f"{monotonicity.get('pct_monotonic_groups', 0):.0f}% groups"]
        })

        return pd.concat([df, summary_row], ignore_index=True)

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
                '% λ significantly positive',
                '% λ significantly negative'
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
                f"{diagnostics.get('pct_significant_positive', 0):.1f}%",
                f"{diagnostics.get('pct_significant_negative', 0):.1f}%"
            ]
        }

        return pd.DataFrame(data)

    def _table_within_issuer_pooled(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 9-10: Within-issuer pooled estimates."""
        pooled = results.get('pooled_estimate', {})
        hypothesis = results.get('hypothesis_test', {})

        data = {
            'Statistic': [
                'Pooled λ',
                'Standard Error',
                '95% CI Lower',
                '95% CI Upper',
                'N Estimates',
                'Test: H0: λ ≤ 0',
                't-statistic',
                'p-value (one-sided)',
                'Reject H0?',
                'Interpretation'
            ],
            'Value': [
                f"{pooled.get('pooled_estimate', np.nan):.6f}",
                f"{pooled.get('pooled_se', np.nan):.6f}",
                f"{pooled.get('ci_lower', np.nan):.6f}",
                f"{pooled.get('ci_upper', np.nan):.6f}",
                f"{pooled.get('n_estimates', 0)}",
                hypothesis.get('test', 'N/A'),
                f"{hypothesis.get('t_statistic', np.nan):.4f}",
                f"{hypothesis.get('p_value', np.nan):.4f}",
                'Yes' if hypothesis.get('reject_null', False) else 'No',
                hypothesis.get('interpretation', 'N/A')
            ]
        }

        return pd.DataFrame(data)

    def _table_sector_regression(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 11-12: Sector regression results."""
        reg = results.get('sector_regression', {})

        if 'lambda' not in reg:
            return pd.DataFrame({'Message': [f'No sector regression for {universe}']})

        data = {
            'Coefficient': [
                'α (Intercept)',
                'λ (Maturity - Industrial)',
                'β_Financial (Financial interaction)',
                'β_Utility (Utility interaction)',
                'β_Energy (Energy interaction)'
            ],
            'Estimate': [
                f"{reg.get('alpha', np.nan):.6f}",
                f"{reg.get('lambda', np.nan):.6f}",
                f"{reg.get('beta_financial', np.nan):.6f}",
                f"{reg.get('beta_utility', np.nan):.6f}",
                f"{reg.get('beta_energy', np.nan):.6f}"
            ],
            'Std Error': [
                '-',
                f"{reg.get('lambda_se', np.nan):.6f}",
                f"{reg.get('beta_financial_se', np.nan):.6f}",
                f"{reg.get('beta_utility_se', np.nan):.6f}",
                f"{reg.get('beta_energy_se', np.nan):.6f}"
            ],
            't-statistic': [
                '-',
                f"{reg.get('lambda_tstat', np.nan):.4f}",
                f"{reg.get('beta_financial_tstat', np.nan):.4f}",
                f"{reg.get('beta_utility_tstat', np.nan):.4f}",
                f"{reg.get('beta_energy_tstat', np.nan):.4f}"
            ],
            'p-value': [
                '-',
                f"{reg.get('lambda_pvalue', np.nan):.4f}",
                f"{reg.get('beta_financial_pvalue', np.nan):.4f}",
                f"{reg.get('beta_utility_pvalue', np.nan):.4f}",
                f"{reg.get('beta_energy_pvalue', np.nan):.4f}"
            ]
        }

        df = pd.DataFrame(data)

        # Add summary stats
        summary = pd.DataFrame({
            'Coefficient': ['R²', 'N Observations', 'N Clusters'],
            'Estimate': [
                f"{reg.get('r_squared', np.nan):.4f}",
                f"{reg.get('n_obs', 0):,}",
                f"{reg.get('n_clusters', 0):,}"
            ],
            'Std Error': ['-', '-', '-'],
            't-statistic': ['-', '-', '-'],
            'p-value': ['-', '-', '-']
        })

        return pd.concat([df, summary], ignore_index=True)

    def _table_sector_tests(self, results: Dict, universe: str) -> pd.DataFrame:
        """Table 13-14: Sector hypothesis tests."""
        joint = results.get('joint_test', {})
        sector_tests = results.get('sector_tests', {})

        data = []

        # Joint test
        data.append({
            'Test': 'Joint F-test',
            'Hypothesis': joint.get('test', 'N/A'),
            'Statistic': f"{joint.get('f_statistic', np.nan):.4f}",
            'p-value': f"{joint.get('p_value', np.nan):.4f}",
            'Reject H0?': 'Yes' if joint.get('reject_null', False) else 'No',
            'Interpretation': joint.get('interpretation', 'N/A')
        })

        # Financial test
        if 'financial_test' in sector_tests:
            fin = sector_tests['financial_test']
            data.append({
                'Test': 'Financial sector',
                'Hypothesis': 'H0: β_fin ≤ 0 vs H1: β_fin > 0',
                'Statistic': f"{fin.get('t_statistic', np.nan):.4f}",
                'p-value': f"{fin.get('p_value', np.nan):.4f}",
                'Reject H0?': 'Yes' if fin.get('reject_null', False) else 'No',
                'Interpretation': fin.get('interpretation', 'N/A')
            })

        # Utility test
        if 'utility_test' in sector_tests:
            util = sector_tests['utility_test']
            data.append({
                'Test': 'Utility sector',
                'Hypothesis': 'H0: β_util ≥ 0 vs H1: β_util < 0',
                'Statistic': f"{util.get('t_statistic', np.nan):.4f}",
                'p-value': f"{util.get('p_value', np.nan):.4f}",
                'Reject H0?': 'Yes' if util.get('reject_null', False) else 'No',
                'Interpretation': util.get('interpretation', 'N/A')
            })

        return pd.DataFrame(data)

    def _table_decision_path(self, synthesis: Dict, universe: str) -> pd.DataFrame:
        """Table 15-16: Decision path summary."""
        path = synthesis.get('decision_path', 0)
        path_name = synthesis.get('path_name', 'Unknown')
        rationale = synthesis.get('rationale', 'N/A')
        criteria = synthesis.get('decision_criteria', {})
        recommendations = synthesis.get('recommendations', {})

        data = {
            'Item': [
                'Decision Path',
                'Path Name',
                'Rationale',
                '',
                'Decision Criteria:',
                '  λ > 0?',
                '  Monotonic?',
                '  Consistent λ?',
                '  Sectors matter?',
                '',
                'Recommendations:',
                '  Stage A',
                '  Stage B',
                '  Stage C',
                '  Stage D',
                '  Stage E'
            ],
            'Value': [
                f"Path {path}",
                path_name,
                rationale,
                '',
                '',
                'Yes' if criteria.get('lambda_positive', False) else 'No',
                'Yes' if criteria.get('monotonic', False) else 'No',
                'Yes' if criteria.get('consistent_lambda', False) else 'No',
                'Yes' if criteria.get('sectors_matter', False) else 'No',
                '',
                '',
                recommendations.get('stage_A', 'N/A'),
                recommendations.get('stage_B', 'N/A'),
                recommendations.get('stage_C', 'N/A'),
                recommendations.get('stage_D', 'N/A'),
                recommendations.get('stage_E', 'N/A')
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

        # λ estimates
        data.append({
            'Category': 'Bucket λ',
            'IG': f"{bucket_results_ig.get('regression_results', {}).get('lambda', np.nan):.6f}",
            'HY': f"{bucket_results_hy.get('regression_results', {}).get('lambda', np.nan):.6f}",
            'Difference': '-',
            'Significant?': '-'
        })

        data.append({
            'Category': 'Within-Issuer λ',
            'IG': f"{within_issuer_results_ig.get('pooled_estimate', {}).get('pooled_estimate', np.nan):.6f}",
            'HY': f"{within_issuer_results_hy.get('pooled_estimate', {}).get('pooled_estimate', np.nan):.6f}",
            'Difference': '-',
            'Significant?': '-'
        })

        data.append({
            'Category': 'Sector (Base) λ',
            'IG': f"{sector_results_ig.get('base_regression', {}).get('lambda', np.nan):.6f}",
            'HY': f"{sector_results_hy.get('base_regression', {}).get('lambda', np.nan):.6f}",
            'Difference': '-',
            'Significant?': '-'
        })

        # Decision paths
        data.append({
            'Category': '',
            'IG': '',
            'HY': '',
            'Difference': '',
            'Significant?': ''
        })

        ig_path = synthesis_ig.get('decision_path', 0)
        hy_path = synthesis_hy.get('decision_path', 0)

        data.append({
            'Category': 'Decision Path',
            'IG': f"Path {ig_path}: {synthesis_ig.get('path_name', 'Unknown')}",
            'HY': f"Path {hy_path}: {synthesis_hy.get('path_name', 'Unknown')}",
            'Difference': 'Same' if ig_path == hy_path else 'Different',
            'Significant?': '-'
        })

        # Unified approach
        data.append({
            'Category': '',
            'IG': '',
            'HY': '',
            'Difference': '',
            'Significant?': ''
        })

        data.append({
            'Category': 'Unified Approach',
            'IG': comparison.get('unified_approach', 'N/A'),
            'HY': '',
            'Difference': '',
            'Significant?': ''
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
            f.write("STAGE 0 REPORTING SUMMARY\n")
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
