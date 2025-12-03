"""
Stage 0: Visualization Module

Creates 10 figures for Stage 0 analysis:
1. Bucket characteristics scatter (s̄ vs T̄)
2. Cross-sectional regression fit
3. Maturity monotonicity by rating-sector
4. Within-issuer λ distribution
5. Within-issuer λ time series
6. Sector interaction coefficients
7. Sector-specific λ comparison
8. Decision path comparison (IG vs HY)
9. λ estimates across all three analyses
10. Diagnostic summary dashboard

Based on visualization requirements from the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class Stage0Plots:
    """
    Visualization utilities for Stage 0 analysis.
    """

    def __init__(self, output_dir: str = "output/stage0_figures"):
        """
        Initialize plotting module.

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette
        self.colors = {
            'IG': '#2E86AB',  # Blue
            'HY': '#A23B72',  # Purple
            'Industrial': '#06A77D',  # Green
            'Financial': '#F18F01',  # Orange
            'Utility': '#C73E1D',  # Red
            'Energy': '#6C757D'  # Gray
        }

    def plot_all_figures(
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
    ):
        """
        Generate all 10 Stage 0 figures.

        Args:
            bucket_results_ig: Bucket analysis for IG
            bucket_results_hy: Bucket analysis for HY
            within_issuer_results_ig: Within-issuer for IG
            within_issuer_results_hy: Within-issuer for HY
            sector_results_ig: Sector analysis for IG
            sector_results_hy: Sector analysis for HY
            synthesis_ig: Synthesis for IG
            synthesis_hy: Synthesis for HY
            comparison: IG vs HY comparison
        """
        print("Generating Stage 0 visualizations...")

        # Figure 1: Bucket characteristics
        self.plot_bucket_characteristics(bucket_results_ig, bucket_results_hy)
        print("  [1/10] Bucket characteristics scatter")

        # Figure 2: Cross-sectional regression
        self.plot_cross_sectional_regression(bucket_results_ig, bucket_results_hy)
        print("  [2/10] Cross-sectional regression fit")

        # Figure 3: Maturity monotonicity
        self.plot_maturity_monotonicity(bucket_results_ig, bucket_results_hy)
        print("  [3/10] Maturity monotonicity")

        # Figure 4: Within-issuer λ distribution
        self.plot_within_issuer_distribution(within_issuer_results_ig, within_issuer_results_hy)
        print("  [4/10] Within-issuer λ distribution")

        # Figure 5: Within-issuer λ time series
        self.plot_within_issuer_time_series(within_issuer_results_ig, within_issuer_results_hy)
        print("  [5/10] Within-issuer λ time series")

        # Figure 6: Sector interaction coefficients
        self.plot_sector_coefficients(sector_results_ig, sector_results_hy)
        print("  [6/10] Sector interaction coefficients")

        # Figure 7: Sector-specific λ comparison
        self.plot_sector_lambda_comparison(sector_results_ig, sector_results_hy)
        print("  [7/10] Sector-specific λ comparison")

        # Figure 8: Decision path comparison
        self.plot_decision_paths(synthesis_ig, synthesis_hy, comparison)
        print("  [8/10] Decision path comparison")

        # Figure 9: λ estimates across analyses
        self.plot_lambda_estimates_comparison(
            bucket_results_ig, bucket_results_hy,
            within_issuer_results_ig, within_issuer_results_hy,
            sector_results_ig, sector_results_hy
        )
        print("  [9/10] λ estimates comparison")

        # Figure 10: Diagnostic dashboard
        self.plot_diagnostic_dashboard(
            bucket_results_ig, bucket_results_hy,
            within_issuer_results_ig, within_issuer_results_hy,
            sector_results_ig, sector_results_hy
        )
        print("  [10/10] Diagnostic dashboard")

        print(f"\nAll figures saved to: {self.output_dir}")

    def plot_bucket_characteristics(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict
    ):
        """
        Figure 1: Scatter plot of bucket characteristics (s̄ vs T̄).
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, universe) in enumerate([(bucket_results_ig, 'IG'), (bucket_results_hy, 'HY')]):
            ax = axes[idx]
            bucket_chars = results.get('bucket_characteristics', pd.DataFrame())

            if len(bucket_chars) > 0:
                ax.scatter(
                    bucket_chars['T_bar'],
                    bucket_chars['s_bar'],
                    s=bucket_chars['n_observations'] / 10,  # Size by observations
                    alpha=0.6,
                    c=self.colors[universe],
                    edgecolors='black',
                    linewidth=0.5
                )

                ax.set_xlabel('Maturity (years)')
                ax.set_ylabel('Spread (bps)')
                ax.set_title(f'{universe}: Bucket Characteristics (s̄ vs T̄)')
                ax.grid(True, alpha=0.3)

                # Add text annotation
                n_buckets = len(bucket_chars)
                ax.text(0.05, 0.95, f'n = {n_buckets} buckets',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_bucket_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cross_sectional_regression(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict
    ):
        """
        Figure 2: Cross-sectional regression fit ln(s̄) vs T̄.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, universe) in enumerate([(bucket_results_ig, 'IG'), (bucket_results_hy, 'HY')]):
            ax = axes[idx]
            bucket_chars = results.get('bucket_characteristics', pd.DataFrame())
            reg_results = results.get('regression_results', {})

            if len(bucket_chars) > 0 and 'lambda' in reg_results:
                # Actual data
                T = bucket_chars['T_bar'].values
                s = bucket_chars['s_bar'].values
                log_s = np.log(s)

                ax.scatter(T, log_s, alpha=0.6, c=self.colors[universe],
                          edgecolors='black', linewidth=0.5, s=50, label='Data')

                # Fitted line
                lambda_est = reg_results['lambda']
                alpha_est = reg_results['alpha']
                T_line = np.linspace(T.min(), T.max(), 100)
                log_s_fit = alpha_est + lambda_est * T_line

                ax.plot(T_line, log_s_fit, 'r--', linewidth=2, label='Fitted')

                ax.set_xlabel('Maturity (years)')
                ax.set_ylabel('ln(Spread)')
                ax.set_title(f'{universe}: ln(s̄) = α + λ·T̄')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add regression statistics
                r2 = reg_results.get('r_squared', np.nan)
                pval = reg_results.get('lambda_pvalue', np.nan)
                text = f'λ = {lambda_est:.4f}\nR² = {r2:.3f}\np = {pval:.4f}'
                ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_cross_sectional_regression.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_maturity_monotonicity(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict
    ):
        """
        Figure 3: Monotonicity test results by rating-sector group.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, universe) in enumerate([(bucket_results_ig, 'IG'), (bucket_results_hy, 'HY')]):
            ax = axes[idx]
            monotonicity = results.get('monotonicity_test', {})
            details = monotonicity.get('details', [])

            if len(details) > 0:
                df = pd.DataFrame(details)
                df['group_label'] = df['rating_bucket'] + '_' + df['sector_group']

                # Bar plot of Spearman correlation
                colors = ['green' if sig else 'red' for sig in df['is_monotonic']]
                ax.barh(df['group_label'], df['spearman_rho'], color=colors, alpha=0.7)

                ax.set_xlabel('Spearman ρ')
                ax.set_ylabel('Rating-Sector Group')
                ax.set_title(f'{universe}: Monotonicity Test (green = significant)')
                ax.axvline(0, color='black', linestyle='--', linewidth=1)
                ax.grid(True, alpha=0.3, axis='x')

                # Add overall percentage
                pct_mono = monotonicity.get('pct_monotonic_groups', 0)
                ax.text(0.95, 0.95, f'{pct_mono:.0f}% monotonic',
                       transform=ax.transAxes, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_maturity_monotonicity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_within_issuer_distribution(
        self,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict
    ):
        """
        Figure 4: Distribution of within-issuer λ estimates.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, universe) in enumerate([(within_issuer_results_ig, 'IG'), (within_issuer_results_hy, 'HY')]):
            ax = axes[idx]
            estimates = results.get('issuer_week_estimates', pd.DataFrame())

            if len(estimates) > 0 and 'lambda' in estimates.columns:
                lambdas = estimates['lambda'].values

                # Histogram
                ax.hist(lambdas, bins=50, alpha=0.7, color=self.colors[universe],
                       edgecolor='black', linewidth=0.5)

                # Add pooled estimate line
                pooled = results.get('pooled_estimate', {}).get('pooled_estimate', np.nan)
                if not np.isnan(pooled):
                    ax.axvline(pooled, color='red', linestyle='--', linewidth=2,
                              label=f'Pooled: {pooled:.4f}')

                ax.set_xlabel('λ (within-issuer)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{universe}: Distribution of Within-Issuer λ')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

                # Add statistics
                mean_lambda = lambdas.mean()
                std_lambda = lambdas.std()
                text = f'Mean: {mean_lambda:.4f}\nStd: {std_lambda:.4f}\nn = {len(lambdas)}'
                ax.text(0.95, 0.95, text, transform=ax.transAxes, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_within_issuer_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_within_issuer_time_series(
        self,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict
    ):
        """
        Figure 5: Time series of within-issuer λ estimates.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        for idx, (results, universe) in enumerate([(within_issuer_results_ig, 'IG'), (within_issuer_results_hy, 'HY')]):
            ax = axes[idx]
            estimates = results.get('issuer_week_estimates', pd.DataFrame())

            if len(estimates) > 0 and 'lambda' in estimates.columns and 'date' in estimates.columns:
                # Aggregate by date (mean and confidence interval)
                weekly = estimates.groupby('date')['lambda'].agg(['mean', 'std', 'count']).reset_index()
                weekly['se'] = weekly['std'] / np.sqrt(weekly['count'])
                weekly['ci_lower'] = weekly['mean'] - 1.96 * weekly['se']
                weekly['ci_upper'] = weekly['mean'] + 1.96 * weekly['se']

                ax.plot(weekly['date'], weekly['mean'], color=self.colors[universe],
                       linewidth=2, label='Mean λ')
                ax.fill_between(weekly['date'], weekly['ci_lower'], weekly['ci_upper'],
                               color=self.colors[universe], alpha=0.3, label='95% CI')

                # Add pooled estimate
                pooled = results.get('pooled_estimate', {}).get('pooled_estimate', np.nan)
                if not np.isnan(pooled):
                    ax.axhline(pooled, color='red', linestyle='--', linewidth=2,
                              label=f'Pooled: {pooled:.4f}')

                ax.set_xlabel('Date')
                ax.set_ylabel('λ (within-issuer)')
                ax.set_title(f'{universe}: Within-Issuer λ Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_within_issuer_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sector_coefficients(
        self,
        sector_results_ig: Dict,
        sector_results_hy: Dict
    ):
        """
        Figure 6: Sector interaction coefficients with confidence intervals.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sectors = ['Financial', 'Utility', 'Energy']
        beta_keys = ['beta_financial', 'beta_utility', 'beta_energy']
        se_keys = ['beta_financial_se', 'beta_utility_se', 'beta_energy_se']

        for idx, (results, universe) in enumerate([(sector_results_ig, 'IG'), (sector_results_hy, 'HY')]):
            ax = axes[idx]
            reg = results.get('sector_regression', {})

            if all(k in reg for k in beta_keys):
                betas = [reg[k] for k in beta_keys]
                ses = [reg[k] for k in se_keys]
                ci_lower = [b - 1.96*se for b, se in zip(betas, ses)]
                ci_upper = [b + 1.96*se for b, se in zip(betas, ses)]

                y_pos = np.arange(len(sectors))
                colors_list = [self.colors[s] for s in sectors]

                ax.barh(y_pos, betas, xerr=None, color=colors_list, alpha=0.7,
                       edgecolor='black', linewidth=0.5)

                # Add confidence intervals
                for i, (b, cl, cu) in enumerate(zip(betas, ci_lower, ci_upper)):
                    ax.plot([cl, cu], [i, i], 'k-', linewidth=2)
                    ax.plot([cl, cl], [i-0.1, i+0.1], 'k-', linewidth=2)
                    ax.plot([cu, cu], [i-0.1, i+0.1], 'k-', linewidth=2)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(sectors)
                ax.set_xlabel('Sector Interaction Coefficient (β)')
                ax.set_title(f'{universe}: Sector Interactions')
                ax.axvline(0, color='black', linestyle='--', linewidth=1)
                ax.grid(True, alpha=0.3, axis='x')

                # Add joint test result
                joint = results.get('joint_test', {})
                pval = joint.get('p_value', np.nan)
                sig = "Significant" if joint.get('reject_null', False) else "Not Significant"
                text = f'Joint test: {sig}\np = {pval:.4f}'
                ax.text(0.95, 0.05, text, transform=ax.transAxes, va='bottom', ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_sector_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sector_lambda_comparison(
        self,
        sector_results_ig: Dict,
        sector_results_hy: Dict
    ):
        """
        Figure 7: Sector-specific λ comparison (base + interactions).
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        sectors = ['Industrial', 'Financial', 'Utility', 'Energy']
        universes = ['IG', 'HY']
        results_list = [sector_results_ig, sector_results_hy]

        x = np.arange(len(sectors))
        width = 0.35

        for idx, (results, universe) in enumerate(zip(results_list, universes)):
            reg = results.get('sector_regression', {})

            if 'lambda' in reg:
                base_lambda = reg['lambda']
                beta_fin = reg.get('beta_financial', 0)
                beta_util = reg.get('beta_utility', 0)
                beta_energy = reg.get('beta_energy', 0)

                # Sector-specific λ = base + interaction
                lambdas = [
                    base_lambda,  # Industrial (base)
                    base_lambda + beta_fin,  # Financial
                    base_lambda + beta_util,  # Utility
                    base_lambda + beta_energy  # Energy
                ]

                offset = width * (idx - 0.5)
                ax.bar(x + offset, lambdas, width, label=universe,
                      color=self.colors[universe], alpha=0.7,
                      edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Sector')
        ax.set_ylabel('Effective λ')
        ax.set_title('Sector-Specific λ (Base + Interaction)')
        ax.set_xticks(x)
        ax.set_xticklabels(sectors)
        ax.legend()
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig7_sector_lambda_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_decision_paths(
        self,
        synthesis_ig: Dict,
        synthesis_hy: Dict,
        comparison: Dict
    ):
        """
        Figure 8: Decision path comparison between IG and HY.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create decision path visualization
        paths = {
            1: "Standard DTS",
            2: "Pure Merton",
            3: "Calibrated Merton",
            4: "Merton + Sectors",
            5: "Theory Fails"
        }

        ig_path = synthesis_ig.get('decision_path', 0)
        hy_path = synthesis_hy.get('decision_path', 0)

        # Bar chart
        y_pos = [0, 1]
        paths_selected = [ig_path, hy_path]
        colors_list = [self.colors['IG'], self.colors['HY']]

        bars = ax.barh(y_pos, paths_selected, color=colors_list, alpha=0.7,
                      edgecolor='black', linewidth=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(['IG', 'HY'])
        ax.set_xlabel('Decision Path')
        ax.set_xlim(0, 6)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(['Path 1\nStandard\nDTS', 'Path 2\nPure\nMerton',
                           'Path 3\nCalibrated\nMerton', 'Path 4\nMerton+\nSectors',
                           'Path 5\nTheory\nFails'], fontsize=9)
        ax.set_title('Stage 0 Decision Paths')
        ax.grid(True, alpha=0.3, axis='x')

        # Add path names as annotations
        for i, (path, color) in enumerate(zip(paths_selected, colors_list)):
            if path > 0:
                ax.text(path, i, f' {paths[path]}', va='center', ha='left',
                       fontweight='bold', fontsize=10)

        # Add comparison text
        unified = comparison.get('unified_approach', '')
        ax.text(0.5, -0.3, f"Recommendation: {unified}",
               ha='center', va='top', transform=ax.transData,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
               fontsize=10, wrap=True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig8_decision_paths.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_lambda_estimates_comparison(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict,
        sector_results_ig: Dict,
        sector_results_hy: Dict
    ):
        """
        Figure 9: λ estimates across all three analyses.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        analyses = ['Bucket', 'Within-Issuer', 'Sector (Base)']

        for idx, (universe, results_tuple) in enumerate([
            ('IG', (bucket_results_ig, within_issuer_results_ig, sector_results_ig)),
            ('HY', (bucket_results_hy, within_issuer_results_hy, sector_results_hy))
        ]):
            ax = axes[idx]
            bucket_res, within_res, sector_res = results_tuple

            # Extract λ estimates
            lambdas = [
                bucket_res.get('regression_results', {}).get('lambda', np.nan),
                within_res.get('pooled_estimate', {}).get('pooled_estimate', np.nan),
                sector_res.get('base_regression', {}).get('lambda', np.nan)
            ]

            # Extract standard errors
            ses = [
                bucket_res.get('regression_results', {}).get('lambda_se', np.nan),
                within_res.get('pooled_estimate', {}).get('pooled_se', np.nan),
                sector_res.get('base_regression', {}).get('lambda_se', np.nan)
            ]

            x_pos = np.arange(len(analyses))

            # Bar plot with error bars
            ax.bar(x_pos, lambdas, yerr=[1.96*se for se in ses],
                  color=self.colors[universe], alpha=0.7,
                  edgecolor='black', linewidth=0.5, capsize=5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(analyses, rotation=15, ha='right')
            ax.set_ylabel('λ Estimate')
            ax.set_title(f'{universe}: λ Across Analyses')
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')

            # Add values as text
            for i, (lam, se) in enumerate(zip(lambdas, ses)):
                if not np.isnan(lam):
                    ax.text(i, lam, f'{lam:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig9_lambda_estimates_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_diagnostic_dashboard(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict,
        sector_results_ig: Dict,
        sector_results_hy: Dict
    ):
        """
        Figure 10: Diagnostic dashboard with key statistics.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Summary statistics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        summary_data = []
        for universe, bucket_res, within_res, sector_res in [
            ('IG', bucket_results_ig, within_issuer_results_ig, sector_results_ig),
            ('HY', bucket_results_hy, within_issuer_results_hy, sector_results_hy)
        ]:
            row = [
                universe,
                bucket_res.get('diagnostics', {}).get('n_buckets_populated', 0),
                within_res.get('diagnostics', {}).get('n_issuer_weeks_with_estimate', 0),
                sector_res.get('diagnostics', {}).get('n_observations', 0),
                f"{bucket_res.get('regression_results', {}).get('lambda', np.nan):.4f}",
                f"{within_res.get('pooled_estimate', {}).get('pooled_estimate', np.nan):.4f}",
                'Yes' if sector_res.get('joint_test', {}).get('reject_null', False) else 'No'
            ]
            summary_data.append(row)

        table = ax1.table(cellText=summary_data,
                         colLabels=['Universe', 'N Buckets', 'N Issuer-Weeks', 'N Obs (Sector)',
                                   'λ (Bucket)', 'λ (Within)', 'Sectors Sig?'],
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax1.set_title('Stage 0 Summary Statistics', fontsize=14, fontweight='bold', pad=20)

        # Additional diagnostic plots (placeholder for 6 more subplots)
        # You can add more detailed diagnostics here

        plt.savefig(self.output_dir / 'fig10_diagnostic_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nDashboard saved with summary statistics")


def create_all_stage0_plots(
    bucket_results_ig: Dict,
    bucket_results_hy: Dict,
    within_issuer_results_ig: Dict,
    within_issuer_results_hy: Dict,
    sector_results_ig: Dict,
    sector_results_hy: Dict,
    synthesis_ig: Dict,
    synthesis_hy: Dict,
    comparison: Dict,
    output_dir: str = "output/stage0_figures"
) -> Stage0Plots:
    """
    Convenience function to create all Stage 0 plots.

    Args:
        bucket_results_ig: Bucket analysis for IG
        bucket_results_hy: Bucket analysis for HY
        within_issuer_results_ig: Within-issuer for IG
        within_issuer_results_hy: Within-issuer for HY
        sector_results_ig: Sector analysis for IG
        sector_results_hy: Sector analysis for HY
        synthesis_ig: Synthesis for IG
        synthesis_hy: Synthesis for HY
        comparison: IG vs HY comparison
        output_dir: Directory to save figures

    Returns:
        Stage0Plots instance
    """
    plotter = Stage0Plots(output_dir=output_dir)

    plotter.plot_all_figures(
        bucket_results_ig, bucket_results_hy,
        within_issuer_results_ig, within_issuer_results_hy,
        sector_results_ig, sector_results_hy,
        synthesis_ig, synthesis_hy,
        comparison
    )

    return plotter
