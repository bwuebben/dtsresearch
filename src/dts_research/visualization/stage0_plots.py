"""
Stage 0: Visualization Module (Evolved Version)

Creates 10 figures for Stage 0 analysis testing β ≈ 1 (Merton prediction):
1. Bucket β vs λ^Merton scatter (empirical vs theoretical)
2. β/λ ratio distribution across buckets
3. Maturity monotonicity of β (should decrease with maturity)
4. Within-issuer β distribution (should center on 1)
5. Within-issuer β time series
6. Sector interaction coefficients (deviations from baseline)
7. Sector-specific total sensitivities (β_0 + β_sector)
8. Decision path comparison (IG vs HY)
9. β estimates across all three analyses (should all be ≈ 1)
10. Diagnostic summary dashboard

Based on visualization requirements from the evolved Stage 0 analysis.
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
    Visualization utilities for evolved Stage 0 analysis.

    Key difference from original: focuses on testing β ≈ 1 (Merton prediction)
    rather than just λ > 0.
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

        # Figure 1: Bucket β vs λ^Merton scatter
        self.plot_beta_vs_lambda_merton(bucket_results_ig, bucket_results_hy)
        print("  [1/10] Bucket β vs λ^Merton scatter")

        # Figure 2: β/λ ratio distribution
        self.plot_beta_lambda_ratio_distribution(bucket_results_ig, bucket_results_hy)
        print("  [2/10] β/λ ratio distribution")

        # Figure 3: Maturity monotonicity of β
        self.plot_maturity_monotonicity(bucket_results_ig, bucket_results_hy)
        print("  [3/10] Maturity monotonicity")

        # Figure 4: Within-issuer β distribution
        self.plot_within_issuer_beta_distribution(within_issuer_results_ig, within_issuer_results_hy)
        print("  [4/10] Within-issuer β distribution")

        # Figure 5: Within-issuer β time series
        self.plot_within_issuer_time_series(within_issuer_results_ig, within_issuer_results_hy)
        print("  [5/10] Within-issuer β time series")

        # Figure 6: Sector interaction coefficients
        self.plot_sector_coefficients(sector_results_ig, sector_results_hy)
        print("  [6/10] Sector interaction coefficients")

        # Figure 7: Sector-specific total sensitivities
        self.plot_sector_sensitivities(sector_results_ig, sector_results_hy)
        print("  [7/10] Sector-specific sensitivities")

        # Figure 8: Decision path comparison
        self.plot_decision_paths(synthesis_ig, synthesis_hy, comparison)
        print("  [8/10] Decision path comparison")

        # Figure 9: β estimates across analyses
        self.plot_beta_comparison_across_analyses(
            bucket_results_ig, bucket_results_hy,
            within_issuer_results_ig, within_issuer_results_hy,
            sector_results_ig, sector_results_hy
        )
        print("  [9/10] β estimates comparison")

        # Figure 10: Diagnostic dashboard
        self.plot_diagnostic_dashboard(
            bucket_results_ig, bucket_results_hy,
            within_issuer_results_ig, within_issuer_results_hy,
            sector_results_ig, sector_results_hy,
            synthesis_ig, synthesis_hy
        )
        print("  [10/10] Diagnostic dashboard")

        print(f"\nAll figures saved to: {self.output_dir}")

    def plot_beta_vs_lambda_merton(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict
    ):
        """
        Figure 1: Scatter plot of empirical β vs theoretical λ^Merton.

        If Merton holds, points should lie on 45-degree line (β = λ^Merton).
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, universe) in enumerate([(bucket_results_ig, 'IG'), (bucket_results_hy, 'HY')]):
            ax = axes[idx]
            bucket_df = results.get('bucket_results', pd.DataFrame())

            if len(bucket_df) > 0 and 'beta' in bucket_df.columns and 'lambda_merton' in bucket_df.columns:
                # Scatter plot
                ax.scatter(
                    bucket_df['lambda_merton'],
                    bucket_df['beta'],
                    s=bucket_df['n_obs'] / 50 if 'n_obs' in bucket_df.columns else 50,
                    alpha=0.6,
                    c=self.colors[universe],
                    edgecolors='black',
                    linewidth=0.5
                )

                # 45-degree line (Merton prediction: β = λ^Merton)
                lims = [
                    min(bucket_df['lambda_merton'].min(), bucket_df['beta'].min()),
                    max(bucket_df['lambda_merton'].max(), bucket_df['beta'].max())
                ]
                ax.plot(lims, lims, 'r--', linewidth=2, label='β = λ^Merton (Merton)')

                # Add 10% and 20% bands
                ax.fill_between(lims, [l*0.9 for l in lims], [l*1.1 for l in lims],
                               alpha=0.2, color='green', label='±10%')

                ax.set_xlabel('λ^Merton (Theoretical)')
                ax.set_ylabel('β (Empirical)')
                ax.set_title(f'{universe}: Empirical β vs Theoretical λ^Merton')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)

                # Add summary statistics
                summary = results.get('summary_statistics', {})
                median_ratio = summary.get('median_beta_lambda_ratio', np.nan)
                pct_within = summary.get('pct_within_20pct', 0)
                text = f'Median β/λ = {median_ratio:.2f}\n{pct_within:.0f}% within ±20%'
                ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_beta_vs_lambda_merton.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_beta_lambda_ratio_distribution(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict
    ):
        """
        Figure 2: Distribution of β/λ^Merton ratios across buckets.

        Should be centered around 1 if Merton holds.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, universe) in enumerate([(bucket_results_ig, 'IG'), (bucket_results_hy, 'HY')]):
            ax = axes[idx]
            bucket_df = results.get('bucket_results', pd.DataFrame())

            if len(bucket_df) > 0 and 'beta_lambda_ratio' in bucket_df.columns:
                ratios = bucket_df['beta_lambda_ratio'].dropna()

                # Histogram
                ax.hist(ratios, bins=30, alpha=0.7, color=self.colors[universe],
                       edgecolor='black', linewidth=0.5, density=True)

                # Reference line at 1 (Merton prediction)
                ax.axvline(1.0, color='red', linestyle='--', linewidth=2,
                          label='Merton prediction (β/λ = 1)')

                # Add 10% bands
                ax.axvline(0.9, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
                ax.axvline(1.1, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='±10% bands')

                ax.set_xlabel('β/λ^Merton Ratio')
                ax.set_ylabel('Density')
                ax.set_title(f'{universe}: Distribution of β/λ Ratios')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

                # Add statistics
                summary = results.get('summary_statistics', {})
                text = (f'Median: {summary.get("median_beta_lambda_ratio", np.nan):.3f}\n'
                       f'Mean: {summary.get("mean_beta_lambda_ratio", np.nan):.3f}\n'
                       f'Std: {summary.get("std_beta_lambda_ratio", np.nan):.3f}\n'
                       f'n = {len(ratios)} buckets')
                ax.text(0.95, 0.95, text, transform=ax.transAxes, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_beta_lambda_ratio_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_maturity_monotonicity(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict
    ):
        """
        Figure 3: β vs maturity by rating group.

        Merton predicts β should DECREASE with maturity (short-maturity bonds
        are more sensitive to market moves).
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, universe) in enumerate([(bucket_results_ig, 'IG'), (bucket_results_hy, 'HY')]):
            ax = axes[idx]
            bucket_df = results.get('bucket_results', pd.DataFrame())
            monotonicity = results.get('monotonicity_test', {})

            if len(bucket_df) > 0 and 'beta' in bucket_df.columns and 'median_maturity' in bucket_df.columns:
                # Plot β vs maturity, colored by rating
                if 'rating_bucket' in bucket_df.columns:
                    for rating in bucket_df['rating_bucket'].unique():
                        subset = bucket_df[bucket_df['rating_bucket'] == rating]
                        subset = subset.sort_values('median_maturity')
                        ax.plot(subset['median_maturity'], subset['beta'],
                               'o-', alpha=0.7, label=rating, markersize=8)
                else:
                    ax.scatter(bucket_df['median_maturity'], bucket_df['beta'],
                              alpha=0.7, c=self.colors[universe], s=50)

                ax.set_xlabel('Maturity (years)')
                ax.set_ylabel('β (DTS Sensitivity)')
                ax.set_title(f'{universe}: β vs Maturity (should decrease)')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)

                # Add monotonicity result
                pct_mono = monotonicity.get('pct_monotonic_groups', 0)
                is_mono = monotonicity.get('overall_monotonic', False)
                status = "✓ Monotonic" if is_mono else "✗ Not Monotonic"
                text = f'{status}\n{pct_mono:.0f}% of groups\nshow β↓ with maturity'
                color = 'green' if is_mono else 'red'
                ax.text(0.05, 0.05, text, transform=ax.transAxes, va='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       color=color, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_maturity_monotonicity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_within_issuer_beta_distribution(
        self,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict
    ):
        """
        Figure 4: Distribution of within-issuer β estimates.

        Should be centered around 1 if Merton holds.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (results, universe) in enumerate([(within_issuer_results_ig, 'IG'), (within_issuer_results_hy, 'HY')]):
            ax = axes[idx]
            estimates = results.get('issuer_week_estimates', pd.DataFrame())

            if len(estimates) > 0 and 'beta' in estimates.columns:
                betas = estimates['beta'].dropna()

                # Clip extreme values for visualization
                betas_clipped = betas.clip(-2, 3)

                # Histogram
                ax.hist(betas_clipped, bins=50, alpha=0.7, color=self.colors[universe],
                       edgecolor='black', linewidth=0.5)

                # Reference line at 1 (Merton prediction)
                ax.axvline(1.0, color='red', linestyle='--', linewidth=2,
                          label='Merton prediction (β = 1)')

                # Add pooled estimate line
                pooled = results.get('pooled_estimate', {}).get('pooled_beta', np.nan)
                if not np.isnan(pooled):
                    ax.axvline(pooled, color='green', linestyle='-', linewidth=2,
                              label=f'Pooled β = {pooled:.3f}')

                ax.set_xlabel('β (Within-Issuer)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{universe}: Distribution of Within-Issuer β')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

                # Add statistics
                diagnostics = results.get('diagnostics', {})
                text = (f'Median: {diagnostics.get("median_beta", np.nan):.3f}\n'
                       f'Mean: {diagnostics.get("mean_beta", np.nan):.3f}\n'
                       f'Std: {diagnostics.get("std_beta", np.nan):.3f}\n'
                       f'{diagnostics.get("pct_beta_in_0_8_1_2", 0):.0f}% in [0.8, 1.2]\n'
                       f'n = {len(betas):,}')
                ax.text(0.95, 0.95, text, transform=ax.transAxes, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_within_issuer_beta_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_within_issuer_time_series(
        self,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict
    ):
        """
        Figure 5: Time series of within-issuer β estimates.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        for idx, (results, universe) in enumerate([(within_issuer_results_ig, 'IG'), (within_issuer_results_hy, 'HY')]):
            ax = axes[idx]
            estimates = results.get('issuer_week_estimates', pd.DataFrame())

            if len(estimates) > 0 and 'beta' in estimates.columns and 'date' in estimates.columns:
                # Aggregate by date (mean and confidence interval)
                weekly = estimates.groupby('date')['beta'].agg(['mean', 'std', 'count']).reset_index()
                weekly['se'] = weekly['std'] / np.sqrt(weekly['count'])
                weekly['ci_lower'] = weekly['mean'] - 1.96 * weekly['se']
                weekly['ci_upper'] = weekly['mean'] + 1.96 * weekly['se']

                ax.plot(weekly['date'], weekly['mean'], color=self.colors[universe],
                       linewidth=2, label='Mean β')
                ax.fill_between(weekly['date'], weekly['ci_lower'], weekly['ci_upper'],
                               color=self.colors[universe], alpha=0.3, label='95% CI')

                # Reference line at 1 (Merton prediction)
                ax.axhline(1.0, color='red', linestyle='--', linewidth=2,
                          label='Merton prediction (β = 1)')

                # Add pooled estimate
                pooled = results.get('pooled_estimate', {}).get('pooled_beta', np.nan)
                if not np.isnan(pooled):
                    ax.axhline(pooled, color='green', linestyle=':', linewidth=2,
                              label=f'Pooled β = {pooled:.3f}')

                ax.set_xlabel('Date')
                ax.set_ylabel('β (Within-Issuer)')
                ax.set_title(f'{universe}: Within-Issuer β Over Time')
                ax.legend(loc='upper right')
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
        Figure 6: Sector interaction coefficients (deviations from Industrial baseline).
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
                ses = [reg.get(k, 0) for k in se_keys]
                ci_lower = [b - 1.96*se for b, se in zip(betas, ses)]
                ci_upper = [b + 1.96*se for b, se in zip(betas, ses)]

                y_pos = np.arange(len(sectors))
                colors_list = [self.colors[s] for s in sectors]

                ax.barh(y_pos, betas, color=colors_list, alpha=0.7,
                       edgecolor='black', linewidth=0.5)

                # Add confidence intervals
                for i, (b, cl, cu) in enumerate(zip(betas, ci_lower, ci_upper)):
                    ax.plot([cl, cu], [i, i], 'k-', linewidth=2)
                    ax.plot([cl, cl], [i-0.1, i+0.1], 'k-', linewidth=2)
                    ax.plot([cu, cu], [i-0.1, i+0.1], 'k-', linewidth=2)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(sectors)
                ax.set_xlabel('Sector Deviation from Industrial (β_sector)')
                ax.set_title(f'{universe}: Sector Deviations from Baseline')
                ax.axvline(0, color='black', linestyle='--', linewidth=1)
                ax.grid(True, alpha=0.3, axis='x')

                # Add base β_0 and joint test
                base_beta = reg.get('beta_0', np.nan)
                joint = results.get('joint_test', {})
                pval = joint.get('p_value', np.nan)
                sig = "Significant" if joint.get('sectors_differ', False) else "Not Significant"
                text = f'Base β₀ = {base_beta:.3f}\nJoint test: {sig}\np = {pval:.4f}'
                ax.text(0.95, 0.05, text, transform=ax.transAxes, va='bottom', ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{universe}: No Data')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_sector_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sector_sensitivities(
        self,
        sector_results_ig: Dict,
        sector_results_hy: Dict
    ):
        """
        Figure 7: Sector-specific total DTS sensitivities (β_0 + β_sector).
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        sectors = ['Industrial', 'Financial', 'Utility', 'Energy']
        sens_keys = ['sensitivity_industrial', 'sensitivity_financial', 'sensitivity_utility', 'sensitivity_energy']
        universes = ['IG', 'HY']
        results_list = [sector_results_ig, sector_results_hy]

        x = np.arange(len(sectors))
        width = 0.35

        for idx, (results, universe) in enumerate(zip(results_list, universes)):
            reg = results.get('sector_regression', {})

            if 'sensitivity_industrial' in reg:
                sensitivities = [reg.get(k, np.nan) for k in sens_keys]

                offset = width * (idx - 0.5)
                bars = ax.bar(x + offset, sensitivities, width, label=universe,
                             color=self.colors[universe], alpha=0.7,
                             edgecolor='black', linewidth=0.5)

                # Add value labels
                for bar, sens in zip(bars, sensitivities):
                    if not np.isnan(sens):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{sens:.2f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Sector')
        ax.set_ylabel('Total DTS Sensitivity (β₀ + β_sector)')
        ax.set_title('Sector-Specific DTS Sensitivities')
        ax.set_xticks(x)
        ax.set_xticklabels(sectors)
        ax.legend()
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Merton (β = 1)')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig7_sector_sensitivities.png', dpi=300, bbox_inches='tight')
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
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left panel: Decision paths as bars
        ax1 = axes[0]

        paths = {
            1: "Standard DTS\n(β ≈ 1 validated)",
            2: "Pure Merton\n(use λ tables)",
            3: "Calibrated Merton\n(scale β to data)",
            4: "Merton + Sectors\n(sector adjustments)",
            5: "Theory Fails\n(alternative models)"
        }

        ig_path = synthesis_ig.get('decision_path', 0)
        hy_path = synthesis_hy.get('decision_path', 0)

        y_pos = [0, 1]
        paths_selected = [ig_path, hy_path]
        colors_list = [self.colors['IG'], self.colors['HY']]

        bars = ax1.barh(y_pos, paths_selected, color=colors_list, alpha=0.7,
                       edgecolor='black', linewidth=2)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(['IG', 'HY'])
        ax1.set_xlabel('Decision Path')
        ax1.set_xlim(0, 6)
        ax1.set_xticks(range(1, 6))
        ax1.set_xticklabels([paths[i] for i in range(1, 6)], fontsize=8)
        ax1.set_title('Stage 0 Decision Paths')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add path names as annotations
        for i, (path, color) in enumerate(zip(paths_selected, colors_list)):
            if path > 0:
                ax1.text(path + 0.1, i, paths[path].replace('\n', ' '),
                        va='center', ha='left', fontweight='bold', fontsize=9)

        # Right panel: Key statistics comparison
        ax2 = axes[1]
        ax2.axis('off')

        # Create comparison table
        ig_stats = synthesis_ig.get('key_statistics', {})
        hy_stats = synthesis_hy.get('key_statistics', {})

        table_data = [
            ['Metric', 'IG', 'HY', 'Status'],
            ['Bucket β/λ ratio',
             f"{ig_stats.get('bucket_median_ratio', np.nan):.2f}",
             f"{hy_stats.get('bucket_median_ratio', np.nan):.2f}",
             '✓' if ig_stats.get('bucket_median_ratio', 0) > 0.8 else '✗'],
            ['Within-issuer β',
             f"{ig_stats.get('within_beta', np.nan):.2f}",
             f"{hy_stats.get('within_beta', np.nan):.2f}",
             '✓' if 0.8 <= ig_stats.get('within_beta', 0) <= 1.2 else '✗'],
            ['Monotonic?',
             'Yes' if ig_stats.get('monotonic', False) else 'No',
             'Yes' if hy_stats.get('monotonic', False) else 'No',
             '✓' if ig_stats.get('monotonic', False) else '✗'],
            ['Sectors differ?',
             'Yes' if ig_stats.get('sectors_differ', False) else 'No',
             'Yes' if hy_stats.get('sectors_differ', False) else 'No',
             '-'],
        ]

        table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center',
                         bbox=[0.1, 0.3, 0.8, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Add unified approach recommendation
        unified = comparison.get('unified_approach', 'N/A')
        ax2.text(0.5, 0.1, f"Recommendation: {unified}",
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                transform=ax2.transAxes)

        ax2.set_title('Key Statistics Comparison', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig8_decision_paths.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_beta_comparison_across_analyses(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict,
        sector_results_ig: Dict,
        sector_results_hy: Dict
    ):
        """
        Figure 9: β estimates across all three analyses.

        All should be ≈ 1 if Merton holds.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        analyses = ['Bucket\n(median β/λ)', 'Within-Issuer\n(pooled β)', 'Sector\n(base β₀)']

        for idx, (universe, results_tuple) in enumerate([
            ('IG', (bucket_results_ig, within_issuer_results_ig, sector_results_ig)),
            ('HY', (bucket_results_hy, within_issuer_results_hy, sector_results_hy))
        ]):
            ax = axes[idx]
            bucket_res, within_res, sector_res = results_tuple

            # Extract β estimates (should all be ≈ 1)
            betas = [
                bucket_res.get('summary_statistics', {}).get('median_beta_lambda_ratio', np.nan),
                within_res.get('pooled_estimate', {}).get('pooled_beta', np.nan),
                sector_res.get('base_regression', {}).get('beta_0', np.nan)
            ]

            # Extract standard errors where available
            ses = [
                bucket_res.get('summary_statistics', {}).get('std_beta_lambda_ratio', np.nan) /
                    np.sqrt(bucket_res.get('summary_statistics', {}).get('n_buckets', 1)),
                within_res.get('pooled_estimate', {}).get('pooled_beta_se', np.nan),
                sector_res.get('base_regression', {}).get('beta_0_se', np.nan)
            ]

            x_pos = np.arange(len(analyses))

            # Handle NaN in error bars
            yerr = [1.96*se if not np.isnan(se) else 0 for se in ses]

            # Bar plot with error bars
            bars = ax.bar(x_pos, betas, yerr=yerr,
                         color=self.colors[universe], alpha=0.7,
                         edgecolor='black', linewidth=0.5, capsize=5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(analyses)
            ax.set_ylabel('β Estimate (should be ≈ 1)')
            ax.set_title(f'{universe}: β Across Analyses')

            # Reference line at 1 (Merton prediction)
            ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Merton (β = 1)')

            # Add shaded region for acceptable range
            ax.axhspan(0.9, 1.1, alpha=0.2, color='green', label='±10% range')

            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

            # Add values as text
            for i, (beta, se) in enumerate(zip(betas, ses)):
                if not np.isnan(beta):
                    ax.text(i, beta + 0.05, f'{beta:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig9_beta_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_diagnostic_dashboard(
        self,
        bucket_results_ig: Dict,
        bucket_results_hy: Dict,
        within_issuer_results_ig: Dict,
        within_issuer_results_hy: Dict,
        sector_results_ig: Dict,
        sector_results_hy: Dict,
        synthesis_ig: Dict,
        synthesis_hy: Dict
    ):
        """
        Figure 10: Diagnostic dashboard with key statistics and validation status.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

        # Top row: Summary statistics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        summary_data = []
        for universe, bucket_res, within_res, sector_res, synth in [
            ('IG', bucket_results_ig, within_issuer_results_ig, sector_results_ig, synthesis_ig),
            ('HY', bucket_results_hy, within_issuer_results_hy, sector_results_hy, synthesis_hy)
        ]:
            bucket_diag = bucket_res.get('diagnostics', {})
            within_diag = within_res.get('diagnostics', {})
            sector_diag = sector_res.get('diagnostics', {})

            row = [
                universe,
                bucket_diag.get('n_buckets_with_regression', 0),
                within_diag.get('n_issuer_weeks_with_estimate', 0),
                sector_diag.get('n_observations', 0),
                f"{bucket_res.get('summary_statistics', {}).get('median_beta_lambda_ratio', np.nan):.3f}",
                f"{within_res.get('pooled_estimate', {}).get('pooled_beta', np.nan):.3f}",
                f"{sector_res.get('base_regression', {}).get('beta_0', np.nan):.3f}",
                'Yes' if sector_res.get('joint_test', {}).get('sectors_differ', False) else 'No',
                f"Path {synth.get('decision_path', 0)}"
            ]
            summary_data.append(row)

        table = ax1.table(cellText=summary_data,
                         colLabels=['Universe', 'N Buckets', 'N Issuer-Wks', 'N Obs (Sector)',
                                   'Bucket β/λ', 'Within β', 'Sector β₀', 'Sectors?', 'Path'],
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax1.set_title('Stage 0 Summary: Testing β ≈ 1 (Merton Prediction)',
                     fontsize=14, fontweight='bold', pad=20)

        # Middle left: Bucket R² distribution
        ax2 = fig.add_subplot(gs[1, 0])
        for results, universe in [(bucket_results_ig, 'IG'), (bucket_results_hy, 'HY')]:
            bucket_df = results.get('bucket_results', pd.DataFrame())
            if len(bucket_df) > 0 and 'r_squared' in bucket_df.columns:
                ax2.hist(bucket_df['r_squared'], bins=20, alpha=0.5,
                        label=universe, color=self.colors[universe])
        ax2.set_xlabel('R² (Bucket Regressions)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bucket Regression Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Middle right: Within-issuer R² distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for results, universe in [(within_issuer_results_ig, 'IG'), (within_issuer_results_hy, 'HY')]:
            estimates = results.get('issuer_week_estimates', pd.DataFrame())
            if len(estimates) > 0 and 'r_squared' in estimates.columns:
                ax3.hist(estimates['r_squared'], bins=20, alpha=0.5,
                        label=universe, color=self.colors[universe])
        ax3.set_xlabel('R² (Within-Issuer Regressions)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Within-Issuer Regression Fit')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom left: Validation checklist
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')

        ig_criteria = synthesis_ig.get('decision_criteria', {})
        hy_criteria = synthesis_hy.get('decision_criteria', {})

        checklist = [
            ['Criterion', 'IG', 'HY'],
            ['β ≈ 1 (bucket level)',
             '✓' if ig_criteria.get('bucket_beta_near_1', False) else '✗',
             '✓' if hy_criteria.get('bucket_beta_near_1', False) else '✗'],
            ['β ≈ 1 (within-issuer)',
             '✓' if ig_criteria.get('within_beta_near_1', False) else '✗',
             '✓' if hy_criteria.get('within_beta_near_1', False) else '✗'],
            ['Monotonic (β ↓ with T)',
             '✓' if ig_criteria.get('monotonic', False) else '✗',
             '✓' if hy_criteria.get('monotonic', False) else '✗'],
            ['Consistent across analyses',
             '✓' if ig_criteria.get('consistent', False) else '✗',
             '✓' if hy_criteria.get('consistent', False) else '✗'],
            ['Theory validated',
             '✓' if ig_criteria.get('theory_validated', False) else '✗',
             '✓' if hy_criteria.get('theory_validated', False) else '✗'],
        ]

        table2 = ax4.table(cellText=checklist[1:], colLabels=checklist[0],
                          cellLoc='center', loc='center',
                          bbox=[0.1, 0.2, 0.8, 0.7])
        table2.auto_set_font_size(False)
        table2.set_fontsize(11)
        table2.scale(1, 2)
        ax4.set_title('Merton Validation Checklist', fontsize=12, fontweight='bold')

        # Bottom right: Recommendations
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        ig_rec = synthesis_ig.get('recommendations', {})
        hy_rec = synthesis_hy.get('recommendations', {})

        rec_text = (
            f"IG Path {synthesis_ig.get('decision_path', 0)}: {synthesis_ig.get('path_name', 'N/A')}\n"
            f"  → Stage A: {ig_rec.get('stage_A', 'N/A')[:50]}...\n\n"
            f"HY Path {synthesis_hy.get('decision_path', 0)}: {synthesis_hy.get('path_name', 'N/A')}\n"
            f"  → Stage A: {hy_rec.get('stage_A', 'N/A')[:50]}...\n"
        )

        ax5.text(0.1, 0.9, rec_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax5.set_title('Recommendations for Stage A', fontsize=12, fontweight='bold')

        plt.savefig(self.output_dir / 'fig10_diagnostic_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nDashboard saved with validation checklist")


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
