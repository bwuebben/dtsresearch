"""
Visualization functions for Stage D deliverables.

Creates Figures D.1, D.2, and D.3 from the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class StageDVisualizer:
    """Creates publication-quality plots for Stage D analysis."""

    def __init__(self, output_dir: str = './output/figures'):
        self.output_dir = output_dir

    def plot_quantile_betas(
        self,
        quantile_results_combined: pd.DataFrame,
        quantile_results_ig: Optional[pd.DataFrame] = None,
        quantile_results_hy: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure D.1: Plot of beta_tau across quantiles.

        Shows how elasticity varies across the distribution of spread changes.

        Args:
            quantile_results_combined: DataFrame with quantile, beta_tau, ci_lower, ci_upper
            quantile_results_ig: Optional IG results
            quantile_results_hy: Optional HY results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        has_regime = quantile_results_ig is not None and quantile_results_hy is not None

        if has_regime:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            ax_ig, ax_hy = axes
        else:
            fig, ax_combined = plt.subplots(1, 1, figsize=(12, 6))

        def plot_regime(ax, results_df, title, color='blue'):
            """Helper to plot one regime."""
            # Main line
            ax.plot(
                results_df['quantile'],
                results_df['beta_tau'],
                'o-',
                color=color,
                linewidth=2,
                markersize=8,
                label='β_τ estimate',
                alpha=0.8
            )

            # Confidence bands
            ax.fill_between(
                results_df['quantile'],
                results_df['ci_lower'],
                results_df['ci_upper'],
                alpha=0.2,
                color=color,
                label='95% CI'
            )

            # Horizontal line at beta = 1 (Merton prediction)
            ax.axhline(
                1.0,
                color='red',
                linestyle='--',
                linewidth=2,
                label='Merton prediction (β = 1)',
                alpha=0.7
            )

            # Median line (vertical)
            ax.axvline(
                0.50,
                color='gray',
                linestyle=':',
                linewidth=1,
                alpha=0.5
            )

            ax.set_xlabel('Quantile (τ)', fontsize=12, fontweight='bold')
            ax.set_ylabel('β_τ', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)

        # Plot regimes
        if has_regime:
            plot_regime(ax_ig, quantile_results_ig, 'Investment Grade (IG)', color='blue')
            plot_regime(ax_hy, quantile_results_hy, 'High Yield (HY)', color='orange')
        else:
            plot_regime(ax_combined, quantile_results_combined, 'Combined (All Bonds)', color='purple')

        fig.suptitle(
            'Figure D.1: Quantile Regression - Beta Across Distribution',
            fontsize=14,
            fontweight='bold',
            y=0.98 if has_regime else 1.0
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_shock_betas(
        self,
        shock_betas_combined: Dict,
        shock_betas_ig: Optional[Dict] = None,
        shock_betas_hy: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure D.2: Bar chart of beta by factor type.

        Shows Global, Sector, and Issuer-specific elasticities.

        Args:
            shock_betas_combined: Dictionary with beta_global, beta_sector, beta_issuer
            shock_betas_ig: Optional IG results
            shock_betas_hy: Optional HY results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        has_regime = shock_betas_ig is not None and shock_betas_hy is not None

        if has_regime:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            ax_combined, ax_ig, ax_hy = axes
        else:
            fig, ax_combined = plt.subplots(1, 1, figsize=(10, 6))

        def plot_regime(ax, shock_betas, title):
            """Helper to plot one regime."""
            factors = ['Global', 'Sector', 'Issuer-specific']
            betas = [
                shock_betas['beta_global'],
                shock_betas['beta_sector'],
                shock_betas['beta_issuer']
            ]
            errors = [
                1.96 * shock_betas['se_global'],
                1.96 * shock_betas['se_sector'],
                1.96 * shock_betas['se_issuer']
            ]

            x_pos = np.arange(len(factors))
            colors = ['steelblue', 'seagreen', 'coral']

            bars = ax.bar(x_pos, betas, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

            # Error bars
            ax.errorbar(
                x_pos,
                betas,
                yerr=errors,
                fmt='none',
                ecolor='black',
                capsize=5,
                capthick=2,
                alpha=0.8
            )

            # Horizontal line at beta = 1
            ax.axhline(
                1.0,
                color='red',
                linestyle='--',
                linewidth=2,
                label='Merton prediction (β = 1)',
                alpha=0.7
            )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(factors, fontsize=11)
            ax.set_ylabel('β', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(betas) * 1.3)

        # Plot regimes
        plot_regime(ax_combined, shock_betas_combined, 'Combined')

        if has_regime:
            plot_regime(ax_ig, shock_betas_ig, 'Investment Grade')
            plot_regime(ax_hy, shock_betas_hy, 'High Yield')

        fig.suptitle(
            'Figure D.2: Shock-Specific Elasticities',
            fontsize=14,
            fontweight='bold',
            y=0.98 if has_regime else 1.0
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_liquidity_improvement(
        self,
        by_liquidity_quartile: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure D.3: Scatter plot of beta improvement vs bid-ask spread.

        Shows if illiquid bonds benefit more from liquidity decomposition.

        Args:
            by_liquidity_quartile: DataFrame with quartile results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_beta, ax_r2 = axes

        if len(by_liquidity_quartile) == 0:
            # No data
            ax_beta.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax_r2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return fig

        # Panel A: Beta improvement
        x = by_liquidity_quartile['Avg_BidAsk'].values
        y = by_liquidity_quartile['beta_def'].values - by_liquidity_quartile['beta_total'].values

        ax_beta.scatter(
            x,
            y,
            s=150,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5,
            color='steelblue'
        )

        # Fit line
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax_beta.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='OLS fit')

        ax_beta.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax_beta.set_xlabel('Average Bid-Ask Spread (bps)', fontsize=12, fontweight='bold')
        ax_beta.set_ylabel('β_def - β_total', fontsize=12, fontweight='bold')
        ax_beta.set_title('Panel A: Beta Improvement vs Liquidity', fontsize=13, fontweight='bold')
        ax_beta.legend(loc='best', fontsize=10)
        ax_beta.grid(True, alpha=0.3)

        # Annotate quartiles
        for idx, row in by_liquidity_quartile.iterrows():
            ax_beta.annotate(
                row['Quartile'],
                xy=(row['Avg_BidAsk'], row['beta_def'] - row['beta_total']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.7
            )

        # Panel B: R² improvement
        y_r2 = by_liquidity_quartile['delta_r2'].values

        ax_r2.scatter(
            x,
            y_r2,
            s=150,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5,
            color='seagreen'
        )

        # Fit line
        if len(x) >= 2:
            z = np.polyfit(x, y_r2, 1)
            p = np.poly1d(z)
            ax_r2.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='OLS fit')

        ax_r2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax_r2.set_xlabel('Average Bid-Ask Spread (bps)', fontsize=12, fontweight='bold')
        ax_r2.set_ylabel('ΔR² (Default - Total)', fontsize=12, fontweight='bold')
        ax_r2.set_title('Panel B: R² Improvement vs Liquidity', fontsize=13, fontweight='bold')
        ax_r2.legend(loc='best', fontsize=10)
        ax_r2.grid(True, alpha=0.3)

        # Annotate quartiles
        for idx, row in by_liquidity_quartile.iterrows():
            ax_r2.annotate(
                row['Quartile'],
                xy=(row['Avg_BidAsk'], row['delta_r2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.7
            )

        fig.suptitle(
            'Figure D.3: Liquidity Adjustment - Improvement by Liquidity Level',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_variance_decomposition(
        self,
        variance_decomp: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Supplementary figure: Variance decomposition pie chart.

        Shows relative importance of Global, Sector, Issuer-specific shocks.

        Args:
            variance_decomp: DataFrame with Component and Pct_of_Total
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        components = variance_decomp['Component'].values
        percentages = variance_decomp['Pct_of_Total'].values

        colors = ['steelblue', 'seagreen', 'coral', 'lightgray']

        wedges, texts, autotexts = ax.pie(
            percentages,
            labels=components,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(
            'Variance Decomposition: Sources of Spread Changes',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_all_stageD_figures(
        self,
        quantile_results_combined: pd.DataFrame,
        shock_betas_combined: Dict,
        by_liquidity_quartile: pd.DataFrame,
        variance_decomp: pd.DataFrame,
        quantile_results_ig: Optional[pd.DataFrame] = None,
        quantile_results_hy: Optional[pd.DataFrame] = None,
        shock_betas_ig: Optional[Dict] = None,
        shock_betas_hy: Optional[Dict] = None,
        output_prefix: str = 'stageD'
    ) -> Dict:
        """
        Generate all Stage D figures at once.

        Args:
            quantile_results_combined: Quantile regression results
            shock_betas_combined: Shock decomposition results
            by_liquidity_quartile: Liquidity quartile analysis
            variance_decomp: Variance decomposition
            quantile_results_ig: Optional IG quantile results
            quantile_results_hy: Optional HY quantile results
            shock_betas_ig: Optional IG shock betas
            shock_betas_hy: Optional HY shock betas
            output_prefix: Prefix for output filenames

        Returns:
            Dictionary with figure objects
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        figures = {}

        # Figure D.1: Quantile betas
        fig1 = self.plot_quantile_betas(
            quantile_results_combined,
            quantile_results_ig,
            quantile_results_hy,
            save_path=f'{self.output_dir}/{output_prefix}_fig1_quantiles.png'
        )
        figures['fig1_quantiles'] = fig1

        # Figure D.2: Shock betas
        fig2 = self.plot_shock_betas(
            shock_betas_combined,
            shock_betas_ig,
            shock_betas_hy,
            save_path=f'{self.output_dir}/{output_prefix}_fig2_shocks.png'
        )
        figures['fig2_shocks'] = fig2

        # Figure D.3: Liquidity improvement
        fig3 = self.plot_liquidity_improvement(
            by_liquidity_quartile,
            save_path=f'{self.output_dir}/{output_prefix}_fig3_liquidity.png'
        )
        figures['fig3_liquidity'] = fig3

        # Supplementary: Variance decomposition
        fig4 = self.plot_variance_decomposition(
            variance_decomp,
            save_path=f'{self.output_dir}/{output_prefix}_fig4_variance.png'
        )
        figures['fig4_variance'] = fig4

        return figures
