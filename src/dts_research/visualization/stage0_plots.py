"""
Visualization functions for Stage 0 deliverables.

Creates Figures 0.1, 0.2, and 0.3 from the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class Stage0Visualizer:
    """Creates publication-quality plots for Stage 0 analysis."""

    def __init__(self, output_dir: str = './output/figures'):
        self.output_dir = output_dir

    def plot_empirical_vs_theoretical(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure 0.1: Scatter plot of empirical beta vs theoretical lambda.

        - 45-degree line for perfect agreement
        - Point size proportional to sample size
        - Color-code by spread level: IG (blue), HY (orange), Distressed (red)
        - Annotate outliers

        Args:
            results_df: Bucket regression results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Define spread regime colors
        def get_color(spread):
            if spread < 300:
                return 'blue'
            elif spread < 1000:
                return 'orange'
            else:
                return 'red'

        results_df['color'] = results_df['median_spread'].apply(get_color)

        # Normalize point sizes
        size_min, size_max = 20, 500
        n_obs = results_df['n_observations']
        sizes = size_min + (size_max - size_min) * (n_obs - n_obs.min()) / (n_obs.max() - n_obs.min())

        # Create scatter plot
        for color, label in [('blue', 'IG (<300 bps)'),
                             ('orange', 'HY (300-1000 bps)'),
                             ('red', 'Distressed (>1000 bps)')]:
            mask = results_df['color'] == color
            ax.scatter(
                results_df.loc[mask, 'lambda_merton'],
                results_df.loc[mask, 'beta'],
                s=sizes[mask],
                c=color,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5,
                label=label
            )

        # 45-degree line
        min_val = min(results_df['lambda_merton'].min(), results_df['beta'].min())
        max_val = max(results_df['lambda_merton'].max(), results_df['beta'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=2, label='Perfect agreement', alpha=0.5)

        # Annotate outliers (ratio > 1.5 or < 0.67)
        outliers = results_df[
            (results_df['beta_lambda_ratio'] > 1.5) |
            (results_df['beta_lambda_ratio'] < 0.67)
        ]

        for idx, row in outliers.iterrows():
            label = f"{row['rating_bucket']}/{row['maturity_bucket']}"
            ax.annotate(
                label,
                xy=(row['lambda_merton'], row['beta']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )

        ax.set_xlabel('Theoretical λ (Merton)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Empirical β (Regression)', fontsize=12, fontweight='bold')
        ax.set_title('Figure 0.1: Empirical vs Theoretical Spread Sensitivities',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_cross_maturity_patterns(
        self,
        results_df: pd.DataFrame,
        ratings: Optional[list] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure 0.2: Cross-maturity patterns by rating.

        Separate panels for different ratings showing:
        - X-axis: Maturity
        - Y-axis: Beta (empirical, solid) and lambda (theoretical, dashed)

        Args:
            results_df: Bucket regression results
            ratings: List of rating buckets to plot (default: all major ratings)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        if ratings is None:
            ratings = ['AAA/AA', 'A', 'BBB', 'BB', 'B']

        # Filter to ratings with sufficient data
        ratings = [r for r in ratings if (results_df['rating_bucket'] == r).sum() > 0]

        n_ratings = len(ratings)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        maturity_order = {'1-2y': 1.5, '2-3y': 2.5, '3-5y': 4,
                         '5-7y': 6, '7-10y': 8.5, '10y+': 12}

        for idx, rating in enumerate(ratings):
            ax = axes[idx]

            # Get data for this rating (aggregate across sectors)
            rating_data = results_df[
                results_df['rating_bucket'] == rating
            ].copy()

            # Group by maturity and average across sectors
            maturity_summary = rating_data.groupby('maturity_bucket').agg({
                'beta': 'mean',
                'lambda_merton': 'mean',
                'se_beta': 'mean'
            }).reset_index()

            maturity_summary['maturity_num'] = maturity_summary['maturity_bucket'].map(maturity_order)
            maturity_summary = maturity_summary.sort_values('maturity_num')

            if len(maturity_summary) < 2:
                ax.text(0.5, 0.5, f'{rating}\nInsufficient data',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # Plot empirical beta
            ax.plot(maturity_summary['maturity_num'],
                   maturity_summary['beta'],
                   'o-', linewidth=2, markersize=8,
                   label='Empirical β', color='darkblue')

            # Plot confidence interval
            ax.fill_between(
                maturity_summary['maturity_num'],
                maturity_summary['beta'] - 1.96 * maturity_summary['se_beta'],
                maturity_summary['beta'] + 1.96 * maturity_summary['se_beta'],
                alpha=0.2, color='darkblue'
            )

            # Plot theoretical lambda
            ax.plot(maturity_summary['maturity_num'],
                   maturity_summary['lambda_merton'],
                   's--', linewidth=2, markersize=6,
                   label='Theoretical λ', color='red', alpha=0.7)

            ax.set_xlabel('Maturity (years)', fontsize=10)
            ax.set_ylabel('Sensitivity', fontsize=10)
            ax.set_title(f'{rating}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Set x-axis labels
            ax.set_xticks(maturity_summary['maturity_num'])
            ax.set_xticklabels(maturity_summary['maturity_bucket'], rotation=45)

        # Remove extra subplots
        for idx in range(n_ratings, len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle('Figure 0.2: Cross-Maturity Patterns by Rating',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_regime_patterns(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure 0.3: Regime patterns showing dispersion vs spread level.

        X-axis: Average spread level of bucket
        Y-axis: Cross-maturity dispersion (std dev of beta)

        Shows whether dispersion declines as spreads widen.

        Args:
            results_df: Bucket regression results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute cross-maturity dispersion for each rating/sector group
        dispersion = results_df.groupby(['rating_bucket', 'sector']).agg({
            'beta': ['std', 'mean'],
            'median_spread': 'mean',
            'is_ig': 'first',
            'n_observations': 'sum'
        }).reset_index()

        dispersion.columns = ['rating_bucket', 'sector', 'beta_std', 'beta_mean',
                             'avg_spread', 'is_ig', 'n_obs']

        # Remove groups with too few maturity buckets
        dispersion = dispersion[dispersion['beta_std'] > 0]

        # Color by IG/HY
        colors = dispersion['is_ig'].map({True: 'blue', False: 'orange'})
        sizes = 50 + 200 * (dispersion['n_obs'] - dispersion['n_obs'].min()) / \
                (dispersion['n_obs'].max() - dispersion['n_obs'].min())

        # Scatter plot
        scatter = ax.scatter(
            dispersion['avg_spread'],
            dispersion['beta_std'],
            c=colors,
            s=sizes,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )

        # Add trend line
        from scipy.stats import linregress
        valid = ~(dispersion['avg_spread'].isna() | dispersion['beta_std'].isna())
        if valid.sum() > 2:
            slope, intercept, r_value, p_value, se = linregress(
                dispersion.loc[valid, 'avg_spread'],
                dispersion.loc[valid, 'beta_std']
            )

            x_line = np.array([dispersion['avg_spread'].min(),
                              dispersion['avg_spread'].max()])
            y_line = slope * x_line + intercept

            ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
                   label=f'Trend (R²={r_value**2:.3f}, p={p_value:.3f})')

        # Add regime boundaries
        ax.axvline(300, color='gray', linestyle=':', alpha=0.5, label='IG/HY boundary')
        ax.axvline(1000, color='gray', linestyle=':', alpha=0.5, label='HY/Distressed boundary')

        ax.set_xlabel('Average Spread Level (bps)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cross-Maturity Dispersion (Std Dev of β)', fontsize=12, fontweight='bold')
        ax.set_title('Figure 0.3: Regime Patterns - Dispersion vs Spread Level',
                    fontsize=14, fontweight='bold', pad=20)

        # Manual legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='IG'),
            Patch(facecolor='orange', alpha=0.6, label='HY')
        ]
        legend1 = ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax.add_artist(legend1)

        if hasattr(ax, 'lines') and len(ax.lines) > 0:
            ax.legend(loc='upper left', fontsize=10)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_all_stage0_figures(
        self,
        results_df: pd.DataFrame,
        output_prefix: str = 'stage0'
    ) -> dict:
        """
        Generate all Stage 0 figures at once.

        Args:
            results_df: Bucket regression results
            output_prefix: Prefix for output filenames

        Returns:
            Dictionary with figure objects
        """
        figures = {}

        # Figure 0.1
        fig1 = self.plot_empirical_vs_theoretical(
            results_df,
            save_path=f'{self.output_dir}/{output_prefix}_fig1_scatter.png'
        )
        figures['fig1_scatter'] = fig1

        # Figure 0.2
        fig2 = self.plot_cross_maturity_patterns(
            results_df,
            save_path=f'{self.output_dir}/{output_prefix}_fig2_crossmaturity.png'
        )
        figures['fig2_crossmaturity'] = fig2

        # Figure 0.3
        fig3 = self.plot_regime_patterns(
            results_df,
            save_path=f'{self.output_dir}/{output_prefix}_fig3_regimes.png'
        )
        figures['fig3_regimes'] = fig3

        return figures
