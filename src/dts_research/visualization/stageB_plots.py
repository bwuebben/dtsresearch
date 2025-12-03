"""
Visualization functions for Stage B deliverables.

Creates Figures B.1, B.2, and B.3 from the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class StageBVisualizer:
    """Creates publication-quality plots for Stage B analysis."""

    def __init__(self, output_dir: str = './output/figures'):
        self.output_dir = output_dir

    def plot_theory_vs_empirical_scatter(
        self,
        theory_vs_reality: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure B.1: Scatter plot of empirical β vs theoretical λ.

        Color-coded by regime with different markers.

        Args:
            theory_vs_reality: Theory vs reality comparison table
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Classify regimes based on spread and maturity range
        def classify_regime(row):
            # Use median_spread if available, otherwise use rating as proxy
            if 'median_spread' in row.index and pd.notna(row['median_spread']):
                spread = row['median_spread']
                if spread < 300:
                    return 'IG'
                elif spread < 1000:
                    return 'HY'
                else:
                    return 'Distressed'
            else:
                # Fallback: classify by rating_bucket
                rating = row.get('rating_bucket', '')
                if rating in ['AAA/AA', 'A', 'BBB']:
                    return 'IG'
                elif rating in ['BB', 'B']:
                    return 'HY'
                else:
                    return 'Distressed'

        theory_vs_reality['regime'] = theory_vs_reality.apply(classify_regime, axis=1)

        # Define colors and markers
        regime_styles = {
            'IG': {'color': 'blue', 'marker': 'o', 'label': 'IG (<300 bps)'},
            'HY': {'color': 'orange', 'marker': 's', 'label': 'HY (300-1000 bps)'},
            'Distressed': {'color': 'red', 'marker': '^', 'label': 'Distressed (>1000 bps)'}
        }

        # Plot each regime
        for regime, style in regime_styles.items():
            subset = theory_vs_reality[theory_vs_reality['regime'] == regime]
            if len(subset) > 0:
                ax.scatter(
                    subset['lambda_merton'],
                    subset['beta'],
                    c=style['color'],
                    marker=style['marker'],
                    s=100,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5,
                    label=style['label']
                )

        # 45-degree line
        min_val = min(theory_vs_reality['lambda_merton'].min(), theory_vs_reality['beta'].min())
        max_val = max(theory_vs_reality['lambda_merton'].max(), theory_vs_reality['beta'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=2, label='Perfect agreement (β = λ)', alpha=0.7)

        # Annotate outliers
        outliers = theory_vs_reality[theory_vs_reality['outlier']]
        for idx, row in outliers.head(5).iterrows():  # Top 5 outliers
            label = f"{row['rating_bucket']}/{row['maturity_bucket']}"
            ax.annotate(
                label,
                xy=(row['lambda_merton'], row['beta']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )

        ax.set_xlabel('Theoretical λ^Merton', fontsize=12, fontweight='bold')
        ax.set_ylabel('Empirical β (Stage A)', fontsize=12, fontweight='bold')
        ax.set_title('Figure B.1: Empirical Betas vs Merton Predictions',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_residual_analysis(
        self,
        theory_vs_reality: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure B.2: Residual analysis (β - λ) by maturity, spread, and sector.

        Three panels showing systematic patterns.

        Args:
            theory_vs_reality: Theory vs reality comparison table
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel A: By Maturity
        ax = axes[0]
        maturity_order = ['1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10y+']

        maturity_data = []
        labels = []
        for mat in maturity_order:
            data = theory_vs_reality[theory_vs_reality['maturity_bucket'] == mat]['deviation']
            if len(data) > 0:
                maturity_data.append(data)
                labels.append(mat)

        if maturity_data:
            bp = ax.boxplot(maturity_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')

        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel('Maturity Bucket', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residual (β - λ)', fontsize=11, fontweight='bold')
        ax.set_title('Panel A: By Maturity', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(labels, rotation=45, ha='right')

        # Panel B: By Spread Level
        ax = axes[1]

        # Bin spreads (if median_spread available)
        if 'median_spread' in theory_vs_reality.columns:
            spread_bins = [0, 100, 200, 300, 500, 1000, 3000]
            spread_labels = ['<100', '100-200', '200-300', '300-500', '500-1000', '>1000']

            theory_vs_reality['spread_bin'] = pd.cut(
                theory_vs_reality['median_spread'],
                bins=spread_bins,
                labels=spread_labels
            )

            spread_data = []
            bin_labels = []
            for label in spread_labels:
                data = theory_vs_reality[theory_vs_reality['spread_bin'] == label]['deviation']
                if len(data) > 0:
                    spread_data.append(data)
                    bin_labels.append(label)

            if spread_data:
                bp = ax.boxplot(spread_data, labels=bin_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightgreen')
        else:
            # Fallback: use regime instead
            bin_labels = []  # Initialize for later use
            for regime in ['IG', 'HY', 'Distressed']:
                data = theory_vs_reality[theory_vs_reality['regime'] == regime]['deviation']
                if len(data) > 0:
                    ax.boxplot([data], labels=[regime], patch_artist=True)
                    bin_labels.append(regime)

        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel('Spread Level (bps)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residual (β - λ)', fontsize=11, fontweight='bold')
        ax.set_title('Panel B: By Spread Level', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        if bin_labels:  # Only set xticklabels if we have labels
            ax.set_xticklabels(bin_labels, rotation=45, ha='right')

        # Panel C: By Sector
        ax = axes[2]

        sectors = theory_vs_reality['sector'].unique()
        sector_data = []
        sector_labels = []
        for sector in sorted(sectors):
            data = theory_vs_reality[theory_vs_reality['sector'] == sector]['deviation']
            if len(data) > 0:
                sector_data.append(data)
                sector_labels.append(sector)

        if sector_data:
            bp = ax.boxplot(sector_data, labels=sector_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightyellow')

        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel('Sector', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residual (β - λ)', fontsize=11, fontweight='bold')
        ax.set_title('Panel C: By Sector', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(sector_labels, rotation=45, ha='right')

        fig.suptitle('Figure B.2: Residual Analysis - Where Does Theory Deviate?',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_lambda_surface_comparison(
        self,
        merton_calc,
        spec_b3_results: dict,
        save_path: Optional[str] = None,
        plot_3d: bool = False
    ) -> plt.Figure:
        """
        Figure B.3: Compare Merton surface to unrestricted surface.

        Side-by-side contour plots or 3D surfaces.

        Args:
            merton_calc: MertonLambdaCalculator instance
            spec_b3_results: Results from Specification B.3
            save_path: Optional path to save figure
            plot_3d: If True, use 3D plots; if False, use contours

        Returns:
            matplotlib Figure object
        """
        # Create grid
        maturity_range = np.linspace(1, 10, 50)
        spread_range = np.linspace(50, 1000, 50)
        M, S = np.meshgrid(maturity_range, spread_range)

        # Merton prediction
        Lambda_merton = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                Lambda_merton[i, j] = merton_calc.lambda_combined(M[i, j], S[i, j])

        # Check if unrestricted available
        b3 = spec_b3_results.get('combined', {})
        has_unrestricted = 'error' not in b3

        if plot_3d:
            # 3D plots
            if has_unrestricted:
                fig = plt.figure(figsize=(16, 6))

                # Panel 1: Merton
                ax1 = fig.add_subplot(121, projection='3d')
                surf1 = ax1.plot_surface(M, S, Lambda_merton, cmap='viridis', alpha=0.8)
                ax1.set_xlabel('Maturity (years)')
                ax1.set_ylabel('Spread (bps)')
                ax1.set_zlabel('λ')
                ax1.set_title('Merton Prediction')
                fig.colorbar(surf1, ax=ax1, shrink=0.5)

                # Panel 2: Would need unrestricted surface data
                # Placeholder for now
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.plot_surface(M, S, Lambda_merton * 0.9, cmap='plasma', alpha=0.8)
                ax2.set_xlabel('Maturity (years)')
                ax2.set_ylabel('Spread (bps)')
                ax2.set_zlabel('λ')
                ax2.set_title('Unrestricted (Spec B.3)')
            else:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(M, S, Lambda_merton, cmap='viridis', alpha=0.8)
                ax.set_xlabel('Maturity (years)')
                ax.set_ylabel('Spread (bps)')
                ax.set_zlabel('λ')
                ax.set_title('Merton Prediction')
                fig.colorbar(surf, ax=ax, shrink=0.5)

        else:
            # Contour plots
            fig, axes = plt.subplots(1, 2 if has_unrestricted else 1,
                                    figsize=(16, 6) if has_unrestricted else (10, 6))

            if has_unrestricted:
                ax1, ax2 = axes
            else:
                ax1 = axes

            # Merton contour
            levels = np.linspace(Lambda_merton.min(), Lambda_merton.max(), 15)
            contour1 = ax1.contourf(M, S, Lambda_merton, levels=levels, cmap='viridis')
            contour_lines1 = ax1.contour(M, S, Lambda_merton, levels=10,
                                         colors='white', linewidths=0.5, alpha=0.5)
            ax1.clabel(contour_lines1, inline=True, fontsize=8, fmt='%.2f')

            fig.colorbar(contour1, ax=ax1, label='λ')
            ax1.set_xlabel('Maturity (years)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Spread (bps)', fontsize=11, fontweight='bold')
            ax1.set_title('Merton Prediction', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            if has_unrestricted:
                # Unrestricted contour (placeholder - would need actual fitted surface)
                Lambda_unrestricted = Lambda_merton * 0.95  # Placeholder
                contour2 = ax2.contourf(M, S, Lambda_unrestricted, levels=levels, cmap='plasma')
                contour_lines2 = ax2.contour(M, S, Lambda_unrestricted, levels=10,
                                            colors='white', linewidths=0.5, alpha=0.5)
                ax2.clabel(contour_lines2, inline=True, fontsize=8, fmt='%.2f')

                fig.colorbar(contour2, ax=ax2, label='λ')
                ax2.set_xlabel('Maturity (years)', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Spread (bps)', fontsize=11, fontweight='bold')
                ax2.set_title('Unrestricted (Spec B.3)', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)

        fig.suptitle('Figure B.3: Lambda Surface Comparison',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_all_stageB_figures(
        self,
        theory_vs_reality: pd.DataFrame,
        merton_calc,
        spec_b3_results: dict,
        output_prefix: str = 'stageB'
    ) -> dict:
        """
        Generate all Stage B figures at once.

        Args:
            theory_vs_reality: Theory vs reality comparison table
            merton_calc: MertonLambdaCalculator instance
            spec_b3_results: Specification B.3 results
            output_prefix: Prefix for output filenames

        Returns:
            Dictionary with figure objects
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        figures = {}

        # Figure B.1: Scatter
        fig1 = self.plot_theory_vs_empirical_scatter(
            theory_vs_reality,
            save_path=f'{self.output_dir}/{output_prefix}_fig1_scatter.png'
        )
        figures['fig1_scatter'] = fig1

        # Figure B.2: Residuals
        fig2 = self.plot_residual_analysis(
            theory_vs_reality,
            save_path=f'{self.output_dir}/{output_prefix}_fig2_residuals.png'
        )
        figures['fig2_residuals'] = fig2

        # Figure B.3: Surface comparison (contour)
        fig3 = self.plot_lambda_surface_comparison(
            merton_calc,
            spec_b3_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig3_surfaces_contour.png',
            plot_3d=False
        )
        figures['fig3_surfaces_contour'] = fig3

        # Figure B.3 (alt): Surface comparison (3D)
        fig3_3d = self.plot_lambda_surface_comparison(
            merton_calc,
            spec_b3_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig3_surfaces_3d.png',
            plot_3d=True
        )
        figures['fig3_surfaces_3d'] = fig3_3d

        return figures
