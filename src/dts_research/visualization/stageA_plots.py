"""
Visualization functions for Stage A deliverables.

Creates Figures A.1 and A.2 from the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class StageAVisualizer:
    """Creates publication-quality plots for Stage A analysis."""

    def __init__(self, output_dir: str = './output/figures'):
        self.output_dir = output_dir

    def plot_beta_heatmap(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure A.1: Heatmap of β^(k) by maturity × rating.

        Separate panels for IG and HY.

        Args:
            results_df: Bucket-level results from Spec A.1
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Define consistent maturity and rating orders
        maturity_order = ['1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10y+']
        rating_order = ['AAA/AA', 'A', 'BBB', 'BB', 'B', 'CCC']

        # IG Heatmap
        ig_data = results_df[results_df['is_ig']].copy()
        ig_pivot = ig_data.groupby(['rating_bucket', 'maturity_bucket'])['beta'].mean().unstack()

        # Reorder
        ig_pivot = ig_pivot.reindex(
            index=[r for r in rating_order if r in ig_pivot.index],
            columns=[m for m in maturity_order if m in ig_pivot.columns]
        )

        sns.heatmap(
            ig_pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            center=1.0,
            vmin=0.5,
            vmax=2.0,
            cbar_kws={'label': 'Beta'},
            ax=ax1,
            linewidths=0.5,
            linecolor='gray'
        )
        ax1.set_title('Investment Grade (IG)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Maturity Bucket', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Rating Bucket', fontsize=12, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

        # HY Heatmap
        hy_data = results_df[~results_df['is_ig']].copy()
        if len(hy_data) > 0:
            hy_pivot = hy_data.groupby(['rating_bucket', 'maturity_bucket'])['beta'].mean().unstack()

            # Reorder
            hy_pivot = hy_pivot.reindex(
                index=[r for r in rating_order if r in hy_pivot.index],
                columns=[m for m in maturity_order if m in hy_pivot.columns]
            )

            sns.heatmap(
                hy_pivot,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn_r',
                center=1.0,
                vmin=0.5,
                vmax=2.0,
                cbar_kws={'label': 'Beta'},
                ax=ax2,
                linewidths=0.5,
                linecolor='gray'
            )
        else:
            ax2.text(0.5, 0.5, 'No HY Data', ha='center', va='center', transform=ax2.transAxes)

        ax2.set_title('High Yield (HY)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Maturity Bucket', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Rating Bucket', fontsize=12, fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        fig.suptitle('Figure A.1: Beta Variation Across Rating and Maturity',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_beta_surface(
        self,
        a2_results: dict,
        save_path: Optional[str] = None,
        plot_3d: bool = False
    ) -> plt.Figure:
        """
        Figure A.2: Implied beta surface from Specification A.2.

        Shows predicted β as function of maturity and spread.

        Args:
            a2_results: Results from Specification A.2
            save_path: Optional path to save figure
            plot_3d: If True, create 3D surface; if False, create contour plot

        Returns:
            matplotlib Figure object
        """
        # Extract coefficients
        combined = a2_results.get('combined', {})
        if 'error' in combined:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f"Error: {combined['error']}",
                   ha='center', va='center', fontsize=14)
            return fig

        gamma_0 = combined['gamma_0']
        gamma_M = combined['gamma_M']
        gamma_s = combined['gamma_s']
        gamma_M2 = combined['gamma_M2']
        gamma_Ms = combined['gamma_Ms']

        # Create grid
        maturity_range = np.linspace(1, 10, 50)
        spread_range = np.linspace(50, 1000, 50)
        M, S = np.meshgrid(maturity_range, spread_range)

        # Predict beta
        Beta = gamma_0 + gamma_M * M + gamma_s * S + gamma_M2 * (M ** 2) + gamma_Ms * M * S

        if plot_3d:
            # 3D surface plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            surf = ax.plot_surface(M, S, Beta, cmap='viridis', alpha=0.8, edgecolor='none')

            ax.set_xlabel('Maturity (years)', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_ylabel('Spread (bps)', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_zlabel('Predicted Beta', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_title('Figure A.2: Implied Beta Surface from Specification A.2',
                        fontsize=14, fontweight='bold', pad=20)

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Beta')

            # Set viewing angle
            ax.view_init(elev=20, azim=45)

        else:
            # Contour plot
            fig, ax = plt.subplots(figsize=(10, 8))

            levels = np.linspace(Beta.min(), Beta.max(), 20)
            contour = ax.contourf(M, S, Beta, levels=levels, cmap='viridis')
            contour_lines = ax.contour(M, S, Beta, levels=10, colors='white',
                                       linewidths=0.5, alpha=0.5)

            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label('Predicted Beta', fontsize=12, fontweight='bold')

            ax.set_xlabel('Maturity (years)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Spread (bps)', fontsize=12, fontweight='bold')
            ax.set_title('Figure A.2: Implied Beta Surface from Specification A.2',
                        fontsize=14, fontweight='bold', pad=20)

            ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_beta_distribution(
        self,
        results_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Additional diagnostic: Distribution of betas across buckets.

        Args:
            results_df: Bucket-level results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel A: Overall distribution
        ax = axes[0, 0]
        ax.hist(results_df['beta'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(results_df['beta'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Mean = {results_df["beta"].mean():.2f}')
        ax.axvline(results_df['beta'].median(), color='green', linestyle='--',
                  linewidth=2, label=f'Median = {results_df["beta"].median():.2f}')
        ax.set_xlabel('Beta', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Panel A: Overall Beta Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel B: IG vs HY
        ax = axes[0, 1]
        ig_betas = results_df[results_df['is_ig']]['beta']
        hy_betas = results_df[~results_df['is_ig']]['beta']

        ax.hist([ig_betas, hy_betas], bins=20, label=['IG', 'HY'],
               edgecolor='black', alpha=0.6, color=['blue', 'orange'])
        ax.set_xlabel('Beta', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Panel B: IG vs HY Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel C: Beta by maturity
        ax = axes[1, 0]
        maturity_order = ['1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10y+']
        maturity_data = []
        labels = []
        for mat in maturity_order:
            data = results_df[results_df['maturity_bucket'] == mat]['beta']
            if len(data) > 0:
                maturity_data.append(data)
                labels.append(mat)

        if maturity_data:
            bp = ax.boxplot(maturity_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')

        ax.set_xlabel('Maturity Bucket', fontsize=11, fontweight='bold')
        ax.set_ylabel('Beta', fontsize=11, fontweight='bold')
        ax.set_title('Panel C: Beta Distribution by Maturity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(labels, rotation=45, ha='right')

        # Panel D: Beta by rating
        ax = axes[1, 1]
        rating_order = ['AAA/AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        rating_data = []
        labels = []
        for rating in rating_order:
            data = results_df[results_df['rating_bucket'] == rating]['beta']
            if len(data) > 0:
                rating_data.append(data)
                labels.append(rating)

        if rating_data:
            bp = ax.boxplot(rating_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')

        ax.set_xlabel('Rating Bucket', fontsize=11, fontweight='bold')
        ax.set_ylabel('Beta', fontsize=11, fontweight='bold')
        ax.set_title('Panel D: Beta Distribution by Rating', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(labels, rotation=45, ha='right')

        fig.suptitle('Stage A: Beta Distribution Diagnostics',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_all_stageA_figures(
        self,
        results_df: pd.DataFrame,
        a2_results: dict,
        output_prefix: str = 'stageA'
    ) -> dict:
        """
        Generate all Stage A figures at once.

        Args:
            results_df: Bucket-level results from A.1
            a2_results: Results from A.2
            output_prefix: Prefix for output filenames

        Returns:
            Dictionary with figure objects
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        figures = {}

        # Figure A.1: Heatmap
        fig1 = self.plot_beta_heatmap(
            results_df,
            save_path=f'{self.output_dir}/{output_prefix}_fig1_heatmap.png'
        )
        figures['fig1_heatmap'] = fig1

        # Figure A.2: Beta surface (contour)
        fig2 = self.plot_beta_surface(
            a2_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig2_surface_contour.png',
            plot_3d=False
        )
        figures['fig2_surface_contour'] = fig2

        # Figure A.2 (alternative): Beta surface (3D)
        fig2_3d = self.plot_beta_surface(
            a2_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig2_surface_3d.png',
            plot_3d=True
        )
        figures['fig2_surface_3d'] = fig2_3d

        # Additional diagnostic
        fig3 = self.plot_beta_distribution(
            results_df,
            save_path=f'{self.output_dir}/{output_prefix}_fig3_distributions.png'
        )
        figures['fig3_distributions'] = fig3

        return figures
