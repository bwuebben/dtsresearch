"""
Visualization functions for Stage E deliverables.

Creates Figures E.1, E.2, and E.3 from the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class StageEVisualizer:
    """Creates publication-quality plots for Stage E analysis."""

    def __init__(self, output_dir: str = './output/figures'):
        self.output_dir = output_dir

    def plot_oos_r2_over_time(
        self,
        oos_results: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure E.1: Out-of-sample R² over rolling windows.

        Shows OOS R² for each specification over time, with crisis periods shaded.

        Args:
            oos_results: OOS validation results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))

        oos_by_window = oos_results.get('oos_by_window', {})

        colors = {
            'Standard DTS': 'gray',
            'Pure Merton': 'blue',
            'Calibrated Merton': 'green',
            'Empirical': 'orange',
            'Time-varying': 'red'
        }

        markers = {
            'Standard DTS': 'o',
            'Pure Merton': 's',
            'Calibrated Merton': '^',
            'Empirical': 'D',
            'Time-varying': 'v'
        }

        # Plot each specification
        for spec_name, windows in oos_by_window.items():
            if len(windows) == 0:
                continue

            # Extract data
            test_starts = [w['test_start'] for w in windows]
            r2_oos = [w['r2_oos'] for w in windows]

            # Plot line
            ax.plot(
                test_starts,
                r2_oos,
                marker=markers.get(spec_name, 'o'),
                color=colors.get(spec_name, 'black'),
                linewidth=2,
                markersize=6,
                label=spec_name,
                alpha=0.8
            )

        # Shade crisis periods (VIX > 30)
        # For illustration, shade 2008-2009 and 2020
        crisis_periods = [
            (pd.Timestamp('2008-09-01'), pd.Timestamp('2009-06-01')),  # Financial crisis
            (pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01'))   # COVID crisis
        ]

        for start, end in crisis_periods:
            ax.axvspan(start, end, color='red', alpha=0.1, label='Crisis' if start == crisis_periods[0][0] else '')

        ax.set_xlabel('Test Window Start Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Out-of-Sample R²', fontsize=12, fontweight='bold')
        ax.set_title('Figure E.1: Out-of-Sample R² Over Rolling Windows', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add horizontal line at 0
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_forecast_error_distribution(
        self,
        df: pd.DataFrame,
        recommended_spec: str,
        hierarchical_results: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure E.2: Forecast error distribution.

        Histogram of forecast errors with normal overlay and Q-Q plot.

        Args:
            df: Full dataset with predictions
            recommended_spec: Name of recommended specification
            hierarchical_results: Hierarchical test results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])

        ax_hist = fig.add_subplot(gs[0])
        ax_qq = fig.add_subplot(gs[1])
        ax_box = fig.add_subplot(gs[2])

        # Compute forecast errors (residuals)
        y_actual = df['oas_pct_change'].values
        y_pred = self._get_predictions(df, recommended_spec, hierarchical_results)

        forecast_errors = y_actual - y_pred

        # Panel A: Histogram
        ax_hist.hist(
            forecast_errors,
            bins=50,
            density=True,
            alpha=0.7,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5,
            label='Forecast errors'
        )

        # Overlay normal distribution
        mu = np.mean(forecast_errors)
        sigma = np.std(forecast_errors)
        x = np.linspace(forecast_errors.min(), forecast_errors.max(), 100)
        normal_pdf = stats.norm.pdf(x, mu, sigma)

        ax_hist.plot(
            x,
            normal_pdf,
            'r--',
            linewidth=2,
            label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})',
            alpha=0.8
        )

        ax_hist.set_xlabel('Forecast Error (Actual − Predicted)', fontsize=11, fontweight='bold')
        ax_hist.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax_hist.set_title('Panel A: Distribution', fontsize=12, fontweight='bold')
        ax_hist.legend(loc='best', fontsize=9)
        ax_hist.grid(True, alpha=0.3)

        # Panel B: Q-Q plot
        stats.probplot(forecast_errors, dist="norm", plot=ax_qq)
        ax_qq.set_title('Panel B: Q-Q Plot', fontsize=12, fontweight='bold')
        ax_qq.grid(True, alpha=0.3)

        # Panel C: Box plot by regime
        # Classify by OAS level
        oas_terciles = pd.qcut(df['oas'], q=3, labels=['IG Narrow', 'IG Wide', 'HY'])
        df_errors = pd.DataFrame({
            'error': forecast_errors,
            'regime': oas_terciles
        })

        df_errors.boxplot(column='error', by='regime', ax=ax_box)
        ax_box.set_xlabel('Regime', fontsize=11, fontweight='bold')
        ax_box.set_ylabel('Forecast Error', fontsize=11, fontweight='bold')
        ax_box.set_title('Panel C: Errors by Regime', fontsize=12, fontweight='bold')
        ax_box.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        plt.sca(ax_box)
        plt.xticks(rotation=0)

        fig.suptitle(
            f'Figure E.2: Forecast Error Distribution ({recommended_spec})',
            fontsize=14,
            fontweight='bold',
            y=1.00
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_predicted_vs_actual(
        self,
        df: pd.DataFrame,
        recommended_spec: str,
        hierarchical_results: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure E.3: Scatter plot of predicted vs actual spread changes.

        Color-coded by regime to show where model works best/worst.

        Args:
            df: Full dataset
            recommended_spec: Name of recommended specification
            hierarchical_results: Hierarchical test results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Get predictions
        y_actual = df['oas_pct_change'].values
        y_pred = self._get_predictions(df, recommended_spec, hierarchical_results)

        # Classify by regime
        # Use OAS and bucket info
        regime_labels = []
        regime_colors = []

        for idx, row in df.iterrows():
            oas = row['oas']
            bucket = row.get('bucket', '')

            if oas < 150:
                regime = 'IG Narrow'
                color = 'blue'
            elif oas < 300:
                regime = 'IG Wide'
                color = 'green'
            elif oas < 600:
                regime = 'HY Moderate'
                color = 'orange'
            else:
                regime = 'Distressed'
                color = 'red'

            regime_labels.append(regime)
            regime_colors.append(color)

        df['regime'] = regime_labels
        df['regime_color'] = regime_colors

        # Scatter plot
        for regime in df['regime'].unique():
            mask = df['regime'] == regime
            ax.scatter(
                y_pred[mask],
                y_actual[mask],
                c=df.loc[mask, 'regime_color'].iloc[0],
                s=15,
                alpha=0.4,
                label=regime,
                edgecolors='none'
            )

        # 45-degree line
        min_val = min(y_pred.min(), y_actual.min())
        max_val = max(y_pred.max(), y_actual.max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'k--',
            linewidth=2,
            label='Perfect fit (45°)',
            alpha=0.7
        )

        # Compute R²
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - ss_res / ss_tot

        ax.set_xlabel('Predicted Spread Change (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Spread Change (%)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Figure E.3: Predicted vs Actual Spread Changes\n{recommended_spec} (R² = {r2:.3f})',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_specification_comparison(
        self,
        oos_results: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Supplementary: Bar chart comparing specifications on OOS metrics.

        Shows OOS R² and RMSE side-by-side for all specifications.

        Args:
            oos_results: OOS validation results
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_r2, ax_rmse = axes

        oos_summary = oos_results.get('oos_summary', {})

        specs = list(oos_summary.keys())
        r2_values = [oos_summary[s]['avg_r2_oos'] for s in specs]
        rmse_values = [oos_summary[s]['avg_rmse_oos'] for s in specs]

        x_pos = np.arange(len(specs))
        colors = ['gray', 'blue', 'green', 'orange', 'red'][:len(specs)]

        # Panel A: R²
        bars_r2 = ax_r2.bar(x_pos, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax_r2.set_xticks(x_pos)
        ax_r2.set_xticklabels(specs, fontsize=10, rotation=15, ha='right')
        ax_r2.set_ylabel('Out-of-Sample R²', fontsize=12, fontweight='bold')
        ax_r2.set_title('Panel A: Forecast Accuracy (R²)', fontsize=13, fontweight='bold', pad=15)
        ax_r2.grid(True, alpha=0.3, axis='y')
        ax_r2.axhline(0, color='black', linestyle=':', linewidth=1)

        # Annotate values
        for i, (bar, val) in enumerate(zip(bars_r2, r2_values)):
            ax_r2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        # Panel B: RMSE
        bars_rmse = ax_rmse.bar(x_pos, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax_rmse.set_xticks(x_pos)
        ax_rmse.set_xticklabels(specs, fontsize=10, rotation=15, ha='right')
        ax_rmse.set_ylabel('Out-of-Sample RMSE', fontsize=12, fontweight='bold')
        ax_rmse.set_title('Panel B: Forecast Error (RMSE)', fontsize=13, fontweight='bold', pad=15)
        ax_rmse.grid(True, alpha=0.3, axis='y')

        # Annotate values
        for i, (bar, val) in enumerate(zip(bars_rmse, rmse_values)):
            ax_rmse.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        fig.suptitle(
            'Specification Comparison: Out-of-Sample Performance',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def _get_predictions(
        self,
        df: pd.DataFrame,
        spec_name: str,
        hierarchical_results: Dict
    ) -> np.ndarray:
        """
        Generate predictions for a given specification.

        Returns predicted spread changes.
        """
        if spec_name == 'Standard DTS':
            y_pred = df['f_DTS'].values

        elif spec_name == 'Pure Merton':
            y_pred = df['lambda_Merton'].values * df['f_DTS'].values

        elif spec_name == 'Calibrated Merton':
            level3 = hierarchical_results.get('level3', {})
            c0 = level3.get('c0', 1.0)
            c_s = level3.get('c_s', -0.25)

            lambda_s_adj = df['lambda_s'] ** (c_s / -0.25)
            lambda_adj = df['lambda_T'] * lambda_s_adj
            y_pred = c0 * lambda_adj * df['f_DTS']

        elif spec_name == 'Empirical':
            # Use lambda_Merton as proxy (in practice would use full empirical)
            y_pred = df['lambda_Merton'].values * df['f_DTS'].values

        elif spec_name == 'Time-varying':
            level5 = hierarchical_results.get('level5', {})
            gamma_vix = level5.get('gamma_vix', 0)
            gamma_oas = level5.get('gamma_oas', 0)

            macro_adj = np.exp(
                gamma_vix * df['vix'] / 100 +
                gamma_oas * np.log(df['oas_index'])
            )

            lambda_tv = df['lambda_Merton'] * macro_adj
            y_pred = lambda_tv * df['f_DTS']

        else:
            # Fallback
            y_pred = df['lambda_Merton'].values * df['f_DTS'].values

        return np.nan_to_num(y_pred, nan=0.0)

    def create_all_stageE_figures(
        self,
        df: pd.DataFrame,
        oos_results: Dict,
        hierarchical_results: Dict,
        recommended_spec: str,
        output_prefix: str = 'stageE'
    ) -> Dict:
        """
        Generate all Stage E figures at once.

        Args:
            df: Full dataset
            oos_results: OOS validation results
            hierarchical_results: Hierarchical test results
            recommended_spec: Name of recommended specification
            output_prefix: Prefix for output filenames

        Returns:
            Dictionary with figure objects
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        figures = {}

        # Figure E.1: OOS R² over time
        fig1 = self.plot_oos_r2_over_time(
            oos_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig1_oos_r2.png'
        )
        figures['fig1_oos_r2'] = fig1

        # Figure E.2: Forecast error distribution
        fig2 = self.plot_forecast_error_distribution(
            df,
            recommended_spec,
            hierarchical_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig2_error_dist.png'
        )
        figures['fig2_error_dist'] = fig2

        # Figure E.3: Predicted vs actual
        fig3 = self.plot_predicted_vs_actual(
            df,
            recommended_spec,
            hierarchical_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig3_pred_vs_actual.png'
        )
        figures['fig3_pred_vs_actual'] = fig3

        # Supplementary: Specification comparison
        fig4 = self.plot_specification_comparison(
            oos_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig4_spec_comparison.png'
        )
        figures['fig4_spec_comparison'] = fig4

        return figures
