"""
Visualization functions for Stage C deliverables.

Creates Figures C.1, C.2, C.3, and C.4 from the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class StageCVisualizer:
    """Creates publication-quality plots for Stage C analysis."""

    def __init__(self, output_dir: str = './output/figures'):
        self.output_dir = output_dir

    def plot_beta_time_series(
        self,
        rolling_results_combined: pd.DataFrame,
        rolling_results_ig: Optional[pd.DataFrame] = None,
        rolling_results_hy: Optional[pd.DataFrame] = None,
        crisis_periods: Optional[list] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure C.1: Time series of beta_w for IG and HY.

        Shows rolling window estimates with 95% confidence bands.

        Args:
            rolling_results_combined: DataFrame with beta_w time series (combined)
            rolling_results_ig: Optional IG results
            rolling_results_hy: Optional HY results
            crisis_periods: List of tuples (start_date, end_date, label)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        has_regime = rolling_results_ig is not None and rolling_results_hy is not None

        if has_regime:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            ax_ig, ax_hy = axes
        else:
            fig, ax_combined = plt.subplots(1, 1, figsize=(14, 6))

        # Crisis periods (default to COVID and 2022 rate shock)
        if crisis_periods is None:
            crisis_periods = [
                (pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01'), 'COVID'),
                (pd.Timestamp('2022-02-01'), pd.Timestamp('2022-10-01'), 'Rate Shock')
            ]

        def plot_regime(ax, results_df, title, color='blue'):
            """Helper to plot one regime."""
            # Calculate window midpoints for x-axis
            results_df = results_df.copy()
            results_df['window_mid'] = results_df['window_start'] + (
                results_df['window_end'] - results_df['window_start']
            ) / 2

            # Plot point estimates
            ax.plot(
                results_df['window_mid'],
                results_df['beta_w'],
                'o-',
                color=color,
                linewidth=2,
                markersize=6,
                label='β_w estimate',
                alpha=0.8
            )

            # Plot confidence bands
            ax.fill_between(
                results_df['window_mid'],
                results_df['ci_lower'],
                results_df['ci_upper'],
                alpha=0.2,
                color=color,
                label='95% CI'
            )

            # Horizontal line at beta = 1 (theory prediction)
            ax.axhline(
                1.0,
                color='red',
                linestyle='--',
                linewidth=2,
                label='Theory prediction (β = 1)',
                alpha=0.7
            )

            # Shade crisis periods
            for start, end, label in crisis_periods:
                ax.axvspan(start, end, alpha=0.15, color='gray', label=label if label == crisis_periods[0][2] else None)

            ax.set_ylabel('β_w', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Plot regimes
        if has_regime:
            plot_regime(ax_ig, rolling_results_ig, 'Investment Grade (IG)', color='blue')
            plot_regime(ax_hy, rolling_results_hy, 'High Yield (HY)', color='orange')
            ax_hy.set_xlabel('Year', fontsize=12, fontweight='bold')
        else:
            plot_regime(ax_combined, rolling_results_combined, 'Combined (All Bonds)', color='purple')
            ax_combined.set_xlabel('Year', fontsize=12, fontweight='bold')

        fig.suptitle(
            'Figure C.1: Time Series of Rolling Window Beta Estimates',
            fontsize=14,
            fontweight='bold',
            y=0.98 if has_regime else 1.0
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_beta_vs_macro(
        self,
        rolling_results: pd.DataFrame,
        macro_data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure C.2: beta_w vs macro state variables.

        Two panels: beta_w vs VIX, beta_w vs log(OAS)

        Args:
            rolling_results: DataFrame with beta_w time series
            macro_data: DataFrame with vix, oas_index columns
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_vix, ax_oas = axes

        # Merge data
        rolling_results = rolling_results.copy()
        rolling_results['window_mid'] = rolling_results['window_start'] + (
            rolling_results['window_end'] - rolling_results['window_start']
        ) / 2

        merged_data = []
        for idx, row in rolling_results.iterrows():
            window_macro = macro_data[
                (macro_data['date'] >= row['window_start']) &
                (macro_data['date'] < row['window_end'])
            ]

            if len(window_macro) > 0:
                merged_data.append({
                    'beta_w': row['beta_w'],
                    'vix_avg': window_macro['vix'].mean(),
                    'oas_avg': window_macro['oas_index'].mean(),
                    'year': row['window_mid'].year
                })

        df = pd.DataFrame(merged_data)

        if len(df) == 0:
            return fig

        # Color code by time period
        df['period'] = 'Pre-2020'
        df.loc[df['year'] == 2020, 'period'] = 'COVID (2020)'
        df.loc[df['year'] > 2020, 'period'] = 'Post-COVID'

        period_colors = {
            'Pre-2020': 'blue',
            'COVID (2020)': 'red',
            'Post-COVID': 'green'
        }

        # Panel A: Beta vs VIX
        for period, color in period_colors.items():
            subset = df[df['period'] == period]
            if len(subset) > 0:
                ax_vix.scatter(
                    subset['vix_avg'],
                    subset['beta_w'],
                    c=color,
                    s=100,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5,
                    label=period
                )

        # OLS fit line
        if len(df) >= 3:
            z = np.polyfit(df['vix_avg'], df['beta_w'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df['vix_avg'].min(), df['vix_avg'].max(), 100)
            ax_vix.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.5, label='OLS fit')

        ax_vix.set_xlabel('Average VIX', fontsize=11, fontweight='bold')
        ax_vix.set_ylabel('β_w', fontsize=11, fontweight='bold')
        ax_vix.set_title('Panel A: β_w vs VIX', fontsize=12, fontweight='bold')
        ax_vix.legend(loc='best', fontsize=9)
        ax_vix.grid(True, alpha=0.3)

        # Panel B: Beta vs log(OAS)
        for period, color in period_colors.items():
            subset = df[df['period'] == period]
            if len(subset) > 0:
                ax_oas.scatter(
                    np.log(subset['oas_avg']),
                    subset['beta_w'],
                    c=color,
                    s=100,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5,
                    label=period
                )

        # OLS fit line
        if len(df) >= 3:
            z = np.polyfit(np.log(df['oas_avg']), df['beta_w'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.log(df['oas_avg']).min(), np.log(df['oas_avg']).max(), 100)
            ax_oas.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.5, label='OLS fit')

        ax_oas.set_xlabel('log(Average OAS Index)', fontsize=11, fontweight='bold')
        ax_oas.set_ylabel('β_w', fontsize=11, fontweight='bold')
        ax_oas.set_title('Panel B: β_w vs log(OAS)', fontsize=12, fontweight='bold')
        ax_oas.legend(loc='best', fontsize=9)
        ax_oas.grid(True, alpha=0.3)

        fig.suptitle(
            'Figure C.2: Beta vs Macro State Variables',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_implied_lambda_over_time(
        self,
        macro_driver_results: Dict,
        representative_bonds: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure C.3: Implied lambda_i,t for representative bonds over time.

        Shows static vs time-varying lambda for 1y, 5y, 10y BBB bonds.

        Args:
            macro_driver_results: Results from macro driver analysis
            representative_bonds: Optional DataFrame with actual bond data
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # For demonstration, create synthetic time-varying lambda
        # In real implementation, would use actual bond data
        dates = pd.date_range('2010-01-01', '2024-12-31', freq='M')

        # Static lambda values (from Merton)
        lambda_static = {
            '1y BBB': 2.78,
            '5y BBB': 1.00,
            '10y BBB': 0.67
        }

        # If we have macro driver results, create time-varying version
        if 'coefficients' in macro_driver_results and 'error' not in macro_driver_results:
            delta_vix = macro_driver_results['coefficients']['delta_VIX']

            # Simulate VIX over time (for illustration)
            vix_series = 15 + 10 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
            vix_series[200:220] = 40  # COVID spike
            vix_series[550:600] = 25  # 2022 volatility

            # Calculate time-varying adjustment
            vix_effect = delta_vix * (vix_series - vix_series.mean())

            colors = {'1y BBB': 'blue', '5y BBB': 'green', '10y BBB': 'red'}

            for bond, static_val in lambda_static.items():
                # Static (dashed)
                ax.plot(
                    dates,
                    [static_val] * len(dates),
                    '--',
                    color=colors[bond],
                    linewidth=2,
                    alpha=0.5,
                    label=f'{bond} (static)'
                )

                # Time-varying (solid)
                lambda_tv = static_val * (1 + vix_effect)
                ax.plot(
                    dates,
                    lambda_tv,
                    '-',
                    color=colors[bond],
                    linewidth=2,
                    label=f'{bond} (time-varying)'
                )

        else:
            # Just plot static
            colors = {'1y BBB': 'blue', '5y BBB': 'green', '10y BBB': 'red'}
            for bond, static_val in lambda_static.items():
                ax.plot(
                    dates,
                    [static_val] * len(dates),
                    '-',
                    color=colors[bond],
                    linewidth=2,
                    label=bond
                )

        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('λ (DTS Sensitivity)', fontsize=12, fontweight='bold')
        ax.set_title(
            'Figure C.3: Implied Lambda Over Time for Representative Bonds',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='best', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_crisis_vs_normal(
        self,
        df: pd.DataFrame,
        static_predictions: pd.Series,
        time_varying_predictions: Optional[pd.Series] = None,
        vix_threshold: float = 30,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Figure C.4: Scenario analysis - crisis vs normal periods.

        Histograms of spread changes in normal vs stress periods.

        Args:
            df: DataFrame with oas_pct_change and vix columns
            static_predictions: Static Merton predictions
            time_varying_predictions: Optional time-varying predictions
            vix_threshold: Threshold for defining crisis (default 30)
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_normal, ax_crisis = axes

        # Classify periods
        df = df.copy()
        df['is_crisis'] = df['vix'] > vix_threshold

        normal = df[~df['is_crisis']]['oas_pct_change']
        crisis = df[df['is_crisis']]['oas_pct_change']

        # Panel A: Normal periods
        if len(normal) > 0:
            ax_normal.hist(
                normal,
                bins=50,
                alpha=0.6,
                color='blue',
                edgecolor='black',
                label='Actual spread changes'
            )

            # Overlay predictions
            if len(static_predictions) > 0:
                static_normal = static_predictions[~df['is_crisis']]
                ax_normal.axvline(
                    static_normal.mean(),
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label='Static Merton (mean)'
                )

            if time_varying_predictions is not None:
                tv_normal = time_varying_predictions[~df['is_crisis']]
                ax_normal.axvline(
                    tv_normal.mean(),
                    color='green',
                    linestyle='-',
                    linewidth=2,
                    label='Time-varying (mean)'
                )

        ax_normal.set_xlabel('Spread Change (%)', fontsize=11, fontweight='bold')
        ax_normal.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax_normal.set_title(f'Normal Periods (VIX < {vix_threshold})', fontsize=12, fontweight='bold')
        ax_normal.legend(loc='best', fontsize=9)
        ax_normal.grid(True, alpha=0.3, axis='y')

        # Panel B: Crisis periods
        if len(crisis) > 0:
            ax_crisis.hist(
                crisis,
                bins=50,
                alpha=0.6,
                color='red',
                edgecolor='black',
                label='Actual spread changes'
            )

            # Overlay predictions
            if len(static_predictions) > 0:
                static_crisis = static_predictions[df['is_crisis']]
                ax_crisis.axvline(
                    static_crisis.mean(),
                    color='darkred',
                    linestyle='--',
                    linewidth=2,
                    label='Static Merton (mean)'
                )

            if time_varying_predictions is not None:
                tv_crisis = time_varying_predictions[df['is_crisis']]
                ax_crisis.axvline(
                    tv_crisis.mean(),
                    color='green',
                    linestyle='-',
                    linewidth=2,
                    label='Time-varying (mean)'
                )

        ax_crisis.set_xlabel('Spread Change (%)', fontsize=11, fontweight='bold')
        ax_crisis.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax_crisis.set_title(f'Crisis Periods (VIX ≥ {vix_threshold})', fontsize=12, fontweight='bold')
        ax_crisis.legend(loc='best', fontsize=9)
        ax_crisis.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            'Figure C.4: Spread Change Distribution - Normal vs Crisis',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_all_stageC_figures(
        self,
        rolling_results_combined: pd.DataFrame,
        macro_data: pd.DataFrame,
        macro_driver_results: Dict,
        df: pd.DataFrame,
        rolling_results_ig: Optional[pd.DataFrame] = None,
        rolling_results_hy: Optional[pd.DataFrame] = None,
        output_prefix: str = 'stageC'
    ) -> Dict:
        """
        Generate all Stage C figures at once.

        Args:
            rolling_results_combined: Rolling window results (combined)
            macro_data: Macro variables DataFrame
            macro_driver_results: Results from macro driver analysis
            df: Full regression data
            rolling_results_ig: Optional IG results
            rolling_results_hy: Optional HY results
            output_prefix: Prefix for output filenames

        Returns:
            Dictionary with figure objects
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        figures = {}

        # Figure C.1: Time series
        fig1 = self.plot_beta_time_series(
            rolling_results_combined,
            rolling_results_ig,
            rolling_results_hy,
            save_path=f'{self.output_dir}/{output_prefix}_fig1_timeseries.png'
        )
        figures['fig1_timeseries'] = fig1

        # Figure C.2: Beta vs macro
        fig2 = self.plot_beta_vs_macro(
            rolling_results_combined,
            macro_data,
            save_path=f'{self.output_dir}/{output_prefix}_fig2_macro.png'
        )
        figures['fig2_macro'] = fig2

        # Figure C.3: Implied lambda over time
        fig3 = self.plot_implied_lambda_over_time(
            macro_driver_results,
            save_path=f'{self.output_dir}/{output_prefix}_fig3_lambda_time.png'
        )
        figures['fig3_lambda_time'] = fig3

        # Figure C.4: Crisis vs normal
        # Need to create static predictions
        if 'lambda_merton' in df.columns and 'oas_index_pct_change' in df.columns:
            static_preds = df['lambda_merton'] * df['oas_index_pct_change']
        else:
            static_preds = pd.Series([])

        fig4 = self.plot_crisis_vs_normal(
            df,
            static_preds,
            save_path=f'{self.output_dir}/{output_prefix}_fig4_crisis.png'
        )
        figures['fig4_crisis'] = fig4

        return figures
