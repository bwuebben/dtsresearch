"""
Test Stage C Plotting and Reporting with Mock Data

Tests the visualization and reporting modules for Stage C
(Does Static Merton Suffice or Do We Need Time-Variation?).

Run with: python tests/test_stageC_plots_reporting.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.visualization.stageC_plots import StageCVisualizer
from dts_research.utils.reportingC import StageCReporter


def generate_mock_rolling_results() -> pd.DataFrame:
    """Generate mock rolling window beta estimates."""
    np.random.seed(42)

    dates = pd.date_range('2015-01-01', '2024-12-31', freq='ME')
    n_windows = len(dates)

    # Beta varies over time with mean ~1.0, higher during crises
    base_beta = np.ones(n_windows)

    # COVID spike (2020-03 to 2020-06)
    covid_start = (dates >= '2020-03-01') & (dates <= '2020-06-01')
    base_beta[covid_start] = 1.3

    # 2022 volatility
    vol_2022 = (dates >= '2022-02-01') & (dates <= '2022-10-01')
    base_beta[vol_2022] = 1.15

    data = []
    for i, date in enumerate(dates):
        beta_w = base_beta[i] + np.random.normal(0, 0.08)
        se_beta = 0.05 + np.random.uniform(0, 0.02)
        # window_start is 12 months before window_end
        window_start = date - pd.DateOffset(months=12)

        data.append({
            'window_start': window_start,
            'window_end': date,
            'beta_w': beta_w,
            'se_beta': se_beta,
            'ci_lower': beta_w - 1.96 * se_beta,
            'ci_upper': beta_w + 1.96 * se_beta,
            't_stat': beta_w / se_beta,
            'r_squared': np.random.uniform(0.65, 0.80),
            'n_obs': np.random.randint(10000, 20000),
            'vix_avg': 15 + 10 * base_beta[i] - 10 + np.random.normal(0, 2),
            'oas_avg': 100 + 50 * base_beta[i] - 50 + np.random.normal(0, 10)
        })

    return pd.DataFrame(data)


def generate_mock_chow_test() -> dict:
    """Generate mock Chow test results."""
    return {
        'f_statistic': 2.85,
        'p_value': 0.015,
        'df_num': 12,
        'df_denom': 150000,
        'reject_h0': True,
        'stable': False,
        'interpretation': 'Chow test rejects stability (p=0.015). Beta varies over time.'
    }


def generate_mock_macro_driver_results() -> dict:
    """Generate mock macro driver analysis results."""
    return {
        'coefficients': {
            'intercept': 0.85,
            'delta_VIX': 0.008,
            'delta_OAS': -0.0003
        },
        'std_errors': {
            'intercept_se': 0.05,
            'delta_VIX_se': 0.002,
            'delta_OAS_se': 0.0001
        },
        'p_values': {
            'p_intercept': 0.001,
            'p_delta_VIX': 0.001,
            'p_delta_OAS': 0.003
        },
        't_stats': {
            't_intercept': 17.0,
            't_delta_VIX': 4.0,
            't_delta_OAS': -3.0
        },
        'r_squared': 0.45,
        'adj_r_squared': 0.43,
        'n_windows': 120,
        'economic_significance': {
            'effect_vix_pct': 24.0,
            'effect_oas_pct': -15.0,
            'vix_1std_effect': 0.08,
            'oas_1std_effect': -0.03
        },
        'interpretation': 'VIX explains 24% of beta variation. Higher VIX → higher beta (flight to quality dampens).'
    }


def generate_mock_maturity_results() -> dict:
    """Generate mock maturity-specific time-variation results."""
    return {
        'by_maturity': {
            '1-2y': {
                'delta_VIX': 0.012,
                'se_delta_VIX': 0.003,
                't_stat': 4.0,
                'p_value': 0.001,
                'effect_pct': 36.0
            },
            '3-5y': {
                'delta_VIX': 0.008,
                'se_delta_VIX': 0.002,
                't_stat': 4.0,
                'p_value': 0.001,
                'effect_pct': 24.0
            },
            '7-10y': {
                'delta_VIX': 0.005,
                'se_delta_VIX': 0.002,
                't_stat': 2.5,
                'p_value': 0.02,
                'effect_pct': 15.0
            }
        },
        'pattern_test': {
            'pattern': 'Short > Medium > Long',
            'confirms_theory': True,
            'spearman_rho': -0.85,
            'p_value': 0.03,
            'interpretation': 'VIX effect decreases with maturity as Merton predicts.'
        }
    }


def generate_mock_macro_data() -> pd.DataFrame:
    """Generate mock macro data."""
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='D')
    n = len(dates)

    np.random.seed(42)

    # VIX
    vix = 15 + 5 * np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.normal(0, 2, n)
    covid_mask = (dates >= '2020-03-01') & (dates <= '2020-06-01')
    vix[covid_mask] = 40 + np.random.normal(0, 5, covid_mask.sum())

    # OAS
    oas = 100 + 30 * np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.normal(0, 10, n)
    oas[covid_mask] = 200 + np.random.normal(0, 20, covid_mask.sum())

    return pd.DataFrame({
        'date': dates,
        'vix': np.maximum(vix, 10),
        'oas_index': np.maximum(oas, 50),
        'r_10y': 2.0 + 0.5 * np.sin(np.arange(n) * 2 * np.pi / (365 * 3)) + np.random.normal(0, 0.1, n)
    })


def generate_mock_decision() -> str:
    """Generate mock Stage C decision."""
    return """
STAGE C DECISION: STATIC MERTON WITH CRISIS OVERLAY
====================================================

Evidence Summary:
- Chow test: p = 0.015 → Beta varies over time
- VIX effect: 24% of beta variation explained
- Pattern: Short maturity more sensitive (confirms Merton)

Interpretation:
Static Merton is adequate for normal periods, but crisis periods
show elevated betas. The pattern of VIX sensitivity by maturity
(short > long) confirms Merton's prediction.

RECOMMENDATION:
Use static λ^Merton as baseline, with optional crisis adjustment:
  - Normal (VIX < 25): λ = λ^Merton
  - Elevated (25 ≤ VIX < 35): λ = 1.1 × λ^Merton
  - Crisis (VIX ≥ 35): λ = 1.25 × λ^Merton

This captures the economic significance of time-variation (24%)
while maintaining simplicity.

Proceed to Stage D (robustness testing).
"""


def main():
    """Generate mock data and run plotting/reporting for Stage C."""
    print("=" * 80)
    print("STAGE C MOCK DATA TEST")
    print("Testing plotting and reporting with mock data")
    print("=" * 80)

    # Generate mock results
    print("\nGenerating mock data...")
    rolling_combined = generate_mock_rolling_results()
    chow_combined = generate_mock_chow_test()
    macro_driver_results = generate_mock_macro_driver_results()
    maturity_results = generate_mock_maturity_results()
    macro_data = generate_mock_macro_data()
    decision = generate_mock_decision()

    # Create mock regression_data for visualizations
    np.random.seed(42)
    regression_data = pd.DataFrame({
        'date': pd.date_range('2015-01-01', periods=1000, freq='W'),
        'spread_regime': np.random.choice(['IG', 'HY'], 1000),
        'oas': np.random.uniform(50, 400, 1000),
        'vix': np.random.uniform(12, 40, 1000),
        'oas_pct_change': np.random.normal(0, 0.02, 1000)
    })

    print(f"  Rolling windows: {len(rolling_combined)}")
    print(f"  Chow test: p = {chow_combined['p_value']:.4f}")
    print(f"  VIX effect: {macro_driver_results['economic_significance']['effect_vix_pct']:.1f}%")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    output_dir_figures = 'output/stageC_test_figures'
    visualizer = StageCVisualizer(output_dir=output_dir_figures)

    # Mock IG/HY rolling results
    rolling_ig = rolling_combined.copy()
    rolling_ig['beta_w'] = rolling_ig['beta_w'] * 0.95
    rolling_ig['ci_lower'] = rolling_ig['beta_w'] - 1.96 * rolling_ig['se_beta']
    rolling_ig['ci_upper'] = rolling_ig['beta_w'] + 1.96 * rolling_ig['se_beta']
    rolling_hy = rolling_combined.copy()
    rolling_hy['beta_w'] = rolling_hy['beta_w'] * 1.10
    rolling_hy['ci_lower'] = rolling_hy['beta_w'] - 1.96 * rolling_hy['se_beta']
    rolling_hy['ci_upper'] = rolling_hy['beta_w'] + 1.96 * rolling_hy['se_beta']

    figures = visualizer.create_all_stageC_figures(
        rolling_combined,
        macro_data,
        macro_driver_results,
        regression_data,
        rolling_ig,
        rolling_hy,
        output_prefix='stageC_test'
    )

    print(f"\nCreated {len(figures)} figures in {output_dir_figures}/")

    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)

    output_dir_reports = 'output/stageC_test_reports'
    reporter = StageCReporter(output_dir=output_dir_reports)

    # Mock chow tests for IG/HY
    chow_ig = chow_combined.copy()
    chow_ig['p_value'] = 0.08
    chow_hy = chow_combined.copy()
    chow_hy['p_value'] = 0.002

    reporter.save_all_reports(
        rolling_combined,
        chow_combined,
        macro_driver_results,
        maturity_results,
        decision,
        rolling_ig,
        rolling_hy,
        chow_ig,
        chow_hy,
        prefix='stageC_test'
    )

    print(f"\nReports saved to {output_dir_reports}/")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nFigures: {output_dir_figures}/")
    print(f"Reports: {output_dir_reports}/")


if __name__ == '__main__':
    main()
