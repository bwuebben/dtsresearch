"""
Test Stage E Plotting and Reporting with Mock Data

Tests the visualization and reporting modules for Stage E
(Production Specification Selection).

Run with: python tests/test_stageE_plots_reporting.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.visualization.stageE_plots import StageEVisualizer
from dts_research.utils.reportingE import StageEReporter


def generate_mock_hierarchical_results() -> dict:
    """Generate mock hierarchical testing results."""
    return {
        'level1': {
            'test': 'Is there cross-sectional variation?',
            'f_statistic': 4.52,
            'p_value': 0.0001,
            'decision': 'REJECT (variation exists)',
            'reasoning': 'F-test strongly rejects equality of betas across buckets.',
            'proceed': True
        },
        'level2': {
            'test': 'Does Merton λ explain variation?',
            'beta_merton': 0.98,
            'p_value_vs_1': 0.50,
            'r_squared': 0.73,
            'decision': 'ACCEPT (β ≈ 1)',
            'reasoning': 'β_Merton = 0.98 not significantly different from 1. Theory validated.',
            'proceed': False  # Could stop here
        },
        'level3': {
            'test': 'Does calibration improve fit?',
            'delta_r2': 0.02,
            'decision': 'MARGINAL (small improvement)',
            'reasoning': 'Calibration improves R² by 2%. Minor benefit.',
            'proceed': True
        },
        'level4': {
            'test': 'Does flexible form beat Merton?',
            'delta_r2_vs_merton': 0.05,
            'delta_aic': -500,
            'decision': 'YES (but marginal)',
            'reasoning': 'Flexible form improves R² by 5% but adds 9 parameters.',
            'proceed': True
        },
        'level5': {
            'test': 'Is time-variation needed?',
            'chow_p_value': 0.015,
            'vix_effect_pct': 24.0,
            'decision': 'MODERATE (crisis adjustment helpful)',
            'reasoning': 'Time-variation significant but economically moderate (24%).',
            'proceed': False
        },
        'recommended_level': 2,
        'recommended_spec': 'Merton λ (Pure Theory)'
    }


def generate_mock_oos_results() -> dict:
    """Generate mock out-of-sample validation results."""
    np.random.seed(42)

    # Generate rolling OOS results
    dates = pd.date_range('2018-01-01', '2024-12-31', freq='QE')
    n_windows = len(dates)

    specs = ['Standard DTS', 'Pure Merton', 'Calibrated Merton', 'Empirical', 'Time-varying']
    base_r2 = {'Standard DTS': 0.70, 'Pure Merton': 0.73, 'Calibrated Merton': 0.74,
               'Empirical': 0.76, 'Time-varying': 0.77}
    base_rmse = {'Standard DTS': 0.055, 'Pure Merton': 0.050, 'Calibrated Merton': 0.048,
                 'Empirical': 0.046, 'Time-varying': 0.045}

    # Create dict keyed by spec name with list of window dicts
    oos_by_window = {}
    for spec in specs:
        windows = []
        for i, date in enumerate(dates):
            windows.append({
                'test_start': date,
                'r2_oos': base_r2[spec] + np.random.normal(0, 0.05),
                'rmse_oos': base_rmse[spec] + np.random.normal(0, 0.005),
                'n_obs_test': np.random.randint(8000, 12000)
            })
        oos_by_window[spec] = windows

    # Summary by spec
    oos_summary = {}
    for spec in specs:
        r2_vals = [w['r2_oos'] for w in oos_by_window[spec]]
        rmse_vals = [w['rmse_oos'] for w in oos_by_window[spec]]
        oos_summary[spec] = {
            'avg_r2_oos': np.mean(r2_vals),
            'std_r2_oos': np.std(r2_vals),
            'avg_rmse_oos': np.mean(rmse_vals),
            'std_rmse_oos': np.std(rmse_vals),
            'n_windows': len(oos_by_window[spec])
        }

    return {
        'oos_by_window': oos_by_window,
        'oos_summary': oos_summary
    }


def generate_mock_regime_results() -> dict:
    """Generate mock performance by regime results."""
    specs = ['Standard DTS', 'Pure Merton', 'Calibrated Merton', 'Empirical', 'Time-varying']

    regime_results = {}
    for spec in specs:
        regime_results[spec] = {
            'Low VIX (< 20)': {
                'avg_r2_oos': 0.75 if spec != 'Standard DTS' else 0.72,
                'avg_rmse_oos': 0.045,
                'n_windows': 15
            },
            'Medium VIX (20-30)': {
                'avg_r2_oos': 0.72 if spec != 'Standard DTS' else 0.68,
                'avg_rmse_oos': 0.052,
                'n_windows': 8
            },
            'High VIX (> 30)': {
                'avg_r2_oos': 0.65 if spec != 'Time-varying' else 0.70,
                'avg_rmse_oos': 0.065 if spec != 'Time-varying' else 0.055,
                'n_windows': 5
            }
        }

    return regime_results


def generate_mock_production_blueprint() -> dict:
    """Generate mock production blueprint."""
    return {
        'specification': 'Merton λ (Pure Theory)',
        'level': 2,
        'parameters': {
            'n_params': 0,
            'description': 'λ = λ^Merton(s, T) from lookup tables'
        },
        'complexity': 'Low',
        'recalibration_frequency': 'None (theory-based)',
        'implementation': 'λ_i,t = f(s_i,t, T_i,t) from Merton lookup tables',
        'performance': {
            'avg_r2_oos': 0.73,
            'avg_rmse_oos': 0.050,
            'std_r2_oos': 0.04
        },
        'advantages': [
            'Theory-based (no estimation)',
            'No recalibration needed',
            'Interpretable',
            'Robust to regime shifts'
        ],
        'limitations': [
            'Cannot capture nonlinearities',
            'No crisis adjustment',
            'Assumes Merton structure correct'
        ],
        'monitoring': {
            'frequency': 'Monthly',
            'metrics': ['Rolling R²', 'Forecast bias', 'β_Merton stability'],
            'trigger_reconsider': 'If β_Merton deviates > 15% from 1.0 for 6+ months'
        }
    }


def generate_mock_regression_data() -> pd.DataFrame:
    """Generate mock regression data for visualizations."""
    np.random.seed(42)
    n = 5000

    dates = pd.date_range('2020-01-01', periods=n // 10, freq='W')
    dates = np.tile(dates, 10)[:n]

    data = {
        'date': dates,
        'spread_change': np.random.normal(0, 0.02, n),
        'dts_factor': np.random.normal(0, 0.015, n),
        'f_DTS': np.random.normal(0, 0.015, n),
        'lambda_Merton': np.random.uniform(0.5, 1.5, n),
        'predicted_ols': np.random.normal(0, 0.018, n),
        'predicted_merton': np.random.normal(0, 0.017, n),
        'vix': np.random.uniform(12, 35, n),
        'oas': np.random.uniform(50, 400, n),
        'oas_index': np.random.uniform(100, 200, n),
        'spread_regime': np.random.choice(['IG', 'HY'], n, p=[0.7, 0.3])
    }

    # Add correlation between predicted and actual
    data['spread_change'] = 0.7 * data['predicted_merton'] + 0.3 * np.random.normal(0, 0.01, n)
    # oas_pct_change is the actual spread change used for evaluation
    data['oas_pct_change'] = data['spread_change']

    return pd.DataFrame(data)


def main():
    """Generate mock data and run plotting/reporting for Stage E."""
    print("=" * 80)
    print("STAGE E MOCK DATA TEST")
    print("Testing plotting and reporting with mock data")
    print("=" * 80)

    # Generate mock results
    print("\nGenerating mock data...")
    hierarchical_results = generate_mock_hierarchical_results()
    oos_results = generate_mock_oos_results()
    regime_results = generate_mock_regime_results()
    production_blueprint = generate_mock_production_blueprint()
    regression_data = generate_mock_regression_data()

    recommended_spec = hierarchical_results['recommended_spec']

    print(f"  Hierarchical testing: Recommended Level {hierarchical_results['recommended_level']}")
    print(f"  Recommended spec: {recommended_spec}")
    print(f"  OOS R²: {oos_results['oos_summary']['Pure Merton']['avg_r2_oos']:.3f}")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    output_dir_figures = 'output/stageE_test_figures'
    visualizer = StageEVisualizer(output_dir=output_dir_figures)

    figures = visualizer.create_all_stageE_figures(
        regression_data,
        oos_results,
        hierarchical_results,
        recommended_spec,
        output_prefix='stageE_test'
    )

    print(f"\nCreated {len(figures)} figures in {output_dir_figures}/")

    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)

    output_dir_reports = 'output/stageE_test_reports'
    reporter = StageEReporter(output_dir=output_dir_reports)

    reporter.save_all_reports(
        hierarchical_results,
        oos_results,
        regime_results,
        production_blueprint,
        prefix='stageE_test'
    )

    print(f"\nReports saved to {output_dir_reports}/")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nFigures: {output_dir_figures}/")
    print(f"Reports: {output_dir_reports}/")


if __name__ == '__main__':
    main()
