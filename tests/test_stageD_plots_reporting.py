"""
Test Stage D Plotting and Reporting with Mock Data

Tests the visualization and reporting modules for Stage D
(Robustness and Extensions).

Run with: python tests/test_stageD_plots_reporting.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.visualization.stageD_plots import StageDVisualizer
from dts_research.utils.reportingD import StageDReporter


def generate_mock_quantile_results() -> dict:
    """Generate mock quantile regression results."""
    np.random.seed(42)

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # Betas increase in tails (amplification)
    betas = {
        0.05: 1.25,  # Left tail amplified
        0.10: 1.15,
        0.25: 1.05,
        0.50: 1.00,  # Median
        0.75: 0.98,
        0.90: 1.02,
        0.95: 1.08   # Right tail slightly amplified
    }

    results_combined = pd.DataFrame([
        {
            'quantile': q,
            'beta_tau': betas[q] + np.random.normal(0, 0.02),
            'se_beta': 0.04 + np.random.uniform(0, 0.02),
            't_stat': (betas[q] + np.random.normal(0, 0.02)) / (0.04 + np.random.uniform(0, 0.02)),
            'ci_lower': betas[q] - 0.08,
            'ci_upper': betas[q] + 0.08,
            'n_obs': 150000
        }
        for q in quantiles
    ])

    tail_tests = {
        'beta_05': 1.25,
        'beta_50': 1.00,
        'beta_95': 1.08,
        'amplification_left': 1.25,
        'amplification_right': 1.08,
        'diff_left_tail': 0.25,
        'diff_right_tail': 0.08,
        'pattern': 'Left tail amplification detected',
        'p_value_left': 0.001,
        'p_value_right': 0.15
    }

    return {
        'results_combined': results_combined,
        'results_ig': results_combined.copy(),
        'results_hy': results_combined.copy(),
        'tail_tests': tail_tests
    }


def generate_mock_shock_results() -> dict:
    """Generate mock shock decomposition results."""
    shock_betas_combined = {
        'beta_global': 0.98,
        'beta_sector': 1.12,
        'beta_issuer': 1.05,
        'se_global': 0.03,
        'se_sector': 0.05,
        'se_issuer': 0.04,
        't_global': 32.67,
        't_sector': 22.40,
        't_issuer': 26.25,
        'p_global': 0.0001,
        'p_sector': 0.0001,
        'p_issuer': 0.0001,
        'r_squared': 0.78,
        'adj_r_squared': 0.77,
        'n_obs': 150000
    }

    variance_decomp = pd.DataFrame([
        {'Component': 'Global (Market)', 'Variance': 0.65, 'Pct_of_Total': 65.0},
        {'Component': 'Sector', 'Variance': 0.20, 'Pct_of_Total': 20.0},
        {'Component': 'Issuer-Specific', 'Variance': 0.10, 'Pct_of_Total': 10.0},
        {'Component': 'Residual', 'Variance': 0.05, 'Pct_of_Total': 5.0}
    ])

    shock_betas_hy = {
        'beta_global': 0.98,
        'beta_sector': 1.18,
        'beta_issuer': 1.10,
        'se_global': 0.04,
        'se_sector': 0.06,
        'se_issuer': 0.05,
        't_global': 24.50,
        't_sector': 19.67,
        't_issuer': 22.00,
        'p_global': 0.0001,
        'p_sector': 0.0001,
        'p_issuer': 0.0001,
        'n_obs': 50000
    }

    return {
        'shock_betas_combined': shock_betas_combined,
        'shock_betas_ig': shock_betas_combined.copy(),
        'shock_betas_hy': shock_betas_hy,
        'variance_decomp': variance_decomp
    }


def generate_mock_liquidity_results() -> dict:
    """Generate mock liquidity adjustment results."""
    liquidity_model = {
        'phi_0': 50.0,
        'phi_bid_ask': 15.5,
        'phi_log_size': -0.02,
        'phi_log_turnover': -0.05,
        'phi_age': 0.8,
        'r_squared': 0.25,
        'adj_r_squared': 0.24,
        'n_obs': 150000
    }

    comparison = {
        'beta_total': 0.98,
        'r2_total': 0.73,
        'beta_def': 1.02,
        'r2_def': 0.78,
        'delta_r2': 0.05,
        'improvement_pct': 6.8,
        'beta_improvement': True
    }

    by_liquidity_quartile = pd.DataFrame([
        {'Quartile': 'Q1 (Most Liquid)', 'Avg_BidAsk': 10, 'beta_total': 0.95, 'beta_def': 0.97, 'r_squared': 0.75, 'delta_r2': 0.02},
        {'Quartile': 'Q2', 'Avg_BidAsk': 25, 'beta_total': 0.98, 'beta_def': 1.02, 'r_squared': 0.74, 'delta_r2': 0.04},
        {'Quartile': 'Q3', 'Avg_BidAsk': 45, 'beta_total': 1.02, 'beta_def': 1.08, 'r_squared': 0.72, 'delta_r2': 0.06},
        {'Quartile': 'Q4 (Least Liquid)', 'Avg_BidAsk': 80, 'beta_total': 1.08, 'beta_def': 1.18, 'r_squared': 0.68, 'delta_r2': 0.10}
    ])

    return {
        'liquidity_model': liquidity_model,
        'comparison': comparison,
        'by_liquidity_quartile': by_liquidity_quartile
    }


def main():
    """Generate mock data and run plotting/reporting for Stage D."""
    print("=" * 80)
    print("STAGE D MOCK DATA TEST")
    print("Testing plotting and reporting with mock data")
    print("=" * 80)

    # Generate mock results
    print("\nGenerating mock data...")
    quantile_results = generate_mock_quantile_results()
    shock_results = generate_mock_shock_results()
    liquidity_results = generate_mock_liquidity_results()

    print(f"  Quantile regression: {len(quantile_results['results_combined'])} quantiles")
    print(f"  Left tail amplification: {quantile_results['tail_tests']['amplification_left']:.2f}x")
    print(f"  Shock decomposition: Global={shock_results['shock_betas_combined']['beta_global']:.2f}, "
          f"Sector={shock_results['shock_betas_combined']['beta_sector']:.2f}")
    print(f"  Liquidity improvement: ΔR² = {liquidity_results['comparison']['delta_r2']:.3f}")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    output_dir_figures = 'output/stageD_test_figures'
    visualizer = StageDVisualizer(output_dir=output_dir_figures)

    figures = visualizer.create_all_stageD_figures(
        quantile_results['results_combined'],
        shock_results['shock_betas_combined'],
        liquidity_results['by_liquidity_quartile'],
        shock_results['variance_decomp'],
        quantile_results.get('results_ig'),
        quantile_results.get('results_hy'),
        shock_results.get('shock_betas_ig'),
        shock_results.get('shock_betas_hy'),
        output_prefix='stageD_test'
    )

    print(f"\nCreated {len(figures)} figures in {output_dir_figures}/")

    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)

    output_dir_reports = 'output/stageD_test_reports'
    reporter = StageDReporter(output_dir=output_dir_reports)

    reporter.save_all_reports(
        quantile_results,
        shock_results,
        liquidity_results,
        prefix='stageD_test'
    )

    print(f"\nReports saved to {output_dir_reports}/")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nFigures: {output_dir_figures}/")
    print(f"Reports: {output_dir_reports}/")


if __name__ == '__main__':
    main()
