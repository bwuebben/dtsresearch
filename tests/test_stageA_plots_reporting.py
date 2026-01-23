"""
Test Stage A Plotting and Reporting with Mock Data

Tests the visualization and reporting modules for Stage A
(Establish Cross-Sectional Variation).

Run with: python tests/test_stageA_plots_reporting.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.visualization.stageA_plots import StageAVisualizer
from dts_research.utils.reportingA import StageAReporter


def generate_mock_spec_a1_results() -> pd.DataFrame:
    """Generate mock bucket-level beta estimates (Specification A.1)."""
    np.random.seed(42)

    # Use rating buckets that match expected format
    ig_ratings = ['AAA/AA', 'A', 'BBB']
    hy_ratings = ['BB', 'B', 'CCC']
    maturities = ['1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10y+']

    data = []

    # IG bonds
    for rating in ig_ratings:
        for mat in maturities:
            rating_effect = {'AAA/AA': 0.75, 'A': 0.85, 'BBB': 0.95}[rating]
            mat_effect = {'1-2y': 1.3, '2-3y': 1.1, '3-5y': 1.0, '5-7y': 0.9, '7-10y': 0.85, '10y+': 0.8}[mat]

            beta = rating_effect * mat_effect + np.random.normal(0, 0.1)
            se_beta = abs(beta) * 0.1 + 0.02

            data.append({
                'rating_bucket': rating,
                'maturity_bucket': mat,
                'bucket_id': f'{rating}_{mat}',
                'beta': beta,
                'se_beta': se_beta,
                't_stat': beta / se_beta,
                'p_value': 0.001 if abs(beta / se_beta) > 2 else 0.1,
                'r_squared': np.random.uniform(0.15, 0.45),
                'n_observations': np.random.randint(500, 5000),
                'mean_spread': np.random.uniform(50, 200),
                'mean_maturity': {'1-2y': 1.5, '2-3y': 2.5, '3-5y': 4.0, '5-7y': 6.0, '7-10y': 8.5, '10y+': 15.0}[mat],
                'is_ig': True
            })

    # HY bonds
    for rating in hy_ratings:
        for mat in maturities:
            rating_effect = {'BB': 1.1, 'B': 1.25, 'CCC': 1.4}[rating]
            mat_effect = {'1-2y': 1.3, '2-3y': 1.1, '3-5y': 1.0, '5-7y': 0.9, '7-10y': 0.85, '10y+': 0.8}[mat]

            beta = rating_effect * mat_effect + np.random.normal(0, 0.15)
            se_beta = abs(beta) * 0.12 + 0.03

            data.append({
                'rating_bucket': rating,
                'maturity_bucket': mat,
                'bucket_id': f'{rating}_{mat}',
                'beta': beta,
                'se_beta': se_beta,
                't_stat': beta / se_beta,
                'p_value': 0.001 if abs(beta / se_beta) > 2 else 0.1,
                'r_squared': np.random.uniform(0.20, 0.50),
                'n_observations': np.random.randint(300, 3000),
                'mean_spread': np.random.uniform(250, 600),
                'mean_maturity': {'1-2y': 1.5, '2-3y': 2.5, '3-5y': 4.0, '5-7y': 6.0, '7-10y': 8.5, '10y+': 15.0}[mat],
                'is_ig': False
            })

    return pd.DataFrame(data)


def generate_mock_f_tests() -> list:
    """Generate mock F-test results for beta equality."""
    tests = [
        {
            'test': 'Overall Beta Equality',
            'hypothesis': 'H0: All betas are equal',
            'f_statistic': 4.52,
            'p_value': 0.0001,
            'df_num': 35,
            'df_denom': 50000,
            'reject_h0': True,
            'interpretation': 'Betas differ significantly across buckets'
        },
        {
            'test': 'Rating Effect (within maturity)',
            'hypothesis': 'H0: Betas equal across ratings',
            'f_statistic': 3.21,
            'p_value': 0.008,
            'df_num': 5,
            'df_denom': 50000,
            'reject_h0': True,
            'interpretation': 'Ratings affect DTS sensitivity'
        },
        {
            'test': 'Maturity Effect (within rating)',
            'hypothesis': 'H0: Betas equal across maturities',
            'f_statistic': 5.67,
            'p_value': 0.0002,
            'df_num': 5,
            'df_denom': 50000,
            'reject_h0': True,
            'interpretation': 'Maturity affects DTS sensitivity'
        },
        {
            'test': 'Rating × Maturity Interaction',
            'hypothesis': 'H0: No interaction effects',
            'f_statistic': 1.45,
            'p_value': 0.12,
            'df_num': 25,
            'df_denom': 50000,
            'reject_h0': False,
            'interpretation': 'Interaction effects not significant'
        }
    ]
    return tests


def generate_mock_spec_a2_results() -> dict:
    """Generate mock continuous characteristic regression results (Specification A.2)."""
    return {
        'combined': {
            'r_squared': 0.35,
            'adj_r_squared': 0.34,
            'n_observations': 15000,
            # Gamma coefficients for beta surface: β = γ₀ + γ_M·M + γ_s·s + γ_M²·M² + γ_Ms·M·s
            'gamma_0': 0.85,
            'gamma_M': -0.05,
            'gamma_s': 0.0008,
            'gamma_M2': 0.002,
            'gamma_Ms': -0.00005,
            'se_gamma_0': 0.05,
            'se_gamma_M': 0.01,
            'se_gamma_s': 0.0002,
            'se_gamma_M2': 0.001,
            'se_gamma_Ms': 0.00002,
            'coefficients': {
                'intercept': 0.85,
                'log_spread': 0.15,
                'log_maturity': -0.12,
                'spread_x_maturity': 0.02
            },
            'std_errors': {
                'intercept': 0.05,
                'log_spread': 0.02,
                'log_maturity': 0.03,
                'spread_x_maturity': 0.01
            },
            'p_values': {
                'intercept': 0.001,
                'log_spread': 0.001,
                'log_maturity': 0.001,
                'spread_x_maturity': 0.05
            }
        },
        'ig': {
            'r_squared': 0.28,
            'adj_r_squared': 0.27,
            'n_observations': 10000,
            'gamma_0': 0.75,
            'gamma_M': -0.04,
            'gamma_s': 0.0006,
            'gamma_M2': 0.001,
            'gamma_Ms': -0.00003
        },
        'hy': {
            'r_squared': 0.42,
            'adj_r_squared': 0.41,
            'n_observations': 5000,
            'gamma_0': 0.95,
            'gamma_M': -0.06,
            'gamma_s': 0.001,
            'gamma_M2': 0.003,
            'gamma_Ms': -0.00008
        }
    }


def generate_mock_econ_significance() -> dict:
    """Generate mock economic significance metrics."""
    return {
        'min_beta': 0.65,
        'max_beta': 1.52,
        'range': 0.87,
        'ratio_max_min': 2.34,
        'mean': 0.98,
        'median': 0.95,
        'std': 0.22,
        'iqr': 0.28,
        'cv': 0.22,
        'p10': 0.72,
        'p25': 0.82,
        'p50': 0.95,
        'p75': 1.10,
        'p90': 1.28,
        'pct_below_0_8': 15.2,
        'pct_above_1_2': 18.5,
        'interpretation': 'Significant variation in DTS betas across buckets. Ratio of 2.34x suggests economically meaningful differences.'
    }


def generate_mock_ig_hy_comparison() -> dict:
    """Generate mock IG vs HY comparison."""
    return {
        'ig_mean': 0.85,
        'ig_std': 0.15,
        'ig_n': 24,
        'hy_mean': 1.15,
        'hy_std': 0.25,
        'hy_n': 12,
        'std_ratio_ig_hy': 0.60,
        'mean_diff': 0.30,
        't_stat': 4.52,
        'p_value': 0.0001,
        'interpretation': 'HY bonds show higher and more variable DTS sensitivity than IG'
    }


def generate_mock_decision() -> str:
    """Generate mock Stage A decision."""
    return """
STAGE A DECISION: PROCEED TO STAGE B
=====================================

Statistical Evidence:
- Overall F-test: F = 4.52, p < 0.0001 → REJECT H0 (betas are equal)
- Rating effect: Significant (p = 0.008)
- Maturity effect: Significant (p = 0.0002)

Economic Significance:
- Beta range: 0.65 to 1.52 (2.3x variation)
- Cross-sectional std: 0.22
- IQR: 0.28

Regime Comparison (IG vs HY):
- IG std: 0.15, HY std: 0.25
- HY shows higher variation → Supports Stage B testing

RECOMMENDATION:
Proceed to Stage B to test whether Merton's structural model
explains the documented cross-sectional variation in DTS betas.

Key questions for Stage B:
1. Does λ^Merton capture the rating effect?
2. Does λ^Merton capture the maturity effect?
3. Is β_Merton ≈ 1 (theory validated)?
"""


def main():
    """Generate mock data and run plotting/reporting for Stage A."""
    print("=" * 80)
    print("STAGE A MOCK DATA TEST")
    print("Testing plotting and reporting with mock data")
    print("=" * 80)

    # Generate mock results
    print("\nGenerating mock data...")
    spec_a1_results = generate_mock_spec_a1_results()
    all_tests = generate_mock_f_tests()
    spec_a2_results = generate_mock_spec_a2_results()
    econ_sig = generate_mock_econ_significance()
    ig_hy_comp = generate_mock_ig_hy_comparison()
    decision = generate_mock_decision()

    print(f"  Spec A.1: {len(spec_a1_results)} bucket betas")
    print(f"  Spec A.2: R² = {spec_a2_results['combined']['r_squared']:.3f}")
    print(f"  F-tests: {len(all_tests)} tests")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    output_dir_figures = 'output/stageA_test_figures'
    visualizer = StageAVisualizer(output_dir=output_dir_figures)

    figures = visualizer.create_all_stageA_figures(
        spec_a1_results,
        spec_a2_results,
        output_prefix='stageA_test'
    )

    print(f"\nCreated {len(figures)} figures in {output_dir_figures}/")

    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)

    output_dir_reports = 'output/stageA_test_reports'
    reporter = StageAReporter(output_dir=output_dir_reports)

    reporter.save_all_reports(
        spec_a1_results,
        all_tests,
        spec_a2_results,
        econ_sig,
        ig_hy_comp,
        decision,
        prefix='stageA_test'
    )

    print(f"\nReports saved to {output_dir_reports}/")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nFigures: {output_dir_figures}/")
    print(f"Reports: {output_dir_reports}/")


if __name__ == '__main__':
    main()
