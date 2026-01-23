"""
Test Stage B Plotting and Reporting with Mock Data

Tests the visualization and reporting modules for Stage B
(Does Merton Explain the Variation?).

Run with: python tests/test_stageB_plots_reporting.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.visualization.stageB_plots import StageBVisualizer
from dts_research.utils.reportingB import StageBReporter
from dts_research.models.merton import MertonLambdaCalculator


def generate_mock_spec_b1() -> dict:
    """Generate mock Specification B.1 results (Merton constrained)."""
    return {
        'combined': {
            'beta_merton': 0.98,
            'se_beta': 0.03,
            't_stat': 32.7,
            'p_value': 0.0001,
            't_stat_vs_1': -0.67,
            't_stat_h0_beta_eq_1': -0.67,
            'p_value_h0_beta_eq_1': 0.50,
            'r_squared': 0.73,
            'adj_r_squared': 0.72,
            'rmse': 0.0245,
            'n_obs': 150000,
            'n_clusters': 200,
            'interpretation': 'β ≈ 1: Theory validated'
        },
        'ig': {
            'beta_merton': 0.95,
            'se_beta': 0.04,
            't_stat_h0_beta_eq_1': -1.25,
            'p_value_h0_beta_eq_1': 0.21,
            'r_squared': 0.68,
            'rmse': 0.0215,
            'n_obs': 100000,
            'interpretation': 'β ≈ 1: Theory validated for IG'
        },
        'hy': {
            'beta_merton': 1.05,
            'se_beta': 0.05,
            't_stat_h0_beta_eq_1': 1.00,
            'p_value_h0_beta_eq_1': 0.32,
            'r_squared': 0.78,
            'rmse': 0.0285,
            'n_obs': 50000,
            'interpretation': 'β ≈ 1: Theory validated for HY'
        }
    }


def generate_mock_spec_b2() -> dict:
    """Generate mock Specification B.2 results (decomposed components)."""
    return {
        'combined': {
            'beta_T': 0.92,
            'se_beta_T': 0.04,
            't_stat_T': 23.0,
            'p_value_T': 0.0001,
            'beta_s': 1.05,
            'se_beta_s': 0.03,
            't_stat_s': 35.0,
            'p_value_s': 0.0001,
            'joint_test_fstat': 1.85,
            'joint_test_pvalue': 0.16,
            'r_squared': 0.74,
            'adj_r_squared': 0.73,
            'rmse': 0.0238,
            'n_obs': 150000,
            'interpretation': 'β_T and β_s ≈ 1: Decomposition validated'
        },
        'ig': {
            'beta_T': 0.88,
            'se_beta_T': 0.05,
            'beta_s': 1.02,
            'se_beta_s': 0.04,
            'joint_test_pvalue': 0.18,
            'r_squared': 0.69,
            'rmse': 0.0210,
            'interpretation': 'Decomposition validated for IG'
        },
        'hy': {
            'beta_T': 0.98,
            'se_beta_T': 0.06,
            'beta_s': 1.10,
            'se_beta_s': 0.05,
            'joint_test_pvalue': 0.12,
            'r_squared': 0.79,
            'rmse': 0.0275,
            'interpretation': 'Decomposition validated for HY'
        }
    }


def generate_mock_spec_b3() -> dict:
    """Generate mock Specification B.3 results (unrestricted)."""
    return {
        'combined': {
            'r_squared': 0.78,
            'adj_r_squared': 0.77,
            'n_parameters': 10,
            'n_obs': 150000,
            'lambda_r_squared': 0.73,
            'coefficients': {
                'intercept': 0.02,
                'log_spread': 0.45,
                'log_maturity': -0.32,
                'log_spread_sq': 0.05,
                'log_maturity_sq': 0.02,
                'spread_x_maturity': -0.08
            }
        },
        'ig': {
            'r_squared': 0.72,
            'n_parameters': 10
        },
        'hy': {
            'r_squared': 0.82,
            'n_parameters': 10
        }
    }


def generate_mock_model_comparison() -> pd.DataFrame:
    """Generate mock model comparison table."""
    data = [
        {'Model': 'Stage A (Buckets)', 'R²': 0.75, 'Adj R²': 0.74, 'RMSE': 0.0250, 'ΔR² vs Stage A': np.nan, 'N Parameters': 36, 'AIC': 125000.0},
        {'Model': 'B.1 (Merton Offset)', 'R²': 0.73, 'Adj R²': 0.72, 'RMSE': 0.0245, 'ΔR² vs Stage A': -0.02, 'N Parameters': 1, 'AIC': 126500.0},
        {'Model': 'B.2 (Decomposed)', 'R²': 0.74, 'Adj R²': 0.73, 'RMSE': 0.0240, 'ΔR² vs Stage A': -0.01, 'N Parameters': 2, 'AIC': 126000.0},
        {'Model': 'B.3 (Unrestricted)', 'R²': 0.78, 'Adj R²': 0.77, 'RMSE': 0.0230, 'ΔR² vs Stage A': 0.03, 'N Parameters': 10, 'AIC': 124000.0}
    ]
    return pd.DataFrame(data)


def generate_mock_theory_vs_reality() -> pd.DataFrame:
    """Generate mock theory vs reality comparison table."""
    np.random.seed(42)

    ig_ratings = ['AAA/AA', 'A', 'BBB']
    hy_ratings = ['BB', 'B', 'CCC']
    maturities = ['1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10+y']
    sectors = ['Industrial', 'Financial', 'Utility']

    data = []

    # IG buckets
    for rating in ig_ratings:
        for mat in maturities:
            spread_base = {'AAA/AA': 40, 'A': 80, 'BBB': 150}[rating]
            mat_val = {'1-2y': 1.5, '2-3y': 2.5, '3-5y': 4.0, '5-7y': 6.0, '7-10y': 8.5, '10+y': 15.0}[mat]

            lambda_merton = 2.0 / (1 + mat_val * 0.2) * (1 + spread_base / 500)
            lambda_merton = max(0.5, min(2.5, lambda_merton + np.random.normal(0, 0.1)))

            beta = lambda_merton * (1 + np.random.normal(0, 0.08))
            ratio = beta / lambda_merton
            deviation = beta - lambda_merton
            pct_deviation = (ratio - 1) * 100

            data.append({
                'rating_bucket': rating,
                'maturity_bucket': mat,
                'beta': beta,
                'lambda_merton': lambda_merton,
                'ratio': ratio,
                'deviation': deviation,
                'pct_deviation': pct_deviation,
                'in_acceptable_range': 0.8 <= ratio <= 1.2,
                'median_spread': spread_base + np.random.normal(0, 10),
                'median_maturity': mat_val,
                'n_observations': np.random.randint(1000, 5000),
                'outlier': abs(ratio - 1) > 0.25,
                'sector': np.random.choice(sectors)
            })

    # HY buckets
    for rating in hy_ratings:
        for mat in maturities:
            spread_base = {'BB': 300, 'B': 500, 'CCC': 800}[rating]
            mat_val = {'1-2y': 1.5, '2-3y': 2.5, '3-5y': 4.0, '5-7y': 6.0, '7-10y': 8.5, '10+y': 15.0}[mat]

            lambda_merton = 2.0 / (1 + mat_val * 0.2) * (1 + spread_base / 500)
            lambda_merton = max(0.5, min(2.5, lambda_merton + np.random.normal(0, 0.15)))

            beta = lambda_merton * (1 + np.random.normal(0, 0.12))
            ratio = beta / lambda_merton
            deviation = beta - lambda_merton
            pct_deviation = (ratio - 1) * 100

            data.append({
                'rating_bucket': rating,
                'maturity_bucket': mat,
                'beta': beta,
                'lambda_merton': lambda_merton,
                'ratio': ratio,
                'deviation': deviation,
                'pct_deviation': pct_deviation,
                'in_acceptable_range': 0.8 <= ratio <= 1.2,
                'median_spread': spread_base + np.random.normal(0, 30),
                'median_maturity': mat_val,
                'n_observations': np.random.randint(500, 3000),
                'outlier': abs(ratio - 1) > 0.25,
                'sector': np.random.choice(sectors)
            })

    df = pd.DataFrame(data)
    # Sort by absolute deviation for the reporting module
    df['abs_deviation'] = df['deviation'].abs()
    df = df.sort_values('abs_deviation', ascending=False).drop('abs_deviation', axis=1)
    return df


def generate_mock_theory_assessment() -> dict:
    """Generate mock theory performance assessment."""
    return {
        'n_buckets': 36,
        'pct_in_acceptable_range': 78.5,
        'median_ratio': 1.02,
        'mean_ratio': 1.01,
        'std_ratio': 0.12,
        'min_ratio': 0.78,
        'max_ratio': 1.28,
        'systematic_bias': 'None detected',
        'rating_bias': 'Slight over-prediction for AAA (ratio = 0.92)',
        'maturity_bias': 'Slight under-prediction for 10+ years (ratio = 1.08)',
        'assessment': 'Theory performs well overall. Merton explains 78.5% of buckets within ±20%.'
    }


def generate_mock_decision() -> str:
    """Generate mock Stage B decision."""
    return """
STAGE B DECISION: PATH 2 - MERTON WITH MINOR CALIBRATION
=========================================================

Evidence Summary:
- β_Merton = 0.98 (not significantly different from 1, p = 0.50)
- R² = 0.73 (vs 0.75 for unrestricted buckets)
- 78.5% of buckets have β/λ ratio in [0.8, 1.2]

Interpretation:
Merton's structural model successfully explains the cross-sectional
variation documented in Stage A. The single-parameter Merton model
(λ^Merton) captures nearly all explanatory power of the 36-parameter
bucket model.

RECOMMENDATION:
Use Merton-based λ with minor calibration (β ≈ 0.98).
Proceed to Stage C to test time-stability.

Implementation:
λ_production = 0.98 × λ^Merton(s, T)

Key advantage: Theory-based, interpretable, parsimonious (1 parameter).
"""


def main():
    """Generate mock data and run plotting/reporting for Stage B."""
    print("=" * 80)
    print("STAGE B MOCK DATA TEST")
    print("Testing plotting and reporting with mock data")
    print("=" * 80)

    # Generate mock results
    print("\nGenerating mock data...")
    spec_b1 = generate_mock_spec_b1()
    spec_b2 = generate_mock_spec_b2()
    spec_b3 = generate_mock_spec_b3()
    model_comparison = generate_mock_model_comparison()
    theory_vs_reality = generate_mock_theory_vs_reality()
    theory_assessment = generate_mock_theory_assessment()
    decision = generate_mock_decision()

    print(f"  Spec B.1: β_Merton = {spec_b1['combined']['beta_merton']:.3f}")
    print(f"  Spec B.2: β_T = {spec_b2['combined']['beta_T']:.3f}, β_s = {spec_b2['combined']['beta_s']:.3f}")
    print(f"  Spec B.3: R² = {spec_b3['combined']['r_squared']:.3f}")
    print(f"  Theory assessment: {theory_assessment['pct_in_acceptable_range']:.1f}% in acceptable range")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    output_dir_figures = 'output/stageB_test_figures'
    Path(output_dir_figures).mkdir(parents=True, exist_ok=True)

    merton_calc = MertonLambdaCalculator()
    visualizer = StageBVisualizer(output_dir=output_dir_figures)

    figures = visualizer.create_all_stageB_figures(
        theory_vs_reality,
        merton_calc,
        spec_b3,
        output_prefix='stageB_test'
    )

    print(f"\nCreated {len(figures)} figures in {output_dir_figures}/")

    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)

    output_dir_reports = 'output/stageB_test_reports'
    Path(output_dir_reports).mkdir(parents=True, exist_ok=True)

    reporter = StageBReporter(output_dir=output_dir_reports)

    reporter.save_all_reports(
        spec_b1,
        spec_b2,
        spec_b3,
        model_comparison,
        theory_vs_reality,
        theory_assessment,
        decision,
        prefix='stageB_test'
    )

    print(f"\nReports saved to {output_dir_reports}/")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nFigures: {output_dir_figures}/")
    print(f"Reports: {output_dir_reports}/")


if __name__ == '__main__':
    main()
