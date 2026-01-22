"""
Test Stage 0 Plotting and Reporting with Mock Data

Generates realistic mock data that matches the evolved Stage 0 analysis output
structures, then runs the plotting and reporting modules to generate figures
and tables for review.

Run with: python tests/test_stage0_plots_reporting.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.visualization.stage0_plots import create_all_stage0_plots
from dts_research.utils.reporting0 import generate_stage0_report


def generate_mock_bucket_results(universe: str = 'IG', scenario: str = 'good_fit') -> dict:
    """
    Generate mock bucket-level analysis results.

    Args:
        universe: 'IG' or 'HY'
        scenario: 'good_fit' (β ≈ λ), 'over_predict' (β < λ), 'under_predict' (β > λ)
    """
    np.random.seed(42 if universe == 'IG' else 123)

    # Generate 72 buckets (6 rating x 6 maturity x 2 sector)
    ratings = ['AAA', 'AA', 'A', 'BBB'] if universe == 'IG' else ['BB', 'B', 'CCC']
    maturities = ['0-2y', '2-5y', '5-7y', '7-10y', '10-15y', '15-30y']
    sectors = ['Industrial', 'Financial']

    bucket_data = []
    for rating in ratings:
        for mat_bucket in maturities:
            for sector in sectors:
                # Generate realistic spread and maturity
                mat_idx = maturities.index(mat_bucket)
                median_maturity = [1, 3.5, 6, 8.5, 12.5, 22.5][mat_idx]

                # Spread depends on rating and maturity
                if universe == 'IG':
                    base_spread = {'AAA': 50, 'AA': 75, 'A': 100, 'BBB': 150}[rating]
                else:
                    base_spread = {'BB': 250, 'B': 400, 'CCC': 700}[rating]

                # Add maturity effect (spreads typically increase with maturity for HY)
                median_spread = base_spread + mat_idx * 10 + np.random.normal(0, 10)

                # Compute theoretical lambda^Merton (simplified)
                # λ^Merton ≈ 1 / T for short maturities, approaches 0 for long
                lambda_merton = max(0.3, 2.0 / (1 + median_maturity * 0.3)) + np.random.normal(0, 0.05)

                # Generate empirical beta based on scenario
                if scenario == 'good_fit':
                    beta = lambda_merton * (1 + np.random.normal(0, 0.1))  # β ≈ λ
                elif scenario == 'over_predict':
                    beta = lambda_merton * 0.7 * (1 + np.random.normal(0, 0.1))  # β < λ
                else:
                    beta = lambda_merton * 1.4 * (1 + np.random.normal(0, 0.1))  # β > λ

                # β should decrease with maturity (Merton prediction)
                beta = beta * (1 - mat_idx * 0.05)

                bucket_data.append({
                    'bucket_id': f'{rating}_{mat_bucket}_{sector}',
                    'rating_bucket': rating,
                    'maturity_bucket': mat_bucket,
                    'sector_group': sector,
                    'median_spread': max(20, median_spread),
                    'median_maturity': median_maturity,
                    'beta': beta,
                    'beta_se': abs(beta) * 0.1 + 0.01,
                    'beta_tstat': beta / (abs(beta) * 0.1 + 0.01),
                    'beta_pvalue': 0.001 if abs(beta) > 0.1 else 0.1,
                    'alpha': np.random.normal(0, 0.001),
                    'lambda_merton': lambda_merton,
                    'beta_lambda_ratio': beta / lambda_merton if lambda_merton > 0 else np.nan,
                    'ratio_diff_from_1': (beta / lambda_merton - 1) if lambda_merton > 0 else np.nan,
                    'r_squared': np.random.uniform(0.1, 0.4),
                    'n_obs': np.random.randint(500, 5000),
                    'n_weeks': np.random.randint(50, 200)
                })

    bucket_df = pd.DataFrame(bucket_data)

    # Compute summary statistics
    ratios = bucket_df['beta_lambda_ratio'].dropna()
    summary_stats = {
        'median_beta_lambda_ratio': ratios.median(),
        'mean_beta_lambda_ratio': ratios.mean(),
        'std_beta_lambda_ratio': ratios.std(),
        'pct_within_10pct': 100.0 * ((ratios >= 0.9) & (ratios <= 1.1)).mean(),
        'pct_within_20pct': 100.0 * ((ratios >= 0.8) & (ratios <= 1.2)).mean(),
        'n_buckets': len(ratios),
        'median_beta': bucket_df['beta'].median(),
        'median_lambda_merton': bucket_df['lambda_merton'].median()
    }

    # Monotonicity test results
    mono_details = []
    for rating in ratings:
        subset = bucket_df[bucket_df['rating_bucket'] == rating].sort_values('median_maturity')
        if len(subset) >= 3:
            # Compute Spearman correlation
            from scipy.stats import spearmanr
            rho, pval = spearmanr(subset['median_maturity'], subset['beta'])
            mono_details.append({
                'rating_bucket': rating,
                'n_buckets': len(subset),
                'spearman_rho': rho,
                'p_value': pval,
                'is_monotonic_decreasing': rho < 0 and pval < 0.10
            })

    mono_df = pd.DataFrame(mono_details)
    pct_mono = 100.0 * mono_df['is_monotonic_decreasing'].mean() if len(mono_df) > 0 else 0

    monotonicity_test = {
        'overall_monotonic': pct_mono > 70,
        'pct_monotonic_groups': pct_mono,
        'n_groups_tested': len(mono_df),
        'details': mono_details,
        'interpretation': "β decreases with maturity as Merton predicts" if pct_mono > 70 else "β does NOT consistently decrease with maturity"
    }

    # Diagnostics
    diagnostics = {
        'n_buckets_with_regression': len(bucket_df),
        'n_buckets_expected': 72,
        'pct_coverage': 100.0 * len(bucket_df) / 72,
        'total_bond_weeks': int(bucket_df['n_obs'].sum()),
        'n_unique_bonds': np.random.randint(5000, 20000),
        'n_unique_weeks': np.random.randint(100, 300),
        'total_observations': int(bucket_df['n_obs'].sum()),
        'mean_obs_per_bucket': bucket_df['n_obs'].mean(),
        'median_obs_per_bucket': bucket_df['n_obs'].median(),
        'min_obs_per_bucket': int(bucket_df['n_obs'].min()),
        'mean_r_squared': bucket_df['r_squared'].mean()
    }

    return {
        'universe': universe,
        'bucket_results': bucket_df,
        'summary_statistics': summary_stats,
        'monotonicity_test': monotonicity_test,
        'diagnostics': diagnostics
    }


def generate_mock_within_issuer_results(universe: str = 'IG', scenario: str = 'good_fit') -> dict:
    """
    Generate mock within-issuer analysis results.
    """
    np.random.seed(43 if universe == 'IG' else 124)

    # Generate issuer-week estimates
    n_estimates = 2000 if universe == 'IG' else 1500
    dates = pd.date_range('2020-01-01', periods=150, freq='W')

    # Beta should be centered around 1 for good_fit
    if scenario == 'good_fit':
        beta_mean = 1.0
    elif scenario == 'over_predict':
        beta_mean = 0.7
    else:
        beta_mean = 1.3

    estimates_data = []
    for i in range(n_estimates):
        date = np.random.choice(dates)
        beta = np.random.normal(beta_mean, 0.4)
        beta = np.clip(beta, -1, 3)  # Clip extreme values

        estimates_data.append({
            'issuer_id': f'issuer_{i % 200}',
            'date': date,
            'beta': beta,
            'beta_se': abs(beta) * 0.2 + 0.05,
            'beta_tstat': beta / (abs(beta) * 0.2 + 0.05),
            'beta_pvalue': np.random.uniform(0, 0.1),
            'alpha': np.random.normal(0, 0.01),
            'n_bonds': np.random.randint(3, 15),
            'r_squared': np.random.uniform(0.2, 0.8),
            'maturity_range': np.random.uniform(2, 10),
            'lambda_range': np.random.uniform(0.2, 1.0),
            'mean_spread': np.random.uniform(100, 500)
        })

    estimates_df = pd.DataFrame(estimates_data)

    # Compute pooled estimate
    betas = estimates_df['beta']
    variances = estimates_df['beta_se']**2

    # Inverse-variance weighted pooling
    weights = 1 / variances
    pooled_beta = (weights * betas).sum() / weights.sum()
    pooled_var = 1 / weights.sum()
    pooled_se = np.sqrt(pooled_var)

    pooled_estimate = {
        'pooled_beta': pooled_beta,
        'pooled_beta_se': pooled_se,
        'ci_lower': pooled_beta - 1.96 * pooled_se,
        'ci_upper': pooled_beta + 1.96 * pooled_se,
        'n_estimates': len(estimates_df)
    }

    # Hypothesis test for β = 1
    t_stat_eq_1 = (pooled_beta - 1.0) / pooled_se
    p_value_eq_1 = 2 * (1 - min(0.9999, abs(t_stat_eq_1) / 10))  # Simplified

    t_stat_pos = pooled_beta / pooled_se
    p_value_pos = 1 - min(0.9999, t_stat_pos / 10)

    beta_in_range = 0.9 <= pooled_beta <= 1.1

    hypothesis_test = {
        'test': 'H0: β = 1 (Merton prediction)',
        'beta_pooled': pooled_beta,
        'beta_se': pooled_se,
        't_statistic_beta_eq_1': t_stat_eq_1,
        'p_value_beta_equals_1': max(0.001, abs(t_stat_eq_1) * 0.1) if abs(t_stat_eq_1) < 2 else 0.001,
        't_statistic_beta_pos': t_stat_pos,
        'p_value_beta_positive': max(0.0001, p_value_pos),
        'reject_beta_equals_1': abs(t_stat_eq_1) > 1.96,
        'beta_in_range_0_9_1_1': beta_in_range,
        'merton_validates': beta_in_range and abs(t_stat_eq_1) < 2,
        'interpretation': f"β = {pooled_beta:.3f} {'is consistent with' if beta_in_range else 'deviates from'} Merton"
    }

    # Diagnostics
    diagnostics = {
        'n_bonds_after_filter': np.random.randint(50000, 200000),
        'n_unique_issuers': 200,
        'n_unique_weeks': 150,
        'n_issuer_weeks_total': 3000,
        'n_issuer_weeks_with_estimate': len(estimates_df),
        'pct_issuer_weeks_with_estimate': 100.0 * len(estimates_df) / 3000,
        'mean_bonds_per_issuer_week': estimates_df['n_bonds'].mean(),
        'mean_maturity_range': estimates_df['maturity_range'].mean(),
        'mean_r_squared': estimates_df['r_squared'].mean(),
        'median_beta': betas.median(),
        'mean_beta': betas.mean(),
        'std_beta': betas.std(),
        'pct_beta_in_0_8_1_2': 100.0 * ((betas >= 0.8) & (betas <= 1.2)).mean(),
        'pct_beta_positive': 100.0 * (betas > 0).mean(),
        'beta_range': (betas.min(), betas.max())
    }

    return {
        'universe': universe,
        'issuer_week_estimates': estimates_df,
        'pooled_estimate': pooled_estimate,
        'hypothesis_test': hypothesis_test,
        'diagnostics': diagnostics,
        'filter_stats': {}
    }


def generate_mock_sector_results(universe: str = 'IG', sectors_differ: bool = True) -> dict:
    """
    Generate mock sector analysis results.
    """
    np.random.seed(44 if universe == 'IG' else 125)

    # Base β_0 (should be ≈ 1)
    base_beta = np.random.normal(1.0, 0.1)

    # Sector deviations
    if sectors_differ:
        beta_fin = np.random.normal(0.15, 0.05)  # Financial amplifies
        beta_util = np.random.normal(-0.12, 0.04)  # Utility dampens
        beta_energy = np.random.normal(0.05, 0.03)  # Energy slightly different
    else:
        beta_fin = np.random.normal(0.02, 0.05)
        beta_util = np.random.normal(-0.01, 0.04)
        beta_energy = np.random.normal(0.01, 0.03)

    n_obs = np.random.randint(100000, 300000)
    n_clusters = np.random.randint(100, 200)

    base_regression = {
        'beta_0': base_beta,
        'beta_0_se': 0.05,
        'beta_0_tstat': base_beta / 0.05,
        'beta_0_pvalue': 0.001,
        'alpha': np.random.normal(0, 0.001),
        'r_squared': np.random.uniform(0.15, 0.3),
        'n_obs': n_obs,
        'n_clusters': n_clusters,
        'interpretation': f"β₀ = {base_beta:.3f}: Merton-scaled DTS explains spread changes"
    }

    sector_regression = {
        'alpha': np.random.normal(0, 0.001),
        'beta_0': base_beta,
        'beta_financial': beta_fin,
        'beta_utility': beta_util,
        'beta_energy': beta_energy,
        'beta_0_se': 0.05,
        'beta_financial_se': 0.04,
        'beta_utility_se': 0.035,
        'beta_energy_se': 0.03,
        'beta_0_tstat': base_beta / 0.05,
        'beta_financial_tstat': beta_fin / 0.04,
        'beta_utility_tstat': beta_util / 0.035,
        'beta_energy_tstat': beta_energy / 0.03,
        'beta_0_pvalue': 0.001,
        'beta_financial_pvalue': 0.001 if abs(beta_fin / 0.04) > 2 else 0.1,
        'beta_utility_pvalue': 0.001 if abs(beta_util / 0.035) > 2 else 0.1,
        'beta_energy_pvalue': 0.1,
        'r_squared': np.random.uniform(0.18, 0.35),
        'n_obs': n_obs,
        'n_clusters': n_clusters,
        'sensitivity_industrial': base_beta,
        'sensitivity_financial': base_beta + beta_fin,
        'sensitivity_utility': base_beta + beta_util,
        'sensitivity_energy': base_beta + beta_energy
    }

    # Joint F-test
    f_stat = 5.2 if sectors_differ else 1.5
    f_pval = 0.001 if sectors_differ else 0.2

    joint_test = {
        'test': 'H0: β_fin = β_util = β_energy = 0 (sectors have same DTS sensitivity)',
        'f_statistic': f_stat,
        'p_value': f_pval,
        'df_numerator': 3,
        'df_denominator': n_obs - 5,
        'reject_null': f_pval < 0.05,
        'sectors_differ': f_pval < 0.05,
        'interpretation': "Sector interactions are significant" if f_pval < 0.05 else "Sectors do not differ"
    }

    # Individual sector tests
    sector_tests = {
        'financial_test': {
            'hypothesis': 'H0: β_fin ≤ 0 vs H1: β_fin > 0 (Financials amplify)',
            'beta_deviation': beta_fin,
            'se': 0.04,
            't_statistic': beta_fin / 0.04,
            'p_value': 0.001 if beta_fin > 0 and sectors_differ else 0.15,
            'reject_null': beta_fin > 0 and sectors_differ,
            'total_sensitivity': base_beta + beta_fin,
            'interpretation': f"Financial deviation = {beta_fin:.3f}: {'amplifies' if beta_fin > 0 else 'dampens'} market moves"
        },
        'utility_test': {
            'hypothesis': 'H0: β_util ≥ 0 vs H1: β_util < 0 (Utilities dampen)',
            'beta_deviation': beta_util,
            'se': 0.035,
            't_statistic': beta_util / 0.035,
            'p_value': 0.001 if beta_util < 0 and sectors_differ else 0.15,
            'reject_null': beta_util < 0 and sectors_differ,
            'total_sensitivity': base_beta + beta_util,
            'interpretation': f"Utility deviation = {beta_util:.3f}: {'dampens' if beta_util < 0 else 'amplifies'} market moves"
        },
        'energy_test': {
            'hypothesis': 'H0: β_energy = 0 (Energy same as Industrial)',
            'beta_deviation': beta_energy,
            'se': 0.03,
            't_statistic': beta_energy / 0.03,
            'p_value': 0.1,
            'reject_null': False,
            'total_sensitivity': base_beta + beta_energy,
            'interpretation': f"Energy deviation = {beta_energy:.3f}"
        },
        'summary': {
            'industrial_baseline': base_beta,
            'financial_total': base_beta + beta_fin,
            'utility_total': base_beta + beta_util,
            'energy_total': base_beta + beta_energy,
            'need_sector_adjustment': sectors_differ
        }
    }

    diagnostics = {
        'n_observations': n_obs,
        'n_clusters': n_clusters,
        'n_unique_bonds': np.random.randint(10000, 30000),
        'n_unique_weeks': n_clusters,
        'r_squared': sector_regression['r_squared'],
        'sector_distribution': {
            'Industrial': n_obs * 0.4,
            'Financial': n_obs * 0.35,
            'Utility': n_obs * 0.15,
            'Energy': n_obs * 0.1
        },
        'pct_industrial': 40,
        'pct_financial': 35,
        'pct_utility': 15,
        'pct_energy': 10
    }

    return {
        'universe': universe,
        'base_regression': base_regression,
        'sector_regression': sector_regression,
        'joint_test': joint_test,
        'sector_tests': sector_tests,
        'diagnostics': diagnostics
    }


def generate_mock_synthesis(
    bucket_results: dict,
    within_issuer_results: dict,
    sector_results: dict,
    universe: str = 'IG'
) -> dict:
    """
    Generate mock synthesis results based on the component analyses.
    """
    # Extract key statistics
    bucket_ratio = bucket_results['summary_statistics']['median_beta_lambda_ratio']
    within_beta = within_issuer_results['pooled_estimate']['pooled_beta']
    base_beta = sector_results['base_regression']['beta_0']
    sectors_differ = sector_results['joint_test']['sectors_differ']
    monotonic = bucket_results['monotonicity_test']['overall_monotonic']

    # Evaluate criteria
    bucket_beta_near_1 = 0.9 <= bucket_ratio <= 1.1
    within_beta_near_1 = 0.9 <= within_beta <= 1.1
    consistent = 0.8 <= bucket_ratio <= 1.2 and 0.8 <= within_beta <= 1.2

    # Determine path
    if not monotonic:
        path = 5
        path_name = "Theory Fails (consider alternative models)"
        rationale = "β does NOT decrease with maturity as Merton predicts"
    elif sectors_differ:
        path = 4
        path_name = "Merton + Sectors (sector-specific adjustments)"
        rationale = "Sector effects significant"
    elif bucket_beta_near_1 and within_beta_near_1 and monotonic:
        path = 1
        path_name = "Standard DTS (theory works well)"
        rationale = f"β ≈ 1 validated: within-issuer β = {within_beta:.2f}, bucket ratio = {bucket_ratio:.2f}"
    elif consistent:
        path = 2
        path_name = "Pure Merton (use theoretical λ tables)"
        rationale = f"Theory works but not perfectly"
    else:
        path = 3
        path_name = "Calibrated Merton (scale β to data)"
        rationale = f"β consistent but ≠ 1"

    key_statistics = {
        'bucket_median_ratio': bucket_ratio,
        'bucket_pct_within_20': bucket_results['summary_statistics']['pct_within_20pct'],
        'bucket_median_beta': bucket_results['summary_statistics']['median_beta'],
        'monotonic': monotonic,
        'pct_monotonic': bucket_results['monotonicity_test']['pct_monotonic_groups'],
        'within_beta': within_beta,
        'within_beta_se': within_issuer_results['pooled_estimate']['pooled_beta_se'],
        'within_validates': within_issuer_results['hypothesis_test']['merton_validates'],
        'within_beta_in_range': within_beta_near_1,
        'base_beta': base_beta,
        'sectors_differ': sectors_differ,
        'need_sector_adjustment': sector_results['sector_tests']['summary']['need_sector_adjustment']
    }

    decision_criteria = {
        'bucket_beta_near_1': bucket_beta_near_1,
        'within_beta_near_1': within_beta_near_1,
        'monotonic': monotonic,
        'sectors_matter': sectors_differ,
        'consistent': consistent,
        'base_beta_near_1': 0.8 <= base_beta <= 1.2,
        'theory_validated': bucket_beta_near_1 and within_beta_near_1 and monotonic
    }

    recommendations = {
        'stage_A': f"Use Merton λ tables (β = {within_beta:.2f})",
        'stage_B': "Proceed with Merton-based specification",
        'stage_C': "Test for time-variation in sensitivity",
        'stage_D': "Standard robustness checks",
        'stage_E': f"Use Specification {path}"
    }

    return {
        'universe': universe,
        'decision_path': path,
        'path_name': path_name,
        'rationale': rationale,
        'key_statistics': key_statistics,
        'decision_criteria': decision_criteria,
        'recommendations': recommendations
    }


def main():
    """
    Generate mock data and run plotting/reporting for Stage 0.
    """
    print("=" * 80)
    print("STAGE 0 MOCK DATA TEST")
    print("Testing plotting and reporting with realistic mock data")
    print("=" * 80)

    # Generate mock results for IG (good fit scenario)
    print("\nGenerating mock data for IG (good fit scenario)...")
    bucket_results_ig = generate_mock_bucket_results('IG', 'good_fit')
    within_issuer_results_ig = generate_mock_within_issuer_results('IG', 'good_fit')
    sector_results_ig = generate_mock_sector_results('IG', sectors_differ=True)
    synthesis_ig = generate_mock_synthesis(bucket_results_ig, within_issuer_results_ig, sector_results_ig, 'IG')

    # Generate mock results for HY (slightly worse fit)
    print("Generating mock data for HY (moderate fit scenario)...")
    bucket_results_hy = generate_mock_bucket_results('HY', 'good_fit')
    within_issuer_results_hy = generate_mock_within_issuer_results('HY', 'good_fit')
    sector_results_hy = generate_mock_sector_results('HY', sectors_differ=True)
    synthesis_hy = generate_mock_synthesis(bucket_results_hy, within_issuer_results_hy, sector_results_hy, 'HY')

    # Comparison
    ig_path = synthesis_ig['decision_path']
    hy_path = synthesis_hy['decision_path']
    comparison = {
        'ig_path': ig_path,
        'ig_path_name': synthesis_ig['path_name'],
        'hy_path': hy_path,
        'hy_path_name': synthesis_hy['path_name'],
        'same_path': ig_path == hy_path,
        'interpretation': f"Both universes follow Path {ig_path}" if ig_path == hy_path else f"Paths differ: IG={ig_path}, HY={hy_path}",
        'unified_approach': f"Use Path {max(ig_path, hy_path)} for both universes"
    }

    # Print summary
    print("\n" + "-" * 80)
    print("MOCK DATA SUMMARY")
    print("-" * 80)
    print(f"\nIG Universe:")
    print(f"  Bucket β/λ ratio: {bucket_results_ig['summary_statistics']['median_beta_lambda_ratio']:.3f}")
    print(f"  Within-issuer β: {within_issuer_results_ig['pooled_estimate']['pooled_beta']:.3f}")
    print(f"  Sector base β₀: {sector_results_ig['base_regression']['beta_0']:.3f}")
    print(f"  Decision Path: {synthesis_ig['decision_path']} - {synthesis_ig['path_name']}")

    print(f"\nHY Universe:")
    print(f"  Bucket β/λ ratio: {bucket_results_hy['summary_statistics']['median_beta_lambda_ratio']:.3f}")
    print(f"  Within-issuer β: {within_issuer_results_hy['pooled_estimate']['pooled_beta']:.3f}")
    print(f"  Sector base β₀: {sector_results_hy['base_regression']['beta_0']:.3f}")
    print(f"  Decision Path: {synthesis_hy['decision_path']} - {synthesis_hy['path_name']}")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    output_dir_figures = 'output/stage0_test_figures'
    create_all_stage0_plots(
        bucket_results_ig, bucket_results_hy,
        within_issuer_results_ig, within_issuer_results_hy,
        sector_results_ig, sector_results_hy,
        synthesis_ig, synthesis_hy,
        comparison,
        output_dir=output_dir_figures
    )

    # Generate tables
    print("\n" + "=" * 80)
    print("GENERATING TABLES")
    print("=" * 80)

    output_dir_tables = 'output/stage0_test_tables'
    generate_stage0_report(
        bucket_results_ig, bucket_results_hy,
        within_issuer_results_ig, within_issuer_results_hy,
        sector_results_ig, sector_results_hy,
        synthesis_ig, synthesis_hy,
        comparison,
        output_dir=output_dir_tables
    )

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {output_dir_figures}/")
    print(f"Tables saved to: {output_dir_tables}/")
    print("\nPlease review the generated figures and tables.")


if __name__ == '__main__':
    main()
