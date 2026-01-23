#!/usr/bin/env python3
"""
Main script to run Stage A analysis.

This orchestrates the complete Stage A pipeline:
1. Load data (use Stage 0 preparation or fresh load)
2. Run Specification A.1: Bucket-level betas
3. Run F-tests for equality across dimensions
4. Run Specification A.2: Continuous characteristics (optional, time-intensive)
5. Generate visualizations
6. Create reports
7. Provide decision recommendation

Usage:
    python run_stageA.py [--mock-data] [--run-spec-a2]
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dts_research.data.loader import BondDataLoader
from dts_research.data.bucket_definitions import classify_bonds_into_buckets
from dts_research.data.sector_classification import SectorClassifier
from dts_research.data.issuer_identification import add_issuer_identification
from dts_research.analysis.stageA import StageAAnalysis
from dts_research.analysis.stage0_synthesis import Stage0Synthesis
from dts_research.analysis.stage0_bucket import BucketLevelAnalysis
from dts_research.analysis.stage0_within_issuer import WithinIssuerAnalysis
from dts_research.analysis.stage0_sector import SectorInteractionAnalysis
from dts_research.visualization.stageA_plots import StageAVisualizer
from dts_research.utils.reportingA import StageAReporter


def prepare_regression_data(bond_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare bond data for regression analysis.

    Adds spread changes, index DTS factor, and required columns.
    """
    df = bond_data.copy()

    # Compute spread changes
    df = df.sort_values(['bond_id', 'date'])
    df['oas_lag'] = df.groupby('bond_id')['oas'].shift(1)
    df['oas_pct_change'] = (df['oas'] - df['oas_lag']) / df['oas_lag']

    # Compute index-level DTS factor
    index_factor = df.groupby('date')['oas'].mean()
    index_factor_pct = index_factor.pct_change()
    df = df.merge(
        index_factor_pct.reset_index().rename(columns={'oas': 'oas_index_pct_change'}),
        on='date',
        how='left'
    )

    # Add spread regime
    df['spread_regime'] = np.where(df['oas'] < 300, 'IG', 'HY')

    # Add week identifier for clustering
    df['week'] = df['date'].dt.isocalendar().week.astype(str) + '_' + df['date'].dt.year.astype(str)

    # Drop NaN
    df = df.dropna(subset=['oas_pct_change', 'oas_index_pct_change'])

    return df


def get_stage0_synthesis(bond_data: pd.DataFrame, universe: str = 'IG') -> dict:
    """
    Run Stage 0 synthesis to get decision path for Stage A.

    This determines whether Stage A should proceed based on Stage 0 results.
    """
    # Run Stage 0 analyses
    bucket_analyzer = BucketLevelAnalysis()
    bucket_results = bucket_analyzer.run_bucket_analysis(bond_data, universe=universe)

    within_analyzer = WithinIssuerAnalysis()
    within_results = within_analyzer.run_within_issuer_analysis(bond_data, universe=universe, verbose=False)

    sector_analyzer = SectorInteractionAnalysis()
    sector_results = sector_analyzer.run_sector_analysis(bond_data, universe=universe, cluster_by='week')

    # Synthesize
    synthesizer = Stage0Synthesis()
    synthesis = synthesizer.synthesize_results(
        bucket_results, within_results, sector_results, universe=universe
    )

    return synthesis


def main():
    """
    Run complete Stage A analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Run Stage A Analysis')
    parser.add_argument('--mock-data', action='store_true', default=True,
                       help='Use mock data (default: True)')
    parser.add_argument('--run-spec-a2', action='store_true', default=False,
                       help='Run Specification A.2 (time-intensive)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE A: ESTABLISH CROSS-SECTIONAL VARIATION")
    print("=" * 80)
    print()
    print("Critical Question: Do DTS betas differ across bonds?")
    print("If NO -> Standard DTS adequate, stop here")
    print("If YES -> Proceed to Stage B to test if Merton explains variation")
    print()

    # Create output directories
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'stageA_figures'
    reports_dir = output_dir / 'stageA_reports'
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load and Prepare Data
    # -------------------------------------------------------------------------
    print("Step 1: Loading and preparing bond data...")

    loader = BondDataLoader()
    if args.mock_data:
        print("  Using mock data for testing")
        bond_data = loader.generate_mock_data(args.start_date, args.end_date, n_bonds=500)
    else:
        print("  Loading from database...")
        bond_data = loader.load_bond_data(args.start_date, args.end_date)

    print(f"  Loaded {len(bond_data):,} bond-week observations")
    print()

    # Add sector classification
    print("  Adding sector classification...")
    sector_classifier = SectorClassifier()
    bond_data = sector_classifier.classify_sector(bond_data, bclass_column='sector_classification')
    bond_data = sector_classifier.add_sector_dummies(bond_data)

    # Add issuer identification
    print("  Adding issuer identification...")
    bond_data = add_issuer_identification(
        bond_data,
        parent_id_col='ultimate_parent_id',
        seniority_col='seniority'
    )

    # Classify into buckets
    print("  Classifying bonds into buckets...")
    bond_data, bucket_stats = classify_bonds_into_buckets(
        bond_data,
        rating_column='rating',
        maturity_column='time_to_maturity',
        sector_column='sector',
        compute_characteristics=True
    )

    # Add required columns to bucket_stats
    if bucket_stats is not None:
        bucket_stats['median_maturity'] = bucket_stats['maturity_median']
        bucket_stats['median_spread'] = bucket_stats['spread_median']
        bucket_stats['sector'] = bucket_stats['sector_group'].map({
            'A': 'Industrial', 'B': 'Financial'
        })
        ig_ratings = ['AAA/AA', 'A', 'BBB']
        bucket_stats['is_ig'] = bucket_stats['rating_bucket'].isin(ig_ratings)

    print(f"  Created {len(bucket_stats)} buckets")
    print()

    # Prepare regression data
    print("  Preparing regression data...")
    regression_data = prepare_regression_data(bond_data)
    print(f"  Regression-ready observations: {len(regression_data):,}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Get Stage 0 Results (for context)
    # -------------------------------------------------------------------------
    print("Step 2: Getting Stage 0 synthesis (for decision context)...")
    stage0_synthesis = get_stage0_synthesis(bond_data, universe='IG')
    print(f"  Stage 0 Decision Path: {stage0_synthesis['decision_path']} - {stage0_synthesis['path_name']}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Run Specification A.1 - Bucket-Level Betas
    # -------------------------------------------------------------------------
    print("Step 3: Running Specification A.1 (bucket-level betas)...")

    stageA_analysis = StageAAnalysis(stage0_results=stage0_synthesis)
    spec_a1_results = stageA_analysis.run_specification_a1(regression_data, bucket_stats)

    print(f"  Estimated betas for {len(spec_a1_results)} buckets")
    if len(spec_a1_results) > 0 and 'beta' in spec_a1_results.columns:
        print(f"  Beta range: {spec_a1_results['beta'].min():.3f} to {spec_a1_results['beta'].max():.3f}")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Run F-Tests for Beta Equality
    # -------------------------------------------------------------------------
    print("Step 4: Testing for beta equality across dimensions...")

    # Overall test
    overall_test = stageA_analysis.test_beta_equality_overall(spec_a1_results)
    print(f"\n  CRITICAL TEST - Overall Beta Equality:")
    print(f"    H0: All betas are equal")
    print(f"    F-statistic: {overall_test.get('f_statistic', np.nan):.2f}")
    print(f"    p-value: {overall_test.get('p_value', np.nan):.4f}")
    print(f"    Reject H0: {'YES' if overall_test.get('reject_h0', False) else 'NO'}")
    print(f"    -> {overall_test.get('interpretation', 'N/A')}")
    print()

    # Dimension-specific tests
    print("  Running dimension-specific tests...")
    all_tests = stageA_analysis.run_all_dimension_tests(spec_a1_results)

    print("\n  Key Dimension Tests:")
    for test in all_tests[1:4]:  # Show first 3 after overall
        if 'error' not in test:
            print(f"    {test.get('test', 'Unknown')}: F={test.get('f_statistic', 0):.2f}, p={test.get('p_value', 1):.4f}")

    print()

    # -------------------------------------------------------------------------
    # Step 5: Economic Significance
    # -------------------------------------------------------------------------
    print("Step 5: Assessing economic significance...")

    econ_sig = stageA_analysis.compute_economic_significance(spec_a1_results)
    print(f"  Beta range: {econ_sig.get('min_beta', np.nan):.3f} to {econ_sig.get('max_beta', np.nan):.3f}")
    print(f"  Ratio (max/min): {econ_sig.get('ratio_max_min', np.nan):.2f}x")
    print(f"  IQR: {econ_sig.get('iqr', np.nan):.3f}")
    print(f"  Standard deviation: {econ_sig.get('std', np.nan):.3f}")
    print()

    # IG vs HY comparison
    print("  Comparing IG vs HY variation...")
    ig_hy_comp = stageA_analysis.compare_ig_vs_hy_variation(spec_a1_results)
    if 'error' not in ig_hy_comp:
        print(f"    IG std: {ig_hy_comp.get('ig_std', np.nan):.3f}")
        print(f"    HY std: {ig_hy_comp.get('hy_std', np.nan):.3f}")
        print(f"    Ratio (IG/HY): {ig_hy_comp.get('std_ratio_ig_hy', np.nan):.2f}")
        print(f"    -> {ig_hy_comp.get('interpretation', 'N/A')}")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Run Specification A.2 (Optional)
    # -------------------------------------------------------------------------
    if args.run_spec_a2:
        print("Step 6: Running Specification A.2 (continuous characteristics)...")
        print("  NOTE: This is time-intensive (rolling window estimation)")
        spec_a2_results = stageA_analysis.run_specification_a2()
        if 'error' not in spec_a2_results.get('combined', {}):
            print(f"  Combined R²: {spec_a2_results['combined'].get('r_squared', np.nan):.3f}")
    else:
        print("Step 6: Skipping Specification A.2 (use --run-spec-a2 to enable)")
        spec_a2_results = {'combined': {'r_squared': np.nan, 'skipped': True}}
    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Decision Recommendation
    # -------------------------------------------------------------------------
    print("Step 7: Generating decision recommendation...")

    decision = stageA_analysis.generate_stage_a_decision(
        overall_test,
        econ_sig,
        spec_a2_results
    )

    print()
    print("=" * 80)
    print("STAGE A DECISION")
    print("=" * 80)
    print(decision)
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Step 8: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 8: Generating visualizations...")

    visualizer = StageAVisualizer(output_dir=str(figures_dir))
    figures = visualizer.create_all_stageA_figures(
        spec_a1_results,
        spec_a2_results,
        output_prefix='stageA'
    )

    print(f"  Created {len(figures)} figures in {figures_dir}")
    print()

    # -------------------------------------------------------------------------
    # Step 9: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 9: Generating reports...")

    reporter = StageAReporter(output_dir=str(reports_dir))
    reporter.save_all_reports(
        spec_a1_results,
        all_tests,
        spec_a2_results,
        econ_sig,
        ig_hy_comp if 'error' not in ig_hy_comp else {},
        decision,
        prefix='stageA'
    )

    print(f"  Reports saved to {reports_dir}")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STAGE A COMPLETE")
    print("=" * 80)
    print()
    print("Key Findings:")
    print(f"  - Overall F-test p-value: {overall_test.get('p_value', np.nan):.4f}")
    print(f"  - Beta range: {econ_sig.get('min_beta', np.nan):.3f} to {econ_sig.get('max_beta', np.nan):.3f}")
    print(f"  - Economic variation: {econ_sig.get('ratio_max_min', np.nan):.2f}x")

    print()
    print("Next steps:")
    if 'STOP' in decision:
        print("  -> Analysis complete: Standard DTS is adequate")
    else:
        print("  -> Review reports for detailed analysis")
        print("  -> Proceed to Stage B to test if Merton explains variation")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
