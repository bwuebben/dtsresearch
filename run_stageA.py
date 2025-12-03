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
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dts_research.data.loader import BondDataLoader
from dts_research.analysis.buckets import BucketClassifier
from dts_research.analysis.stage0 import Stage0Analysis
from dts_research.analysis.stageA import StageAAnalysis
from dts_research.visualization.stageA_plots import StageAVisualizer
from dts_research.utils.reportingA import StageAReporter


def main():
    """
    Run complete Stage A analysis pipeline.
    """
    print("="*80)
    print("STAGE A: ESTABLISH CROSS-SECTIONAL VARIATION")
    print("="*80)
    print()
    print("Critical Question: Do DTS betas differ across bonds?")
    print("If NO → Standard DTS adequate, stop here")
    print("If YES → Proceed to Stage B to test if Merton explains variation")
    print()

    # Configuration
    start_date = '2010-01-01'
    end_date = '2024-12-31'
    use_mock_data = True
    run_spec_a2 = False  # Set to True to run time-intensive rolling window estimation (~30-40 min)

    # Create output directories
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load and Prepare Data
    # -------------------------------------------------------------------------
    print("Step 1: Loading and preparing bond data...")

    if use_mock_data:
        print("  Using mock data for testing")
        loader = BondDataLoader()
        bond_data = loader.generate_mock_data(start_date, end_date, n_bonds=500)
        index_data_ig = loader.generate_mock_index_data(start_date, end_date, index_type='IG')
    else:
        print("  Loading from database...")
        connection_string = "your_connection_string_here"
        loader = BondDataLoader(connection_string)
        loader.connect()
        bond_data = loader.load_bond_data(start_date, end_date)
        index_data_ig = loader.load_index_data(start_date, end_date, index_type='IG')
        loader.close()

    print(f"  Loaded {len(bond_data):,} bond-week observations")
    print()

    # Classify into buckets
    print("  Classifying bonds into buckets...")
    classifier = BucketClassifier()
    bond_data = classifier.classify_bonds(bond_data)
    bucket_stats = classifier.compute_bucket_characteristics(bond_data, min_observations=50)
    print(f"  Created {len(bucket_stats)} buckets")
    print()

    # Prepare regression data
    print("  Preparing regression data...")
    stage0_analysis = Stage0Analysis()
    index_data = index_data_ig.copy()
    regression_data = stage0_analysis.prepare_regression_data(bond_data, index_data)
    print(f"  Regression-ready observations: {len(regression_data):,}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Run Specification A.1 - Bucket-Level Betas
    # -------------------------------------------------------------------------
    print("Step 2: Running Specification A.1 (bucket-level betas)...")

    stageA_analysis = StageAAnalysis()
    spec_a1_results = stageA_analysis.run_specification_a1(regression_data, bucket_stats)

    print(f"  Estimated betas for {len(spec_a1_results)} buckets")
    print(f"  Beta range: {spec_a1_results['beta'].min():.3f} to {spec_a1_results['beta'].max():.3f}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Run F-Tests for Beta Equality
    # -------------------------------------------------------------------------
    print("Step 3: Testing for beta equality across dimensions...")

    # Overall test
    overall_test = stageA_analysis.test_beta_equality_overall(spec_a1_results)
    print(f"\n  CRITICAL TEST - Overall Beta Equality:")
    print(f"    H0: All betas are equal")
    print(f"    F-statistic: {overall_test['f_statistic']:.2f}")
    print(f"    p-value: {overall_test['p_value']:.4f}")
    print(f"    Reject H0: {'YES' if overall_test['reject_h0'] else 'NO'}")
    print(f"    → {overall_test['interpretation']}")
    print()

    # Dimension-specific tests
    print("  Running dimension-specific tests...")
    all_tests = stageA_analysis.run_all_dimension_tests(spec_a1_results)

    # Show a few key tests
    print("\n  Key Dimension Tests:")
    for test in all_tests[1:4]:  # Show first 3 after overall
        if 'error' not in test:
            print(f"    {test['test']}: F={test.get('f_statistic', 0):.2f}, p={test.get('p_value', 1):.4f}")

    print()

    # -------------------------------------------------------------------------
    # Step 4: Economic Significance
    # -------------------------------------------------------------------------
    print("Step 4: Assessing economic significance...")

    econ_sig = stageA_analysis.compute_economic_significance(spec_a1_results)
    print(f"  Beta range: {econ_sig['min_beta']:.3f} to {econ_sig['max_beta']:.3f}")
    print(f"  Ratio (max/min): {econ_sig['ratio_max_min']:.2f}x")
    print(f"  IQR: {econ_sig['iqr']:.3f}")
    print(f"  Standard deviation: {econ_sig['std']:.3f}")
    print()

    # IG vs HY comparison
    print("  Comparing IG vs HY variation (Regime 2 test)...")
    ig_hy_comp = stageA_analysis.compare_ig_vs_hy_variation(spec_a1_results)
    if 'error' not in ig_hy_comp:
        print(f"    IG std: {ig_hy_comp['ig_std']:.3f}")
        print(f"    HY std: {ig_hy_comp['hy_std']:.3f}")
        print(f"    Ratio (IG/HY): {ig_hy_comp['std_ratio_ig_hy']:.2f}")
        print(f"    → {ig_hy_comp['interpretation']}")
    print()

    # -------------------------------------------------------------------------
    # Step 5: Run Specification A.2 - Continuous Characteristics (Optional)
    # -------------------------------------------------------------------------
    if run_spec_a2:
        print("Step 5: Running Specification A.2 (continuous characteristics)...")
        print("  NOTE: This is time-intensive (rolling window estimation)")
        print()

        # Step 1: Estimate bond-specific betas
        print("  Step 5.1: Estimating bond-specific betas (2-year rolling windows)...")
        print("    This may take several minutes...")

        bond_betas = stageA_analysis.estimate_bond_specific_betas(
            bond_data,
            index_data,
            window_weeks=104  # 2 years
        )

        if len(bond_betas) > 0:
            print(f"    Estimated {len(bond_betas):,} bond-window betas")
            print(f"    Covering {bond_betas['bond_id'].nunique()} bonds")
            print()

            # Step 2: Cross-sectional regression
            print("  Step 5.2: Cross-sectional regression of betas on characteristics...")
            spec_a2_results = stageA_analysis.run_specification_a2_by_regime(bond_betas)

            if 'error' not in spec_a2_results['combined']:
                print(f"    Combined R²: {spec_a2_results['combined']['r_squared']:.3f}")
                print(f"    IG R²: {spec_a2_results.get('ig', {}).get('r_squared', 0):.3f}")
                print(f"    HY R²: {spec_a2_results.get('hy', {}).get('r_squared', 0):.3f}")
                print()
        else:
            print("    WARNING: No bond-specific betas estimated (insufficient data)")
            spec_a2_results = {'combined': {'error': 'Insufficient data'}}
            print()
    else:
        print("Step 5: Skipping Specification A.2 (set run_spec_a2=True to enable)")
        spec_a2_results = {'combined': {'error': 'Skipped by user'}}
        print()

    # -------------------------------------------------------------------------
    # Step 6: Generate Decision Recommendation
    # -------------------------------------------------------------------------
    print("Step 6: Generating decision recommendation...")

    decision = stageA_analysis.generate_stage_a_decision(
        overall_test,
        econ_sig,
        spec_a2_results
    )

    print()
    print("="*80)
    print("STAGE A DECISION")
    print("="*80)
    print(decision)
    print("="*80)
    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 7: Generating visualizations...")

    visualizer = StageAVisualizer(output_dir='./output/figures')
    figures = visualizer.create_all_stageA_figures(
        spec_a1_results,
        spec_a2_results,
        output_prefix='stageA'
    )

    print(f"  Created {len(figures)} figures:")
    print("    - Figure A.1: Beta heatmap (rating × maturity)")
    print("    - Figure A.2: Beta surface from Spec A.2 (contour)")
    print("    - Figure A.2 (alt): Beta surface (3D)")
    print("    - Additional: Beta distribution diagnostics")
    print()

    # -------------------------------------------------------------------------
    # Step 8: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 8: Generating reports...")

    reporter = StageAReporter(output_dir='./output/reports')
    reporter.save_all_reports(
        spec_a1_results,
        all_tests,
        spec_a2_results,
        econ_sig,
        ig_hy_comp if 'error' not in ig_hy_comp else {},
        decision,
        prefix='stageA'
    )

    print("  Created reports:")
    print("    - Table A.1: Bucket-level beta estimates")
    print("    - Table A.2: Tests of beta equality")
    print("    - Table A.3: Continuous characteristic regression")
    print("    - Written summary (2 pages)")
    print("    - Full results CSV")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("="*80)
    print("STAGE A COMPLETE")
    print("="*80)
    print()
    print("Key Findings:")
    print(f"  • Overall F-test p-value: {overall_test['p_value']:.4f}")
    print(f"  • Beta range: {econ_sig['min_beta']:.3f} to {econ_sig['max_beta']:.3f}")
    print(f"  • Economic variation: {econ_sig['ratio_max_min']:.2f}x")

    if run_spec_a2 and 'error' not in spec_a2_results['combined']:
        print(f"  • Spec A.2 R²: {spec_a2_results['combined']['r_squared']:.3f}")

    print()
    print("Next steps:")
    if 'STOP' in decision:
        print("  ✓ Analysis complete: Standard DTS is adequate")
        print("  → Report findings and conclude research")
    else:
        print("  → Review output/reports/stageA_summary.txt for detailed analysis")
        print("  → Examine output/figures/ for visual diagnostics")
        print("  → Proceed to Stage B to test if Merton explains variation")

    print()
    print("="*80)


if __name__ == '__main__':
    main()
