#!/usr/bin/env python3
"""
Main script to run Stage 0 analysis.

This orchestrates the complete Stage 0 pipeline:
1. Load data (mock or from database)
2. Classify bonds into buckets
3. Run pooled regressions
4. Perform statistical tests
5. Generate visualizations
6. Create reports
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dts_research.data.loader import BondDataLoader
from dts_research.analysis.buckets import BucketClassifier
from dts_research.analysis.stage0 import Stage0Analysis
from dts_research.visualization.stage0_plots import Stage0Visualizer
from dts_research.utils.reporting import Stage0Reporter


def main():
    """
    Run complete Stage 0 analysis pipeline.
    """
    print("="*80)
    print("STAGE 0: RAW VALIDATION USING BUCKET-LEVEL ANALYSIS")
    print("="*80)
    print()

    # Configuration
    start_date = '2010-01-01'
    end_date = '2024-12-31'
    use_mock_data = True  # Set to False when using real database

    # Create output directories
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("Step 1: Loading bond data...")

    if use_mock_data:
        print("  Using mock data for testing")
        loader = BondDataLoader()
        bond_data = loader.generate_mock_data(start_date, end_date, n_bonds=500)
        index_data_ig = loader.generate_mock_index_data(start_date, end_date, index_type='IG')
        index_data_hy = loader.generate_mock_index_data(start_date, end_date, index_type='HY', seed=43)
    else:
        print("  Loading from database...")
        # TODO: User fills in connection string
        connection_string = "your_connection_string_here"
        loader = BondDataLoader(connection_string)
        loader.connect()
        bond_data = loader.load_bond_data(start_date, end_date)
        index_data_ig = loader.load_index_data(start_date, end_date, index_type='IG')
        index_data_hy = loader.load_index_data(start_date, end_date, index_type='HY')
        loader.close()

    print(f"  Loaded {len(bond_data):,} bond-week observations")
    print(f"  Bonds: {bond_data['bond_id'].nunique():,}")
    print(f"  Date range: {bond_data['date'].min()} to {bond_data['date'].max()}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Classify into Buckets
    # -------------------------------------------------------------------------
    print("Step 2: Classifying bonds into buckets...")

    classifier = BucketClassifier()
    bond_data = classifier.classify_bonds(bond_data)

    bucket_stats = classifier.compute_bucket_characteristics(bond_data, min_observations=50)

    print(f"  Total buckets: {len(bucket_stats)}")
    print(f"  IG buckets: {bucket_stats['is_ig'].sum()}")
    print(f"  HY buckets: {(~bucket_stats['is_ig']).sum()}")
    print()

    # Show bucket coverage
    print("  Bucket coverage (observations by rating x maturity):")
    coverage = classifier.summarize_bucket_coverage(bucket_stats)
    print(coverage)
    print()

    # -------------------------------------------------------------------------
    # Step 3: Run Regressions
    # -------------------------------------------------------------------------
    print("Step 3: Running pooled regressions for each bucket...")

    # Combine IG and HY index data (use appropriate index for each bond)
    bond_data['index_type'] = bond_data['is_ig'].map({True: 'IG', False: 'HY'})

    # For simplicity, use IG index for all bonds in this example
    # In production, merge appropriate index based on bond's IG/HY classification
    index_data = index_data_ig.copy()

    analysis = Stage0Analysis()
    regression_data = analysis.prepare_regression_data(bond_data, index_data)

    print(f"  Regression-ready observations: {len(regression_data):,}")

    results_df = analysis.run_all_bucket_regressions(regression_data, bucket_stats)

    print(f"  Successfully estimated {len(results_df)} bucket regressions")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Statistical Tests
    # -------------------------------------------------------------------------
    print("Step 4: Running statistical tests...")

    # Test 1: Aggregate level test
    print("  Test 1: Aggregate level test (H0: mean deviation = 0)")
    aggregate_test = analysis.aggregate_level_test(results_df)
    print(f"    Mean deviation: {aggregate_test['mean_deviation']:.4f}")
    print(f"    Median ratio: {aggregate_test['median_ratio']:.3f}")
    print(f"    p-value: {aggregate_test['p_value']:.4f}")
    print(f"    Buckets in range [0.8, 1.2]: {aggregate_test['pct_in_range']:.1f}%")
    print()

    # Test 2: Cross-maturity pattern tests
    print("  Test 2: Cross-maturity pattern tests")
    cross_maturity_tests = []

    # Test key rating/sector combinations
    test_combinations = [
        ('BBB', 'Industrial'),
        ('A', 'Financial'),
        ('BB', 'Industrial'),
    ]

    for rating, sector in test_combinations:
        test = analysis.test_cross_maturity_pattern(results_df, rating, sector)
        if 'error' not in test:
            cross_maturity_tests.append(test)
            print(f"    {rating}/{sector}: {test['interpretation']}")
            print(f"      Spearman ρ: {test['spearman_rho_maturity_beta']:.3f} (p={test['p_value_maturity']:.3f})")

    print()

    # Test 3: Regime pattern test
    print("  Test 3: Regime pattern test (dispersion vs spread level)")
    regime_test = analysis.test_regime_pattern(results_df)
    print(f"    {regime_test['interpretation']}")
    print(f"    IG dispersion: {regime_test['ig_avg_dispersion']:.3f}")
    print(f"    HY dispersion: {regime_test['hy_avg_dispersion']:.3f}")
    print()

    # Identify outliers
    print("  Identifying outlier buckets...")
    outliers = analysis.identify_outliers(results_df, threshold=1.5)
    print(f"    Found {len(outliers)} outlier buckets (ratio outside [0.67, 1.5])")
    print()

    # -------------------------------------------------------------------------
    # Step 5: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 5: Generating visualizations...")

    visualizer = Stage0Visualizer(output_dir='./output/figures')
    figures = visualizer.create_all_stage0_figures(results_df, output_prefix='stage0')

    print(f"  Created {len(figures)} figures:")
    print("    - Figure 0.1: Empirical vs Theoretical scatter plot")
    print("    - Figure 0.2: Cross-maturity patterns by rating")
    print("    - Figure 0.3: Regime patterns (dispersion vs spread)")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 6: Generating reports...")

    reporter = Stage0Reporter(output_dir='./output/reports')
    reporter.save_all_reports(
        results_df,
        aggregate_test,
        cross_maturity_tests,
        regime_test,
        outliers,
        prefix='stage0'
    )

    print("  Created reports:")
    print("    - Table 0.1: Bucket-level results")
    print("    - Table 0.2: Cross-maturity patterns")
    print("    - Written summary (2-3 pages)")
    print("    - Full results CSV")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Decision Recommendation
    # -------------------------------------------------------------------------
    print("="*80)
    print("STAGE 0 COMPLETE - DECISION RECOMMENDATION")
    print("="*80)

    recommendation = analysis.generate_decision_recommendation(results_df)
    print(recommendation)
    print()

    print("Next steps:")
    print("  - Review output/reports/stage0_summary.txt for detailed analysis")
    print("  - Examine output/figures/ for visual diagnostics")
    print("  - Proceed to Stage A based on decision recommendation")
    print()

    print("="*80)
    print("All Stage 0 deliverables completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
