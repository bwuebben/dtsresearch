#!/usr/bin/env python3
"""
Main script to run Stage C analysis.

PREREQUISITE: Stage B showed Merton explains variation (Paths 1-3)

This orchestrates the complete Stage C pipeline:
1. Load data and Stage B results
2. Run rolling window stability test (Chow test)
3. If unstable, run macro driver analysis
4. If unstable, run maturity-specific analysis
5. Generate visualizations
6. Create reports
7. Provide decision recommendation
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dts_research.data.loader import BondDataLoader
from dts_research.analysis.buckets import BucketClassifier
from dts_research.analysis.stage0 import Stage0Analysis
from dts_research.analysis.stageB import StageBAnalysis
from dts_research.analysis.stageC import StageCAnalysis
from dts_research.visualization.stageC_plots import StageCVisualizer
from dts_research.utils.reportingC import StageCReporter


def generate_mock_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate mock macro data for testing.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with vix, oas_index, r_10y columns
    """
    dates = pd.date_range(start_date, end_date, freq='D')

    # VIX: Mean ~15, spikes to 40+ in 2020 COVID
    vix_base = 15 + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    vix_noise = np.random.normal(0, 2, len(dates))
    vix = vix_base + vix_noise

    # COVID spike (2020-03 to 2020-06)
    covid_mask = (dates >= '2020-03-01') & (dates <= '2020-06-01')
    vix[covid_mask] = 40 + np.random.normal(0, 5, covid_mask.sum())

    # 2022 volatility (2022-02 to 2022-10)
    vol_2022_mask = (dates >= '2022-02-01') & (dates <= '2022-10-01')
    vix[vol_2022_mask] = 25 + np.random.normal(0, 3, vol_2022_mask.sum())

    # OAS index: Mean ~100 for IG, wider in crises
    oas_base = 100 + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    oas_noise = np.random.normal(0, 10, len(dates))
    oas_index = oas_base + oas_noise

    # COVID widening
    oas_index[covid_mask] = 200 + np.random.normal(0, 20, covid_mask.sum())

    # 10-year rate
    r_10y_base = 2.0 + 1.0 * np.sin(np.arange(len(dates)) * 2 * np.pi / (365 * 3))
    r_10y_noise = np.random.normal(0, 0.1, len(dates))
    r_10y = r_10y_base + r_10y_noise

    return pd.DataFrame({
        'date': dates,
        'vix': np.maximum(vix, 10),  # Floor at 10
        'oas_index': np.maximum(oas_index, 50),  # Floor at 50
        'r_10y': np.maximum(r_10y, 0.5)  # Floor at 0.5%
    })


def main():
    """
    Run complete Stage C analysis pipeline.
    """
    print("="*80)
    print("STAGE C: DOES STATIC MERTON SUFFICE OR DO WE NEED TIME-VARIATION?")
    print("="*80)
    print()
    print("Critical Question: Is the relationship between lambda and (s, T)")
    print("                  stable over time, or do macro variables induce")
    print("                  time-variation?")
    print()
    print("Key Principle: Don't add time-variation until you've proven the")
    print("              simple static model fails.")
    print()

    # Configuration
    start_date = '2010-01-01'
    end_date = '2024-12-31'
    use_mock_data = True
    window_years = 1  # 1-year rolling windows

    # Create output directories
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Data and Run Prerequisites
    # -------------------------------------------------------------------------
    print("Step 1: Loading data and running prerequisite analyses...")
    print("  (Stage C requires Stage B results as input)")
    print()

    if use_mock_data:
        print("  Using mock data for testing")
        loader = BondDataLoader()
        bond_data = loader.generate_mock_data(start_date, end_date, n_bonds=500)
        index_data = loader.generate_mock_index_data(start_date, end_date, index_type='IG')
        macro_data = generate_mock_macro_data(start_date, end_date)
    else:
        print("  Loading from database...")
        connection_string = "your_connection_string_here"
        loader = BondDataLoader(connection_string)
        loader.connect()
        bond_data = loader.load_bond_data(start_date, end_date)
        index_data = loader.load_index_data(start_date, end_date, index_type='IG')
        # Load macro data from your database
        macro_data = loader.load_macro_data(start_date, end_date)
        loader.close()

    print(f"  Loaded {len(bond_data):,} bond-week observations")
    print(f"  Loaded {len(macro_data):,} days of macro data")
    print()

    # Classify and prepare
    print("  Running prerequisite analyses (Stage 0 + Stage B)...")
    classifier = BucketClassifier()
    bond_data = classifier.classify_bonds(bond_data)
    bucket_stats = classifier.compute_bucket_characteristics(bond_data)

    stage0 = Stage0Analysis()
    regression_data = stage0.prepare_regression_data(bond_data, index_data)

    # Add VIX from macro data (merge by date)
    regression_data = regression_data.merge(
        macro_data[['date', 'vix']],
        on='date',
        how='left'
    )
    regression_data['vix'] = regression_data['vix'].fillna(15)  # Default VIX if missing

    # Add spread_regime column (IG/HY based on OAS)
    regression_data['spread_regime'] = regression_data['oas'].apply(
        lambda x: 'IG' if x < 300 else 'HY'
    )

    # Run Stage B (need for context)
    stageB = StageBAnalysis()
    spec_b1 = stageB.run_specification_b1(regression_data, by_regime=True)

    b1_combined = spec_b1.get('combined', {})
    if 'error' in b1_combined:
        print("  ERROR: Stage B prerequisite failed")
        print(f"  {b1_combined['error']}")
        return

    print(f"  Stage B: β_Merton = {b1_combined['beta_merton']:.3f}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Run Rolling Window Stability Test
    # -------------------------------------------------------------------------
    print("Step 2: Running rolling window stability test...")
    print(f"  Window size: {window_years} year(s)")
    print("  Testing H0: beta_1 = beta_2 = ... = beta_W")
    print()

    stageC = StageCAnalysis()
    stability_results = stageC.rolling_window_stability_test(
        regression_data,
        window_years=window_years,
        by_regime=True,
        by_maturity=True
    )

    rolling_combined = stability_results['results_combined']
    chow_combined = stability_results['chow_test_combined']

    print(f"  Estimated {len(rolling_combined)} windows")
    print(f"  Chow test: F = {chow_combined['f_statistic']:.2f}, p = {chow_combined['p_value']:.4f}")
    print(f"  {chow_combined['interpretation']}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Macro Driver Analysis (if unstable)
    # -------------------------------------------------------------------------
    macro_driver_results = None
    stable = chow_combined.get('stable', True)

    if not stable:
        print("Step 3: Running macro driver analysis...")
        print("  (Relationship is unstable, investigating drivers)")
        print()

        macro_driver_results = stageC.macro_driver_analysis(
            rolling_combined,
            macro_data
        )

        if 'error' not in macro_driver_results:
            print(f"  VIX effect: δ_VIX = {macro_driver_results['coefficients']['delta_VIX']:.4f}")
            print(f"             (p = {macro_driver_results['p_values']['p_delta_VIX']:.4f})")
            print(f"             Economic: {macro_driver_results['economic_significance']['effect_vix_pct']:.1f}%")
            print()
            print(f"  OAS effect: δ_OAS = {macro_driver_results['coefficients']['delta_OAS']:.4f}")
            print(f"             (p = {macro_driver_results['p_values']['p_delta_OAS']:.4f})")
            print(f"             Economic: {macro_driver_results['economic_significance']['effect_oas_pct']:.1f}%")
            print()
        else:
            print(f"  ERROR: {macro_driver_results['error']}")
            print()
    else:
        print("Step 3: Macro driver analysis SKIPPED")
        print("  (Relationship is stable, no need to explain time-variation)")
        print()

    # -------------------------------------------------------------------------
    # Step 4: Maturity-Specific Analysis (if unstable)
    # -------------------------------------------------------------------------
    maturity_results = None

    if not stable:
        print("Step 4: Running maturity-specific time-variation analysis...")
        print("  Theory predicts: δ_VIX,1y > δ_VIX,5y > δ_VIX,10y")
        print()

        maturity_results = stageC.maturity_specific_time_variation(
            regression_data,
            macro_data,
            window_years=window_years
        )

        if 'error' not in maturity_results.get('by_maturity', {}):
            by_mat = maturity_results['by_maturity']
            for bucket in ['1-2y', '3-5y', '7-10y']:
                if bucket in by_mat:
                    result = by_mat[bucket]
                    print(f"  {bucket}: δ_VIX = {result['delta_VIX']:.4f}, effect = {result['effect_pct']:.1f}%")

            pattern_test = maturity_results['pattern_test']
            if 'error' not in pattern_test:
                confirms = pattern_test.get('confirms_theory', False)
                print()
                print(f"  Pattern: {pattern_test['pattern']}")
                print(f"  Confirms theory? {'✓ YES' if confirms else '✗ NO'}")
        else:
            print("  Insufficient data for maturity-specific analysis")

        print()
    else:
        print("Step 4: Maturity-specific analysis SKIPPED")
        print("  (Relationship is stable)")
        print()

    # -------------------------------------------------------------------------
    # Step 5: Generate Decision
    # -------------------------------------------------------------------------
    print("Step 5: Generating decision recommendation...")
    print()

    decision = stability_results['decision']

    print("="*80)
    print("STAGE C DECISION")
    print("="*80)
    print(decision)
    print("="*80)
    print()

    # -------------------------------------------------------------------------
    # Step 6: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 6: Generating visualizations...")

    visualizer = StageCVisualizer(output_dir='./output/figures')

    rolling_ig = stability_results.get('results_ig')
    rolling_hy = stability_results.get('results_hy')

    figures = visualizer.create_all_stageC_figures(
        rolling_combined,
        macro_data,
        macro_driver_results if macro_driver_results else {},
        regression_data,
        rolling_ig,
        rolling_hy,
        output_prefix='stageC'
    )

    print(f"  Created {len(figures)} figures:")
    print("    - Figure C.1: Beta time series (rolling windows)")
    print("    - Figure C.2: Beta vs macro variables")
    print("    - Figure C.3: Implied lambda over time")
    print("    - Figure C.4: Crisis vs normal period analysis")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 7: Generating reports...")

    reporter = StageCReporter(output_dir='./output/reports')

    chow_ig = stability_results.get('chow_test_ig')
    chow_hy = stability_results.get('chow_test_hy')

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
        prefix='stageC'
    )

    print("  Created reports:")
    print("    - Table C.1: Rolling window stability test")
    if macro_driver_results and 'error' not in macro_driver_results:
        print("    - Table C.2: Macro driver regression")
    if maturity_results and 'error' not in maturity_results.get('by_maturity', {}):
        print("    - Table C.3: Maturity-specific time-variation")
    print("    - Full rolling window results CSV")
    print("    - Written summary (3-4 pages)")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("="*80)
    print("STAGE C COMPLETE")
    print("="*80)
    print()
    print("Key Findings:")
    print(f"  • Chow test: p = {chow_combined['p_value']:.4f}")
    print(f"  • Stability: {'✓ STABLE' if stable else '✗ UNSTABLE'}")

    if not stable and macro_driver_results and 'error' not in macro_driver_results:
        effect_vix = macro_driver_results['economic_significance']['effect_vix_pct']
        effect_oas = macro_driver_results['economic_significance']['effect_oas_pct']
        print(f"  • VIX effect: {effect_vix:.1f}%")
        print(f"  • OAS effect: {effect_oas:.1f}%")

    print()
    print("Next steps:")

    if stable:
        print("  → Use static lambda (no time-varying adjustments needed)")
        print("  → Proceed to Stage D to test robustness")
        print("  → Stage E will select parsimonious production spec")
    else:
        if macro_driver_results and 'error' not in macro_driver_results:
            max_effect = max(
                abs(macro_driver_results['economic_significance']['effect_vix_pct']),
                abs(macro_driver_results['economic_significance']['effect_oas_pct'])
            )

            if max_effect > 20:
                print("  → Time-variation is ECONOMICALLY SIGNIFICANT (> 20%)")
                print("  → Consider time-varying lambda in production")
                print("  → Stage E should incorporate macro state variables")
            elif max_effect > 10:
                print("  → Moderate time-variation (10-20%)")
                print("  → Consider hybrid: static baseline + crisis adjustments")
            else:
                print("  → Time-variation statistically significant but small (< 10%)")
                print("  → Use static lambda despite statistical test")
        else:
            print("  → Time-variation detected but drivers unclear")
            print("  → Review Figure C.1 for visual assessment")

        print("  → Proceed to Stage D (robustness)")

    print()
    print("="*80)


if __name__ == '__main__':
    main()
