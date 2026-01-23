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

Usage:
    python run_stageC.py [--mock-data]
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
from dts_research.analysis.stageC import StageCAnalysis
from dts_research.analysis.stage0_synthesis import Stage0Synthesis
from dts_research.analysis.stage0_bucket import BucketLevelAnalysis
from dts_research.analysis.stage0_within_issuer import WithinIssuerAnalysis
from dts_research.analysis.stage0_sector import SectorInteractionAnalysis
from dts_research.models.merton import MertonLambdaCalculator
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

    np.random.seed(42)

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


def prepare_regression_data(bond_data: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare bond data for regression analysis.

    Adds spread changes, index DTS factor, Merton lambda, and required columns.
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
    df['f_DTS'] = df['oas_index_pct_change']

    # Compute Merton lambda
    merton_calc = MertonLambdaCalculator()
    df['lambda_Merton'] = df.apply(
        lambda row: merton_calc.lambda_combined(row['time_to_maturity'], row['oas']),
        axis=1
    )

    # Compute Merton-scaled factor
    df['f_merton'] = df['lambda_Merton'] * df['f_DTS']

    # Merge macro data (VIX)
    df = df.merge(
        macro_data[['date', 'vix']],
        on='date',
        how='left'
    )
    df['vix'] = df['vix'].fillna(15)  # Default VIX if missing

    # Add spread regime
    df['spread_regime'] = np.where(df['oas'] < 300, 'IG', 'HY')

    # Add week identifier for clustering
    df['week'] = df['date'].dt.isocalendar().week.astype(str) + '_' + df['date'].dt.year.astype(str)

    # Drop NaN
    df = df.dropna(subset=['oas_pct_change', 'oas_index_pct_change', 'lambda_Merton'])

    return df


def get_stage0_synthesis(bond_data: pd.DataFrame, universe: str = 'IG') -> dict:
    """Run Stage 0 synthesis to get decision path."""
    bucket_analyzer = BucketLevelAnalysis()
    bucket_results = bucket_analyzer.run_bucket_analysis(bond_data, universe=universe)

    within_analyzer = WithinIssuerAnalysis()
    within_results = within_analyzer.run_within_issuer_analysis(bond_data, universe=universe, verbose=False)

    sector_analyzer = SectorInteractionAnalysis()
    sector_results = sector_analyzer.run_sector_analysis(bond_data, universe=universe, cluster_by='week')

    synthesizer = Stage0Synthesis()
    synthesis = synthesizer.synthesize_results(
        bucket_results, within_results, sector_results, universe=universe
    )

    return synthesis


def main():
    """
    Run complete Stage C analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Run Stage C Analysis')
    parser.add_argument('--mock-data', action='store_true', default=True,
                       help='Use mock data (default: True)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--window-years', type=int, default=1,
                       help='Rolling window size in years')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE C: DOES STATIC MERTON SUFFICE OR DO WE NEED TIME-VARIATION?")
    print("=" * 80)
    print()
    print("Critical Question: Is the relationship between lambda and (s, T)")
    print("                  stable over time, or do macro variables induce")
    print("                  time-variation?")
    print()
    print("Key Principle: Don't add time-variation until you've proven the")
    print("              simple static model fails.")
    print()

    # Create output directories
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'stageC_figures'
    reports_dir = output_dir / 'stageC_reports'
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Data and Run Prerequisites
    # -------------------------------------------------------------------------
    print("Step 1: Loading data and running prerequisite analyses...")
    print("  (Stage C requires Stage B results as input)")
    print()

    loader = BondDataLoader()
    if args.mock_data:
        print("  Using mock data for testing")
        bond_data = loader.generate_mock_data(args.start_date, args.end_date, n_bonds=500)
        macro_data = generate_mock_macro_data(args.start_date, args.end_date)
    else:
        print("  Loading from database...")
        bond_data = loader.load_bond_data(args.start_date, args.end_date)
        macro_data = loader.load_macro_data(args.start_date, args.end_date)

    print(f"  Loaded {len(bond_data):,} bond-week observations")
    print(f"  Loaded {len(macro_data):,} days of macro data")
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
    bond_data, bucket_stats = classify_bonds_into_buckets(
        bond_data,
        rating_column='rating',
        maturity_column='time_to_maturity',
        sector_column='sector',
        compute_characteristics=True
    )

    # Prepare regression data
    print("  Preparing regression data with Merton components...")
    regression_data = prepare_regression_data(bond_data, macro_data)
    print(f"  Regression-ready observations: {len(regression_data):,}")
    print()

    # Get Stage 0 synthesis
    print("  Getting Stage 0 synthesis...")
    stage0_synthesis = get_stage0_synthesis(bond_data, universe='IG')
    print(f"  Stage 0 Path: {stage0_synthesis['decision_path']}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Run Rolling Window Stability Test
    # -------------------------------------------------------------------------
    print("Step 2: Running rolling window stability test...")
    print(f"  Window size: {args.window_years} year(s)")
    print("  Testing H0: beta_1 = beta_2 = ... = beta_W")
    print()

    stageC = StageCAnalysis(stage0_results=stage0_synthesis)
    stability_results = stageC.rolling_window_stability_test(
        regression_data,
        window_years=args.window_years,
        by_regime=True,
        by_maturity=True
    )

    # The results use keys: results_combined, chow_test_combined, results_ig, chow_test_ig, etc.
    rolling_combined = stability_results.get('results_combined', pd.DataFrame())
    chow_combined = stability_results.get('chow_test_combined', {})

    if len(rolling_combined) > 0:
        print(f"  Estimated {len(rolling_combined)} windows")
    print(f"  Chow test: F = {chow_combined.get('f_statistic', np.nan):.2f}, p = {chow_combined.get('p_value', np.nan):.4f}")
    print(f"  {chow_combined.get('interpretation', 'N/A')}")
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

        try:
            macro_driver_results = stageC.macro_driver_analysis(
                rolling_combined,
                macro_data
            )

            if 'error' not in macro_driver_results:
                coeffs = macro_driver_results.get('coefficients', {})
                pvals = macro_driver_results.get('p_values', {})
                econ = macro_driver_results.get('economic_significance', {})

                print(f"  VIX effect: delta_VIX = {coeffs.get('delta_VIX', np.nan):.4f}")
                print(f"             (p = {pvals.get('p_delta_VIX', np.nan):.4f})")
                print(f"             Economic: {econ.get('effect_vix_pct', np.nan):.1f}%")
                print()
                print(f"  OAS effect: delta_OAS = {coeffs.get('delta_OAS', np.nan):.4f}")
                print(f"             (p = {pvals.get('p_delta_OAS', np.nan):.4f})")
                print(f"             Economic: {econ.get('effect_oas_pct', np.nan):.1f}%")
            else:
                print(f"  ERROR: {macro_driver_results.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ERROR: Macro driver analysis failed: {str(e)[:50]}")
            macro_driver_results = {'error': str(e)}
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
        print("  Theory predicts: delta_VIX,1y > delta_VIX,5y > delta_VIX,10y")
        print()

        try:
            maturity_results = stageC.maturity_specific_time_variation(
                regression_data,
                macro_data,
                window_years=args.window_years
            )

            if 'error' not in maturity_results.get('by_maturity', {}):
                by_mat = maturity_results.get('by_maturity', {})
                for bucket in ['1-2y', '3-5y', '7-10y']:
                    if bucket in by_mat:
                        result = by_mat[bucket]
                        print(f"  {bucket}: delta_VIX = {result.get('delta_VIX', np.nan):.4f}, effect = {result.get('effect_pct', np.nan):.1f}%")

                pattern_test = maturity_results.get('pattern_test', {})
                if 'error' not in pattern_test:
                    confirms = pattern_test.get('confirms_theory', False)
                    print()
                    print(f"  Pattern: {pattern_test.get('pattern', 'N/A')}")
                    print(f"  Confirms theory? {'YES' if confirms else 'NO'}")
        except Exception as e:
            print(f"  ERROR: Maturity-specific analysis failed: {str(e)[:50]}")
            maturity_results = {'error': str(e)}
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

    decision = stability_results.get('decision', 'No decision generated')

    print("=" * 80)
    print("STAGE C DECISION")
    print("=" * 80)
    print(decision)
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Step 6: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 6: Generating visualizations...")

    visualizer = StageCVisualizer(output_dir=str(figures_dir))

    # Get IG and HY rolling results (keys are results_ig, results_hy)
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

    print(f"  Created {len(figures)} figures in {figures_dir}")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 7: Generating reports...")

    reporter = StageCReporter(output_dir=str(reports_dir))

    # Get IG and HY chow tests (keys are chow_test_ig, chow_test_hy)
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

    print(f"  Reports saved to {reports_dir}")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STAGE C COMPLETE")
    print("=" * 80)
    print()
    print("Key Findings:")
    print(f"  - Chow test: p = {chow_combined.get('p_value', np.nan):.4f}")
    print(f"  - Stability: {'STABLE' if stable else 'UNSTABLE'}")

    if not stable and macro_driver_results and 'error' not in macro_driver_results:
        econ = macro_driver_results.get('economic_significance', {})
        print(f"  - VIX effect: {econ.get('effect_vix_pct', np.nan):.1f}%")
        print(f"  - OAS effect: {econ.get('effect_oas_pct', np.nan):.1f}%")

    print()
    print("Next steps:")

    if stable:
        print("  -> Use static lambda (no time-varying adjustments needed)")
        print("  -> Proceed to Stage D to test robustness")
    else:
        print("  -> Time-variation detected")
        print("  -> Review Figure C.1 for visual assessment")
        print("  -> Proceed to Stage D (robustness)")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
