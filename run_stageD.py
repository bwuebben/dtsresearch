#!/usr/bin/env python3
"""
Main script to run Stage D analysis.

PREREQUISITE: Stages A-C completed

This orchestrates the complete Stage D pipeline:
1. Load data and prerequisites
2. Run D.1: Tail behavior (quantile regression)
3. Run D.2: Shock decomposition
4. Run D.3: Liquidity adjustment
5. Generate visualizations
6. Create reports
7. Provide recommendations

Usage:
    python run_stageD.py [--mock-data]
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
from dts_research.analysis.stageD import StageDAnalysis
from dts_research.analysis.stage0_synthesis import Stage0Synthesis
from dts_research.analysis.stage0_bucket import BucketLevelAnalysis
from dts_research.analysis.stage0_within_issuer import WithinIssuerAnalysis
from dts_research.analysis.stage0_sector import SectorInteractionAnalysis
from dts_research.models.merton import MertonLambdaCalculator
from dts_research.visualization.stageD_plots import StageDVisualizer
from dts_research.utils.reportingD import StageDReporter


def prepare_regression_data(bond_data: pd.DataFrame) -> pd.DataFrame:
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
    Run complete Stage D analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Run Stage D Analysis')
    parser.add_argument('--mock-data', action='store_true', default=True,
                       help='Use mock data (default: True)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE D: ROBUSTNESS AND EXTENSIONS")
    print("=" * 80)
    print()
    print("Objective: Test robustness across:")
    print("  1. Tail events (quantile regression)")
    print("  2. Shock types (systematic vs idiosyncratic)")
    print("  3. Spread components (default vs liquidity)")
    print()
    print("Key Framing: These are SECONDARY tests.")
    print("  - If Stages A-C validated Merton -> Confirm not just mean effect")
    print("  - If Stages A-C showed failure -> Diagnose WHY")
    print()

    # Create output directories
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'stageD_figures'
    reports_dir = output_dir / 'stageD_reports'
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Data and Run Prerequisites
    # -------------------------------------------------------------------------
    print("Step 1: Loading data...")
    print()

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
    bond_data, bucket_stats = classify_bonds_into_buckets(
        bond_data,
        rating_column='rating',
        maturity_column='time_to_maturity',
        sector_column='sector',
        compute_characteristics=True
    )

    # Prepare regression data
    print("  Preparing regression data...")
    regression_data = prepare_regression_data(bond_data)
    print(f"  Prepared {len(regression_data):,} observations for analysis")
    print()

    # Get Stage 0 synthesis
    print("  Getting Stage 0 synthesis...")
    stage0_synthesis = get_stage0_synthesis(bond_data, universe='IG')
    print(f"  Stage 0 Path: {stage0_synthesis['decision_path']}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Run D.1 - Tail Behavior (Quantile Regression)
    # -------------------------------------------------------------------------
    print("Step 2: Running D.1 - Tail Behavior (Quantile Regression)...")
    print("  Testing if Merton holds across distribution of spread changes")
    print()

    stageD = StageDAnalysis(stage0_results=stage0_synthesis)
    quantile_results = stageD.quantile_regression_analysis(
        regression_data,
        by_regime=True
    )

    tail_tests = quantile_results.get('tail_tests', {})
    print(f"  Pattern: {tail_tests.get('pattern', 'N/A')}")
    print(f"  Left tail amplification: {tail_tests.get('amplification_left', np.nan):.2f}x")
    print(f"  Right tail amplification: {tail_tests.get('amplification_right', np.nan):.2f}x")
    print()

    if tail_tests.get('amplification_left', 1.0) > 1.3:
        print("  WARNING: LEFT TAIL AMPLIFICATION DETECTED")
        print(f"    -> Tail risk {(tail_tests.get('amplification_left', 1.0)-1)*100:.0f}% larger than mean")
    else:
        print("  OK: No significant tail amplification")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Run D.2 - Shock Decomposition
    # -------------------------------------------------------------------------
    print("Step 3: Running D.2 - Shock Decomposition...")
    print("  Decomposing into Global, Sector, and Issuer-specific shocks")
    print()

    shock_results = stageD.shock_decomposition_analysis(
        regression_data,
        by_regime=True
    )

    shock_betas = shock_results.get('shock_betas_combined', {})
    variance_decomp = shock_results.get('variance_decomp', pd.DataFrame())

    print("  Variance Decomposition:")
    if len(variance_decomp) > 0:
        for _, row in variance_decomp.iterrows():
            print(f"    - {row.get('Component', 'N/A')}: {row.get('Pct_of_Total', np.nan):.1f}%")
    print()

    print("  Shock-Specific Elasticities:")
    print(f"    - beta^(G) (Global) = {shock_betas.get('beta_global', np.nan):.3f}")
    print(f"    - beta^(S) (Sector) = {shock_betas.get('beta_sector', np.nan):.3f}")
    print(f"    - beta^(I) (Issuer) = {shock_betas.get('beta_issuer', np.nan):.3f}")
    print()

    all_near_one = all(
        0.9 <= shock_betas.get(f'beta_{k}', 1.0) <= 1.1
        for k in ['global', 'sector', 'issuer']
    )

    if all_near_one:
        print("  OK: All shock types respect Merton elasticities")
    else:
        if shock_betas.get('beta_sector', 1.0) > 1.2:
            print(f"  WARNING: Sector shocks amplified ({shock_betas.get('beta_sector', np.nan):.2f}x)")
        if shock_betas.get('beta_issuer', 1.0) > 1.2:
            print(f"  WARNING: Issuer-specific shocks amplified ({shock_betas.get('beta_issuer', np.nan):.2f}x)")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Run D.3 - Liquidity Adjustment
    # -------------------------------------------------------------------------
    print("Step 4: Running D.3 - Liquidity Adjustment...")
    print("  Decomposing OAS into default and liquidity components")
    print()

    liquidity_results = stageD.liquidity_adjustment_analysis(
        regression_data
    )

    liq_model = liquidity_results.get('liquidity_model', {})
    comparison = liquidity_results.get('comparison', {})

    print(f"  Liquidity Model R²: {liq_model.get('r_squared', np.nan):.3f}")
    print()

    print("  Merton Fit Comparison:")
    print(f"    - Total OAS: beta = {comparison.get('beta_total', np.nan):.3f}, R² = {comparison.get('r2_total', np.nan):.3f}")
    print(f"    - Default component: beta = {comparison.get('beta_def', np.nan):.3f}, R² = {comparison.get('r2_def', np.nan):.3f}")
    print(f"    - Improvement: dR² = {comparison.get('delta_r2', np.nan):.3f} ({comparison.get('improvement_pct', np.nan):.1f}%)")
    print()

    delta_r2 = comparison.get('delta_r2', 0)
    if delta_r2 > 0.05:
        print("  OK: Liquidity adjustment materially improves fit")
        print("    -> Decompose OAS for HY and illiquid bonds")
    elif delta_r2 < 0.02:
        print("  -> Liquidity adjustment has minimal impact")
        print("    -> Use total OAS (simpler)")
    else:
        print("  -> Marginal benefit from liquidity adjustment")
    print()

    # -------------------------------------------------------------------------
    # Step 5: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 5: Generating visualizations...")

    visualizer = StageDVisualizer(output_dir=str(figures_dir))

    figures = visualizer.create_all_stageD_figures(
        quantile_results.get('results_combined', pd.DataFrame()),
        shock_results.get('shock_betas_combined', {}),
        liquidity_results.get('by_liquidity_quartile', pd.DataFrame()),
        shock_results.get('variance_decomp', pd.DataFrame()),
        quantile_results.get('results_ig'),
        quantile_results.get('results_hy'),
        shock_results.get('shock_betas_ig'),
        shock_results.get('shock_betas_hy'),
        output_prefix='stageD'
    )

    print(f"  Created {len(figures)} figures in {figures_dir}")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 6: Generating reports...")

    reporter = StageDReporter(output_dir=str(reports_dir))

    reporter.save_all_reports(
        quantile_results,
        shock_results,
        liquidity_results,
        prefix='stageD'
    )

    print(f"  Reports saved to {reports_dir}")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Recommendations
    # -------------------------------------------------------------------------
    print("Step 7: Generating production recommendations...")
    print()

    print("=" * 80)
    print("STAGE D RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Tail adjustments
    print("TAIL ADJUSTMENTS:")
    if tail_tests.get('amplification_left', 1.0) > 1.3:
        print(f"  WARNING: Use tail-specific lambda for VaR/ES:")
        print(f"    lambda^VaR = {tail_tests.get('amplification_left', np.nan):.2f} x lambda^Merton")
        print(f"    (Tail risk {(tail_tests.get('amplification_left', 1.0)-1)*100:.0f}% larger than mean)")
    else:
        print("  OK: Standard Merton lambda adequate for VaR/ES")
        print("    No tail-specific adjustments needed")
    print()

    # Shock-type adjustments
    print("SHOCK-TYPE CONSIDERATIONS:")
    if all_near_one:
        print("  OK: Use uniform lambda across shock types")
        print("    All shocks respect Merton elasticities")
    else:
        if shock_betas.get('beta_sector', 1.0) > 1.2:
            print(f"  WARNING: Sector shocks amplified ({shock_betas.get('beta_sector', np.nan):.2f}x)")
            print("    -> Consider sector-specific risk factors")

        if shock_betas.get('beta_issuer', 1.0) > 1.2:
            print(f"  WARNING: Issuer-specific shocks amplified ({shock_betas.get('beta_issuer', np.nan):.2f}x)")
            print("    -> Idiosyncratic risk larger than Merton predicts")
    print()

    # Liquidity adjustments
    print("LIQUIDITY DECOMPOSITION:")
    if delta_r2 > 0.05:
        print("  OK: Decompose OAS for HY and illiquid bonds")
        print("    -> Use lambda^def from Merton for default component")
        print("    -> Add separate lambda^liq empirically estimated")
    elif delta_r2 < 0.02:
        print("  -> Use total OAS (simpler)")
        print("    Liquidity decomposition not worth complexity")
    else:
        print("  -> Consider for precision-critical applications")
        print("    Marginal benefit, may not justify operational cost")

    print()
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STAGE D COMPLETE")
    print("=" * 80)
    print()
    print("Key Findings:")
    print(f"  - Tail amplification: {tail_tests.get('amplification_left', np.nan):.2f}x (left), {tail_tests.get('amplification_right', np.nan):.2f}x (right)")
    print(f"  - Global shock beta: {shock_betas.get('beta_global', np.nan):.3f}")
    print(f"  - Sector shock beta: {shock_betas.get('beta_sector', np.nan):.3f}")
    print(f"  - Issuer shock beta: {shock_betas.get('beta_issuer', np.nan):.3f}")
    print(f"  - Liquidity adjustment dR²: {comparison.get('delta_r2', np.nan):.3f}")
    print()

    print("Next steps:")
    print("  -> Review reports for detailed analysis")
    print("  -> Examine Figures D.1-D.3 for visual confirmation")
    print("  -> Incorporate findings into Stage E production specification")
    print("  -> Document any tail/shock/liquidity adjustments needed")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
