#!/usr/bin/env python3
"""
Main script to run Stage B analysis.

PREREQUISITE: Stage A must show variation exists (F-test p < 0.10)

This orchestrates the complete Stage B pipeline:
1. Load data and Stage A results
2. Run Specification B.1: Merton as offset (constrained)
3. Run Specification B.2: Decomposed components (maturity vs credit)
4. Run Specification B.3: Unrestricted (comparison baseline)
5. Compare all models
6. Create theory vs reality table
7. Generate visualizations
8. Create reports
9. Provide decision recommendation

Usage:
    python run_stageB.py [--mock-data]
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
from dts_research.analysis.stageB import StageBAnalysis
from dts_research.analysis.stage0_synthesis import Stage0Synthesis
from dts_research.analysis.stage0_bucket import BucketLevelAnalysis
from dts_research.analysis.stage0_within_issuer import WithinIssuerAnalysis
from dts_research.analysis.stage0_sector import SectorInteractionAnalysis
from dts_research.models.merton import MertonLambdaCalculator
from dts_research.visualization.stageB_plots import StageBVisualizer
from dts_research.utils.reportingB import StageBReporter


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

    # Compute Merton lambda components
    merton_calc = MertonLambdaCalculator()
    df['lambda_Merton'] = df.apply(
        lambda row: merton_calc.lambda_combined(row['time_to_maturity'], row['oas']),
        axis=1
    )
    df['lambda_T'] = df.apply(
        lambda row: merton_calc.lambda_T(row['time_to_maturity'], row['oas']),
        axis=1
    )
    df['lambda_s'] = df.apply(
        lambda row: merton_calc.lambda_s(row['oas']),
        axis=1
    )

    # Compute Merton-scaled factors
    df['f_merton'] = df['lambda_Merton'] * df['f_DTS']
    df['f_T'] = df['lambda_T'] * df['f_DTS']
    df['f_s'] = df['lambda_s'] * df['f_DTS']

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
    Run complete Stage B analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Run Stage B Analysis')
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
    print("STAGE B: DOES MERTON EXPLAIN THE VARIATION?")
    print("=" * 80)
    print()
    print("Critical Question: Does Merton's structural model explain the")
    print("                  cross-sectional variation documented in Stage A?")
    print()
    print("Three specifications:")
    print("  B.1: Merton as offset (constrained) - test if beta_Merton = 1")
    print("  B.2: Decomposed components - test beta_T and beta_s separately")
    print("  B.3: Unrestricted - fully flexible comparison")
    print()

    # Create output directories
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'stageB_figures'
    reports_dir = output_dir / 'stageB_reports'
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Data and Run Prerequisites
    # -------------------------------------------------------------------------
    print("Step 1: Loading data and running prerequisite analyses...")
    print("  (Stage B requires Stage A results as input)")
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
    print("  Classifying bonds into buckets...")
    bond_data, bucket_stats = classify_bonds_into_buckets(
        bond_data,
        rating_column='rating',
        maturity_column='time_to_maturity',
        sector_column='sector',
        compute_characteristics=True
    )

    # Add required columns to bucket_stats
    merton_calc = MertonLambdaCalculator()
    if bucket_stats is not None:
        bucket_stats['median_maturity'] = bucket_stats['maturity_median']
        bucket_stats['median_spread'] = bucket_stats['spread_median']
        bucket_stats['sector'] = bucket_stats['sector_group'].map({
            'A': 'Industrial', 'B': 'Financial'
        })
        ig_ratings = ['AAA/AA', 'A', 'BBB']
        bucket_stats['is_ig'] = bucket_stats['rating_bucket'].isin(ig_ratings)

        # Compute Merton lambda for each bucket based on median characteristics
        bucket_stats['lambda_merton'] = bucket_stats.apply(
            lambda row: merton_calc.lambda_combined(
                row['median_maturity'] if pd.notna(row['median_maturity']) else 5.0,
                row['median_spread'] if pd.notna(row['median_spread']) else 100.0
            ),
            axis=1
        )

    print(f"  Created {len(bucket_stats)} buckets")
    print()

    # Prepare regression data
    print("  Preparing regression data with Merton components...")
    regression_data = prepare_regression_data(bond_data)
    print(f"  Regression-ready observations: {len(regression_data):,}")
    print()

    # Get Stage 0 synthesis
    print("  Getting Stage 0 synthesis...")
    stage0_synthesis = get_stage0_synthesis(bond_data, universe='IG')
    print(f"  Stage 0 Path: {stage0_synthesis['decision_path']}")

    # Run Stage A to get bucket betas
    print("  Running Stage A for comparison...")
    stageA = StageAAnalysis(stage0_results=stage0_synthesis)
    stage_a_results = stageA.run_specification_a1(regression_data, bucket_stats)
    print(f"  Stage A: {len(stage_a_results)} bucket betas estimated")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Run Specification B.1 - Merton as Offset
    # -------------------------------------------------------------------------
    print("Step 2: Running Specification B.1 (Merton constrained)...")
    print("  y_i,t = alpha + beta_Merton * [lambda^Merton_i,t * f_DTS,t] + epsilon")
    print("  Theory prediction: beta_Merton = 1")
    print()

    stageB = StageBAnalysis(stage0_results=stage0_synthesis)
    spec_b1 = stageB.run_specification_b1(regression_data, by_regime=True)

    b1_combined = spec_b1.get('combined', {})
    if 'error' not in b1_combined:
        print(f"  beta_Merton = {b1_combined.get('beta_merton', np.nan):.3f} (SE = {b1_combined.get('se_beta', np.nan):.3f})")
        print(f"  Test H0: beta=1, p-value = {b1_combined.get('p_value_h0_beta_eq_1', np.nan):.4f}")
        print(f"  R² = {b1_combined.get('r_squared', np.nan):.3f}")
        print(f"  -> {b1_combined.get('interpretation', 'N/A')}")
    else:
        print(f"  ERROR: {b1_combined.get('error', 'Unknown error')}")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Run Specification B.2 - Decomposed Components
    # -------------------------------------------------------------------------
    print("Step 3: Running Specification B.2 (decomposed components)...")
    print("  y_i,t = alpha + beta_T*[lambda_T * f_DTS] + beta_s*[lambda_s * f_DTS] + epsilon")
    print("  Theory prediction: beta_T ~ 1 and beta_s ~ 1")
    print()

    spec_b2 = stageB.run_specification_b2(regression_data, by_regime=True)

    b2_combined = spec_b2.get('combined', {})
    if 'error' not in b2_combined:
        print(f"  beta_T (maturity) = {b2_combined.get('beta_T', np.nan):.3f} (SE = {b2_combined.get('se_beta_T', np.nan):.3f})")
        print(f"  beta_s (credit) = {b2_combined.get('beta_s', np.nan):.3f} (SE = {b2_combined.get('se_beta_s', np.nan):.3f})")
        print(f"  Joint test p-value = {b2_combined.get('joint_test_pvalue', np.nan):.4f}")
        print(f"  R² = {b2_combined.get('r_squared', np.nan):.3f}")
        print(f"  -> {b2_combined.get('interpretation', 'N/A')}")
    else:
        print(f"  ERROR: {b2_combined.get('error', 'Unknown error')}")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Run Specification B.3 - Unrestricted
    # -------------------------------------------------------------------------
    print("Step 4: Running Specification B.3 (unrestricted)...")
    print("  Fully flexible functional form with polynomials and interactions")
    print()

    spec_b3 = stageB.run_specification_b3(regression_data, by_regime=True)

    b3_combined = spec_b3.get('combined', {})
    if 'error' not in b3_combined:
        print(f"  R² = {b3_combined.get('r_squared', np.nan):.3f}")
        print(f"  Parameters = {b3_combined.get('n_parameters', 'N/A')}")
        print(f"  Lambda model R² = {b3_combined.get('lambda_r_squared', np.nan):.3f}")
    else:
        print(f"  ERROR: {b3_combined.get('error', 'Unknown error')}")
    print()

    # -------------------------------------------------------------------------
    # Step 5: Model Comparison
    # -------------------------------------------------------------------------
    print("Step 5: Comparing all models...")

    model_comparison = stageB.compare_models(stage_a_results, spec_b1, spec_b2, spec_b3)

    print("\n  Model Comparison:")
    for idx, row in model_comparison.iterrows():
        print(f"    {row['Model']}: R² = {row['R²']}, dR² = {row['ΔR² vs Stage A']}")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Theory vs Reality
    # -------------------------------------------------------------------------
    print("Step 6: Creating theory vs reality comparison...")

    theory_vs_reality = stageB.create_theory_vs_reality_table(stage_a_results, bucket_stats)
    theory_assessment = stageB.assess_theory_performance(theory_vs_reality)

    print(f"  Buckets in acceptable range [0.8, 1.2]: {theory_assessment.get('pct_in_acceptable_range', np.nan):.1f}%")
    print(f"  Median ratio (beta/lambda): {theory_assessment.get('median_ratio', np.nan):.3f}")
    print(f"  Systematic bias: {theory_assessment.get('systematic_bias', 'N/A')}")
    print(f"  -> {theory_assessment.get('assessment', 'N/A')}")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Decision
    # -------------------------------------------------------------------------
    print("Step 7: Generating decision recommendation...")

    decision = stageB.generate_stage_b_decision(spec_b1, model_comparison, theory_assessment)

    print()
    print("=" * 80)
    print("STAGE B DECISION")
    print("=" * 80)
    print(decision)
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Step 8: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 8: Generating visualizations...")

    merton_calc = MertonLambdaCalculator()
    visualizer = StageBVisualizer(output_dir=str(figures_dir))
    figures = visualizer.create_all_stageB_figures(
        theory_vs_reality,
        merton_calc,
        spec_b3,
        output_prefix='stageB'
    )

    print(f"  Created {len(figures)} figures in {figures_dir}")
    print()

    # -------------------------------------------------------------------------
    # Step 9: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 9: Generating reports...")

    reporter = StageBReporter(output_dir=str(reports_dir))
    reporter.save_all_reports(
        spec_b1,
        spec_b2,
        spec_b3,
        model_comparison,
        theory_vs_reality,
        theory_assessment,
        decision,
        prefix='stageB'
    )

    print(f"  Reports saved to {reports_dir}")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STAGE B COMPLETE")
    print("=" * 80)
    print()
    print("Key Findings:")
    if 'error' not in b1_combined:
        print(f"  - beta_Merton = {b1_combined.get('beta_merton', np.nan):.3f} (H0: beta=1, p={b1_combined.get('p_value_h0_beta_eq_1', np.nan):.4f})")
        print(f"  - R² = {b1_combined.get('r_squared', np.nan):.3f}")
    if 'error' not in b2_combined:
        print(f"  - beta_T = {b2_combined.get('beta_T', np.nan):.3f}, beta_s = {b2_combined.get('beta_s', np.nan):.3f}")
    print(f"  - Theory explains {theory_assessment.get('pct_in_acceptable_range', np.nan):.0f}% of buckets well")
    print()

    print("Next steps:")
    if 'PATH 1' in decision or 'PATH 2' in decision:
        print("  -> Proceed to Stage C to test time-variation")
    elif 'PATH 3' in decision:
        print("  -> Stage C should run both theory and unrestricted tracks")
    else:
        print("  -> Theory fundamentally fails - proceed to Stage D")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
