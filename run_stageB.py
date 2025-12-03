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
from dts_research.analysis.stageB import StageBAnalysis
from dts_research.models.merton import MertonLambdaCalculator
from dts_research.visualization.stageB_plots import StageBVisualizer
from dts_research.utils.reportingB import StageBReporter


def main():
    """
    Run complete Stage B analysis pipeline.
    """
    print("="*80)
    print("STAGE B: DOES MERTON EXPLAIN THE VARIATION?")
    print("="*80)
    print()
    print("Critical Question: Does Merton's structural model explain the")
    print("                  cross-sectional variation documented in Stage A?")
    print()
    print("Three specifications:")
    print("  B.1: Merton as offset (constrained) - test if β_Merton = 1")
    print("  B.2: Decomposed components - test β_T and β_s separately")
    print("  B.3: Unrestricted - fully flexible comparison")
    print()

    # Configuration
    start_date = '2010-01-01'
    end_date = '2024-12-31'
    use_mock_data = True

    # Create output directories
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Data and Run Prerequisites
    # -------------------------------------------------------------------------
    print("Step 1: Loading data and running prerequisite analyses...")
    print("  (Stage B requires Stage A results as input)")
    print()

    if use_mock_data:
        print("  Using mock data for testing")
        loader = BondDataLoader()
        bond_data = loader.generate_mock_data(start_date, end_date, n_bonds=500)
        index_data = loader.generate_mock_index_data(start_date, end_date, index_type='IG')
    else:
        print("  Loading from database...")
        connection_string = "your_connection_string_here"
        loader = BondDataLoader(connection_string)
        loader.connect()
        bond_data = loader.load_bond_data(start_date, end_date)
        index_data = loader.load_index_data(start_date, end_date, index_type='IG')
        loader.close()

    print(f"  Loaded {len(bond_data):,} bond-week observations")
    print()

    # Classify and prepare
    print("  Running prerequisite analyses (Stage 0 + Stage A)...")
    classifier = BucketClassifier()
    bond_data = classifier.classify_bonds(bond_data)
    bucket_stats = classifier.compute_bucket_characteristics(bond_data)

    # Debug: Check bucket_stats columns
    print(f"  Bucket stats columns: {list(bucket_stats.columns)}")
    print(f"  Bucket stats shape: {bucket_stats.shape}")

    stage0 = Stage0Analysis()
    regression_data = stage0.prepare_regression_data(bond_data, index_data)

    # Run Stage A (need results for comparison)
    stageA = StageAAnalysis()
    stage_a_results = stageA.run_specification_a1(regression_data, bucket_stats)

    print(f"  Stage A: {len(stage_a_results)} bucket betas estimated")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Run Specification B.1 - Merton as Offset
    # -------------------------------------------------------------------------
    print("Step 2: Running Specification B.1 (Merton constrained)...")
    print("  y_i,t = α + β_Merton · [λ^Merton_i,t · f_DTS,t] + ε")
    print("  Theory prediction: β_Merton = 1")
    print()

    stageB = StageBAnalysis()
    spec_b1 = stageB.run_specification_b1(regression_data, by_regime=True)

    b1_combined = spec_b1.get('combined', {})
    if 'error' not in b1_combined:
        print(f"  β_Merton = {b1_combined['beta_merton']:.3f} (SE = {b1_combined['se_beta']:.3f})")
        print(f"  Test H0: β=1, p-value = {b1_combined['p_value_h0_beta_eq_1']:.4f}")
        print(f"  R² = {b1_combined['r_squared']:.3f}")
        print(f"  → {b1_combined['interpretation']}")
    else:
        print(f"  ERROR: {b1_combined['error']}")

    print()

    # -------------------------------------------------------------------------
    # Step 3: Run Specification B.2 - Decomposed Components
    # -------------------------------------------------------------------------
    print("Step 3: Running Specification B.2 (decomposed components)...")
    print("  y_i,t = α + β_T·[λ_T · f_DTS] + β_s·[λ_s · f_DTS] + ε")
    print("  Theory prediction: β_T ≈ 1 and β_s ≈ 1")
    print()

    spec_b2 = stageB.run_specification_b2(regression_data, by_regime=True)

    b2_combined = spec_b2.get('combined', {})
    if 'error' not in b2_combined:
        print(f"  β_T (maturity) = {b2_combined['beta_T']:.3f} (SE = {b2_combined['se_beta_T']:.3f})")
        print(f"  β_s (credit) = {b2_combined['beta_s']:.3f} (SE = {b2_combined['se_beta_s']:.3f})")
        print(f"  Joint test p-value = {b2_combined['joint_test_pvalue']:.4f}")
        print(f"  R² = {b2_combined['r_squared']:.3f}")
        print(f"  → {b2_combined['interpretation']}")
    else:
        print(f"  ERROR: {b2_combined['error']}")

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
        print(f"  R² = {b3_combined['r_squared']:.3f}")
        print(f"  Parameters = {b3_combined.get('n_parameters', 'N/A')}")
        print(f"  Lambda model R² = {b3_combined.get('lambda_r_squared', 'N/A'):.3f}")
    else:
        print(f"  ERROR: {b3_combined['error']}")

    print()

    # -------------------------------------------------------------------------
    # Step 5: Model Comparison
    # -------------------------------------------------------------------------
    print("Step 5: Comparing all models...")

    model_comparison = stageB.compare_models(stage_a_results, spec_b1, spec_b2, spec_b3)

    print("\n  Model Comparison:")
    for idx, row in model_comparison.iterrows():
        print(f"    {row['Model']}: R² = {row['R²']}, ΔR² = {row['ΔR² vs Stage A']}")

    print()

    # -------------------------------------------------------------------------
    # Step 6: Theory vs Reality
    # -------------------------------------------------------------------------
    print("Step 6: Creating theory vs reality comparison...")

    theory_vs_reality = stageB.create_theory_vs_reality_table(stage_a_results, bucket_stats)
    theory_assessment = stageB.assess_theory_performance(theory_vs_reality)

    print(f"  Buckets in acceptable range [0.8, 1.2]: {theory_assessment['pct_in_acceptable_range']:.1f}%")
    print(f"  Median ratio (β/λ): {theory_assessment['median_ratio']:.3f}")
    print(f"  Systematic bias: {theory_assessment['systematic_bias']}")
    print(f"  → {theory_assessment['assessment']}")

    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Decision
    # -------------------------------------------------------------------------
    print("Step 7: Generating decision recommendation...")

    decision = stageB.generate_stage_b_decision(spec_b1, model_comparison, theory_assessment)

    print()
    print("="*80)
    print("STAGE B DECISION")
    print("="*80)
    print(decision)
    print("="*80)
    print()

    # -------------------------------------------------------------------------
    # Step 8: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 8: Generating visualizations...")

    merton_calc = MertonLambdaCalculator()
    visualizer = StageBVisualizer(output_dir='./output/figures')
    figures = visualizer.create_all_stageB_figures(
        theory_vs_reality,
        merton_calc,
        spec_b3,
        output_prefix='stageB'
    )

    print(f"  Created {len(figures)} figures:")
    print("    - Figure B.1: Empirical vs theoretical scatter")
    print("    - Figure B.2: Residual analysis (3 panels)")
    print("    - Figure B.3: Lambda surface comparison (contour)")
    print("    - Figure B.3 (alt): Lambda surface comparison (3D)")
    print()

    # -------------------------------------------------------------------------
    # Step 9: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 9: Generating reports...")

    reporter = StageBReporter(output_dir='./output/reports')
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

    print("  Created reports:")
    print("    - Table B.1: Constrained Merton specifications")
    print("    - Table B.2: Model comparison")
    print("    - Table B.3: Theory vs reality")
    print("    - Full theory vs reality CSV")
    print("    - Written summary (3-4 pages)")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("="*80)
    print("STAGE B COMPLETE")
    print("="*80)
    print()
    print("Key Findings:")
    if 'error' not in b1_combined:
        print(f"  • β_Merton = {b1_combined['beta_merton']:.3f} (H0: β=1, p={b1_combined['p_value_h0_beta_eq_1']:.4f})")
        print(f"  • R² = {b1_combined['r_squared']:.3f}")
    if 'error' not in b2_combined:
        print(f"  • β_T = {b2_combined['beta_T']:.3f}, β_s = {b2_combined['beta_s']:.3f}")
    print(f"  • Theory explains {theory_assessment['pct_in_acceptable_range']:.0f}% of buckets well")
    print()

    print("Next steps:")
    if 'PATH 1' in decision or 'PATH 2' in decision:
        print("  → Review output/reports/stageB_summary.txt")
        print("  → Examine theory vs reality table")
        print("  → Proceed to Stage C to test time-variation")
    elif 'PATH 3' in decision:
        print("  → Theory captures structure but incomplete")
        print("  → Stage C should run both theory and unrestricted tracks")
    else:
        print("  → Theory fundamentally fails")
        print("  → Skip Stage C, proceed to Stage D (diagnostics)")

    print()
    print("="*80)


if __name__ == '__main__':
    main()
