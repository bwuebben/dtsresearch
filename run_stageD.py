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
from dts_research.analysis.stageD import StageDAnalysis
from dts_research.visualization.stageD_plots import StageDVisualizer
from dts_research.utils.reportingD import StageDReporter
from dts_research.data.sector_classification import SectorClassifier
from dts_research.data.issuer_identification import add_issuer_identification


def main():
    """
    Run complete Stage D analysis pipeline.
    """
    print("="*80)
    print("STAGE D: ROBUSTNESS AND EXTENSIONS")
    print("="*80)
    print()
    print("Objective: Test robustness across:")
    print("  1. Tail events (quantile regression)")
    print("  2. Shock types (systematic vs idiosyncratic)")
    print("  3. Spread components (default vs liquidity)")
    print()
    print("Key Framing: These are SECONDARY tests.")
    print("  - If Stages A-C validated Merton → Confirm not just mean effect")
    print("  - If Stages A-C showed failure → Diagnose WHY")
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
    print("Step 1: Loading data...")
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
    print("  Preparing data...")
    classifier = BucketClassifier()
    bond_data = classifier.classify_bonds(bond_data)

    # Add sector classification and issuer identification (required for Stage 0 evolved specs)
    print("  Adding sector classification...")
    sector_classifier = SectorClassifier()
    bond_data = sector_classifier.classify_sector(bond_data, bclass_column='sector_classification')
    bond_data = sector_classifier.add_sector_dummies(bond_data)

    print("  Adding issuer identification...")
    bond_data = add_issuer_identification(
        bond_data,
        parent_id_col='ultimate_parent_id',
        seniority_col='seniority'
    )

    stage0 = Stage0Analysis()
    regression_data = stage0.prepare_regression_data(bond_data, index_data)

    # Add spread_regime column (IG/HY based on OAS)
    regression_data['spread_regime'] = regression_data['oas'].apply(
        lambda x: 'IG' if x < 300 else 'HY'
    )

    print(f"  Prepared {len(regression_data):,} observations for analysis")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Run D.1 - Tail Behavior (Quantile Regression)
    # -------------------------------------------------------------------------
    print("Step 2: Running D.1 - Tail Behavior (Quantile Regression)...")
    print("  Testing if Merton holds across distribution of spread changes")
    print()

    stageD = StageDAnalysis()
    quantile_results = stageD.quantile_regression_analysis(
        regression_data,
        by_regime=True
    )

    tail_tests = quantile_results['tail_tests']
    print(f"  Pattern: {tail_tests['pattern']}")
    print(f"  Left tail amplification: {tail_tests['amplification_left']:.2f}x")
    print(f"  Right tail amplification: {tail_tests['amplification_right']:.2f}x")
    print()

    if tail_tests['amplification_left'] > 1.3:
        print("  ⚠ LEFT TAIL AMPLIFICATION DETECTED")
        print(f"    → Tail risk {(tail_tests['amplification_left']-1)*100:.0f}% larger than mean")
    else:
        print("  ✓ No significant tail amplification")

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

    shock_betas = shock_results['shock_betas_combined']
    variance_decomp = shock_results['variance_decomp']

    print("  Variance Decomposition:")
    for _, row in variance_decomp.iterrows():
        print(f"    - {row['Component']}: {row['Pct_of_Total']:.1f}%")
    print()

    print("  Shock-Specific Elasticities:")
    print(f"    - β^(G) (Global) = {shock_betas['beta_global']:.3f}")
    print(f"    - β^(S) (Sector) = {shock_betas['beta_sector']:.3f}")
    print(f"    - β^(I) (Issuer) = {shock_betas['beta_issuer']:.3f}")
    print()

    all_near_one = all(
        0.9 <= shock_betas[f'beta_{k}'] <= 1.1
        for k in ['global', 'sector', 'issuer']
    )

    if all_near_one:
        print("  ✓ All shock types respect Merton elasticities")
    else:
        if shock_betas['beta_sector'] > 1.2:
            print(f"  ⚠ Sector shocks amplified ({shock_betas['beta_sector']:.2f}x)")
        if shock_betas['beta_issuer'] > 1.2:
            print(f"  ⚠ Issuer-specific shocks amplified ({shock_betas['beta_issuer']:.2f}x)")

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

    liq_model = liquidity_results['liquidity_model']
    comparison = liquidity_results['comparison']

    print(f"  Liquidity Model R²: {liq_model['r_squared']:.3f}")
    print()

    print("  Merton Fit Comparison:")
    print(f"    - Total OAS: β = {comparison['beta_total']:.3f}, R² = {comparison['r2_total']:.3f}")
    print(f"    - Default component: β = {comparison['beta_def']:.3f}, R² = {comparison['r2_def']:.3f}")
    print(f"    - Improvement: ΔR² = {comparison['delta_r2']:.3f} ({comparison['improvement_pct']:.1f}%)")
    print()

    if comparison['delta_r2'] > 0.05:
        print("  ✓ Liquidity adjustment materially improves fit")
        print("    → Decompose OAS for HY and illiquid bonds")
    elif comparison['delta_r2'] < 0.02:
        print("  → Liquidity adjustment has minimal impact")
        print("    → Use total OAS (simpler)")
    else:
        print("  → Marginal benefit from liquidity adjustment")

    print()

    # -------------------------------------------------------------------------
    # Step 5: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 5: Generating visualizations...")

    visualizer = StageDVisualizer(output_dir='./output/figures')

    figures = visualizer.create_all_stageD_figures(
        quantile_results['results_combined'],
        shock_results['shock_betas_combined'],
        liquidity_results.get('by_liquidity_quartile', pd.DataFrame()),
        shock_results['variance_decomp'],
        quantile_results.get('results_ig'),
        quantile_results.get('results_hy'),
        shock_results.get('shock_betas_ig'),
        shock_results.get('shock_betas_hy'),
        output_prefix='stageD'
    )

    print(f"  Created {len(figures)} figures:")
    print("    - Figure D.1: Quantile regression (beta across distribution)")
    print("    - Figure D.2: Shock-specific elasticities")
    print("    - Figure D.3: Liquidity adjustment improvement")
    print("    - Figure D.4: Variance decomposition (supplementary)")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 6: Generating reports...")

    reporter = StageDReporter(output_dir='./output/reports')

    reporter.save_all_reports(
        quantile_results,
        shock_results,
        liquidity_results,
        prefix='stageD'
    )

    print("  Created reports:")
    print("    - Table D.1: Quantile-specific betas")
    print("    - Table D.2: Tail amplification factors")
    print("    - Table D.3: Variance decomposition")
    print("    - Table D.4: Shock-specific elasticities")
    print("    - Table D.5: Liquidity model estimates")
    print("    - Table D.6: Merton fit comparison (total vs default)")
    print("    - Table D.7: Improvement by liquidity quartile")
    print("    - Written summary (3-4 pages)")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Recommendations
    # -------------------------------------------------------------------------
    print("Step 7: Generating production recommendations...")
    print()

    print("="*80)
    print("STAGE D RECOMMENDATIONS")
    print("="*80)
    print()

    # Tail adjustments
    print("TAIL ADJUSTMENTS:")
    if tail_tests['amplification_left'] > 1.3:
        print(f"  ⚠ Use tail-specific lambda for VaR/ES:")
        print(f"    λ^VaR = {tail_tests['amplification_left']:.2f} × λ^Merton")
        print(f"    (Tail risk {(tail_tests['amplification_left']-1)*100:.0f}% larger than mean)")
    else:
        print("  ✓ Standard Merton λ adequate for VaR/ES")
        print("    No tail-specific adjustments needed")
    print()

    # Shock-type adjustments
    print("SHOCK-TYPE CONSIDERATIONS:")
    if all_near_one:
        print("  ✓ Use uniform λ across shock types")
        print("    All shocks respect Merton elasticities")
    else:
        if shock_betas['beta_sector'] > 1.2:
            print(f"  ⚠ Sector shocks amplified ({shock_betas['beta_sector']:.2f}x)")
            print("    → Consider sector-specific risk factors")

        if shock_betas['beta_issuer'] > 1.2:
            print(f"  ⚠ Issuer-specific shocks amplified ({shock_betas['beta_issuer']:.2f}x)")
            print("    → Idiosyncratic risk larger than Merton predicts")
    print()

    # Liquidity adjustments
    print("LIQUIDITY DECOMPOSITION:")
    if comparison['delta_r2'] > 0.05:
        print("  ✓ Decompose OAS for HY and illiquid bonds")
        print("    → Use λ^def from Merton for default component")
        print("    → Add separate λ^liq empirically estimated")
    elif comparison['delta_r2'] < 0.02:
        print("  → Use total OAS (simpler)")
        print("    Liquidity decomposition not worth complexity")
    else:
        print("  → Consider for precision-critical applications")
        print("    Marginal benefit, may not justify operational cost")

    print()
    print("="*80)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("="*80)
    print("STAGE D COMPLETE")
    print("="*80)
    print()
    print("Key Findings:")
    print(f"  • Tail amplification: {tail_tests['amplification_left']:.2f}x (left), {tail_tests['amplification_right']:.2f}x (right)")
    print(f"  • Global shock β: {shock_betas['beta_global']:.3f}")
    print(f"  • Sector shock β: {shock_betas['beta_sector']:.3f}")
    print(f"  • Issuer shock β: {shock_betas['beta_issuer']:.3f}")
    print(f"  • Liquidity adjustment ΔR²: {comparison['delta_r2']:.3f}")
    print()

    print("Next steps:")
    print("  → Review output/reports/stageD_summary.txt for detailed analysis")
    print("  → Examine Figures D.1-D.3 for visual confirmation")
    print("  → Incorporate findings into Stage E production specification")
    print("  → Document any tail/shock/liquidity adjustments needed")

    print()
    print("="*80)


if __name__ == '__main__':
    main()
