"""
Stage 0 Runner Script

Orchestrates the complete Stage 0 analysis:
1. Load and prepare data
2. Run bucket-level analysis (IG & HY)
3. Run within-issuer analysis (IG & HY)
4. Run sector interaction analysis (IG & HY)
5. Synthesize results and determine decision path
6. Generate all figures (10)
7. Generate all tables (17)
8. Create summary report

Usage:
    python run_stage0.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--output-dir path]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dts_research.data.loader import BondDataLoader
from dts_research.analysis.stage0_bucket import run_bucket_analysis_both_universes
from dts_research.analysis.stage0_within_issuer import run_within_issuer_analysis_both_universes
from dts_research.analysis.stage0_sector import run_sector_analysis_both_universes
from dts_research.analysis.stage0_synthesis import run_stage0_synthesis
from dts_research.visualization.stage0_plots import create_all_stage0_plots
from dts_research.utils.reporting0 import generate_stage0_report


def main():
    """Run complete Stage 0 analysis."""
    parser = argparse.ArgumentParser(description='Run Stage 0 Analysis')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--mock-data', action='store_true',
                       help='Use mock data (default if no connection)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 0: EVOLVED DTS ANALYSIS")
    print("=" * 80)
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print("[Step 1/8] Loading bond data...")
    loader = BondDataLoader()
    bond_data = loader.load_bond_data(args.start_date, args.end_date)
    print(f"  Loaded {len(bond_data):,} observations")
    print(f"  Date range: {bond_data['date'].min()} to {bond_data['date'].max()}")
    print(f"  Unique bonds: {bond_data['bond_id'].nunique():,}")
    print()

    # Add sector classification
    print("  Classifying sectors...")
    from dts_research.data.sector_classification import SectorClassifier
    from dts_research.data.issuer_identification import add_issuer_identification

    classifier = SectorClassifier()
    bond_data = classifier.classify_sector(bond_data, bclass_column='sector_classification')
    bond_data = classifier.add_sector_dummies(bond_data)

    # Add issuer identification (needed for all analyses)
    bond_data = add_issuer_identification(
        bond_data,
        parent_id_col='ultimate_parent_id',
        seniority_col='seniority'
    )

    print(f"  Sector distribution:")
    for sector, count in bond_data['sector'].value_counts().items():
        pct = 100.0 * count / len(bond_data)
        print(f"    {sector}: {count:,} ({pct:.1f}%)")
    print()

    # Step 2: Bucket-level analysis
    print("[Step 2/8] Running bucket-level analysis...")
    bucket_results = run_bucket_analysis_both_universes(bond_data)

    bucket_ig = bucket_results['IG']
    bucket_hy = bucket_results['HY']

    # Use correct keys from diagnostics
    ig_n_buckets = bucket_ig['diagnostics'].get('n_buckets_with_regression', 0)
    hy_n_buckets = bucket_hy['diagnostics'].get('n_buckets_with_regression', 0)

    # Get summary stats for lambda estimates
    ig_summary = bucket_ig.get('summary_statistics', {})
    hy_summary = bucket_hy.get('summary_statistics', {})

    print(f"  IG: {ig_n_buckets} buckets with regressions")
    print(f"      Median β/λ ratio = {ig_summary.get('median_ratio', np.nan):.3f}")
    print(f"  HY: {hy_n_buckets} buckets with regressions")
    print(f"      Median β/λ ratio = {hy_summary.get('median_ratio', np.nan):.3f}")
    print()

    # Step 3: Within-issuer analysis
    print("[Step 3/8] Running within-issuer analysis...")
    within_results = run_within_issuer_analysis_both_universes(bond_data, verbose=args.verbose)

    within_ig = within_results['IG']
    within_hy = within_results['HY']

    # Use correct keys - pooled_beta and pooled_beta_se
    ig_pooled = within_ig['pooled_estimate']
    hy_pooled = within_hy['pooled_estimate']

    print(f"  IG: {within_ig['diagnostics'].get('n_issuer_weeks_with_estimate', 0)} issuer-weeks analyzed")
    print(f"      Pooled β = {ig_pooled.get('pooled_beta', np.nan):.6f} "
          f"± {ig_pooled.get('pooled_beta_se', np.nan):.6f}")
    print(f"  HY: {within_hy['diagnostics'].get('n_issuer_weeks_with_estimate', 0)} issuer-weeks analyzed")
    print(f"      Pooled β = {hy_pooled.get('pooled_beta', np.nan):.6f} "
          f"± {hy_pooled.get('pooled_beta_se', np.nan):.6f}")
    print()

    # Step 4: Sector interaction analysis
    print("[Step 4/8] Running sector interaction analysis...")
    sector_results = run_sector_analysis_both_universes(bond_data, cluster_by='week')

    sector_ig = sector_results['IG']
    sector_hy = sector_results['HY']

    # Use correct keys - beta_0 instead of lambda
    ig_base = sector_ig['base_regression']
    hy_base = sector_hy['base_regression']

    print(f"  IG: Base β = {ig_base.get('beta_0', np.nan):.6f}")
    print(f"      Sectors significant? {sector_ig['joint_test'].get('reject_null', False)}")
    print(f"  HY: Base β = {hy_base.get('beta_0', np.nan):.6f}")
    print(f"      Sectors significant? {sector_hy['joint_test'].get('reject_null', False)}")
    print()

    # Step 5: Synthesis and decision framework
    print("[Step 5/8] Synthesizing results and determining decision paths...")
    synthesis_results = run_stage0_synthesis(
        bucket_ig, bucket_hy,
        within_ig, within_hy,
        sector_ig, sector_hy
    )

    synthesis_ig = synthesis_results['IG']
    synthesis_hy = synthesis_results['HY']
    comparison = synthesis_results['comparison']

    print(f"  IG Decision: Path {synthesis_ig['decision_path']} - {synthesis_ig['path_name']}")
    print(f"     {synthesis_ig['rationale']}")
    print(f"  HY Decision: Path {synthesis_hy['decision_path']} - {synthesis_hy['path_name']}")
    print(f"     {synthesis_hy['rationale']}")
    print(f"  Unified Approach: {comparison['unified_approach']}")
    print()

    # Step 6: Generate figures
    print("[Step 6/8] Generating visualizations (10 figures)...")
    figures_dir = output_dir / 'stage0_figures'
    create_all_stage0_plots(
        bucket_ig, bucket_hy,
        within_ig, within_hy,
        sector_ig, sector_hy,
        synthesis_ig, synthesis_hy,
        comparison,
        output_dir=str(figures_dir)
    )
    print(f"  Figures saved to: {figures_dir}")
    print()

    # Step 7: Generate tables
    print("[Step 7/8] Generating tables (17 tables)...")
    tables_dir = output_dir / 'stage0_tables'
    generate_stage0_report(
        bucket_ig, bucket_hy,
        within_ig, within_hy,
        sector_ig, sector_hy,
        synthesis_ig, synthesis_hy,
        comparison,
        output_dir=str(tables_dir)
    )
    print(f"  Tables saved to: {tables_dir}")
    print()

    # Step 8: Create summary report
    print("[Step 8/8] Creating summary report...")
    create_summary_report(
        synthesis_ig, synthesis_hy, comparison,
        output_dir / 'STAGE_0_SUMMARY.txt'
    )
    print(f"  Summary saved to: {output_dir / 'STAGE_0_SUMMARY.txt'}")
    print()

    print("=" * 80)
    print("STAGE 0 ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Tables: {tables_dir}")
    print(f"  - Summary: {output_dir / 'STAGE_0_SUMMARY.txt'}")
    print()
    print("Next steps:")
    print(f"  IG: {synthesis_ig['recommendations']['stage_A']}")
    print(f"  HY: {synthesis_hy['recommendations']['stage_A']}")
    print()


def create_summary_report(
    synthesis_ig: dict,
    synthesis_hy: dict,
    comparison: dict,
    output_path: Path
):
    """Create executive summary report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 0: EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # IG Summary
        f.write("-" * 80 + "\n")
        f.write("INVESTMENT GRADE (IG) UNIVERSE\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Decision Path: Path {synthesis_ig['decision_path']} - {synthesis_ig['path_name']}\n\n")
        f.write(f"Rationale:\n{synthesis_ig['rationale']}\n\n")

        f.write("Key Statistics:\n")
        stats_ig = synthesis_ig['key_statistics']
        f.write(f"  Bucket median β:  {stats_ig.get('bucket_median_beta', np.nan):.6f}\n")
        f.write(f"  Within-Issuer β:  {stats_ig.get('within_beta', np.nan):.6f}\n")
        f.write(f"  Sector Base β:    {stats_ig.get('base_beta', np.nan):.6f}\n")
        f.write(f"  Monotonic?        {'Yes' if stats_ig.get('monotonic', False) else 'No'}\n")
        f.write(f"  Sectors differ?   {'Yes' if stats_ig.get('sectors_differ', False) else 'No'}\n\n")

        f.write("Recommendations for Subsequent Stages:\n")
        for stage, rec in synthesis_ig['recommendations'].items():
            f.write(f"  {stage.upper()}: {rec}\n")
        f.write("\n")

        # HY Summary
        f.write("-" * 80 + "\n")
        f.write("HIGH YIELD (HY) UNIVERSE\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Decision Path: Path {synthesis_hy['decision_path']} - {synthesis_hy['path_name']}\n\n")
        f.write(f"Rationale:\n{synthesis_hy['rationale']}\n\n")

        f.write("Key Statistics:\n")
        stats_hy = synthesis_hy['key_statistics']
        f.write(f"  Bucket median β:  {stats_hy.get('bucket_median_beta', np.nan):.6f}\n")
        f.write(f"  Within-Issuer β:  {stats_hy.get('within_beta', np.nan):.6f}\n")
        f.write(f"  Sector Base β:    {stats_hy.get('base_beta', np.nan):.6f}\n")
        f.write(f"  Monotonic?        {'Yes' if stats_hy.get('monotonic', False) else 'No'}\n")
        f.write(f"  Sectors differ?   {'Yes' if stats_hy.get('sectors_differ', False) else 'No'}\n\n")

        f.write("Recommendations for Subsequent Stages:\n")
        for stage, rec in synthesis_hy['recommendations'].items():
            f.write(f"  {stage.upper()}: {rec}\n")
        f.write("\n")

        # Comparison
        f.write("-" * 80 + "\n")
        f.write("CROSS-UNIVERSE COMPARISON\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Same path? {comparison['same_path']}\n")
        f.write(f"Interpretation: {comparison['interpretation']}\n\n")
        f.write(f"UNIFIED APPROACH:\n{comparison['unified_approach']}\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")


if __name__ == '__main__':
    main()
