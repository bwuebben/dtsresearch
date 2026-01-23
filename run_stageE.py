#!/usr/bin/env python3
"""
Main script to run Stage E analysis.

PREREQUISITE: Stages 0, A, B, C, D completed

This orchestrates the complete Stage E pipeline:
1. Load prerequisite results from Stages A-D
2. Run hierarchical testing (Levels 1-5)
3. Conduct out-of-sample validation
4. Analyze performance by regime
5. Generate production blueprint
6. Create visualizations
7. Generate reports
8. Provide final recommendations

Usage:
    python run_stageE.py [--mock-data]
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
from dts_research.analysis.stageE import StageEAnalysis
from dts_research.analysis.stage0_synthesis import Stage0Synthesis
from dts_research.analysis.stage0_bucket import BucketLevelAnalysis
from dts_research.analysis.stage0_within_issuer import WithinIssuerAnalysis
from dts_research.analysis.stage0_sector import SectorInteractionAnalysis
from dts_research.models.merton import MertonLambdaCalculator
from dts_research.visualization.stageE_plots import StageEVisualizer
from dts_research.utils.reportingE import StageEReporter


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

    # Compute Merton-scaled factor
    df['f_merton'] = df['lambda_Merton'] * df['f_DTS']

    # Add spread regime
    df['spread_regime'] = np.where(df['oas'] < 300, 'IG', 'HY')

    # Add week identifier for clustering
    df['week'] = df['date'].dt.isocalendar().week.astype(str) + '_' + df['date'].dt.year.astype(str)

    # Add mock VIX and OAS index for time-varying tests
    np.random.seed(42)
    base_vix = 20 + 5 * np.random.randn(len(df))
    crisis_mask = (df['date'] >= '2020-03-01') & (df['date'] <= '2020-06-01')
    base_vix[crisis_mask] = base_vix[crisis_mask] + 30
    df['vix'] = np.clip(base_vix, 10, 80)

    df['oas_index'] = 200 + 50 * np.random.randn(len(df))
    df['oas_index'] = np.clip(df['oas_index'], 50, 500)

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


def load_prerequisite_results() -> tuple:
    """
    Load results from Stages A, B, C, D.

    In production, these would be saved/pickled from previous runs.
    For this implementation, we'll simulate with mock results.
    """
    print("Loading prerequisite results from Stages A-D...")
    print()

    # Mock Stage A results
    stage_a_results = {
        'f_test_all_buckets': {
            'f_statistic': 4.52,
            'p_value': 0.0001
        }
    }

    # Mock Stage B results
    stage_b_results = {
        'spec_a1_buckets': {
            'r_squared': 0.75
        },
        'spec_b1': {
            'beta_Merton': 0.98,
            'p_value_vs_1': 0.65,
            'r_squared': 0.73
        },
        'spec_b3': {
            'r_squared': 0.78,
            'n_params': 10
        }
    }

    # Mock Stage C results
    stage_c_results = {
        'stability_test': {
            'chow_p_value': 0.15
        },
        'macro_drivers': {
            'coef_vix': 0.012,
            'coef_oas': -0.008,
            'vix_effect_pct': 25
        }
    }

    # Mock Stage D results
    stage_d_results = {
        'tail_tests': {
            'amplification_left': 1.15
        }
    }

    print("  Loaded prerequisite results")
    print()

    return stage_a_results, stage_b_results, stage_c_results, stage_d_results


def main():
    """
    Run complete Stage E analysis pipeline.
    """
    parser = argparse.ArgumentParser(description='Run Stage E Analysis')
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
    print("STAGE E: PRODUCTION SPECIFICATION SELECTION")
    print("=" * 80)
    print()
    print("Objective: Select parsimonious production model via hierarchical testing")
    print()
    print("Key principle: Stop at the simplest adequate model. Don't over-engineer.")
    print()
    print("Philosophy: Theory provides strong prior. Only deviate when data")
    print("            strongly reject it. Burden of proof on complex model.")
    print()

    # Create output directories
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'stageE_figures'
    reports_dir = output_dir / 'stageE_reports'
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load Data and Prerequisites
    # -------------------------------------------------------------------------
    print("Step 1: Loading data and prerequisites...")
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

    # Load prerequisite results
    stage_a_results, stage_b_results, stage_c_results, stage_d_results = load_prerequisite_results()

    # -------------------------------------------------------------------------
    # Step 2: Run Hierarchical Testing (Levels 1-5)
    # -------------------------------------------------------------------------
    print("Step 2: Running hierarchical testing framework...")
    print()

    stageE = StageEAnalysis(stage0_results=stage0_synthesis)

    hierarchical_results = stageE.hierarchical_testing(
        regression_data,
        stage_a_results,
        stage_b_results,
        stage_c_results,
        stage_d_results
    )

    recommended_level = hierarchical_results.get('recommended_level', 2)
    recommended_spec = hierarchical_results.get('recommended_spec', 'Pure Merton')

    print()
    print("=" * 80)
    print("HIERARCHICAL TEST RESULTS")
    print("=" * 80)
    print()

    # Display results for each level tested
    for level_num in range(1, 6):
        level_key = f'level{level_num}'
        if level_key in hierarchical_results:
            level_result = hierarchical_results[level_key]
            print(f"Level {level_num}: {level_result.get('test', 'N/A')}")
            print(f"  Decision: {level_result.get('decision', 'N/A')}")
            print(f"  Reasoning: {level_result.get('reasoning', 'N/A')}")
            print()

    print("=" * 80)
    print(f"RECOMMENDED SPECIFICATION: {recommended_spec} (Level {recommended_level})")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Step 3: Out-of-Sample Validation
    # -------------------------------------------------------------------------
    print("Step 3: Running out-of-sample validation...")
    print()

    oos_results = stageE.out_of_sample_validation(
        regression_data,
        recommended_level,
        hierarchical_results,
        train_window_years=3,
        test_window_years=1
    )

    print()
    print("Out-of-Sample Performance Summary:")
    print()

    oos_summary = oos_results.get('oos_summary', {})
    for spec_name, metrics in oos_summary.items():
        print(f"  {spec_name}:")
        print(f"    Avg OOS R²: {metrics.get('avg_r2_oos', np.nan):.3f}")
        print(f"    Avg OOS RMSE: {metrics.get('avg_rmse_oos', np.nan):.3f}")
        print()

    # -------------------------------------------------------------------------
    # Step 4: Performance by Regime
    # -------------------------------------------------------------------------
    print("Step 4: Analyzing performance by regime...")
    print()

    regime_results = stageE.performance_by_regime(
        regression_data,
        oos_results,
        vix_thresholds=(20, 30)
    )

    print("Performance by Regime:")
    print()

    for spec_name, regime_data in regime_results.items():
        if spec_name == recommended_spec:
            print(f"{spec_name} (RECOMMENDED):")
            for regime_name, metrics in regime_data.items():
                if metrics.get('n_windows', 0) > 0:
                    print(f"  {regime_name}:")
                    print(f"    R² = {metrics.get('avg_r2_oos', np.nan):.3f}, RMSE = {metrics.get('avg_rmse_oos', np.nan):.3f}")
            print()

    # -------------------------------------------------------------------------
    # Step 5: Generate Production Blueprint
    # -------------------------------------------------------------------------
    print("Step 5: Generating production blueprint...")
    print()

    production_blueprint = stageE.generate_production_blueprint(
        recommended_level,
        hierarchical_results,
        oos_results
    )

    print("Production Specification:")
    print(f"  Name: {production_blueprint.get('specification', 'N/A')}")
    print(f"  Level: {production_blueprint.get('level', 'N/A')}")
    print(f"  Parameters: {production_blueprint.get('parameters', {}).get('n_params', 'N/A')}")
    print(f"  Complexity: {production_blueprint.get('complexity', 'N/A')}")
    print(f"  Recalibration: {production_blueprint.get('recalibration_frequency', 'N/A')}")
    print()

    perf = production_blueprint.get('performance', {})
    print(f"  Expected OOS R²: {perf.get('avg_r2_oos', np.nan):.3f}")
    print(f"  Expected OOS RMSE: {perf.get('avg_rmse_oos', np.nan):.3f}")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Generate Visualizations
    # -------------------------------------------------------------------------
    print("Step 6: Generating visualizations...")
    print()

    visualizer = StageEVisualizer(output_dir=str(figures_dir))

    figures = visualizer.create_all_stageE_figures(
        regression_data,
        oos_results,
        hierarchical_results,
        recommended_spec,
        output_prefix='stageE'
    )

    print(f"  Created {len(figures)} figures in {figures_dir}")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Generate Reports
    # -------------------------------------------------------------------------
    print("Step 7: Generating reports...")
    print()

    reporter = StageEReporter(output_dir=str(reports_dir))

    reporter.save_all_reports(
        hierarchical_results,
        oos_results,
        regime_results,
        production_blueprint,
        prefix='stageE'
    )

    print(f"  Reports saved to {reports_dir}")
    print()

    # -------------------------------------------------------------------------
    # Step 8: Final Recommendations
    # -------------------------------------------------------------------------
    print("Step 8: Final recommendations...")
    print()

    print("=" * 80)
    print("STAGE E FINAL RECOMMENDATIONS")
    print("=" * 80)
    print()

    print(f"RECOMMENDED PRODUCTION SPECIFICATION: {recommended_spec}")
    print("=" * 80)
    print()

    # Implementation details
    impl_formula = production_blueprint.get('implementation', 'N/A')
    print("Implementation Formula:")
    print(f"  {impl_formula}")
    print()

    # Expected performance
    print("Expected Performance:")
    print(f"  - Out-of-Sample R²: {perf.get('avg_r2_oos', np.nan):.3f}")
    print(f"  - Out-of-Sample RMSE: {perf.get('avg_rmse_oos', np.nan):.3f}")
    print()

    # Complexity and maintenance
    print("Implementation Details:")
    print(f"  - Complexity: {production_blueprint.get('complexity', 'N/A')}")
    print(f"  - Parameters: {production_blueprint.get('parameters', {}).get('n_params', 'N/A')}")
    print(f"  - Recalibration: {production_blueprint.get('recalibration_frequency', 'N/A')}")
    print()

    # Key advantages
    print("Key Advantages:")
    for adv in production_blueprint.get('advantages', []):
        print(f"  - {adv}")
    print()

    # Next steps
    print("Next Steps:")
    print("  1. Review implementation blueprint")
    print("  2. Examine Figures E.1-E.3 for visual confirmation")
    print("  3. Validate on hold-out sample before production deployment")
    print("  4. Set up monitoring infrastructure")
    print("  5. Establish recalibration schedule")
    print()

    print("=" * 80)
    print("STAGE E COMPLETE")
    print("=" * 80)
    print()

    print("Summary:")
    print(f"  - Hierarchical testing completed through Level {recommended_level}")
    print(f"  - Recommended: {recommended_spec}")
    print(f"  - Expected OOS R²: {perf.get('avg_r2_oos', np.nan):.3f}")
    print(f"  - Implementation complexity: {production_blueprint.get('complexity', 'N/A')}")
    print()

    print("The research program is now COMPLETE!")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
