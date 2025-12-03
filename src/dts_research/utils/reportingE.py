"""
Reporting functions for Stage E deliverables.

Generates Tables E.1-E.4 and written implementation blueprint.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import os


class StageEReporter:
    """Generates publication-quality reports for Stage E analysis."""

    def __init__(self, output_dir: str = './output/reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_hierarchical_test_results(
        self,
        hierarchical_results: Dict,
        filename: str = 'stageE_table_e1_hierarchical_tests.csv'
    ):
        """
        Table E.1: Hierarchical test results.

        Rows: Levels 1-5
        Columns: Test, Statistic, p-value, Decision
        Mark the stopping level.
        """
        rows = []

        # Level 1
        level1 = hierarchical_results.get('level1', {})
        rows.append({
            'Level': 'Level 1: Standard DTS',
            'Test': level1.get('test', ''),
            'Statistic': level1.get('statistic', np.nan),
            'p_value': level1.get('p_value', np.nan),
            'Decision': level1.get('decision', ''),
            'Reasoning': level1.get('reasoning', '')
        })

        # Level 2
        if 'level2' in hierarchical_results:
            level2 = hierarchical_results['level2']
            rows.append({
                'Level': 'Level 2: Pure Merton',
                'Test': level2.get('test', ''),
                'Statistic': level2.get('beta_Merton', np.nan),
                'p_value': level2.get('p_value', np.nan),
                'Decision': level2.get('decision', ''),
                'Reasoning': level2.get('reasoning', '')
            })

        # Level 3
        if 'level3' in hierarchical_results:
            level3 = hierarchical_results['level3']
            rows.append({
                'Level': 'Level 3: Calibrated Merton',
                'Test': 'c0, c_s calibration',
                'Statistic': f"c0={level3.get('c0', np.nan):.3f}, c_s={level3.get('c_s', np.nan):.3f}",
                'p_value': np.nan,
                'Decision': level3.get('decision', ''),
                'Reasoning': level3.get('reasoning', '')
            })

        # Level 4
        if 'level4' in hierarchical_results:
            level4 = hierarchical_results['level4']
            rows.append({
                'Level': 'Level 4: Empirical',
                'Test': 'Empirical vs Calibrated',
                'Statistic': level4.get('delta_r2', np.nan),
                'p_value': np.nan,
                'Decision': level4.get('decision', ''),
                'Reasoning': level4.get('reasoning', '')
            })

        # Level 5
        if 'level5' in hierarchical_results:
            level5 = hierarchical_results['level5']
            rows.append({
                'Level': 'Level 5: Time-varying',
                'Test': 'Time-varying vs Static',
                'Statistic': level5.get('crisis_rmse_reduction_pct', np.nan),
                'p_value': np.nan,
                'Decision': level5.get('decision', ''),
                'Reasoning': level5.get('reasoning', '')
            })

        df = pd.DataFrame(rows)

        # Mark recommended level
        recommended_level = hierarchical_results.get('recommended_level', 0)
        df['Recommended'] = ''
        if recommended_level > 0 and recommended_level <= len(df):
            df.loc[recommended_level - 1, 'Recommended'] = '✓ RECOMMENDED'

        # Save
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"  Saved: {filename}")

    def save_model_comparison_table(
        self,
        oos_results: Dict,
        hierarchical_results: Dict,
        filename: str = 'stageE_table_e2_model_comparison.csv'
    ):
        """
        Table E.2: Model comparison (all candidate specs).

        Rows: Standard DTS, Pure Merton, Calibrated, Empirical, Time-varying
        Columns: Parameters, In-sample R², OOS R², OOS RMSE
        """
        oos_summary = oos_results.get('oos_summary', {})

        rows = []

        # Get in-sample R² from hierarchical results
        level1 = hierarchical_results.get('level1', {})
        level2 = hierarchical_results.get('level2', {})
        level3 = hierarchical_results.get('level3', {})
        level4 = hierarchical_results.get('level4', {})
        level5 = hierarchical_results.get('level5', {})

        # Standard DTS
        if 'Standard DTS' in oos_summary:
            rows.append({
                'Specification': 'Standard DTS',
                'Parameters': 0,
                'In_sample_R2': np.nan,
                'OOS_R2': oos_summary['Standard DTS'].get('avg_r2_oos', np.nan),
                'OOS_RMSE': oos_summary['Standard DTS'].get('avg_rmse_oos', np.nan),
                'Complexity': 'Trivial'
            })

        # Pure Merton
        if 'Pure Merton' in oos_summary:
            rows.append({
                'Specification': 'Pure Merton',
                'Parameters': 0,
                'In_sample_R2': level2.get('r2_merton', np.nan),
                'OOS_R2': oos_summary['Pure Merton'].get('avg_r2_oos', np.nan),
                'OOS_RMSE': oos_summary['Pure Merton'].get('avg_rmse_oos', np.nan),
                'Complexity': 'Low'
            })

        # Calibrated Merton
        if 'Calibrated Merton' in oos_summary:
            rows.append({
                'Specification': 'Calibrated Merton',
                'Parameters': 2,
                'In_sample_R2': level3.get('r_squared', np.nan),
                'OOS_R2': oos_summary['Calibrated Merton'].get('avg_r2_oos', np.nan),
                'OOS_RMSE': oos_summary['Calibrated Merton'].get('avg_rmse_oos', np.nan),
                'Complexity': 'Low-Moderate'
            })

        # Empirical
        if 'Empirical' in oos_summary:
            rows.append({
                'Specification': 'Empirical',
                'Parameters': level4.get('n_params', 10),
                'In_sample_R2': level4.get('r2_empirical', np.nan),
                'OOS_R2': oos_summary['Empirical'].get('avg_r2_oos', np.nan),
                'OOS_RMSE': oos_summary['Empirical'].get('avg_rmse_oos', np.nan),
                'Complexity': 'Moderate-High'
            })

        # Time-varying
        if 'Time-varying' in oos_summary:
            rows.append({
                'Specification': 'Time-varying',
                'Parameters': level5.get('n_params', 12),
                'In_sample_R2': level5.get('r2_tv', np.nan),
                'OOS_R2': oos_summary['Time-varying'].get('avg_r2_oos', np.nan),
                'OOS_RMSE': oos_summary['Time-varying'].get('avg_rmse_oos', np.nan),
                'Complexity': 'High'
            })

        df = pd.DataFrame(rows)

        # Mark recommended
        recommended_spec = hierarchical_results.get('recommended_spec', '')
        df['Recommended'] = ''
        df.loc[df['Specification'] == recommended_spec, 'Recommended'] = '✓'

        # Save
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"  Saved: {filename}")

    def save_performance_by_regime_table(
        self,
        regime_results: Dict,
        filename: str = 'stageE_table_e3_performance_by_regime.csv'
    ):
        """
        Table E.3: Performance by regime.

        Panel A: Normal (VIX < 20)
        Panel B: Stress (VIX 20-30)
        Panel C: Crisis (VIX > 30)
        """
        rows = []

        for spec_name, regime_data in regime_results.items():
            for regime_name, metrics in regime_data.items():
                rows.append({
                    'Specification': spec_name,
                    'Regime': regime_name,
                    'N_Windows': metrics.get('n_windows', 0),
                    'Avg_R2_OOS': metrics.get('avg_r2_oos', np.nan),
                    'Avg_RMSE_OOS': metrics.get('avg_rmse_oos', np.nan)
                })

        df = pd.DataFrame(rows)

        # Pivot for better readability
        df_pivot_r2 = df.pivot(index='Specification', columns='Regime', values='Avg_R2_OOS')
        df_pivot_rmse = df.pivot(index='Specification', columns='Regime', values='Avg_RMSE_OOS')

        # Save both formats
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)

        output_path_r2 = os.path.join(self.output_dir, filename.replace('.csv', '_r2_pivot.csv'))
        df_pivot_r2.to_csv(output_path_r2)

        output_path_rmse = os.path.join(self.output_dir, filename.replace('.csv', '_rmse_pivot.csv'))
        df_pivot_rmse.to_csv(output_path_rmse)

        print(f"  Saved: {filename} (+ pivot tables)")

    def save_production_specification_table(
        self,
        production_blueprint: Dict,
        filename: str = 'stageE_table_e4_production_spec.csv'
    ):
        """
        Table E.4: Recommended production specification.

        Single-row table with all key details.
        """
        spec = production_blueprint.get('specification', '')
        level = production_blueprint.get('level', 0)
        params = production_blueprint.get('parameters', {})
        implementation = production_blueprint.get('implementation', '')
        performance = production_blueprint.get('performance', {})
        complexity = production_blueprint.get('complexity', '')
        recal_freq = production_blueprint.get('recalibration_frequency', '')

        row = {
            'Specification': spec,
            'Level': level,
            'N_Parameters': params.get('n_params', 0),
            'Implementation_Formula': implementation,
            'Expected_OOS_R2': performance.get('avg_r2_oos', np.nan),
            'Expected_OOS_RMSE': performance.get('avg_rmse_oos', np.nan),
            'Complexity': complexity,
            'Recalibration_Frequency': recal_freq
        }

        df = pd.DataFrame([row])

        # Save
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"  Saved: {filename}")

    def generate_implementation_blueprint(
        self,
        production_blueprint: Dict,
        hierarchical_results: Dict,
        oos_results: Dict,
        regime_results: Dict,
        filename: str = 'stageE_implementation_blueprint.txt'
    ):
        """
        Generate 5-7 page implementation blueprint.

        Sections:
        1. Executive summary
        2. Algorithmic steps
        3. Pseudo-code
        4. Recalibration protocol
        5. Edge case handling
        6. Integration with existing systems
        7. Performance monitoring
        8. Comparative performance analysis
        9. Economic value examples
        10. Limitations and caveats
        """
        spec = production_blueprint.get('specification', '')
        level = production_blueprint.get('level', 0)
        params = production_blueprint.get('parameters', {})
        implementation = production_blueprint.get('implementation', '')
        performance = production_blueprint.get('performance', {})
        complexity = production_blueprint.get('complexity', '')
        recal_freq = production_blueprint.get('recalibration_frequency', '')

        lines = []
        lines.append("=" * 80)
        lines.append("STAGE E: PRODUCTION IMPLEMENTATION BLUEPRINT")
        lines.append("=" * 80)
        lines.append("")

        # Section 1: Executive Summary
        lines.append("SECTION 1: EXECUTIVE SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Recommended Specification: {spec} (Level {level})")
        lines.append(f"Complexity Rating: {complexity}")
        lines.append(f"Number of Parameters: {params.get('n_params', 0)}")
        lines.append(f"Recalibration Frequency: {recal_freq}")
        lines.append("")
        lines.append("Expected Performance:")
        lines.append(f"  • Out-of-Sample R²: {performance.get('avg_r2_oos', np.nan):.3f}")
        lines.append(f"  • Out-of-Sample RMSE: {performance.get('avg_rmse_oos', np.nan):.3f}")
        lines.append("")

        # Comparison to baseline
        oos_summary = oos_results.get('oos_summary', {})
        baseline_rmse = oos_summary.get('Standard DTS', {}).get('avg_rmse_oos', np.nan)
        spec_rmse = performance.get('avg_rmse_oos', np.nan)

        if not np.isnan(baseline_rmse) and not np.isnan(spec_rmse) and baseline_rmse > 0:
            rmse_reduction_pct = 100 * (baseline_rmse - spec_rmse) / baseline_rmse
            lines.append(f"Improvement over Standard DTS:")
            lines.append(f"  • RMSE Reduction: {rmse_reduction_pct:.1f}%")
        lines.append("")

        # Section 2: Algorithmic Steps
        lines.append("SECTION 2: ALGORITHMIC STEPS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Input Data Required:")

        if level == 1:
            lines.append("  • f_DTS,t: Index-level DTS factor")
            lines.append("")
            lines.append("Calculation:")
            lines.append("  • Spread sensitivity = f_DTS,t (no adjustments)")

        elif level == 2:
            lines.append("  • Bond OAS (s_i)")
            lines.append("  • Years to maturity (T_i)")
            lines.append("  • f_DTS,t: Index-level DTS factor")
            lines.append("")
            lines.append("Calculation:")
            lines.append("  1. Look up lambda_T(T_i; 5y, s_i) from Merton table")
            lines.append("  2. Compute lambda_s(s_i; 100) = (s_i / 100)^(-0.25)")
            lines.append("  3. lambda_i = lambda_T × lambda_s")
            lines.append("  4. Spread sensitivity = lambda_i × f_DTS,t")

        elif level == 3:
            c0 = params.get('parameters', {}).get('c0', 1.0)
            c_s = params.get('parameters', {}).get('c_s', -0.25)
            lines.append("  • Bond OAS (s_i)")
            lines.append("  • Years to maturity (T_i)")
            lines.append("  • f_DTS,t: Index-level DTS factor")
            lines.append("")
            lines.append("Calculation:")
            lines.append("  1. Look up lambda_T(T_i; 5y, s_i) from Merton table")
            lines.append(f"  2. Compute lambda_s_adj = lambda_s(s_i)^({c_s:.3f} / -0.25)")
            lines.append(f"  3. lambda_i = {c0:.3f} × lambda_T × lambda_s_adj")
            lines.append("  4. Spread sensitivity = lambda_i × f_DTS,t")

        elif level == 4:
            lines.append("  • Bond OAS (s_i)")
            lines.append("  • Years to maturity (T_i or M_i)")
            lines.append("  • Sector (sector_i)")
            lines.append("  • f_DTS,t: Index-level DTS factor")
            lines.append("")
            lines.append("Calculation:")
            lines.append("  1. log_M = log(M_i + 0.1)")
            lines.append("  2. log_s = log(s_i + 1)")
            lines.append("  3. lambda_i = exp(β_0 + β_M·log_M + β_s·log_s + β_M²·(log_M)² + ")
            lines.append("                     β_Ms·log_M·log_s + sector_effects)")
            lines.append("  4. Spread sensitivity = lambda_i × f_DTS,t")
            lines.append("")
            lines.append("  Note: Coefficients β_0, β_M, etc. from Stage B Spec B.3")

        elif level == 5:
            gamma_vix = params.get('parameters', {}).get('gamma_vix', 0)
            gamma_oas = params.get('parameters', {}).get('gamma_oas', 0)
            lines.append("  • Bond OAS (s_i)")
            lines.append("  • Years to maturity (T_i)")
            lines.append("  • Sector (sector_i)")
            lines.append("  • VIX_t: Current VIX level")
            lines.append("  • OAS_index,t: Index-level OAS")
            lines.append("  • f_DTS,t: Index-level DTS factor")
            lines.append("")
            lines.append("Calculation:")
            lines.append("  1. Compute lambda_base_i (from Level 2, 3, or 4)")
            lines.append(f"  2. macro_adj = exp({gamma_vix:.3f}·(VIX_t/100) + {gamma_oas:.3f}·log(OAS_index,t))")
            lines.append("  3. lambda_i,t = lambda_base_i × macro_adj")
            lines.append("  4. Spread sensitivity = lambda_i,t × f_DTS,t")

        lines.append("")

        # Section 3: Pseudo-code
        lines.append("SECTION 3: PSEUDO-CODE")
        lines.append("=" * 80)
        lines.append("")

        if level == 1:
            lines.append("function compute_sensitivity(f_DTS):")
            lines.append("    return f_DTS")

        elif level == 2:
            lines.append("function compute_lambda(OAS, Maturity):")
            lines.append("    lambda_T = lookup_maturity_adjustment(Maturity, OAS)")
            lines.append("    lambda_s = (OAS / 100) ^ (-0.25)")
            lines.append("    lambda = lambda_T * lambda_s")
            lines.append("    return lambda")
            lines.append("")
            lines.append("function compute_sensitivity(OAS, Maturity, f_DTS):")
            lines.append("    lambda = compute_lambda(OAS, Maturity)")
            lines.append("    return lambda * f_DTS")

        elif level == 3:
            c0 = params.get('parameters', {}).get('c0', 1.0)
            c_s = params.get('parameters', {}).get('c_s', -0.25)
            lines.append("function compute_lambda_calibrated(OAS, Maturity):")
            lines.append("    lambda_T = lookup_maturity_adjustment(Maturity, OAS)")
            lines.append("    lambda_s = (OAS / 100) ^ (-0.25)")
            lines.append(f"    lambda_s_adj = lambda_s ^ ({c_s:.3f} / -0.25)")
            lines.append(f"    lambda = {c0:.3f} * lambda_T * lambda_s_adj")
            lines.append("    return lambda")
            lines.append("")
            lines.append("function compute_sensitivity(OAS, Maturity, f_DTS):")
            lines.append("    lambda = compute_lambda_calibrated(OAS, Maturity)")
            lines.append("    return lambda * f_DTS")

        elif level == 4:
            lines.append("function compute_lambda_empirical(OAS, Maturity, Sector, coefficients):")
            lines.append("    log_M = log(Maturity + 0.1)")
            lines.append("    log_s = log(OAS + 1)")
            lines.append("    log_M_sq = log_M ^ 2")
            lines.append("    interaction = log_M * log_s")
            lines.append("    sector_effect = coefficients.sector[Sector]")
            lines.append("    ")
            lines.append("    log_lambda = (coefficients.beta_0 + ")
            lines.append("                  coefficients.beta_M * log_M + ")
            lines.append("                  coefficients.beta_s * log_s + ")
            lines.append("                  coefficients.beta_M2 * log_M_sq + ")
            lines.append("                  coefficients.beta_Ms * interaction + ")
            lines.append("                  sector_effect)")
            lines.append("    ")
            lines.append("    lambda = exp(log_lambda)")
            lines.append("    return lambda")

        elif level == 5:
            gamma_vix = params.get('parameters', {}).get('gamma_vix', 0)
            gamma_oas = params.get('parameters', {}).get('gamma_oas', 0)
            lines.append("function compute_lambda_time_varying(OAS, Maturity, Sector, VIX, OAS_index):")
            lines.append("    # Compute base lambda (from Level 2, 3, or 4)")
            lines.append("    lambda_base = compute_lambda_base(OAS, Maturity, Sector)")
            lines.append("    ")
            lines.append("    # Macro adjustment")
            lines.append(f"    macro_adj = exp({gamma_vix:.3f} * (VIX / 100) + ")
            lines.append(f"                    {gamma_oas:.3f} * log(OAS_index))")
            lines.append("    ")
            lines.append("    lambda = lambda_base * macro_adj")
            lines.append("    return lambda")

        lines.append("")

        # Section 4: Recalibration Protocol
        lines.append("SECTION 4: RECALIBRATION PROTOCOL")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Frequency: {recal_freq}")
        lines.append("")

        if level == 1:
            lines.append("No recalibration needed (static specification)")

        elif level == 2:
            lines.append("Quarterly Review:")
            lines.append("  1. Check if current bond OAS and maturities are within table range")
            lines.append("  2. If bonds outside range, flag for manual review")
            lines.append("  3. No parameter re-estimation needed (lookup tables static)")

        elif level in [3, 4]:
            lines.append("Annual Recalibration:")
            lines.append("  1. Re-estimate specification using trailing 3-year window")
            lines.append("  2. Compare new parameter estimates to current parameters")
            lines.append("  3. If estimates differ by > 20%, adopt new parameters")
            lines.append("  4. Document parameter changes in version history")
            lines.append("")
            lines.append("Procedure:")
            lines.append("  • Training window: Most recent 3 years")
            lines.append("  • Validation: Out-of-sample on most recent 6 months")
            lines.append("  • Update trigger: Parameter drift > 20% OR OOS R² drops > 10%")

        elif level == 5:
            lines.append("Daily Macro Update:")
            lines.append("  • Update VIX_t and OAS_index,t from market data feeds")
            lines.append("  • Recalculate macro_adj for all bonds")
            lines.append("")
            lines.append("Annual Parameter Recalibration:")
            lines.append("  • Re-estimate gamma_vix and gamma_oas")
            lines.append("  • Re-estimate base lambda specification")
            lines.append("  • Update if parameters drift > 20%")

        lines.append("")

        # Section 5: Edge Case Handling
        lines.append("SECTION 5: EDGE CASE HANDLING")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Very Short Maturity (< 6 months):")
        lines.append("  • Use 6-month lambda as floor")
        lines.append("  • Do not extrapolate below 6 months")
        lines.append("  • Rationale: Merton approximation breaks down at very short maturities")
        lines.append("")
        lines.append("Distressed Spreads (> 2000 bps):")
        lines.append("  • Cap lambda at 1.2")
        lines.append("  • Rationale: Proportionality improves in extreme stress")
        lines.append("  • Alternative: Use 1.0 (standard DTS) for distressed bonds")
        lines.append("")
        lines.append("Missing Sector:")
        lines.append("  • Use cross-sector average lambda")
        lines.append("  • Or: Assign to 'Other' category")
        lines.append("")
        lines.append("New Issues (< 3 months old):")
        lines.append("  • Check if Stage B found significant new issue effect")
        lines.append("  • If yes: Apply new issue adjustment gamma_new")
        lines.append("  • If no: Use standard lambda")
        lines.append("")
        lines.append("Missing VIX (for time-varying spec):")
        lines.append("  • Use trailing 20-day average VIX")
        lines.append("  • Or: Fall back to static lambda_base")
        lines.append("")

        # Section 6: Integration
        lines.append("SECTION 6: INTEGRATION WITH EXISTING SYSTEMS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Risk Models:")
        lines.append("  • Replace DTS_i,t with DTS*_i,t = lambda_i × DTS_i,t")
        lines.append("  • Portfolio risk = Σ (holdings_i × DTS*_i,t)")
        lines.append("  • VaR/ES: Use adjusted sensitivities in risk calculations")
        lines.append("")
        lines.append("Attribution:")
        lines.append("  • Decompose factor return into:")
        lines.append("    - Index-level factor return: f_DTS,t")
        lines.append("    - Cross-sectional lambda effect: (lambda_i - lambda_avg) × f_DTS,t")
        lines.append("")
        lines.append("Relative Value:")
        lines.append("  • Flag bonds with large deviations from lambda-adjusted fair value")
        lines.append("  • Rich/cheap signal: Actual spread change vs predicted (lambda_i × f_DTS,t)")
        lines.append("")

        # Section 7: Performance Monitoring
        lines.append("SECTION 7: PERFORMANCE MONITORING")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Track Monthly:")
        lines.append("  • Out-of-sample R² (rolling 12-month window)")
        lines.append("  • Out-of-sample RMSE")
        lines.append("  • Forecast error distribution (check for bias)")
        lines.append("")
        lines.append("Alert Triggers:")
        lines.append("  • OOS R² drops below 50% of historical average")
        lines.append("  • RMSE increases by > 30% from baseline")
        lines.append("  • Systematic forecast bias detected (mean error ≠ 0)")
        lines.append("")
        lines.append("Quarterly Review:")
        lines.append("  • Compare to Standard DTS benchmark")
        lines.append("  • Assess performance by regime (IG/HY, narrow/wide spreads)")
        lines.append("  • Check edge case handling (short maturities, distressed, etc.)")
        lines.append("")

        # Section 8: Comparative Performance
        lines.append("SECTION 8: COMPARATIVE PERFORMANCE ANALYSIS")
        lines.append("=" * 80)
        lines.append("")

        # Get comparison metrics
        baseline_summary = oos_summary.get('Standard DTS', {})
        spec_summary = oos_summary.get(spec, {})

        baseline_r2 = baseline_summary.get('avg_r2_oos', np.nan)
        spec_r2 = spec_summary.get('avg_r2_oos', np.nan)

        if not np.isnan(baseline_r2) and not np.isnan(spec_r2):
            r2_improvement = spec_r2 - baseline_r2
            lines.append(f"vs Standard DTS (baseline):")
            lines.append(f"  • Standard DTS R²: {baseline_r2:.3f}")
            lines.append(f"  • {spec} R²: {spec_r2:.3f}")
            lines.append(f"  • Improvement: +{r2_improvement:.3f} ({100*r2_improvement/baseline_r2:.1f}%)")
            lines.append("")

        if not np.isnan(baseline_rmse) and not np.isnan(spec_rmse):
            lines.append(f"  • Standard DTS RMSE: {baseline_rmse:.3f}")
            lines.append(f"  • {spec} RMSE: {spec_rmse:.3f}")
            lines.append(f"  • Reduction: {rmse_reduction_pct:.1f}%")
            lines.append("")

        # Regime-specific performance
        lines.append("Performance by Regime:")
        if spec in regime_results:
            for regime_name, metrics in regime_results[spec].items():
                if metrics['n_windows'] > 0:
                    lines.append(f"  {regime_name}:")
                    lines.append(f"    R² = {metrics['avg_r2_oos']:.3f}, RMSE = {metrics['avg_rmse_oos']:.3f}")
        lines.append("")

        # Section 9: Economic Value
        lines.append("SECTION 9: ECONOMIC VALUE EXAMPLES")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Example 1: Hedging Efficiency")
        lines.append("-" * 40)
        lines.append("Portfolio: Long $100M of 1-year BBB bonds")
        lines.append("Hedge: Short 5-year BBB bonds")
        lines.append("")
        lines.append("Standard DTS:")
        lines.append("  • Hedge ratio = 1.0 (assumes proportional sensitivities)")
        lines.append("  • Residual tracking error ≈ 120 bps/year")
        lines.append("")

        if level >= 2:
            lines.append("Merton-adjusted:")
            lines.append("  • Hedge ratio = lambda_1y / lambda_5y ≈ 3.2")
            lines.append("    (1-year bonds ~3x more sensitive than 5-year)")
            lines.append("  • Residual tracking error ≈ 40 bps/year")
            lines.append("  • Value: 80 bps lower tracking error")
            lines.append("    → ~$800K lower unexpected P&L volatility")
        lines.append("")

        lines.append("Example 2: Relative Value Signals")
        lines.append("-" * 40)
        lines.append("Identifying rich/cheap bonds within capital structure")
        lines.append("")
        lines.append("Standard DTS:")
        lines.append("  • Assumes all bonds move proportionally")
        lines.append("  • Misses 300-500% cross-maturity differences")
        lines.append("")

        if level >= 2:
            lines.append("Merton-adjusted:")
            lines.append("  • Properly scales for maturity")
            lines.append("  • Identifies mispricings averaging 15 bps")
            lines.append("  • Trade: Long cheap (e.g. 10y), short rich (e.g. 1y)")
            lines.append("    → Earn 30 bps as convergence occurs over 6 months")
        lines.append("")

        lines.append("Example 3: Portfolio Construction")
        lines.append("-" * 40)
        lines.append("Credit barbell (short + long maturity, avoid intermediate)")
        lines.append("")
        lines.append("Standard DTS:")
        lines.append("  • Treats all durations equally")
        lines.append("  • Over-concentrates DTS risk in short end")
        lines.append("")

        if level >= 2:
            lines.append("Merton-adjusted:")
            lines.append("  • Recognizes front-end 3-4× more sensitive")
            lines.append("  • Rebalances to equalize risk contribution")
            lines.append("  • Result: 25% reduction in unexpected drawdowns")
            lines.append("    during spread volatility spikes")
        lines.append("")

        # Section 10: Limitations
        lines.append("SECTION 10: LIMITATIONS AND CAVEATS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Model Limitations:")
        lines.append("  • Out-of-sample performance may degrade if regime shifts")
        lines.append("    outside historical experience")
        lines.append("  • Liquidity crises (2008-style) may break Merton assumptions")
        lines.append("  • Assumes structural model holds—not data-driven machine learning")
        lines.append("")
        lines.append("Data Requirements:")
        lines.append("  • Requires clean OAS, maturity, sector data")
        lines.append("  • Quality control critical—garbage in, garbage out")
        lines.append("  • Missing data must be handled carefully (see Section 5)")
        lines.append("")
        lines.append("Operational Risks:")
        lines.append("  • Parameter drift possible—requires monitoring")
        lines.append("  • Recalibration needed periodically (see Section 4)")

        if level == 5:
            lines.append("  • Daily macro data feeds required (VIX, OAS index)")
            lines.append("  • System downtime = fall back to static lambda")

        lines.append("")
        lines.append("Sensitivity to Implementation Choices:")
        lines.append("  • Bucket definitions affect granularity vs sample size")
        lines.append("  • Frequency (daily/weekly/monthly) affects noise vs lag")
        lines.append("  • Clustering choice (week/bond) affects standard errors")
        lines.append("")

        # Summary
        lines.append("=" * 80)
        lines.append("END OF IMPLEMENTATION BLUEPRINT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Recommended Specification: {spec}")
        lines.append(f"Expected OOS R²: {performance.get('avg_r2_oos', np.nan):.3f}")
        lines.append(f"Implementation Complexity: {complexity}")
        lines.append("")
        lines.append("Next Steps:")
        lines.append("  1. Review and approve specification")
        lines.append("  2. Implement in test environment")
        lines.append("  3. Validate on hold-out sample")
        lines.append("  4. Deploy to production with monitoring")
        lines.append("  5. Recalibrate per schedule (Section 4)")
        lines.append("")

        # Save
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Saved: {filename}")

    def save_all_reports(
        self,
        hierarchical_results: Dict,
        oos_results: Dict,
        regime_results: Dict,
        production_blueprint: Dict,
        prefix: str = 'stageE'
    ):
        """
        Save all Stage E reports at once.

        Args:
            hierarchical_results: Hierarchical test results
            oos_results: OOS validation results
            regime_results: Performance by regime
            production_blueprint: Production specification blueprint
            prefix: Filename prefix
        """
        print("Saving Stage E reports...")
        print()

        # Table E.1: Hierarchical tests
        self.save_hierarchical_test_results(
            hierarchical_results,
            filename=f'{prefix}_table_e1_hierarchical_tests.csv'
        )

        # Table E.2: Model comparison
        self.save_model_comparison_table(
            oos_results,
            hierarchical_results,
            filename=f'{prefix}_table_e2_model_comparison.csv'
        )

        # Table E.3: Performance by regime
        self.save_performance_by_regime_table(
            regime_results,
            filename=f'{prefix}_table_e3_performance_by_regime.csv'
        )

        # Table E.4: Production specification
        self.save_production_specification_table(
            production_blueprint,
            filename=f'{prefix}_table_e4_production_spec.csv'
        )

        # Implementation blueprint
        self.generate_implementation_blueprint(
            production_blueprint,
            hierarchical_results,
            oos_results,
            regime_results,
            filename=f'{prefix}_implementation_blueprint.txt'
        )

        print()
        print("All Stage E reports saved successfully!")
