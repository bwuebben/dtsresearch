"""
Reporting functions for Stage C deliverables.

Creates Tables C.1, C.2, C.3 and written summary.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class StageCReporter:
    """Creates publication-quality reports for Stage C analysis."""

    def __init__(self, output_dir: str = './output/reports'):
        self.output_dir = output_dir

    def create_table_c1_stability_test(
        self,
        rolling_results_combined: pd.DataFrame,
        rolling_results_ig: Optional[pd.DataFrame] = None,
        rolling_results_hy: Optional[pd.DataFrame] = None,
        chow_test_combined: Optional[Dict] = None,
        chow_test_ig: Optional[Dict] = None,
        chow_test_hy: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Table C.1: Rolling window stability test.

        Rows: Time windows
        Columns: beta_w, standard error, 95% CI, sample size
        Separate panels for IG and HY
        Chow test: F-statistic, p-value

        Args:
            rolling_results_combined: Combined rolling window results
            rolling_results_ig: Optional IG results
            rolling_results_hy: Optional HY results
            chow_test_combined: Chow test for combined
            chow_test_ig: Chow test for IG
            chow_test_hy: Chow test for HY

        Returns:
            DataFrame formatted for publication
        """
        tables = []

        def format_results(results_df, regime_label):
            """Format one regime's results."""
            df = results_df.copy()

            # Format window periods
            df['Period'] = df.apply(
                lambda row: f"{row['window_start'].year}-{row['window_end'].year}",
                axis=1
            )

            # Format beta with SE in parentheses
            df['β_w (SE)'] = df.apply(
                lambda row: f"{row['beta_w']:.3f} ({row['se_beta']:.3f})",
                axis=1
            )

            # Format CI
            df['95% CI'] = df.apply(
                lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
                axis=1
            )

            # Format R²
            df['R²'] = df['r_squared'].apply(lambda x: f"{x:.3f}")

            # Sample size
            df['N'] = df['n_obs'].astype(int)

            # Select columns
            table = df[['Period', 'β_w (SE)', '95% CI', 'R²', 'N']].copy()
            table.insert(0, 'Regime', regime_label)

            return table

        # Combined
        tables.append(format_results(rolling_results_combined, 'Combined'))

        # IG
        if rolling_results_ig is not None:
            tables.append(format_results(rolling_results_ig, 'IG'))

        # HY
        if rolling_results_hy is not None:
            tables.append(format_results(rolling_results_hy, 'HY'))

        # Combine all
        full_table = pd.concat(tables, ignore_index=True)

        # Add Chow test results as footer rows
        chow_rows = []

        if chow_test_combined is not None:
            chow_rows.append({
                'Regime': 'Combined',
                'Period': 'Chow Test',
                'β_w (SE)': f"F = {chow_test_combined.get('f_statistic', np.nan):.2f}",
                '95% CI': f"p = {chow_test_combined.get('p_value', np.nan):.4f}",
                'R²': chow_test_combined.get('interpretation', ''),
                'N': ''
            })

        if chow_test_ig is not None:
            chow_rows.append({
                'Regime': 'IG',
                'Period': 'Chow Test',
                'β_w (SE)': f"F = {chow_test_ig.get('f_statistic', np.nan):.2f}",
                '95% CI': f"p = {chow_test_ig.get('p_value', np.nan):.4f}",
                'R²': chow_test_ig.get('interpretation', ''),
                'N': ''
            })

        if chow_test_hy is not None:
            chow_rows.append({
                'Regime': 'HY',
                'Period': 'Chow Test',
                'β_w (SE)': f"F = {chow_test_hy.get('f_statistic', np.nan):.2f}",
                '95% CI': f"p = {chow_test_hy.get('p_value', np.nan):.4f}",
                'R²': chow_test_hy.get('interpretation', ''),
                'N': ''
            })

        if chow_rows:
            chow_df = pd.DataFrame(chow_rows)
            full_table = pd.concat([full_table, chow_df], ignore_index=True)

        return full_table

    def create_table_c2_macro_drivers(
        self,
        macro_driver_results: Dict
    ) -> pd.DataFrame:
        """
        Table C.2: Macro driver regression.

        Coefficients, standard errors, t-statistics, R²
        Economic significance: Effect of 1 SD change

        Args:
            macro_driver_results: Results from macro driver analysis

        Returns:
            DataFrame formatted for publication
        """
        if 'error' in macro_driver_results:
            return pd.DataFrame({'Error': [macro_driver_results['error']]})

        coeffs = macro_driver_results.get('coefficients', {})
        ses = macro_driver_results.get('std_errors', {})
        t_stats = macro_driver_results.get('t_stats', {})
        p_vals = macro_driver_results.get('p_values', {})
        econ_sig = macro_driver_results.get('economic_significance', {})

        rows = []

        # Intercept
        rows.append({
            'Variable': 'Intercept',
            'Coefficient': f"{coeffs.get('delta_0', np.nan):.4f}",
            'Std Error': f"{ses.get('se_delta_0', np.nan):.4f}",
            't-stat': '',
            'p-value': '',
            '1-SD Effect (%)': ''
        })

        # VIX
        rows.append({
            'Variable': 'VIX',
            'Coefficient': f"{coeffs.get('delta_VIX', np.nan):.4f}",
            'Std Error': f"{ses.get('se_delta_VIX', np.nan):.4f}",
            't-stat': f"{t_stats.get('t_delta_VIX', np.nan):.2f}",
            'p-value': f"{p_vals.get('p_delta_VIX', np.nan):.4f}",
            '1-SD Effect (%)': f"{econ_sig.get('effect_vix_pct', np.nan):.1f}%"
        })

        # OAS
        rows.append({
            'Variable': 'log(OAS)',
            'Coefficient': f"{coeffs.get('delta_OAS', np.nan):.4f}",
            'Std Error': f"{ses.get('se_delta_OAS', np.nan):.4f}",
            't-stat': f"{t_stats.get('t_delta_OAS', np.nan):.2f}",
            'p-value': f"{p_vals.get('p_delta_OAS', np.nan):.4f}",
            '1-SD Effect (%)': f"{econ_sig.get('effect_oas_pct', np.nan):.1f}%"
        })

        # 10y rate (if available)
        if 'delta_r' in coeffs:
            rows.append({
                'Variable': '10y Rate',
                'Coefficient': f"{coeffs.get('delta_r', np.nan):.4f}",
                'Std Error': f"{ses.get('se_delta_r', np.nan):.4f}",
                't-stat': f"{t_stats.get('t_delta_r', np.nan):.2f}",
                'p-value': f"{p_vals.get('p_delta_r', np.nan):.4f}",
                '1-SD Effect (%)': ''
            })

        table = pd.DataFrame(rows)

        # Add R² as footer
        r_squared = macro_driver_results.get('r_squared', np.nan)
        n_windows = macro_driver_results.get('n_windows', '')

        footer = pd.DataFrame([{
            'Variable': 'R²',
            'Coefficient': f"{r_squared:.3f}",
            'Std Error': '',
            't-stat': '',
            'p-value': '',
            '1-SD Effect (%)': f'N = {n_windows} windows'
        }])

        table = pd.concat([table, footer], ignore_index=True)

        return table

    def create_table_c3_maturity_specific(
        self,
        maturity_results: Dict
    ) -> pd.DataFrame:
        """
        Table C.3: Maturity-specific time-variation.

        Rows: Maturity buckets
        Columns: delta_VIX, standard error, t-statistic

        Args:
            maturity_results: Results from maturity-specific analysis

        Returns:
            DataFrame formatted for publication
        """
        if 'error' in maturity_results.get('by_maturity', {}):
            return pd.DataFrame({'Error': ['Insufficient data for maturity-specific analysis']})

        by_mat = maturity_results.get('by_maturity', {})

        if len(by_mat) == 0:
            return pd.DataFrame({'Error': ['No maturity-specific results available']})

        rows = []

        for bucket in ['1-2y', '3-5y', '7-10y']:
            if bucket in by_mat:
                result = by_mat[bucket]
                rows.append({
                    'Maturity Bucket': bucket,
                    'δ_VIX': f"{result['delta_VIX']:.4f}",
                    'Std Error': f"{result['se_delta_VIX']:.4f}",
                    't-stat': f"{result['t_stat']:.2f}",
                    'p-value': f"{result['p_value']:.4f}",
                    'Effect (%)': f"{result['effect_pct']:.1f}%"
                })

        table = pd.DataFrame(rows)

        # Add pattern test
        pattern_test = maturity_results.get('pattern_test', {})
        if 'error' not in pattern_test:
            footer = pd.DataFrame([{
                'Maturity Bucket': 'Pattern Test',
                'δ_VIX': pattern_test.get('pattern', ''),
                'Std Error': '',
                't-stat': '✓' if pattern_test.get('confirms_theory', False) else '✗',
                'p-value': 'Theory confirmed' if pattern_test.get('confirms_theory', False) else 'Pattern unclear',
                'Effect (%)': ''
            }])

            table = pd.concat([table, footer], ignore_index=True)

        return table

    def generate_written_summary(
        self,
        chow_test_combined: Dict,
        macro_driver_results: Optional[Dict],
        maturity_results: Optional[Dict],
        decision: str
    ) -> str:
        """
        Generate 3-4 page written summary for Stage C.

        Args:
            chow_test_combined: Chow test results
            macro_driver_results: Macro driver analysis results
            maturity_results: Maturity-specific results
            decision: Decision text from analysis

        Returns:
            Multi-page summary text
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary = f"""
{'='*80}
STAGE C: STABILITY ANALYSIS SUMMARY
{'='*80}

Generated: {timestamp}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

Stage C tests whether the relationship between Merton lambda and bond
characteristics (spread, maturity) is stable over time, or whether macro
state variables induce time-variation.

KEY PRINCIPLE: Don't add time-variation until you've proven the simple
static model fails.

{'='*80}
1. IS THE RELATIONSHIP STABLE OVER TIME?
{'='*80}

Chow Test for Structural Break:
  H0: beta_1 = beta_2 = ... = beta_W (all windows have same beta)

Results (Combined):
  - F-statistic: {chow_test_combined.get('f_statistic', np.nan):.2f}
  - p-value: {chow_test_combined.get('p_value', np.nan):.4f}
  - Interpretation: {chow_test_combined.get('interpretation', 'N/A')}

Beta Statistics Across Windows:
  - Mean: {chow_test_combined.get('mean_beta', np.nan):.3f}
  - Std Dev: {chow_test_combined.get('std_beta', np.nan):.3f}
  - Min: {chow_test_combined.get('min_beta', np.nan):.3f}
  - Max: {chow_test_combined.get('max_beta', np.nan):.3f}
  - Range: {chow_test_combined.get('max_beta', 1) - chow_test_combined.get('min_beta', 1):.3f}

Assessment:
"""

        stable = chow_test_combined.get('stable', None)

        if stable:
            summary += """
✓ STABLE: The relationship is statistically stable across time periods.

  Static Merton lambda provides a reliable baseline across normal markets
  and crisis periods. No need for time-varying adjustments.

  See Figure C.1 for visual confirmation: confidence bands should overlap
  beta = 1 throughout most of the sample.
"""
        elif stable is False and chow_test_combined.get('p_value', 1) < 0.01:
            summary += """
✗ UNSTABLE: Strong evidence of time-variation (p < 0.01)

  The relationship changes significantly across time periods. This suggests:
  - Macro state variables affect sensitivity patterns
  - Static lambda may misprice risk in certain periods
  - Time-varying specification may be necessary

  NEXT: Examine macro driver analysis to identify sources of variation.
"""
        else:
            summary += """
⚠ MARGINAL: Some evidence of instability (0.01 < p < 0.10)

  There is weak statistical evidence of time-variation, but not overwhelming.

  NEXT: Check economic significance. Even if statistically detectable,
  time-variation may be too small to matter for practical purposes.
"""

        # Macro driver analysis
        if macro_driver_results and 'error' not in macro_driver_results:
            summary += f"""

{'='*80}
2. WHAT DRIVES TIME-VARIATION? (MACRO DRIVER ANALYSIS)
{'='*80}

Second-stage regression:
  beta_w = delta_0 + delta_VIX * VIX_w + delta_OAS * log(OAS_w) + eta_w

Results:

VIX Effect:
  - Coefficient (δ_VIX): {macro_driver_results['coefficients']['delta_VIX']:.4f}
  - t-statistic: {macro_driver_results['t_stats']['t_delta_VIX']:.2f}
  - p-value: {macro_driver_results['p_values']['p_delta_VIX']:.4f}
  - Economic effect: {macro_driver_results['economic_significance']['effect_vix_pct']:.1f}% change for 1 SD VIX move

  Theory prediction: δ_VIX > 0 (high volatility amplifies sensitivity)
"""

            delta_vix = macro_driver_results['coefficients']['delta_VIX']
            p_vix = macro_driver_results['p_values']['p_delta_VIX']

            if p_vix < 0.05 and delta_vix > 0:
                summary += "  ✓ Significant and consistent with theory\n"
            elif p_vix < 0.05 and delta_vix < 0:
                summary += "  ✗ Significant but OPPOSITE sign from theory (investigate)\n"
            else:
                summary += "  ○ Not statistically significant\n"

            summary += f"""
OAS Effect:
  - Coefficient (δ_OAS): {macro_driver_results['coefficients']['delta_OAS']:.4f}
  - t-statistic: {macro_driver_results['t_stats']['t_delta_OAS']:.2f}
  - p-value: {macro_driver_results['p_values']['p_delta_OAS']:.4f}
  - Economic effect: {macro_driver_results['economic_significance']['effect_oas_pct']:.1f}% change for 1 SD OAS move

  Theory prediction: δ_OAS < 0 (wide spreads compress dispersion)
"""

            delta_oas = macro_driver_results['coefficients']['delta_OAS']
            p_oas = macro_driver_results['p_values']['p_delta_OAS']

            if p_oas < 0.05 and delta_oas < 0:
                summary += "  ✓ Significant and consistent with theory\n"
            elif p_oas < 0.05 and delta_oas > 0:
                summary += "  ✗ Significant but OPPOSITE sign from theory (investigate)\n"
            else:
                summary += "  ○ Not statistically significant\n"

            summary += f"""
Regression Fit:
  - R²: {macro_driver_results['r_squared']:.3f}
  - N (windows): {macro_driver_results['n_windows']}

{macro_driver_results.get('interpretation', '')}
"""

        else:
            summary += """

{'='*80}
2. MACRO DRIVER ANALYSIS
{'='*80}

Not applicable (relationship is stable, no need to explain time-variation).
"""

        # Maturity-specific
        if maturity_results and 'error' not in maturity_results.get('by_maturity', {}):
            summary += f"""

{'='*80}
3. MATURITY-SPECIFIC TIME-VARIATION
{'='*80}

Theory predicts: Short-maturity bonds more sensitive to macro state than
long-maturity bonds (front-end whipsaws more during crises).

Test: δ_VIX,1y > δ_VIX,5y > δ_VIX,10y

Results:
"""

            by_mat = maturity_results.get('by_maturity', {})
            for bucket in ['1-2y', '3-5y', '7-10y']:
                if bucket in by_mat:
                    result = by_mat[bucket]
                    summary += f"""
  {bucket}:
    - δ_VIX: {result['delta_VIX']:.4f} (t = {result['t_stat']:.2f}, p = {result['p_value']:.4f})
    - Economic effect: {result['effect_pct']:.1f}%
"""

            pattern_test = maturity_results.get('pattern_test', {})
            if 'error' not in pattern_test:
                summary += f"""
Pattern Assessment:
  - Pattern observed: {pattern_test.get('pattern', 'N/A')}
  - Confirms theory? {' ✓ YES' if pattern_test.get('confirms_theory', False) else '✗ NO'}
"""

        else:
            summary += """

{'='*80}
3. MATURITY-SPECIFIC TIME-VARIATION
{'='*80}

Insufficient data for maturity-specific analysis.
"""

        # Decision
        summary += f"""

{'='*80}
4. PRACTICAL IMPLICATIONS
{'='*80}

Question: Even if time-variation is statistically significant, does it
matter for portfolio management?

Consider:
  1. Risk model accuracy: Does time-varying lambda reduce tracking error?
  2. Crisis performance: Did static lambda severely misprice during 2020 COVID?
  3. Operational complexity: Worth daily macro feeds and recalibration?

Recommendation Framework:
  - Effect < 10%: Use static lambda (time-variation in noise)
  - Effect 10-20%: Hybrid (static baseline + crisis adjustments)
  - Effect > 20%: Implement time-varying lambda with macro state

{'='*80}
5. STAGE C DECISION
{'='*80}

{decision}

{'='*80}
6. IMPLICATIONS FOR STAGE E (PRODUCTION SPECIFICATION)
{'='*80}
"""

        if stable:
            summary += """
Static lambda is sufficient. Stage E will select among:
  - Level 2: Pure Merton (if Stage B showed beta ≈ 1)
  - Level 3: Calibrated Merton (if Stage B showed systematic bias)
  - Level 4: Empirical (if Stage B showed Merton inadequate)

Time-variation is NOT a production feature. Use static baseline.
"""
        else:
            summary += """
Time-varying lambda may be necessary. Stage E will need to decide:
  - Is the operational complexity justified?
  - Which macro variables to include (VIX? OAS? Rates?)
  - Update frequency (daily? weekly? monthly?)

If effects are concentrated in crises (VIX > 30), consider:
  - Static lambda for normal markets
  - Crisis overlay when VIX spikes
  - Avoid continuous recalibration
"""

        summary += f"""

{'='*80}
7. NEXT STEPS
{'='*80}
"""

        if stable:
            summary += """
1. Review Figure C.1 to visually confirm stability
2. Proceed to Stage D (robustness and extensions)
3. Test tail behavior, shock types, liquidity decomposition
4. Stage E will select parsimonious production specification
"""
        else:
            summary += """
1. Review Figures C.1-C.4 to understand time-variation patterns
2. Assess economic significance in crisis periods (Figure C.4)
3. Decide if operational complexity is warranted
4. Proceed to Stage D with awareness that time-variation exists
5. Stage E will incorporate macro state if effects > 20%
"""

        summary += f"""

{'='*80}
REFERENCES
{'='*80}

- Wuebben (2025): Theoretical foundation for time-variation predictions
- Paper Section: Stage C (stability testing methodology)
- Tables: C.1 (rolling windows), C.2 (macro drivers), C.3 (maturity-specific)
- Figures: C.1-C.4 (visualizations)

{'='*80}
END OF STAGE C SUMMARY
{'='*80}
"""

        return summary

    def save_all_reports(
        self,
        rolling_results_combined: pd.DataFrame,
        chow_test_combined: Dict,
        macro_driver_results: Optional[Dict],
        maturity_results: Optional[Dict],
        decision: str,
        rolling_results_ig: Optional[pd.DataFrame] = None,
        rolling_results_hy: Optional[pd.DataFrame] = None,
        chow_test_ig: Optional[Dict] = None,
        chow_test_hy: Optional[Dict] = None,
        prefix: str = 'stageC'
    ):
        """
        Save all Stage C reports to CSV and text files.

        Args:
            rolling_results_combined: Combined rolling window results
            chow_test_combined: Chow test results (combined)
            macro_driver_results: Macro driver analysis results
            maturity_results: Maturity-specific results
            decision: Decision text
            rolling_results_ig: Optional IG results
            rolling_results_hy: Optional HY results
            chow_test_ig: Optional Chow test for IG
            chow_test_hy: Optional Chow test for HY
            prefix: Filename prefix
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Table C.1: Stability test
        table_c1 = self.create_table_c1_stability_test(
            rolling_results_combined,
            rolling_results_ig,
            rolling_results_hy,
            chow_test_combined,
            chow_test_ig,
            chow_test_hy
        )
        table_c1.to_csv(
            f'{self.output_dir}/{prefix}_table_c1_stability.csv',
            index=False
        )

        # Table C.2: Macro drivers (if applicable)
        if macro_driver_results and 'error' not in macro_driver_results:
            table_c2 = self.create_table_c2_macro_drivers(macro_driver_results)
            table_c2.to_csv(
                f'{self.output_dir}/{prefix}_table_c2_macro_drivers.csv',
                index=False
            )

        # Table C.3: Maturity-specific (if applicable)
        if maturity_results and 'error' not in maturity_results.get('by_maturity', {}):
            table_c3 = self.create_table_c3_maturity_specific(maturity_results)
            table_c3.to_csv(
                f'{self.output_dir}/{prefix}_table_c3_maturity_specific.csv',
                index=False
            )

        # Full rolling window results
        rolling_results_combined.to_csv(
            f'{self.output_dir}/{prefix}_rolling_windows_full.csv',
            index=False
        )

        # Written summary
        summary = self.generate_written_summary(
            chow_test_combined,
            macro_driver_results,
            maturity_results,
            decision
        )

        with open(f'{self.output_dir}/{prefix}_summary.txt', 'w') as f:
            f.write(summary)

        print(f"  Saved reports to {self.output_dir}/")
