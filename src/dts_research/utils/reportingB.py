"""
Reporting module for Stage B deliverables.

Generates tables and written summaries as specified in the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class StageBReporter:
    """
    Generates Stage B deliverables:
    - Table B.1: Constrained Merton specifications
    - Table B.2: Model comparison
    - Table B.3: Theory vs Reality
    - Written summary (3-4 pages)
    """

    def __init__(self, output_dir: str = './output/reports'):
        self.output_dir = output_dir

    def create_table_b1_specifications(
        self,
        spec_b1: Dict,
        spec_b2: Dict
    ) -> pd.DataFrame:
        """
        Table B.1: Constrained Merton specifications.

        Shows B.1 (constrained) and B.2 (decomposed) results.

        Args:
            spec_b1: Specification B.1 results
            spec_b2: Specification B.2 results

        Returns:
            Formatted DataFrame
        """
        rows = []

        # Spec B.1 results
        for regime in ['combined', 'ig', 'hy']:
            b1 = spec_b1.get(regime, {})
            if 'error' in b1:
                continue

            regime_label = {'combined': 'Combined', 'ig': 'IG', 'hy': 'HY'}[regime]

            rows.append({
                'Specification': 'B.1: Merton Constrained',
                'Regime': regime_label,
                'β_Merton': f"{b1['beta_merton']:.3f}",
                'SE': f"({b1['se_beta']:.3f})",
                't-stat (β=1)': f"{b1['t_stat_h0_beta_eq_1']:.2f}",
                'p-value': f"{b1['p_value_h0_beta_eq_1']:.4f}",
                'R²': f"{b1['r_squared']:.3f}",
                'RMSE': f"{b1['rmse']:.4f}",
                'Interpretation': b1['interpretation']
            })

        # Spec B.2 results
        for regime in ['combined', 'ig', 'hy']:
            b2 = spec_b2.get(regime, {})
            if 'error' in b2:
                continue

            regime_label = {'combined': 'Combined', 'ig': 'IG', 'hy': 'HY'}[regime]

            rows.append({
                'Specification': 'B.2: Decomposed',
                'Regime': regime_label,
                'β_T': f"{b2['beta_T']:.3f} ({b2['se_beta_T']:.3f})",
                'β_s': f"{b2['beta_s']:.3f} ({b2['se_beta_s']:.3f})",
                'Joint Test p-value': f"{b2['joint_test_pvalue']:.4f}",
                'R²': f"{b2['r_squared']:.3f}",
                'RMSE': f"{b2['rmse']:.4f}",
                'Interpretation': b2['interpretation']
            })

        return pd.DataFrame(rows)

    def create_table_b2_model_comparison(
        self,
        model_comparison: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Table B.2: Model comparison.

        Args:
            model_comparison: Model comparison dataframe

        Returns:
            Formatted DataFrame
        """
        # Format for display
        formatted = model_comparison.copy()

        for col in ['R²', 'Adj R²', 'RMSE', 'ΔR² vs Stage A']:
            if col in formatted.columns:
                formatted[col] = formatted[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "—"
                )

        formatted['AIC'] = formatted['AIC'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )

        return formatted

    def create_table_b3_theory_vs_reality(
        self,
        theory_vs_reality: pd.DataFrame,
        top_n: int = 30
    ) -> pd.DataFrame:
        """
        Table B.3: Theory vs Reality comparison.

        Shows top buckets by absolute deviation.

        Args:
            theory_vs_reality: Theory vs reality comparison
            top_n: Number of top buckets to show

        Returns:
            Formatted DataFrame
        """
        # Select top deviations
        table = theory_vs_reality.head(top_n).copy()

        # Format for display
        display = pd.DataFrame({
            'Bucket': table['rating_bucket'] + ' ' + table['maturity_bucket'],
            'Sector': table['sector'],
            'β (Stage A)': table['beta'].apply(lambda x: f"{x:.3f}"),
            'λ (Merton)': table['lambda_merton'].apply(lambda x: f"{x:.3f}"),
            'Ratio (β/λ)': table['ratio'].apply(lambda x: f"{x:.3f}"),
            'Deviation': table['deviation'].apply(lambda x: f"{x:+.3f}"),
            '% Dev': table['pct_deviation'].apply(lambda x: f"{x:+.1f}%"),
            'Outlier?': table['outlier'].map({True: '***', False: ''}),'
            'N': table['n_observations'].astype(int)
        })

        return display

    def generate_written_summary(
        self,
        spec_b1: Dict,
        spec_b2: Dict,
        spec_b3: Dict,
        model_comparison: pd.DataFrame,
        theory_assessment: Dict,
        decision: str
    ) -> str:
        """
        Generate written summary (3-4 pages) as specified in the paper.

        Args:
            spec_b1: Specification B.1 results
            spec_b2: Specification B.2 results
            spec_b3: Specification B.3 results
            model_comparison: Model comparison table
            theory_assessment: Theory performance assessment
            decision: Decision recommendation

        Returns:
            Formatted string with summary
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        b1 = spec_b1.get('combined', {})
        b2 = spec_b2.get('combined', {})
        b3 = spec_b3.get('combined', {})

        summary = f"""
{'='*80}
STAGE B: DOES MERTON EXPLAIN THE VARIATION?
Generated: {timestamp}
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}

Core Question: Does Merton's structural model explain the cross-sectional
              variation in betas documented in Stage A?

KEY FINDINGS:
• Specification B.1 (Constrained Merton):
  β_Merton = {b1.get('beta_merton', np.nan):.3f} (SE = {b1.get('se_beta', np.nan):.3f})
  R² = {b1.get('r_squared', np.nan):.3f}
  Test H0: β=1, p-value = {b1.get('p_value_h0_beta_eq_1', np.nan):.4f}

• Theory Assessment:
  {theory_assessment['pct_in_acceptable_range']:.0f}% of buckets in acceptable range [0.8, 1.2]
  Median ratio (β/λ) = {theory_assessment['median_ratio']:.3f}
  {theory_assessment['assessment']}

DECISION:
{decision}

{'='*80}
1. DOES MERTON WORK? (Specification B.1 Results)
{'='*80}

Specification B.1 tests: y_i,t = α + β_Merton · [λ^Merton_i,t · f_DTS,t] + ε

Theory prediction: If Merton is exactly correct, β_Merton = 1

COMBINED SAMPLE:
  β_Merton = {b1.get('beta_merton', np.nan):.3f} (SE = {b1.get('se_beta', np.nan):.3f})
  t-statistic (H0: β=1) = {b1.get('t_stat_h0_beta_eq_1', np.nan):.2f}
  p-value = {b1.get('p_value_h0_beta_eq_1', np.nan):.4f}
  {'Reject H0' if b1.get('reject_h0_beta_eq_1', False) else 'Fail to reject H0'}

  R² = {b1.get('r_squared', np.nan):.3f}
  Adj R² = {b1.get('adj_r_squared', np.nan):.3f}
  RMSE = {b1.get('rmse', np.nan):.4f}
  N = {b1.get('n_observations', 0):,}

  Interpretation: {b1.get('interpretation', 'N/A')}

REGIME BREAKDOWN:
"""

        # Add IG/HY breakdown
        for regime, label in [('ig', 'Investment Grade'), ('hy', 'High Yield')]:
            result = spec_b1.get(regime, {})
            if 'error' not in result:
                summary += f"""
  {label}:
    β_Merton = {result['beta_merton']:.3f} (p-value = {result['p_value_h0_beta_eq_1']:.4f})
    R² = {result['r_squared']:.3f}
    {result['interpretation']}
"""

        summary += f"""
{'='*80}
2. WHICH COMPONENT DRIVES FIT? (Specification B.2 Results)
{'='*80}

Specification B.2 decomposes into maturity and credit quality effects:
y_i,t = α + β_T·[λ_T · f_DTS] + β_s·[λ_s · f_DTS] + ε

Theory predictions: β_T ≈ 1 (maturity adjustment works)
                   β_s ≈ 1 (credit quality adjustment works)

COMBINED SAMPLE:
  β_T = {b2.get('beta_T', np.nan):.3f} (SE = {b2.get('se_beta_T', np.nan):.3f})
  β_s = {b2.get('beta_s', np.nan):.3f} (SE = {b2.get('se_beta_s', np.nan):.3f})

  Individual tests:
    H0: β_T = 1, p-value = {b2.get('p_value_T_eq_1', np.nan):.4f}
    H0: β_s = 1, p-value = {b2.get('p_value_s_eq_1', np.nan):.4f}

  Joint test H0: (β_T, β_s) = (1, 1):
    Test statistic = {b2.get('joint_test_statistic', np.nan):.2f}
    p-value = {b2.get('joint_test_pvalue', np.nan):.4f}

  R² = {b2.get('r_squared', np.nan):.3f}

  Interpretation: {b2.get('interpretation', 'N/A')}

IMPLICATIONS:
"""
        # Interpret components
        if 'error' not in b2:
            beta_T = b2['beta_T']
            beta_s = b2['beta_s']

            if 0.9 <= beta_T <= 1.1 and 0.9 <= beta_s <= 1.1:
                summary += "  ✓ Both maturity and credit quality adjustments are empirically valid\n"
            elif 0.9 <= beta_T <= 1.1:
                summary += "  ⚠ Maturity adjustment works, but credit quality needs recalibration\n"
            elif 0.9 <= beta_s <= 1.1:
                summary += "  ⚠ Credit quality adjustment works, but maturity functional form wrong\n"
            else:
                summary += "  ✗ Both components deviate - need to reconsider Merton structure\n"

        summary += f"""
{'='*80}
3. WHERE DOES THEORY SUCCEED? (Theory vs Reality)
{'='*80}

Direct comparison of empirical betas to Merton predictions:

AGGREGATE PERFORMANCE:
  Total buckets analyzed: {theory_assessment['n_buckets']}
  Buckets in acceptable range [0.8, 1.2]: {theory_assessment['pct_in_acceptable_range']:.1f}%

  Median ratio (β/λ): {theory_assessment['median_ratio']:.3f}
  Mean ratio: {theory_assessment['mean_ratio']:.3f}
  Std deviation of ratios: {theory_assessment['std_ratio']:.3f}

  Range: {theory_assessment['min_ratio']:.3f} to {theory_assessment['max_ratio']:.3f}

  Systematic bias: {theory_assessment['systematic_bias']}

  Overall assessment: {theory_assessment['assessment']}

SUCCESS PATTERNS:
"""

        # Identify where theory works best
        if theory_assessment['pct_in_acceptable_range'] >= 70:
            summary += "  ✓ Theory works well in majority of buckets\n"
        elif theory_assessment['pct_in_acceptable_range'] >= 50:
            summary += "  ⚠ Theory works in about half of buckets\n"
        else:
            summary += "  ✗ Theory works in minority of buckets\n"

        summary += f"""
{'='*80}
4. WHERE DOES THEORY FAIL? (Residual Patterns)
{'='*80}

Residual analysis examines β - λ by maturity, spread, and sector.

See Figure B.2 for visual analysis of systematic patterns.

KEY DIAGNOSTICS:
• Zero line = perfect Merton prediction
• Systematic patterns above/below zero indicate model deficiencies
• Outliers indicate specific buckets where theory fails

COMMON FAILURE MODES:
1. Short maturity, low spread: Merton often over-predicts
2. Long maturity, high spread: Merton may under-predict
3. Sector effects: Some sectors systematically deviate

[Detailed residual statistics would go here based on actual data]

{'='*80}
5. IS UNRESTRICTED NECESSARY? (Model Comparison)
{'='*80}

Comparison of model performance:
"""

        # Add model comparison
        for idx, row in model_comparison.iterrows():
            summary += f"""
{row['Model']}:
  R² = {row['R²']}
  RMSE = {row['RMSE']}
  Parameters = {row['N Parameters']}
  ΔR² vs Stage A = {row['ΔR² vs Stage A']}
"""

        summary += f"""
PARAMETER EFFICIENCY:
"""
        # Calculate parameters vs R²
        b1_r2 = model_comparison[model_comparison['Model'].str.contains('B.1')]['R²'].values
        b3_r2 = model_comparison[model_comparison['Model'].str.contains('B.3')]['R²'].values

        if len(b1_r2) > 0 and len(b3_r2) > 0:
            try:
                r2_b1 = float(b1_r2[0])
                r2_b3 = float(b3_r2[0])
                if r2_b3 > r2_b1:
                    improvement = r2_b3 - r2_b1
                    summary += f"  Unrestricted gains {improvement:.3f} R² points over Merton\n"
                    if improvement < 0.05:
                        summary += "  → Minimal gain, Merton preferable for simplicity\n"
                    else:
                        summary += "  → Substantial gain, unrestricted may be necessary\n"
            except:
                pass

        summary += f"""
{'='*80}
6. PRACTICAL RECOMMENDATION
{'='*80}

Based on Stage B results, recommended approach for production systems:
{decision}

{'='*80}
7. IMPLICATIONS FOR STAGE C
{'='*80}

"""

        # Stage C implications
        if 'PATH 1' in decision or 'PATH 2' in decision:
            summary += """
Proceed to Stage C to test time-variation:
• Does static β_Merton suffice or do we need time-varying parameters?
• Test stability across different market regimes
• Rolling window estimation to detect parameter drift

Stage C will determine production specification:
- Pure Merton with static tables
- Calibrated Merton with time-varying β_Merton(t)
- Regime-dependent Merton (IG vs HY vs stress)
"""
        elif 'PATH 3' in decision:
            summary += """
Stage C should run BOTH tracks in parallel:
1. Theory-guided: Test if Merton+calibration stable over time
2. Unrestricted: Test if empirical patterns stable over time

Compare performance to determine final production spec.
"""
        else:
            summary += """
SKIP Stage C - theory fundamentally fails

Proceed directly to:
• Stage D: Robustness tests to diagnose WHY theory fails
• Stage E: Production spec selection (unrestricted only)

Report: Structural models inadequate for DTS adjustments in this market.
"""

        summary += """
{'='*80}
"""
        return summary

    def save_all_reports(
        self,
        spec_b1: Dict,
        spec_b2: Dict,
        spec_b3: Dict,
        model_comparison: pd.DataFrame,
        theory_vs_reality: pd.DataFrame,
        theory_assessment: Dict,
        decision: str,
        prefix: str = 'stageB'
    ):
        """
        Save all Stage B reports to files.

        Args:
            spec_b1: Specification B.1 results
            spec_b2: Specification B.2 results
            spec_b3: Specification B.3 results
            model_comparison: Model comparison table
            theory_vs_reality: Theory vs reality comparison
            theory_assessment: Theory performance assessment
            decision: Decision recommendation
            prefix: Filename prefix
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Table B.1
        table1 = self.create_table_b1_specifications(spec_b1, spec_b2)
        table1.to_csv(f'{self.output_dir}/{prefix}_table_b1_specifications.csv', index=False)

        # Table B.2
        table2 = self.create_table_b2_model_comparison(model_comparison)
        table2.to_csv(f'{self.output_dir}/{prefix}_table_b2_model_comparison.csv', index=False)

        # Table B.3
        table3 = self.create_table_b3_theory_vs_reality(theory_vs_reality)
        table3.to_csv(f'{self.output_dir}/{prefix}_table_b3_theory_vs_reality.csv', index=False)

        # Full theory vs reality
        theory_vs_reality.to_csv(f'{self.output_dir}/{prefix}_theory_vs_reality_full.csv', index=False)

        # Written summary
        summary = self.generate_written_summary(
            spec_b1, spec_b2, spec_b3, model_comparison,
            theory_assessment, decision
        )
        with open(f'{self.output_dir}/{prefix}_summary.txt', 'w') as f:
            f.write(summary)

        print(f"Stage B reports saved to {self.output_dir}/")
