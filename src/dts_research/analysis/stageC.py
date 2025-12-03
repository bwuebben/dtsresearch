"""
Stage C analysis: Test stability of Merton predictions over time.

Implements rolling window stability tests and macro driver analysis.

Integration with Stage 0:
- Skips if Path 5 (theory fails) or Path 4 (mixed evidence)
- Focuses on relevant tests based on Stage 0 findings
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


class StageCAnalysis:
    """
    Implements Stage C: Does Static Merton Suffice or Do We Need Time-Variation?

    Tests whether the relationship between lambda and (s, T) is stable over time,
    or whether macro state variables induce time-variation.

    Integrates with Stage 0 to skip if theory fails or shows mixed evidence.
    """

    def __init__(self, stage0_results: Optional[Dict] = None):
        """
        Initialize Stage C analysis.

        Args:
            stage0_results: Optional Stage 0 results dictionary
        """
        from ..models.merton import MertonLambdaCalculator
        self.merton_calc = MertonLambdaCalculator()
        self.stage0_results = stage0_results
        self.stage0_path = stage0_results.get('decision_path') if stage0_results else None

    def should_skip_stage_c(self) -> Tuple[bool, str]:
        """
        Determine if Stage C should be skipped based on Stage 0.

        Returns:
            (should_skip, reason) tuple
        """
        if self.stage0_path is None:
            return False, "No Stage 0 results available"

        if self.stage0_path == 5:
            return True, (
                "Stage 0 Decision Path 5: Theory Fails\n"
                "Stage C (theory-driven time-variation tests) is not applicable.\n"
                "Recommendation: Skip Stage C."
            )

        if self.stage0_path == 4:
            return True, (
                "Stage 0 Decision Path 4: Mixed Evidence\n"
                "Theory shows weak/inconsistent support.\n"
                "Stage C time-variation tests may not be informative.\n"
                "Recommendation: Skip Stage C or run selectively."
            )

        return False, f"Stage 0 Path {self.stage0_path}: Proceed with Stage C"

    def rolling_window_stability_test(
        self,
        df: pd.DataFrame,
        window_years: int = 1,
        by_regime: bool = True,
        by_maturity: bool = False
    ) -> Dict:
        """
        Estimate beta_w for each rolling window and test for stability.

        For each window w:
            y_i,t = alpha_w + beta_w * [lambda^Merton_i,t * f_DTS,t] + epsilon

        Then Chow test: H0: beta_1 = beta_2 = ... = beta_W

        Args:
            df: Regression data with columns:
                - date
                - oas_pct_change (y_i,t)
                - oas_index_pct_change (f_DTS,t)
                - time_to_maturity
                - oas (spread level)
                - spread_regime (IG/HY)
                - maturity_bucket (for maturity-specific analysis)
            window_years: Window size in years (default 1 year)
            by_regime: If True, run separately for IG and HY
            by_maturity: If True, also run by maturity bucket

        Returns:
            Dictionary with:
                - results_combined: DataFrame with beta_w time series
                - results_ig: DataFrame (if by_regime=True)
                - results_hy: DataFrame (if by_regime=True)
                - results_by_maturity: Dict[str, DataFrame] (if by_maturity=True)
                - chow_test_combined: Chow test results
                - chow_test_ig: Results (if by_regime)
                - chow_test_hy: Results (if by_regime)
                - decision: Text recommendation
        """
        # Add lambda_merton
        df = df.copy()
        df['lambda_merton'] = df.apply(
            lambda row: self.merton_calc.lambda_combined(
                row['time_to_maturity'], row['oas']
            ),
            axis=1
        )
        df['f_merton'] = df['lambda_merton'] * df['oas_index_pct_change']

        results = {}

        # Combined
        results['results_combined'] = self._estimate_rolling_betas(df, window_years)
        results['chow_test_combined'] = self._chow_test(results['results_combined'])

        # By regime
        if by_regime:
            df_ig = df[df['spread_regime'] == 'IG'].copy()
            df_hy = df[df['spread_regime'] == 'HY'].copy()

            if len(df_ig) > 0:
                results['results_ig'] = self._estimate_rolling_betas(df_ig, window_years)
                results['chow_test_ig'] = self._chow_test(results['results_ig'])

            if len(df_hy) > 0:
                results['results_hy'] = self._estimate_rolling_betas(df_hy, window_years)
                results['chow_test_hy'] = self._chow_test(results['results_hy'])

        # By maturity
        if by_maturity and 'maturity_bucket' in df.columns:
            results['results_by_maturity'] = {}
            for bucket in ['1-2y', '3-5y', '7-10y']:
                df_mat = df[df['maturity_bucket'] == bucket].copy()
                if len(df_mat) > 0:
                    results['results_by_maturity'][bucket] = self._estimate_rolling_betas(
                        df_mat, window_years
                    )

        # Generate decision
        results['decision'] = self._generate_stability_decision(results)

        return results

    def _estimate_rolling_betas(
        self,
        df: pd.DataFrame,
        window_years: int
    ) -> pd.DataFrame:
        """
        Estimate beta for each rolling window.

        Args:
            df: Data with date, oas_pct_change, f_merton
            window_years: Window size in years

        Returns:
            DataFrame with columns: window_start, window_end, beta_w, se_beta,
                                    ci_lower, ci_upper, n_obs, r_squared
        """
        # Convert dates
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Create windows
        min_date = df['date'].min()
        max_date = df['date'].max()

        windows = []
        current_start = min_date

        while current_start < max_date:
            window_end = current_start + pd.DateOffset(years=window_years)
            if window_end > max_date:
                window_end = max_date

            # Get data for window
            window_data = df[
                (df['date'] >= current_start) & (df['date'] < window_end)
            ].copy()

            if len(window_data) >= 50:  # Minimum observations
                # Estimate regression
                y = window_data['oas_pct_change'].values
                X = window_data['f_merton'].values
                X_with_const = sm.add_constant(X)

                # Cluster by week
                window_data['week'] = window_data['date'].dt.to_period('W')
                clusters = window_data['week']

                try:
                    model = OLS(y, X_with_const)
                    results = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

                    beta_w = results.params[1]
                    se_beta = results.bse[1]
                    ci_lower = results.conf_int()[1, 0]
                    ci_upper = results.conf_int()[1, 1]
                    r_squared = results.rsquared

                    windows.append({
                        'window_start': current_start,
                        'window_end': window_end,
                        'beta_w': beta_w,
                        'se_beta': se_beta,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'n_obs': len(window_data),
                        'r_squared': r_squared
                    })
                except:
                    pass

            # Move to next window (non-overlapping)
            current_start = window_end

        return pd.DataFrame(windows)

    def _chow_test(self, rolling_results: pd.DataFrame) -> Dict:
        """
        Perform Chow test for structural break.

        H0: beta_1 = beta_2 = ... = beta_W

        Args:
            rolling_results: DataFrame from _estimate_rolling_betas

        Returns:
            Dictionary with:
                - f_statistic: Chow F-statistic
                - p_value: P-value
                - interpretation: Text interpretation
                - stable: Boolean (True if p > 0.10)
        """
        if len(rolling_results) < 2:
            return {
                'f_statistic': np.nan,
                'p_value': np.nan,
                'interpretation': 'Insufficient windows for Chow test',
                'stable': None
            }

        # Simple F-test for equality of betas
        # Use inverse variance weighting
        betas = rolling_results['beta_w'].values
        ses = rolling_results['se_beta'].values
        weights = 1 / (ses ** 2)

        # Weighted mean
        beta_pooled = np.average(betas, weights=weights)

        # Chi-square test statistic
        chi_sq = np.sum(weights * (betas - beta_pooled) ** 2)
        df = len(betas) - 1

        # F-statistic (approximate)
        f_stat = chi_sq / df
        p_value = 1 - stats.chi2.cdf(chi_sq, df)

        # Interpretation
        stable = p_value > 0.10

        if stable:
            interpretation = f"Static lambda sufficient (p={p_value:.4f})"
        elif p_value < 0.01:
            interpretation = f"Significant time-variation (p={p_value:.4f})"
        else:
            interpretation = f"Marginal instability (p={p_value:.4f})"

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'interpretation': interpretation,
            'stable': stable,
            'mean_beta': beta_pooled,
            'std_beta': np.std(betas),
            'min_beta': np.min(betas),
            'max_beta': np.max(betas)
        }

    def macro_driver_analysis(
        self,
        rolling_results: pd.DataFrame,
        macro_data: pd.DataFrame
    ) -> Dict:
        """
        Second-stage regression of beta_w on macro variables.

        Only run if Chow test rejects stability.

        Regression:
            beta_w = delta_0 + delta_VIX * VIX_w + delta_OAS * log(OAS_w) +
                     delta_r * r_10y_w + eta_w

        Args:
            rolling_results: DataFrame with beta_w time series
            macro_data: DataFrame with columns:
                - date
                - vix (average VIX)
                - oas_index (average OAS index level)
                - r_10y (average 10-year treasury yield)

        Returns:
            Dictionary with:
                - coefficients: Dict of delta estimates
                - std_errors: Dict of standard errors
                - t_stats: Dict of t-statistics
                - p_values: Dict of p-values
                - r_squared: R-squared
                - economic_significance: % change in beta for 1 SD change in macro vars
                - interpretation: Text
        """
        # Merge rolling results with macro data
        rolling_results = rolling_results.copy()

        # Calculate window midpoint
        rolling_results['window_mid'] = rolling_results['window_start'] + (
            rolling_results['window_end'] - rolling_results['window_start']
        ) / 2

        # Merge with macro data (average over window)
        merged_data = []
        for idx, row in rolling_results.iterrows():
            window_macro = macro_data[
                (macro_data['date'] >= row['window_start']) &
                (macro_data['date'] < row['window_end'])
            ]

            if len(window_macro) > 0:
                merged_data.append({
                    'beta_w': row['beta_w'],
                    'se_beta_w': row['se_beta'],
                    'vix_avg': window_macro['vix'].mean(),
                    'oas_avg': window_macro['oas_index'].mean(),
                    'r_10y_avg': window_macro['r_10y'].mean() if 'r_10y' in window_macro.columns else np.nan
                })

        df_macro = pd.DataFrame(merged_data)

        if len(df_macro) < 5:
            return {'error': 'Insufficient data for macro driver analysis'}

        # Prepare regression
        y = df_macro['beta_w'].values
        X = pd.DataFrame({
            'const': 1,
            'vix': df_macro['vix_avg'].values,
            'log_oas': np.log(df_macro['oas_avg'].values),
        })

        if not df_macro['r_10y_avg'].isna().all():
            X['r_10y'] = df_macro['r_10y_avg'].values

        # Run regression
        model = OLS(y, X)
        results = model.fit()

        # Extract results
        coefficients = {
            'delta_0': results.params['const'],
            'delta_VIX': results.params['vix'],
            'delta_OAS': results.params['log_oas']
        }

        std_errors = {
            'se_delta_0': results.bse['const'],
            'se_delta_VIX': results.bse['vix'],
            'se_delta_OAS': results.bse['log_oas']
        }

        t_stats = {
            't_delta_VIX': results.tvalues['vix'],
            't_delta_OAS': results.tvalues['log_oas']
        }

        p_values = {
            'p_delta_VIX': results.pvalues['vix'],
            'p_delta_OAS': results.pvalues['log_oas']
        }

        if 'r_10y' in X.columns:
            coefficients['delta_r'] = results.params['r_10y']
            std_errors['se_delta_r'] = results.bse['r_10y']
            t_stats['t_delta_r'] = results.tvalues['r_10y']
            p_values['p_delta_r'] = results.pvalues['r_10y']

        # Economic significance
        vix_sd = df_macro['vix_avg'].std()
        oas_sd = df_macro['oas_avg'].std()

        effect_vix = coefficients['delta_VIX'] * vix_sd
        effect_oas = coefficients['delta_OAS'] * np.log(oas_sd)

        economic_significance = {
            'effect_vix_1sd': effect_vix,
            'effect_vix_pct': effect_vix * 100,
            'effect_oas_1sd': effect_oas,
            'effect_oas_pct': effect_oas * 100,
            'vix_range': (df_macro['vix_avg'].min(), df_macro['vix_avg'].max()),
            'oas_range': (df_macro['oas_avg'].min(), df_macro['oas_avg'].max())
        }

        # Interpretation
        interpretation = self._interpret_macro_drivers(
            coefficients, p_values, economic_significance
        )

        return {
            'coefficients': coefficients,
            'std_errors': std_errors,
            't_stats': t_stats,
            'p_values': p_values,
            'r_squared': results.rsquared,
            'economic_significance': economic_significance,
            'interpretation': interpretation,
            'n_windows': len(df_macro)
        }

    def maturity_specific_time_variation(
        self,
        df: pd.DataFrame,
        macro_data: pd.DataFrame,
        window_years: int = 1
    ) -> Dict:
        """
        Estimate maturity-specific time-variation.

        Theory predicts: delta_VIX,1y > delta_VIX,5y > delta_VIX,10y
        (short bonds more regime-dependent)

        Args:
            df: Regression data
            macro_data: Macro variables
            window_years: Window size

        Returns:
            Dictionary with maturity-specific results
        """
        results = {}

        maturity_buckets = ['1-2y', '3-5y', '7-10y']

        for bucket in maturity_buckets:
            df_mat = df[df['maturity_bucket'] == bucket].copy()

            if len(df_mat) < 100:
                continue

            # Get rolling betas for this maturity
            rolling_results = self._estimate_rolling_betas(df_mat, window_years)

            if len(rolling_results) < 5:
                continue

            # Run macro driver analysis
            macro_results = self.macro_driver_analysis(rolling_results, macro_data)

            if 'error' not in macro_results:
                results[bucket] = {
                    'delta_VIX': macro_results['coefficients']['delta_VIX'],
                    'se_delta_VIX': macro_results['std_errors']['se_delta_VIX'],
                    't_stat': macro_results['t_stats']['t_delta_VIX'],
                    'p_value': macro_results['p_values']['p_delta_VIX'],
                    'effect_pct': macro_results['economic_significance']['effect_vix_pct']
                }

        # Test pattern: delta_VIX decreases with maturity
        if len(results) >= 2:
            pattern_test = self._test_maturity_pattern(results)
        else:
            pattern_test = {'error': 'Insufficient maturity buckets'}

        return {
            'by_maturity': results,
            'pattern_test': pattern_test
        }

    def _test_maturity_pattern(self, maturity_results: Dict) -> Dict:
        """
        Test if delta_VIX decreases with maturity.

        Args:
            maturity_results: Dict with maturity-specific delta_VIX estimates

        Returns:
            Test results
        """
        buckets = ['1-2y', '3-5y', '7-10y']
        available = [b for b in buckets if b in maturity_results]

        if len(available) < 2:
            return {'error': 'Need at least 2 maturity buckets'}

        deltas = [maturity_results[b]['delta_VIX'] for b in available]

        # Simple monotonicity test
        is_decreasing = all(deltas[i] >= deltas[i+1] for i in range(len(deltas)-1))

        return {
            'pattern': 'Decreasing with maturity' if is_decreasing else 'Not monotonic',
            'confirms_theory': is_decreasing,
            'deltas': {b: maturity_results[b]['delta_VIX'] for b in available}
        }

    def _generate_stability_decision(self, results: Dict) -> str:
        """
        Generate decision recommendation based on stability tests.

        Args:
            results: Full Stage C results

        Returns:
            Decision text
        """
        chow_combined = results.get('chow_test_combined', {})
        p_value = chow_combined.get('p_value', 1.0)

        if p_value > 0.10:
            decision = """
DECISION: STATIC LAMBDA SUFFICIENT

Chow test p-value = {:.4f} (> 0.10)

The relationship between lambda and (s, T) is stable over time.
No evidence of systematic time-variation.

RECOMMENDATION:
→ Use static lambda from Stage B (no time-varying adjustments needed)
→ Merton provides stable baseline across time periods
→ Proceed to Stage D (robustness) with confidence in static specification
→ No need for macro state variables in production system

NEXT STEPS:
- Use pure Merton or calibrated Merton (from Stage B)
- Test robustness in Stage D
- Select production specification in Stage E
""".format(p_value)

        elif 0.01 < p_value <= 0.10:
            decision = """
DECISION: MARGINAL INSTABILITY

Chow test p-value = {:.4f} (between 0.01 and 0.10)

Some evidence of time-variation, but not overwhelming.

RECOMMENDATION:
→ Examine macro driver analysis to understand source
→ Assess economic significance (> 20% effect?)
→ If effects < 20%, treat as robustness consideration, not core feature
→ Consider hybrid: static baseline with manual crisis adjustments

NEXT STEPS:
- Review Figure C.1 for visual assessment
- Check macro driver coefficients in Table C.2
- Evaluate crisis vs normal period performance
- Decide if operational complexity of time-varying lambda is warranted
""".format(p_value)

        else:  # p_value < 0.01
            beta_std = chow_combined.get('std_beta', 0)
            beta_range = (chow_combined.get('max_beta', 1) - chow_combined.get('min_beta', 1))

            decision = """
DECISION: SIGNIFICANT TIME-VARIATION

Chow test p-value = {:.4f} (< 0.01)

Strong evidence that relationship is NOT stable over time.
Beta varies substantially across windows:
  - Standard deviation: {:.3f}
  - Range: {:.3f}

RECOMMENDATION:
→ Proceed to macro driver analysis (essential)
→ Identify which macro variables drive time-variation (VIX? OAS? Rates?)
→ Assess economic significance: does beta change > 20% across sample?
→ If economically large, incorporate time-varying lambda in production

NEXT STEPS:
- Run macro driver analysis (Table C.2)
- Examine maturity-specific patterns (Table C.3)
- Review crisis period performance (Figure C.4)
- If effects > 20%, add macro state to Stage E production specification
- If effects concentrated in crises, consider hybrid approach
""".format(p_value, beta_std, beta_range)

        return decision

    def _interpret_macro_drivers(
        self,
        coefficients: Dict,
        p_values: Dict,
        economic_significance: Dict
    ) -> str:
        """
        Interpret macro driver regression results.

        Args:
            coefficients: Estimated coefficients
            p_values: P-values
            economic_significance: Economic effect sizes

        Returns:
            Interpretation text
        """
        delta_vix = coefficients['delta_VIX']
        delta_oas = coefficients['delta_OAS']
        p_vix = p_values['p_delta_VIX']
        p_oas = p_values['p_delta_OAS']

        effect_vix_pct = economic_significance['effect_vix_pct']
        effect_oas_pct = economic_significance['effect_oas_pct']

        interpretation = f"""
Macro Driver Analysis:

VIX Effect:
  - Coefficient: {delta_vix:.4f} (p={p_vix:.4f})
  - Economic effect: {effect_vix_pct:.1f}% change in beta for 1 SD change in VIX
  - Theory prediction: delta_VIX > 0 (high volatility amplifies sensitivity)
  - """

        if p_vix < 0.05:
            if delta_vix > 0:
                interpretation += "✓ Significant and consistent with theory\n"
            else:
                interpretation += "✗ Significant but OPPOSITE sign from theory\n"
        else:
            interpretation += "Insignificant (no clear VIX effect)\n"

        interpretation += f"""
OAS Effect:
  - Coefficient: {delta_oas:.4f} (p={p_oas:.4f})
  - Economic effect: {effect_oas_pct:.1f}% change in beta for 1 SD change in log(OAS)
  - Theory prediction: delta_OAS < 0 (wide spreads reduce dispersion)
  - """

        if p_oas < 0.05:
            if delta_oas < 0:
                interpretation += "✓ Significant and consistent with theory\n"
            else:
                interpretation += "✗ Significant but OPPOSITE sign from theory\n"
        else:
            interpretation += "Insignificant (no clear OAS effect)\n"

        # Overall assessment
        interpretation += "\nOverall Assessment:\n"

        if abs(effect_vix_pct) > 20 or abs(effect_oas_pct) > 20:
            interpretation += "→ ECONOMICALLY SIGNIFICANT time-variation (> 20% effect)\n"
            interpretation += "→ Recommend incorporating macro state in production specification\n"
        elif abs(effect_vix_pct) > 10 or abs(effect_oas_pct) > 10:
            interpretation += "→ Moderate time-variation (10-20% effect)\n"
            interpretation += "→ Consider hybrid: static baseline with crisis adjustments\n"
        else:
            interpretation += "→ Small effects (< 10%)\n"
            interpretation += "→ Time-variation not economically meaningful\n"
            interpretation += "→ Use static lambda despite statistical significance\n"

        return interpretation
