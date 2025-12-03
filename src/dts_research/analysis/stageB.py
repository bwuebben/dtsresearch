"""
Stage B: Does Merton Explain the Variation?

Tests whether Merton's structural predictions explain the cross-sectional variation
documented in Stage A. This is the CORE empirical test of the theoretical framework.

Three specifications:
B.1: Merton as offset (constrained) - single β_Merton parameter
B.2: Decomposed components - separate β_T and β_s
B.3: Unrestricted - fully flexible functional form
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from scipy import stats
from ..models.merton import MertonLambdaCalculator


class StageBAnalysis:
    """
    Implements Stage B: Test whether Merton explains variation.

    Critical objective: Does theory explain the variation documented in Stage A?
    """

    def __init__(self):
        self.merton_calc = MertonLambdaCalculator()
        self.spec_b1_results = None
        self.spec_b2_results = None
        self.spec_b3_results = None

    # =========================================================================
    # Specification B.1: Merton as Offset (Constrained)
    # =========================================================================

    def run_specification_b1(
        self,
        df: pd.DataFrame,
        by_regime: bool = True
    ) -> Dict:
        """
        Specification B.1: y_i,t = α + β_Merton · [λ^Merton_i,t · f_DTS,t] + ε

        Theory prediction: β_Merton = 1 if Merton is exactly correct.

        Args:
            df: Regression-ready dataframe with bond characteristics
            by_regime: If True, run separately for IG and HY

        Returns:
            Dictionary with regression results
        """
        # Calculate Merton lambda for each bond-week observation
        df = df.copy()
        df['lambda_merton'] = self.merton_calc.lambda_combined(
            df['time_to_maturity'].values,
            df['oas'].values
        )

        # Create adjusted DTS factor
        df['f_merton'] = df['lambda_merton'] * df['oas_index_pct_change']

        results = {}

        # Run for all data
        results['combined'] = self._estimate_b1_model(df, 'Combined')

        if by_regime:
            # Split by spread level (IG < 300, HY >= 300)
            ig_df = df[df['oas'] < 300].copy()
            hy_df = df[df['oas'] >= 300].copy()

            if len(ig_df) > 100:
                results['ig'] = self._estimate_b1_model(ig_df, 'IG')

            if len(hy_df) > 100:
                results['hy'] = self._estimate_b1_model(hy_df, 'HY')

        self.spec_b1_results = results
        return results

    def _estimate_b1_model(self, df: pd.DataFrame, label: str) -> Dict:
        """
        Helper to estimate B.1 model.

        Args:
            df: Data with f_merton calculated
            label: Label for this estimation

        Returns:
            Dictionary with results
        """
        # Prepare regression
        y = df['oas_pct_change'].values
        X = df['f_merton'].values
        X = sm.add_constant(X)

        # Cluster by week and issuer
        clusters = df['date'].astype(str) + '_' + df['issuer_id'].astype(str)

        try:
            model = sm.OLS(y, X)
            reg_results = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

            beta_merton = reg_results.params[1]
            se_beta = reg_results.bse[1]

            # Wald test: H0: β = 1
            t_stat_h0_1 = (beta_merton - 1.0) / se_beta
            p_value_h0_1 = 2 * (1 - stats.t.cdf(abs(t_stat_h0_1), reg_results.df_resid))

            return {
                'label': label,
                'beta_merton': beta_merton,
                'se_beta': se_beta,
                't_stat': reg_results.tvalues[1],
                'p_value': reg_results.pvalues[1],
                't_stat_h0_beta_eq_1': t_stat_h0_1,
                'p_value_h0_beta_eq_1': p_value_h0_1,
                'reject_h0_beta_eq_1': p_value_h0_1 < 0.05,
                'alpha': reg_results.params[0],
                'se_alpha': reg_results.bse[0],
                'r_squared': reg_results.rsquared,
                'adj_r_squared': reg_results.rsquared_adj,
                'rmse': np.sqrt(np.mean(reg_results.resid ** 2)),
                'aic': reg_results.aic,
                'n_observations': len(y),
                'interpretation': self._interpret_b1_result(beta_merton, p_value_h0_1)
            }

        except Exception as e:
            return {
                'label': label,
                'error': f'Regression failed: {str(e)}'
            }

    def _interpret_b1_result(self, beta: float, p_value: float) -> str:
        """Interpret B.1 results based on β_Merton value."""
        if 0.9 <= beta <= 1.1 and p_value > 0.05:
            return "✓ Merton predictions unbiased - theory works"
        elif 0.8 <= beta <= 1.2:
            return "⚠ Close enough for practical purposes"
        elif beta > 1.2:
            return "✗ Systematic upward bias - need calibration"
        elif beta < 0.8:
            return "✗ Systematic downward bias - need calibration"
        else:
            return "⚠ Review required"

    # =========================================================================
    # Specification B.2: Decomposed Components
    # =========================================================================

    def run_specification_b2(
        self,
        df: pd.DataFrame,
        by_regime: bool = True
    ) -> Dict:
        """
        Specification B.2: Decompose into maturity and credit quality effects.

        y_i,t = α + β_T·[λ_T · f_DTS] + β_s·[λ_s · f_DTS] + ε

        Theory prediction: β_T ≈ 1 and β_s ≈ 1

        Args:
            df: Regression-ready dataframe
            by_regime: If True, run separately for IG and HY

        Returns:
            Dictionary with regression results
        """
        df = df.copy()

        # Calculate lambda components separately
        df['lambda_T'] = self.merton_calc.lambda_T(
            df['time_to_maturity'].values,
            df['oas'].values
        )

        df['lambda_s'] = self.merton_calc.lambda_s(
            df['oas'].values
        )

        # Create adjusted factors
        df['f_T'] = df['lambda_T'] * df['oas_index_pct_change']
        df['f_s'] = df['lambda_s'] * df['oas_index_pct_change']

        results = {}

        # Run for all data
        results['combined'] = self._estimate_b2_model(df, 'Combined')

        if by_regime:
            ig_df = df[df['oas'] < 300].copy()
            hy_df = df[df['oas'] >= 300].copy()

            if len(ig_df) > 100:
                results['ig'] = self._estimate_b2_model(ig_df, 'IG')

            if len(hy_df) > 100:
                results['hy'] = self._estimate_b2_model(hy_df, 'HY')

        self.spec_b2_results = results
        return results

    def _estimate_b2_model(self, df: pd.DataFrame, label: str) -> Dict:
        """Helper to estimate B.2 model."""
        y = df['oas_pct_change'].values
        X = df[['f_T', 'f_s']].values
        X = sm.add_constant(X)

        clusters = df['date'].astype(str) + '_' + df['issuer_id'].astype(str)

        try:
            model = sm.OLS(y, X)
            reg_results = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

            beta_T = reg_results.params[1]
            beta_s = reg_results.params[2]
            se_beta_T = reg_results.bse[1]
            se_beta_s = reg_results.bse[2]

            # Individual tests
            t_stat_T_eq_1 = (beta_T - 1.0) / se_beta_T
            p_value_T_eq_1 = 2 * (1 - stats.t.cdf(abs(t_stat_T_eq_1), reg_results.df_resid))

            t_stat_s_eq_1 = (beta_s - 1.0) / se_beta_s
            p_value_s_eq_1 = 2 * (1 - stats.t.cdf(abs(t_stat_s_eq_1), reg_results.df_resid))

            # Joint test: H0: (β_T, β_s) = (1, 1)
            restrictions = np.array([[0, 1, 0], [0, 0, 1]])  # Test params 1 and 2
            values = np.array([1.0, 1.0])
            joint_test = reg_results.wald_test(restrictions, values)

            return {
                'label': label,
                'beta_T': beta_T,
                'beta_s': beta_s,
                'se_beta_T': se_beta_T,
                'se_beta_s': se_beta_s,
                't_stat_T': reg_results.tvalues[1],
                't_stat_s': reg_results.tvalues[2],
                'p_value_T': reg_results.pvalues[1],
                'p_value_s': reg_results.pvalues[2],
                't_stat_T_eq_1': t_stat_T_eq_1,
                'p_value_T_eq_1': p_value_T_eq_1,
                't_stat_s_eq_1': t_stat_s_eq_1,
                'p_value_s_eq_1': p_value_s_eq_1,
                'joint_test_statistic': joint_test.statistic,
                'joint_test_pvalue': joint_test.pvalue,
                'alpha': reg_results.params[0],
                'r_squared': reg_results.rsquared,
                'adj_r_squared': reg_results.rsquared_adj,
                'rmse': np.sqrt(np.mean(reg_results.resid ** 2)),
                'aic': reg_results.aic,
                'n_observations': len(y),
                'interpretation': self._interpret_b2_result(beta_T, beta_s, p_value_T_eq_1, p_value_s_eq_1)
            }

        except Exception as e:
            return {
                'label': label,
                'error': f'Regression failed: {str(e)}'
            }

    def _interpret_b2_result(
        self,
        beta_T: float,
        beta_s: float,
        p_T: float,
        p_s: float
    ) -> str:
        """Interpret B.2 decomposed results."""
        T_ok = 0.9 <= beta_T <= 1.1 and p_T > 0.05
        s_ok = 0.9 <= beta_s <= 1.1 and p_s > 0.05

        if T_ok and s_ok:
            return "✓ Both maturity and credit quality effects work"
        elif T_ok and not s_ok:
            return "⚠ Maturity correct, credit quality needs recalibration"
        elif not T_ok and s_ok:
            return "⚠ Credit quality correct, maturity functional form wrong"
        else:
            return "✗ Need to reconsider entire Merton structure"

    # =========================================================================
    # Specification B.3: Unrestricted
    # =========================================================================

    def run_specification_b3(
        self,
        df: pd.DataFrame,
        by_regime: bool = True
    ) -> Dict:
        """
        Specification B.3: Fully flexible functional form.

        λ = β₀ + β_M·M + β_M²·M² + β_s·s + β_s²·s² + β_Ms·M·s + rating + sector dummies

        Args:
            df: Regression-ready dataframe
            by_regime: If True, run separately for IG and HY

        Returns:
            Dictionary with regression results
        """
        df = df.copy()

        # Create polynomial and interaction terms
        df['maturity'] = df['time_to_maturity']
        df['spread'] = df['oas']
        df['maturity_sq'] = df['maturity'] ** 2
        df['spread_sq'] = df['spread'] ** 2
        df['maturity_spread'] = df['maturity'] * df['spread']

        results = {}

        # Run for all data
        results['combined'] = self._estimate_b3_model(df, 'Combined')

        if by_regime:
            ig_df = df[df['oas'] < 300].copy()
            hy_df = df[df['oas'] >= 300].copy()

            if len(ig_df) > 100:
                results['ig'] = self._estimate_b3_model(ig_df, 'IG')

            if len(hy_df) > 100:
                results['hy'] = self._estimate_b3_model(hy_df, 'HY')

        self.spec_b3_results = results
        return results

    def _estimate_b3_model(self, df: pd.DataFrame, label: str) -> Dict:
        """Helper to estimate B.3 unrestricted model."""
        # Create rating and sector dummies
        rating_dummies = pd.get_dummies(df['rating'], prefix='rating', drop_first=True)
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector', drop_first=True)

        # Combine predictors
        X_chars = df[['maturity', 'spread', 'maturity_sq', 'spread_sq', 'maturity_spread']].values

        # Calculate implied lambda for each observation
        X_full = np.hstack([X_chars, rating_dummies.values, sector_dummies.values])
        X_full = sm.add_constant(X_full)

        # First estimate lambda
        y_lambda = df['oas_pct_change'] / df['oas_index_pct_change']
        y_lambda = y_lambda.replace([np.inf, -np.inf], np.nan).fillna(1.0)  # Handle division issues

        clusters = df['date'].astype(str) + '_' + df['issuer_id'].astype(str)

        try:
            # Estimate implied lambda
            model_lambda = sm.OLS(y_lambda, X_full)
            lambda_results = model_lambda.fit(cov_type='cluster', cov_kwds={'groups': clusters})

            # Predict lambda
            lambda_hat = lambda_results.predict(X_full)

            # Now estimate main regression with predicted lambda
            df['f_unrestricted'] = lambda_hat * df['oas_index_pct_change']

            y = df['oas_pct_change'].values
            X = df['f_unrestricted'].values
            X = sm.add_constant(X)

            model = sm.OLS(y, X)
            reg_results = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

            n_params = X_full.shape[1]  # Number of parameters in lambda estimation

            return {
                'label': label,
                'beta_unrestricted': reg_results.params[1],
                'se_beta': reg_results.bse[1],
                't_stat': reg_results.tvalues[1],
                'p_value': reg_results.pvalues[1],
                'r_squared': reg_results.rsquared,
                'adj_r_squared': reg_results.rsquared_adj,
                'rmse': np.sqrt(np.mean(reg_results.resid ** 2)),
                'aic': reg_results.aic,
                'n_observations': len(y),
                'n_parameters': n_params,
                'lambda_r_squared': lambda_results.rsquared
            }

        except Exception as e:
            return {
                'label': label,
                'error': f'Regression failed: {str(e)}'
            }

    # =========================================================================
    # Model Comparison
    # =========================================================================

    def compare_models(
        self,
        stage_a_results: pd.DataFrame,
        spec_b1: Dict,
        spec_b2: Dict,
        spec_b3: Dict
    ) -> pd.DataFrame:
        """
        Compare all specifications: Stage A buckets, B.1, B.2, B.3.

        Args:
            stage_a_results: Bucket-level results from Stage A
            spec_b1: Results from Specification B.1
            spec_b2: Results from Specification B.2
            spec_b3: Results from Specification B.3

        Returns:
            DataFrame with model comparison
        """
        # Compute Stage A metrics (pooled across all buckets)
        # This is an approximation - ideally we'd re-run bucket regressions and pool
        stage_a_r2 = stage_a_results['r_squared'].mean()  # Rough approximation

        models = []

        # Stage A (bucket baseline)
        models.append({
            'Model': 'Stage A (Buckets)',
            'Description': 'Fully flexible bucket-level betas',
            'R²': stage_a_r2,
            'Adj R²': np.nan,
            'RMSE': np.nan,
            'AIC': np.nan,
            'N Parameters': len(stage_a_results),  # One beta per bucket
            'ΔR² vs Stage A': 0.0
        })

        # Spec B.1
        b1_combined = spec_b1.get('combined', {})
        if 'error' not in b1_combined:
            models.append({
                'Model': 'Spec B.1 (Merton Constrained)',
                'Description': f"β_Merton = {b1_combined['beta_merton']:.3f}",
                'R²': b1_combined['r_squared'],
                'Adj R²': b1_combined['adj_r_squared'],
                'RMSE': b1_combined['rmse'],
                'AIC': b1_combined['aic'],
                'N Parameters': 2,  # alpha + beta_merton
                'ΔR² vs Stage A': b1_combined['r_squared'] - stage_a_r2
            })

        # Spec B.2
        b2_combined = spec_b2.get('combined', {})
        if 'error' not in b2_combined:
            models.append({
                'Model': 'Spec B.2 (Decomposed)',
                'Description': f"β_T = {b2_combined['beta_T']:.3f}, β_s = {b2_combined['beta_s']:.3f}",
                'R²': b2_combined['r_squared'],
                'Adj R²': b2_combined['adj_r_squared'],
                'RMSE': b2_combined['rmse'],
                'AIC': b2_combined['aic'],
                'N Parameters': 3,  # alpha + beta_T + beta_s
                'ΔR² vs Stage A': b2_combined['r_squared'] - stage_a_r2
            })

        # Spec B.3
        b3_combined = spec_b3.get('combined', {})
        if 'error' not in b3_combined:
            models.append({
                'Model': 'Spec B.3 (Unrestricted)',
                'Description': 'Fully flexible functional form',
                'R²': b3_combined['r_squared'],
                'Adj R²': b3_combined['adj_r_squared'],
                'RMSE': b3_combined['rmse'],
                'AIC': b3_combined['aic'],
                'N Parameters': b3_combined.get('n_parameters', np.nan),
                'ΔR² vs Stage A': b3_combined['r_squared'] - stage_a_r2
            })

        return pd.DataFrame(models)

    # =========================================================================
    # Theory vs Reality Comparison
    # =========================================================================

    def create_theory_vs_reality_table(
        self,
        stage_a_results: pd.DataFrame,
        bucket_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create Theory vs Reality comparison table.

        Compares empirical betas from Stage A to theoretical Merton lambdas.

        Args:
            stage_a_results: Bucket-level betas from Stage A
            bucket_stats: Bucket statistics with Merton lambdas

        Returns:
            DataFrame with comparison
        """
        # Merge Stage A betas with bucket stats
        # Select only columns that exist in bucket_stats
        bucket_cols = ['bucket_id', 'lambda_merton']
        if 'median_maturity' in bucket_stats.columns:
            bucket_cols.append('median_maturity')
        if 'median_spread' in bucket_stats.columns:
            bucket_cols.append('median_spread')

        comparison = stage_a_results.merge(
            bucket_stats[bucket_cols],
            on='bucket_id',
            how='inner'
        )

        # Calculate ratio and deviation
        comparison['ratio'] = comparison['beta'] / comparison['lambda_merton']
        comparison['deviation'] = comparison['beta'] - comparison['lambda_merton']
        comparison['abs_deviation'] = np.abs(comparison['deviation'])
        comparison['pct_deviation'] = 100 * comparison['deviation'] / comparison['lambda_merton']

        # Flag outliers
        comparison['outlier'] = (comparison['ratio'] < 0.8) | (comparison['ratio'] > 1.2)

        # Sort by deviation
        comparison = comparison.sort_values('abs_deviation', ascending=False)

        # Return only columns that exist
        return_cols = [
            'bucket_id', 'rating_bucket', 'maturity_bucket', 'sector',
            'beta', 'lambda_merton', 'ratio', 'deviation', 'pct_deviation',
            'outlier', 'n_observations'
        ]
        if 'median_maturity' in comparison.columns:
            return_cols.insert(-1, 'median_maturity')
        if 'median_spread' in comparison.columns:
            return_cols.insert(-1, 'median_spread')

        return comparison[return_cols]

    def assess_theory_performance(self, theory_vs_reality: pd.DataFrame) -> Dict:
        """
        Assess how well Merton theory matches empirical betas.

        Args:
            theory_vs_reality: Theory vs reality comparison table

        Returns:
            Dictionary with assessment metrics
        """
        ratios = theory_vs_reality['ratio'].values

        # Count buckets in acceptable range [0.8, 1.2]
        in_range = ((ratios >= 0.8) & (ratios <= 1.2)).sum()
        pct_in_range = 100 * in_range / len(ratios)

        # Systematic bias check
        median_ratio = np.median(ratios)
        mean_ratio = np.mean(ratios)

        # Dispersion
        std_ratio = np.std(ratios)

        return {
            'n_buckets': len(ratios),
            'pct_in_acceptable_range': pct_in_range,
            'median_ratio': median_ratio,
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'min_ratio': ratios.min(),
            'max_ratio': ratios.max(),
            'systematic_bias': 'Upward' if median_ratio > 1.2 else ('Downward' if median_ratio < 0.8 else 'None'),
            'assessment': self._assess_theory_quality(pct_in_range, median_ratio, std_ratio)
        }

    def _assess_theory_quality(self, pct_in_range: float, median: float, std: float) -> str:
        """Generate overall assessment of theory performance."""
        if pct_in_range >= 90 and 0.9 <= median <= 1.1:
            return "✓ EXCELLENT: Merton provides excellent baseline"
        elif pct_in_range >= 70 and 0.8 <= median <= 1.2:
            return "✓ GOOD: Merton provides good baseline with minor calibration"
        elif median > 1.2 or median < 0.8:
            return "⚠ SYSTEMATIC BIAS: Recalibrate with β_Merton scaling factor"
        elif std > 0.3:
            return "⚠ HIGH DISPERSION: Heterogeneity beyond Merton dimensions"
        else:
            return "⚠ MIXED: Theory partially explains variation"

    # =========================================================================
    # Decision Framework
    # =========================================================================

    def generate_stage_b_decision(
        self,
        spec_b1: Dict,
        model_comparison: pd.DataFrame,
        theory_assessment: Dict
    ) -> str:
        """
        Generate Stage B decision and recommendation.

        Args:
            spec_b1: Specification B.1 results
            model_comparison: Model comparison table
            theory_assessment: Theory performance assessment

        Returns:
            Decision string
        """
        b1 = spec_b1.get('combined', {})

        if 'error' in b1:
            return "❌ ERROR: Specification B.1 failed to estimate"

        beta_merton = b1['beta_merton']
        r2_merton = b1['r_squared']
        p_value = b1['p_value_h0_beta_eq_1']

        # Get Stage A R²
        stage_a_row = model_comparison[model_comparison['Model'] == 'Stage A (Buckets)']
        r2_stage_a = stage_a_row['R²'].values[0] if len(stage_a_row) > 0 else np.nan

        r2_ratio = r2_merton / r2_stage_a if not np.isnan(r2_stage_a) and r2_stage_a > 0 else np.nan

        # Decision logic from paper
        if 0.9 <= beta_merton <= 1.1 and p_value > 0.05 and r2_ratio > 0.85:
            return (
                "✓ PATH 1: THEORY WORKS WELL\n\n"
                f"β_Merton = {beta_merton:.3f} ∈ [0.9, 1.1]\n"
                f"p-value (H0: β=1) = {p_value:.3f} > 0.05\n"
                f"R² ratio (Merton/Buckets) = {r2_ratio:.2%}\n\n"
                "RECOMMENDATION:\n"
                "• Use pure Merton tables (λ^Merton from theory)\n"
                "• Proceed to Stage C to test time-variation\n"
                "• High confidence in theoretical foundation\n"
                "• Production systems can rely on Merton structure"
            )
        elif (0.8 <= beta_merton <= 1.2) and r2_ratio > 0.80:
            return (
                "✓ PATH 2: THEORY NEEDS CALIBRATION\n\n"
                f"β_Merton = {beta_merton:.3f} outside [0.9, 1.1] but patterns match\n"
                f"R² ratio = {r2_ratio:.2%}\n\n"
                "RECOMMENDATION:\n"
                f"• Use calibrated Merton: λ^prod = {beta_merton:.3f} × λ^Merton\n"
                "• Proceed to Stage C to test stability of β_Merton over time\n"
                "• Theory has right structure, needs scaling adjustment\n"
                "• Simple one-parameter calibration suffices"
            )
        elif 0.6 <= r2_ratio <= 0.85:
            return (
                "⚠ PATH 3: THEORY CAPTURES STRUCTURE BUT MISSES DETAILS\n\n"
                f"β_Merton = {beta_merton:.3f}\n"
                f"R² ratio = {r2_ratio:.2%} (moderate)\n"
                f"Theory explains {theory_assessment['pct_in_acceptable_range']:.0f}% of buckets well\n\n"
                "RECOMMENDATION:\n"
                "• Proceed to Stage C with BOTH tracks:\n"
                "  1. Theory-guided (calibrated Merton)\n"
                "  2. Unrestricted empirical\n"
                "• Compare performance in Stage C\n"
                "• Theory provides useful prior but incomplete"
            )
        else:
            return (
                "✗ PATH 4: THEORY FUNDAMENTALLY FAILS\n\n"
                f"β_Merton = {beta_merton:.3f}\n"
                f"R² ratio = {r2_ratio:.2%} (< 50%)\n"
                f"Wrong patterns or low explanatory power\n\n"
                "RECOMMENDATION:\n"
                "• SKIP Stage C (no point testing time-variation of failed model)\n"
                "• Proceed to Stage D (robustness) to diagnose WHY theory fails\n"
                "• Then Stage E with unrestricted specification only\n"
                "• Report: Structural models inadequate for this market"
            )
