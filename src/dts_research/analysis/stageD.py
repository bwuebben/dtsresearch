"""
Stage D analysis: Robustness and Extensions.

Implements quantile regression, shock decomposition, and liquidity adjustment.

Integration with Stage 0:
- Skips theory-driven tests if Path 5 (theory fails)
- Focuses on model-free robustness for Path 5
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.quantile_regression import QuantReg


class StageDAnalysis:
    """
    Implements Stage D: Robustness and Extensions.

    Tests robustness across:
    1. Tail events (quantile regression)
    2. Shock types (systematic vs idiosyncratic)
    3. Spread components (default vs liquidity)

    Integrates with Stage 0 to focus on model-free robustness if theory fails.
    """

    def __init__(self, stage0_results: Optional[Dict] = None):
        """
        Initialize Stage D analysis.

        Args:
            stage0_results: Optional Stage 0 results dictionary
        """
        from ..models.merton import MertonLambdaCalculator
        self.merton_calc = MertonLambdaCalculator()
        self.stage0_results = stage0_results
        self.stage0_path = stage0_results.get('decision_path') if stage0_results else None

    def should_focus_model_free(self) -> Tuple[bool, str]:
        """
        Determine if Stage D should focus on model-free robustness.

        Returns:
            (focus_model_free, reason) tuple
        """
        if self.stage0_path is None:
            return False, "No Stage 0 results available"

        if self.stage0_path == 5:
            return True, (
                "Stage 0 Decision Path 5: Theory Fails\n"
                "Stage D should focus on model-free robustness checks.\n"
                "Skip Merton-specific tests, emphasize quantile regression and shock decomposition."
            )

        return False, f"Stage 0 Path {self.stage0_path}: Run full Stage D (theory + robustness)"

    # =========================================================================
    # D.1: TAIL BEHAVIOR (QUANTILE REGRESSION)
    # =========================================================================

    def quantile_regression_analysis(
        self,
        df: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        by_regime: bool = True
    ) -> Dict:
        """
        D.1: Estimate beta_tau for different quantiles of spread changes.

        For quantiles tau in {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}:
            Q_tau(y_i,t | f_DTS) = alpha_tau + beta_tau * [lambda^Merton * f_DTS]

        Args:
            df: Regression data with columns:
                - oas_pct_change (y_i,t)
                - oas_index_pct_change (f_DTS,t)
                - time_to_maturity
                - oas (spread level)
                - spread_regime (IG/HY)
            quantiles: List of quantiles to estimate (default: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
            by_regime: If True, run separately for IG and HY

        Returns:
            Dictionary with:
                - results_combined: DataFrame with beta_tau estimates
                - results_ig: DataFrame (if by_regime=True)
                - results_hy: DataFrame (if by_regime=True)
                - tail_tests: Statistical tests for tail amplification
                - interpretation: Text summary
        """
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

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
        results['results_combined'] = self._estimate_quantile_betas(
            df, quantiles
        )

        # By regime
        if by_regime:
            df_ig = df[df['spread_regime'] == 'IG'].copy()
            df_hy = df[df['spread_regime'] == 'HY'].copy()

            if len(df_ig) > 100:
                results['results_ig'] = self._estimate_quantile_betas(
                    df_ig, quantiles
                )

            if len(df_hy) > 100:
                results['results_hy'] = self._estimate_quantile_betas(
                    df_hy, quantiles
                )

        # Tail tests
        results['tail_tests'] = self._test_tail_amplification(
            results['results_combined']
        )

        # Interpretation
        results['interpretation'] = self._interpret_quantile_results(
            results['results_combined'],
            results['tail_tests']
        )

        return results

    def _estimate_quantile_betas(
        self,
        df: pd.DataFrame,
        quantiles: List[float]
    ) -> pd.DataFrame:
        """
        Estimate beta_tau for each quantile.

        Args:
            df: Data with oas_pct_change, f_merton
            quantiles: List of quantiles

        Returns:
            DataFrame with tau, beta_tau, se_beta, ci_lower, ci_upper
        """
        y = df['oas_pct_change'].values
        X = df['f_merton'].values
        X_with_const = sm.add_constant(X)

        results_list = []

        for tau in quantiles:
            try:
                model = QuantReg(y, X_with_const)
                result = model.fit(q=tau)

                beta_tau = result.params[1]
                se_beta = result.bse[1]
                ci_lower, ci_upper = result.conf_int()[1, :]

                results_list.append({
                    'quantile': tau,
                    'beta_tau': beta_tau,
                    'se_beta': se_beta,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    't_stat': beta_tau / se_beta if se_beta > 0 else np.nan
                })
            except:
                # Quantile regression can fail for extreme quantiles
                results_list.append({
                    'quantile': tau,
                    'beta_tau': np.nan,
                    'se_beta': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    't_stat': np.nan
                })

        return pd.DataFrame(results_list)

    def _test_tail_amplification(self, quantile_results: pd.DataFrame) -> Dict:
        """
        Test if left tail (0.05) differs from median (0.50).

        Args:
            quantile_results: DataFrame from _estimate_quantile_betas

        Returns:
            Dictionary with test results
        """
        beta_05 = quantile_results[quantile_results['quantile'] == 0.05]['beta_tau'].values[0]
        beta_50 = quantile_results[quantile_results['quantile'] == 0.50]['beta_tau'].values[0]
        beta_95 = quantile_results[quantile_results['quantile'] == 0.95]['beta_tau'].values[0]

        se_05 = quantile_results[quantile_results['quantile'] == 0.05]['se_beta'].values[0]
        se_50 = quantile_results[quantile_results['quantile'] == 0.50]['se_beta'].values[0]
        se_95 = quantile_results[quantile_results['quantile'] == 0.95]['se_beta'].values[0]

        # Test: beta_05 = beta_50
        diff_left = beta_05 - beta_50
        se_diff_left = np.sqrt(se_05**2 + se_50**2)
        t_stat_left = diff_left / se_diff_left if se_diff_left > 0 else np.nan
        p_val_left = 2 * (1 - stats.t.cdf(abs(t_stat_left), df=1000)) if not np.isnan(t_stat_left) else np.nan

        # Test: beta_95 = beta_50
        diff_right = beta_95 - beta_50
        se_diff_right = np.sqrt(se_95**2 + se_50**2)
        t_stat_right = diff_right / se_diff_right if se_diff_right > 0 else np.nan
        p_val_right = 2 * (1 - stats.t.cdf(abs(t_stat_right), df=1000)) if not np.isnan(t_stat_right) else np.nan

        # Amplification ratios
        ratio_left = beta_05 / beta_50 if beta_50 != 0 else np.nan
        ratio_right = beta_95 / beta_50 if beta_50 != 0 else np.nan

        return {
            'beta_05': beta_05,
            'beta_50': beta_50,
            'beta_95': beta_95,
            'diff_left_tail': diff_left,
            'p_value_left': p_val_left,
            'diff_right_tail': diff_right,
            'p_value_right': p_val_right,
            'amplification_left': ratio_left,
            'amplification_right': ratio_right,
            'pattern': self._classify_tail_pattern(ratio_left, ratio_right, p_val_left, p_val_right)
        }

    def _classify_tail_pattern(
        self,
        ratio_left: float,
        ratio_right: float,
        p_left: float,
        p_right: float
    ) -> str:
        """Classify the tail pattern based on amplification ratios."""
        if p_left > 0.05 and p_right > 0.05:
            return "Symmetric (no tail amplification)"
        elif ratio_left > 1.2 and p_left < 0.05:
            if ratio_right > 1.2 and p_right < 0.05:
                return "U-shaped (both tails amplified)"
            else:
                return "Left tail amplified (jump-to-default risk)"
        elif ratio_right > 1.2 and p_right < 0.05:
            return "Right tail amplified (momentum in rallies)"
        else:
            return "Weak asymmetry"

    def _interpret_quantile_results(
        self,
        quantile_results: pd.DataFrame,
        tail_tests: Dict
    ) -> str:
        """Generate interpretation of quantile regression results."""
        pattern = tail_tests['pattern']
        ratio_left = tail_tests['amplification_left']

        interpretation = f"""
Quantile Regression Analysis:

Pattern: {pattern}

Left tail (5th percentile):
  - β_0.05 = {tail_tests['beta_05']:.3f}
  - β_0.50 = {tail_tests['beta_50']:.3f}
  - Amplification ratio: {ratio_left:.2f}x
  - p-value: {tail_tests['p_value_left']:.4f}

Right tail (95th percentile):
  - β_0.95 = {tail_tests['beta_95']:.3f}
  - Amplification ratio: {tail_tests['amplification_right']:.2f}x
  - p-value: {tail_tests['p_value_right']:.4f}

Interpretation:
"""

        if "no tail" in pattern.lower():
            interpretation += "  ✓ Merton works uniformly across distribution\n"
            interpretation += "  → Use standard lambda for VaR/ES calculations\n"
        elif "left tail" in pattern.lower():
            interpretation += "  ⚠ Left tail shows amplified sensitivity\n"
            interpretation += f"  → Tail risk {(ratio_left-1)*100:.0f}% larger than mean-based models\n"
            interpretation += "  → Use β_0.05 × lambda for stress testing\n"
        elif "u-shaped" in pattern.lower():
            interpretation += "  ⚠ Both tails deviate from median\n"
            interpretation += "  → Extreme moves behave differently (non-linearity)\n"
            interpretation += "  → Consider quantile-specific lambdas for risk models\n"

        return interpretation

    # =========================================================================
    # D.2: SHOCK DECOMPOSITION
    # =========================================================================

    def shock_decomposition_analysis(
        self,
        df: pd.DataFrame,
        by_regime: bool = True
    ) -> Dict:
        """
        D.2: Decompose spread changes into Global, Sector, and Issuer-specific components.

        Tests if different shock types exhibit different elasticities.

        Args:
            df: Regression data with columns:
                - oas_pct_change
                - oas_index_pct_change (global factor)
                - sector
                - bond_id
                - date
            by_regime: If True, run separately for IG and HY

        Returns:
            Dictionary with:
                - factors: DataFrame with decomposed factors
                - variance_decomp: Variance decomposition by bucket
                - shock_betas: Beta estimates for each shock type
                - shock_betas_ig: IG results (if by_regime)
                - shock_betas_hy: HY results (if by_regime)
                - interpretation: Text summary
        """
        df = df.copy()

        # Step 1: Construct factors
        factors_df = self._construct_shock_factors(df)

        # Step 2: Variance decomposition
        variance_decomp = self._variance_decomposition(factors_df)

        # Step 3: Estimate shock-specific betas
        shock_betas_combined = self._estimate_shock_betas(factors_df)

        results = {
            'factors': factors_df,
            'variance_decomp': variance_decomp,
            'shock_betas_combined': shock_betas_combined
        }

        # By regime
        if by_regime:
            factors_ig = factors_df[factors_df['spread_regime'] == 'IG'].copy()
            factors_hy = factors_df[factors_df['spread_regime'] == 'HY'].copy()

            if len(factors_ig) > 100:
                results['shock_betas_ig'] = self._estimate_shock_betas(factors_ig)

            if len(factors_hy) > 100:
                results['shock_betas_hy'] = self._estimate_shock_betas(factors_hy)

        # Interpretation
        results['interpretation'] = self._interpret_shock_decomposition(
            shock_betas_combined,
            variance_decomp
        )

        return results

    def _construct_shock_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose spread changes into orthogonal factors.

        Global -> Sector (orthogonalized) -> Issuer-specific (residual)
        """
        df = df.copy()

        # Global factor (already have)
        df['f_global'] = df['oas_index_pct_change']

        # Sector factors (orthogonalized to global)
        # For each sector, compute average spread change
        sector_means = df.groupby(['date', 'sector'])['oas_pct_change'].mean().reset_index()
        sector_means.rename(columns={'oas_pct_change': 'sector_avg'}, inplace=True)

        df = df.merge(sector_means, on=['date', 'sector'], how='left')

        # Orthogonalize: sector factor = sector_avg - global
        df['f_sector'] = df['sector_avg'] - df['f_global']

        # Issuer-specific (residual)
        df['f_issuer'] = df['oas_pct_change'] - df['f_global'] - df['f_sector']

        return df

    def _variance_decomposition(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute variance decomposition by bucket.

        Shows % variance from Global, Sector, Issuer-specific components.
        """
        # Total variance of oas_pct_change
        total_var = factors_df['oas_pct_change'].var()

        # Component variances
        var_global = factors_df['f_global'].var()
        var_sector = factors_df['f_sector'].var()
        var_issuer = factors_df['f_issuer'].var()

        # Residual
        var_residual = total_var - (var_global + var_sector + var_issuer)

        decomp = {
            'Component': ['Global', 'Sector', 'Issuer-specific', 'Residual'],
            'Variance': [var_global, var_sector, var_issuer, max(0, var_residual)],
            'Pct_of_Total': [
                100 * var_global / total_var,
                100 * var_sector / total_var,
                100 * var_issuer / total_var,
                100 * max(0, var_residual) / total_var
            ]
        }

        return pd.DataFrame(decomp)

    def _estimate_shock_betas(self, factors_df: pd.DataFrame) -> Dict:
        """
        Estimate beta for each shock type.

        y_i,t = beta_G * [lambda * f_global] + beta_S * [lambda * f_sector] +
                beta_I * [lambda * f_issuer] + epsilon
        """
        # Add lambda
        factors_df = factors_df.copy()
        factors_df['lambda_merton'] = factors_df.apply(
            lambda row: self.merton_calc.lambda_combined(
                row['time_to_maturity'], row['oas']
            ),
            axis=1
        )

        # Construct regressors
        factors_df['f_global_weighted'] = factors_df['lambda_merton'] * factors_df['f_global']
        factors_df['f_sector_weighted'] = factors_df['lambda_merton'] * factors_df['f_sector']
        factors_df['f_issuer_weighted'] = factors_df['lambda_merton'] * factors_df['f_issuer']

        # Regression
        y = factors_df['oas_pct_change'].values
        X = factors_df[['f_global_weighted', 'f_sector_weighted', 'f_issuer_weighted']].values
        X_with_const = sm.add_constant(X)

        # Cluster by week
        factors_df['week'] = pd.to_datetime(factors_df['date']).dt.to_period('W')
        clusters = factors_df['week']

        model = OLS(y, X_with_const)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

        return {
            'beta_global': result.params[1],
            'beta_sector': result.params[2],
            'beta_issuer': result.params[3],
            'se_global': result.bse[1],
            'se_sector': result.bse[2],
            'se_issuer': result.bse[3],
            't_global': result.tvalues[1],
            't_sector': result.tvalues[2],
            't_issuer': result.tvalues[3],
            'p_global': result.pvalues[1],
            'p_sector': result.pvalues[2],
            'p_issuer': result.pvalues[3],
            'r_squared': result.rsquared,
            'n_obs': len(y)
        }

    def _interpret_shock_decomposition(
        self,
        shock_betas: Dict,
        variance_decomp: pd.DataFrame
    ) -> str:
        """Generate interpretation of shock decomposition results."""
        beta_g = shock_betas['beta_global']
        beta_s = shock_betas['beta_sector']
        beta_i = shock_betas['beta_issuer']

        var_pcts = dict(zip(variance_decomp['Component'], variance_decomp['Pct_of_Total']))

        interpretation = f"""
Shock Decomposition Analysis:

Variance Decomposition:
  - Global shocks: {var_pcts.get('Global', 0):.1f}%
  - Sector shocks: {var_pcts.get('Sector', 0):.1f}%
  - Issuer-specific: {var_pcts.get('Issuer-specific', 0):.1f}%

Shock-Specific Elasticities:
  - β^(G) (Global) = {beta_g:.3f} (p = {shock_betas['p_global']:.4f})
  - β^(S) (Sector) = {beta_s:.3f} (p = {shock_betas['p_sector']:.4f})
  - β^(I) (Issuer) = {beta_i:.3f} (p = {shock_betas['p_issuer']:.4f})

Merton Prediction: All β ≈ 1 (shocks operate through firm value)

Assessment:
"""

        # Check if all betas ≈ 1
        all_near_one = all(0.9 <= b <= 1.1 for b in [beta_g, beta_s, beta_i])

        if all_near_one:
            interpretation += "  ✓ All shock types have β ≈ 1\n"
            interpretation += "  → Merton applies uniformly across shock types\n"
        else:
            if beta_s > 1.2:
                interpretation += f"  ⚠ Sector shocks amplified ({beta_s:.2f}x)\n"
                interpretation += "  → Suggests contagion or common liquidity factors\n"

            if beta_i > 1.2:
                interpretation += f"  ⚠ Issuer-specific shocks amplified ({beta_i:.2f}x)\n"
                interpretation += "  → Information asymmetry or forced selling\n"

            if beta_g < 1 and beta_i > 1:
                interpretation += "  ⚠ Under-react to macro, over-react to idiosyncratic\n"

        return interpretation

    # =========================================================================
    # D.3: LIQUIDITY ADJUSTMENT
    # =========================================================================

    def liquidity_adjustment_analysis(
        self,
        df: pd.DataFrame,
        liquidity_vars: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        D.3: Decompose OAS into default and liquidity components.

        Tests if Merton works better on default component than total OAS.

        Args:
            df: Regression data with liquidity variables:
                - bid_ask (or proxy)
                - issue_size
                - turnover (or proxy)
                - age
            liquidity_vars: Dictionary mapping variable names

        Returns:
            Dictionary with:
                - liquidity_model: Cross-sectional regression results
                - default_betas: Beta estimates on default component
                - comparison: Total vs default component comparison
                - by_liquidity_quartile: Results by liquidity level
                - interpretation: Text summary
        """
        # For mock data, create synthetic liquidity variables if not present
        df = self._add_mock_liquidity_vars(df)

        # Step 1: Estimate liquidity component
        liquidity_model = self._estimate_liquidity_component(df)

        # Step 2: Compute default component
        df_with_default = self._compute_default_component(df, liquidity_model)

        # Step 3: Re-estimate Merton on default component
        default_betas = self._estimate_on_default_component(df_with_default)

        # Step 4: Compare total vs default
        comparison = self._compare_total_vs_default(df_with_default, default_betas)

        # Step 5: By liquidity quartile
        by_quartile = self._analysis_by_liquidity_quartile(df_with_default)

        results = {
            'liquidity_model': liquidity_model,
            'default_betas': default_betas,
            'comparison': comparison,
            'by_liquidity_quartile': by_quartile,
            'interpretation': self._interpret_liquidity_adjustment(
                liquidity_model,
                default_betas,
                comparison
            )
        }

        return results

    def _add_mock_liquidity_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic liquidity variables for testing."""
        df = df.copy()

        if 'bid_ask' not in df.columns:
            # Bid-ask wider for HY, smaller size, higher spreads
            np.random.seed(42)
            base_ba = np.where(df['spread_regime'] == 'IG', 30, 150)
            df['bid_ask'] = base_ba + np.random.normal(0, 10, len(df))
            df['bid_ask'] = np.maximum(df['bid_ask'], 5)

        if 'issue_size' not in df.columns:
            # Issue size (in billions)
            df['issue_size'] = np.random.lognormal(0, 1, len(df))

        if 'turnover' not in df.columns:
            # Turnover (higher for IG)
            base_turnover = np.where(df['spread_regime'] == 'IG', 0.5, 0.2)
            df['turnover'] = base_turnover + np.random.normal(0, 0.1, len(df))
            df['turnover'] = np.maximum(df['turnover'], 0.01)

        if 'age' not in df.columns:
            # Age in years
            df['age'] = np.random.uniform(0, 10, len(df))

        return df

    def _estimate_liquidity_component(self, df: pd.DataFrame) -> Dict:
        """
        Cross-sectional regression:
        s^liq = phi_0 + phi_1*BidAsk + phi_2*log(Size) + phi_3*log(Turnover) + phi_4*Age
        """
        # Prepare data
        df_liq = df[['oas', 'bid_ask', 'issue_size', 'turnover', 'age']].dropna()

        y = df_liq['oas'].values
        X = pd.DataFrame({
            'const': 1,
            'bid_ask': df_liq['bid_ask'].values,
            'log_size': np.log(df_liq['issue_size'].values),
            'log_turnover': np.log(df_liq['turnover'].values),
            'age': df_liq['age'].values
        })

        model = OLS(y, X)
        result = model.fit()

        return {
            'phi_0': result.params['const'],
            'phi_bid_ask': result.params['bid_ask'],
            'phi_log_size': result.params['log_size'],
            'phi_log_turnover': result.params['log_turnover'],
            'phi_age': result.params['age'],
            'r_squared': result.rsquared,
            'n_obs': len(y)
        }

    def _compute_default_component(
        self,
        df: pd.DataFrame,
        liquidity_model: Dict
    ) -> pd.DataFrame:
        """Compute OAS_def = OAS - OAS_liq."""
        df = df.copy()

        # Predict liquidity component
        df['oas_liq'] = (
            liquidity_model['phi_0'] +
            liquidity_model['phi_bid_ask'] * df['bid_ask'] +
            liquidity_model['phi_log_size'] * np.log(df['issue_size']) +
            liquidity_model['phi_log_turnover'] * np.log(df['turnover']) +
            liquidity_model['phi_age'] * df['age']
        )

        # Default component
        df['oas_def'] = df['oas'] - df['oas_liq']
        df['oas_def'] = np.maximum(df['oas_def'], 10)  # Floor at 10 bps

        # Compute default-based spread changes
        df = df.sort_values(['bond_id', 'date'])
        df['oas_def_lag'] = df.groupby('bond_id')['oas_def'].shift(1)
        df['oas_def_pct_change'] = (df['oas_def'] - df['oas_def_lag']) / df['oas_def_lag'] * 100

        return df

    def _estimate_on_default_component(self, df: pd.DataFrame) -> Dict:
        """Re-run Stage B regression on default component."""
        df_clean = df.dropna(subset=['oas_def_pct_change'])

        # Add lambda
        df_clean = df_clean.copy()
        df_clean['lambda_merton'] = df_clean.apply(
            lambda row: self.merton_calc.lambda_combined(
                row['time_to_maturity'], row['oas_def']  # Use default OAS for lambda
            ),
            axis=1
        )

        # Index-level default factor (simplified - use total index for now)
        df_clean['f_def'] = df_clean['oas_index_pct_change']  # Approximation

        # Regression
        y = df_clean['oas_def_pct_change'].values
        X = (df_clean['lambda_merton'] * df_clean['f_def']).values
        X_with_const = sm.add_constant(X)

        # Cluster by week
        df_clean['week'] = pd.to_datetime(df_clean['date']).dt.to_period('W')
        clusters = df_clean['week']

        model = OLS(y, X_with_const)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

        return {
            'beta_def': result.params[1],
            'se_beta': result.bse[1],
            'r_squared': result.rsquared,
            'rmse': np.sqrt(result.mse_resid),
            'n_obs': len(y)
        }

    def _compare_total_vs_default(
        self,
        df: pd.DataFrame,
        default_betas: Dict
    ) -> Dict:
        """Compare Merton fit on total OAS vs default component."""
        # Estimate on total OAS for comparison
        df_clean = df.dropna(subset=['oas_pct_change'])
        df_clean = df_clean.copy()

        df_clean['lambda_merton'] = df_clean.apply(
            lambda row: self.merton_calc.lambda_combined(
                row['time_to_maturity'], row['oas']
            ),
            axis=1
        )

        y = df_clean['oas_pct_change'].values
        X = (df_clean['lambda_merton'] * df_clean['oas_index_pct_change']).values
        X_with_const = sm.add_constant(X)

        df_clean['week'] = pd.to_datetime(df_clean['date']).dt.to_period('W')
        clusters = df_clean['week']

        model = OLS(y, X_with_const)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

        beta_total = result.params[1]
        r2_total = result.rsquared

        beta_def = default_betas['beta_def']
        r2_def = default_betas['r_squared']

        return {
            'beta_total': beta_total,
            'r2_total': r2_total,
            'beta_def': beta_def,
            'r2_def': r2_def,
            'delta_r2': r2_def - r2_total,
            'improvement_pct': 100 * (r2_def - r2_total) / r2_total if r2_total > 0 else 0,
            'beta_improvement': abs(beta_def - 1) < abs(beta_total - 1)
        }

    def _analysis_by_liquidity_quartile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split by liquidity quartiles and compare."""
        df = df.copy()
        df['liq_quartile'] = pd.qcut(df['bid_ask'], q=4, labels=['Q1 (Liquid)', 'Q2', 'Q3', 'Q4 (Illiquid)'])

        results_list = []

        for quartile in df['liq_quartile'].unique():
            df_q = df[df['liq_quartile'] == quartile].copy()

            if len(df_q) < 50:
                continue

            # Estimate on total
            df_q['lambda_merton'] = df_q.apply(
                lambda row: self.merton_calc.lambda_combined(row['time_to_maturity'], row['oas']),
                axis=1
            )

            # Drop NaNs from both y and X together to keep them aligned
            df_total = df_q[['oas_pct_change', 'lambda_merton', 'oas_index_pct_change']].dropna()
            if len(df_total) < 50:
                continue

            y_total = df_total['oas_pct_change'].values
            X_total = (df_total['lambda_merton'] * df_total['oas_index_pct_change']).values
            X_total_const = sm.add_constant(X_total)
            model_total = OLS(y_total, X_total_const).fit()

            # Estimate on default
            df_def = df_q[['oas_def_pct_change', 'lambda_merton', 'oas_index_pct_change']].dropna()
            if len(df_def) < 50:
                continue

            y_def = df_def['oas_def_pct_change'].values
            X_def = (df_def['lambda_merton'] * df_def['oas_index_pct_change']).values
            X_def_const = sm.add_constant(X_def)
            model_def = OLS(y_def, X_def_const).fit()

            results_list.append({
                'Quartile': quartile,
                'Avg_BidAsk': df_q['bid_ask'].mean(),
                'beta_total': model_total.params[1],
                'r2_total': model_total.rsquared,
                'beta_def': model_def.params[1],
                'r2_def': model_def.rsquared,
                'delta_r2': model_def.rsquared - model_total.rsquared
            })

        return pd.DataFrame(results_list)

    def _interpret_liquidity_adjustment(
        self,
        liquidity_model: Dict,
        default_betas: Dict,
        comparison: Dict
    ) -> str:
        """Generate interpretation of liquidity adjustment results."""
        r2_liq = liquidity_model['r_squared']
        beta_total = comparison['beta_total']
        beta_def = comparison['beta_def']
        delta_r2 = comparison['delta_r2']

        interpretation = f"""
Liquidity Adjustment Analysis:

Liquidity Model (Cross-sectional):
  - R² = {r2_liq:.3f}
  - Coefficients: Bid-ask {liquidity_model['phi_bid_ask']:.3f}, Size {liquidity_model['phi_log_size']:.2f}

Merton Fit Comparison:
  - Total OAS: β = {beta_total:.3f}, R² = {comparison['r2_total']:.3f}
  - Default component: β = {beta_def:.3f}, R² = {comparison['r2_def']:.3f}
  - ΔR² = {delta_r2:.3f} ({comparison['improvement_pct']:.1f}% improvement)

Assessment:
"""

        if abs(delta_r2) < 0.02:
            interpretation += "  → Liquidity adjustment has minimal impact\n"
            interpretation += "  → Use total OAS (simpler)\n"
        elif delta_r2 > 0.05:
            interpretation += "  ✓ Significant improvement with liquidity adjustment\n"
            interpretation += "  → Decompose OAS for HY and illiquid bonds\n"
        elif beta_def > beta_total and abs(beta_def - 1) < abs(beta_total - 1):
            interpretation += "  ✓ Default component closer to Merton prediction\n"
            interpretation += "  → Consider liquidity adjustment for precision\n"
        else:
            interpretation += "  ○ Marginal benefit from liquidity adjustment\n"

        return interpretation
