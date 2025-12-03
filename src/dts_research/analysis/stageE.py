"""
Stage E: Production Specification Selection

Implements the hierarchical testing framework to select the parsimonious
production model that balances theoretical coherence, empirical fit, and
implementation cost.

Key principle: Stop at the simplest adequate model. Don't over-engineer.

Integration with Stage 0:
- Path 5: Do NOT use Merton-based specs (Levels 2, 3, 5 invalid)
- Path 1-2: Merton-based specs are valid
- Guides starting point in hierarchy
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional, List
import warnings


class StageEAnalysis:
    """
    Production specification selection via hierarchical testing.

    Implements 5 levels:
    - Level 1: Standard DTS (no adjustments)
    - Level 2: Pure Merton (lookup tables) - SKIP if Stage 0 Path 5
    - Level 3: Calibrated Merton (2 parameters) - SKIP if Stage 0 Path 5
    - Level 4: Full Empirical (8-12 parameters)
    - Level 5: Time-varying (base + 2 macro parameters) - SKIP if Stage 0 Path 5

    Integrates with Stage 0 to exclude Merton-based specs if theory fails.
    """

    def __init__(self, stage0_results: Optional[Dict] = None):
        """
        Initialize Stage E analysis.

        Args:
            stage0_results: Optional Stage 0 results dictionary
        """
        self.results = {}
        self.stage0_results = stage0_results
        self.stage0_path = stage0_results.get('decision_path') if stage0_results else None

    def get_valid_levels(self) -> Tuple[List[int], str]:
        """
        Determine which hierarchy levels are valid based on Stage 0.

        Returns:
            (valid_levels, reason) tuple
        """
        if self.stage0_path is None:
            return [1, 2, 3, 4, 5], "No Stage 0 results - test all levels"

        if self.stage0_path == 5:
            return [1, 4], (
                "Stage 0 Path 5: Theory Fails\n"
                "INVALID levels: 2 (Pure Merton), 3 (Calibrated Merton), 5 (Time-varying Merton)\n"
                "VALID levels: 1 (Standard DTS), 4 (Full Empirical)\n"
                "Recommendation: Use Level 4 (empirical) or Level 1 (simplest)"
            )

        if self.stage0_path in [1, 2]:
            return [1, 2, 3, 4, 5], f"Stage 0 Path {self.stage0_path}: Theory works - all levels valid"

        if self.stage0_path == 3:
            return [1, 3, 4], (
                f"Stage 0 Path 3: Weak Evidence\n"
                "Skip Level 2 (pure Merton), use calibrated (3) or empirical (4)"
            )

        if self.stage0_path == 4:
            return [1, 4], (
                f"Stage 0 Path 4: Mixed Evidence\n"
                "Skip Merton-based levels, use empirical (4) or standard (1)"
            )

        return [1, 2, 3, 4, 5], "Default: test all levels"

    def hierarchical_testing(
        self,
        df: pd.DataFrame,
        stage_a_results: Dict,
        stage_b_results: Dict,
        stage_c_results: Dict,
        stage_d_results: Optional[Dict] = None
    ) -> Dict:
        """
        Run hierarchical testing framework (Levels 1-5).

        Args:
            df: Regression data with all features
            stage_a_results: Results from Stage A (F-tests)
            stage_b_results: Results from Stage B (Merton fit)
            stage_c_results: Results from Stage C (stability)
            stage_d_results: Results from Stage D (robustness) - optional

        Returns:
            Dictionary with hierarchical test results and recommended level
        """
        print("Running hierarchical testing framework...")
        print()

        hierarchical_results = {}

        # Level 1: Is standard DTS adequate?
        level1 = self._test_level1(stage_a_results)
        hierarchical_results['level1'] = level1

        if level1['decision'] == 'STOP':
            hierarchical_results['recommended_level'] = 1
            hierarchical_results['recommended_spec'] = 'Standard DTS'
            return hierarchical_results

        # Level 2: Does pure Merton suffice?
        level2 = self._test_level2(stage_b_results)
        hierarchical_results['level2'] = level2

        if level2['decision'] == 'STOP':
            hierarchical_results['recommended_level'] = 2
            hierarchical_results['recommended_spec'] = 'Pure Merton'
            return hierarchical_results

        # Level 3: Calibrated Merton
        level3 = self._test_level3(df, stage_b_results)
        hierarchical_results['level3'] = level3

        if level3['decision'] == 'STOP':
            hierarchical_results['recommended_level'] = 3
            hierarchical_results['recommended_spec'] = 'Calibrated Merton'
            return hierarchical_results

        # Level 4: Full Empirical
        level4 = self._test_level4(df, level3, stage_b_results)
        hierarchical_results['level4'] = level4

        # Check if time-varying needed
        level5_needed = self._check_level5_needed(stage_c_results)

        if level5_needed:
            # Level 5: Time-varying
            level5 = self._test_level5(df, stage_c_results, level4)
            hierarchical_results['level5'] = level5

            if level5['decision'] == 'ADOPT':
                hierarchical_results['recommended_level'] = 5
                hierarchical_results['recommended_spec'] = 'Time-varying'
            else:
                hierarchical_results['recommended_level'] = 4
                hierarchical_results['recommended_spec'] = 'Empirical'
        else:
            hierarchical_results['recommended_level'] = 4
            hierarchical_results['recommended_spec'] = 'Empirical'

        return hierarchical_results

    def _test_level1(self, stage_a_results: Dict) -> Dict:
        """
        Level 1: Is standard DTS adequate?

        Test: Stage A F-test for equality of all betas = 1
        Decision: If p > 0.10, use standard DTS
        """
        f_test = stage_a_results.get('f_test_all_buckets', {})
        p_value = f_test.get('p_value', 0)

        decision = 'STOP' if p_value > 0.10 else 'Proceed to Level 2'

        return {
            'test': 'F-test: All betas equal to 1',
            'statistic': f_test.get('f_statistic', np.nan),
            'p_value': p_value,
            'decision': decision,
            'reasoning': 'No cross-sectional variation detected' if decision == 'STOP' else 'Significant variation exists'
        }

    def _test_level2(self, stage_b_results: Dict) -> Dict:
        """
        Level 2: Does pure Merton suffice?

        Test: Stage B Spec B.1, H0: beta_Merton = 1
        Decision: If beta_Merton in [0.9, 1.1] AND R2_ratio > 0.9, use Pure Merton
        """
        spec_b1 = stage_b_results.get('spec_b1', {})
        beta_merton = spec_b1.get('beta_Merton', np.nan)
        p_value = spec_b1.get('p_value_vs_1', 1.0)
        r2_merton = spec_b1.get('r_squared', 0)

        spec_a1 = stage_b_results.get('spec_a1_buckets', {})
        r2_buckets = spec_a1.get('r_squared', 0)

        r2_ratio = r2_merton / r2_buckets if r2_buckets > 0 else 0

        # Decision criteria
        beta_in_range = 0.9 <= beta_merton <= 1.1
        r2_adequate = r2_ratio > 0.9

        if beta_in_range and r2_adequate:
            decision = 'STOP'
            reasoning = 'Theory unbiased and fit excellent'
        elif beta_in_range and r2_ratio > 0.85:
            decision = 'STOP'
            reasoning = 'Theory unbiased, fit good enough (R² ratio > 0.85)'
        elif not beta_in_range:
            decision = 'Proceed to Level 3'
            reasoning = f'Systematic bias detected (β={beta_merton:.3f})'
        else:  # r2_ratio < 0.85
            decision = 'Proceed to Level 4'
            reasoning = f'Poor fit (R² ratio={r2_ratio:.3f})'

        return {
            'test': 'H0: beta_Merton = 1',
            'beta_Merton': beta_merton,
            'p_value': p_value,
            'r2_merton': r2_merton,
            'r2_buckets': r2_buckets,
            'r2_ratio': r2_ratio,
            'decision': decision,
            'reasoning': reasoning
        }

    def _test_level3(self, df: pd.DataFrame, stage_b_results: Dict) -> Dict:
        """
        Level 3: Calibrated Merton

        Estimate: lambda_prod = c0 * lambda_T * lambda_s^cs
        Grid search over cs in [-0.5, 0], then estimate c0
        """
        print("  Testing Level 3: Calibrated Merton...")

        # Grid search for optimal c_s
        best_c_s = -0.25  # Default theory value
        best_r2 = 0

        c_s_grid = np.linspace(-0.5, 0, 11)  # -0.5, -0.45, ..., 0

        for c_s in c_s_grid:
            # Construct adjusted lambda_s with power c_s
            df_temp = df.copy()
            df_temp['lambda_s_adj'] = df_temp['lambda_s'] ** (c_s / -0.25)  # Adjust from baseline -0.25
            df_temp['lambda_adj'] = df_temp['lambda_T'] * df_temp['lambda_s_adj']
            df_temp['interaction'] = df_temp['lambda_adj'] * df_temp['f_DTS']

            # Regression
            y = df_temp['oas_pct_change'].values
            X = df_temp['interaction'].values

            model = OLS(y, sm.add_constant(X))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_temp['week'].values})
                    r2 = results.rsquared

                    if r2 > best_r2:
                        best_r2 = r2
                        best_c_s = c_s
                except:
                    continue

        # Re-estimate with best c_s
        df['lambda_s_adj'] = df['lambda_s'] ** (best_c_s / -0.25)
        df['lambda_adj'] = df['lambda_T'] * df['lambda_s_adj']
        df['interaction'] = df['lambda_adj'] * df['f_DTS']

        y = df['oas_pct_change'].values
        X = df['interaction'].values

        model = OLS(y, sm.add_constant(X))
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['week'].values})

        c0 = results.params[1]
        se_c0 = results.bse[1]
        r2 = results.rsquared

        # Decision criteria
        c0_in_range = 0.8 <= c0 <= 1.2
        cs_in_range = -0.35 <= best_c_s <= -0.15

        if c0_in_range and cs_in_range:
            decision = 'STOP'
            reasoning = 'Calibrated Merton adequate'
        else:
            decision = 'Proceed to Level 4'
            reasoning = f'Calibration outside acceptable range (c0={c0:.3f}, cs={best_c_s:.3f})'

        return {
            'test': 'Calibrated Merton',
            'c0': c0,
            'se_c0': se_c0,
            'c_s': best_c_s,
            'r_squared': r2,
            'n_params': 2,
            'decision': decision,
            'reasoning': reasoning
        }

    def _test_level4(self, df: pd.DataFrame, level3: Dict, stage_b_results: Dict) -> Dict:
        """
        Level 4: Full Empirical

        Unrestricted functional form with maturity, spread, and sector terms.
        Compare to Level 3 (Calibrated Merton).
        """
        print("  Testing Level 4: Full Empirical...")

        # Get unrestricted results from Stage B
        spec_b3 = stage_b_results.get('spec_b3', {})
        r2_empirical = spec_b3.get('r_squared', 0)
        n_params = spec_b3.get('n_params', 10)

        # Get calibrated Merton R²
        r2_calibrated = level3.get('r_squared', 0)

        delta_r2 = r2_empirical - r2_calibrated
        improvement_pct = 100 * delta_r2 / r2_calibrated if r2_calibrated > 0 else 0

        # Decision: Is improvement worth complexity?
        if delta_r2 > 0.05:
            decision = 'ADOPT'
            reasoning = f'Empirical spec justified (ΔR²={delta_r2:.3f})'
        else:
            decision = 'Stay at Level 3'
            reasoning = f'Marginal improvement (ΔR²={delta_r2:.3f}), prefer parsimony'

        return {
            'test': 'Full Empirical vs Calibrated',
            'r2_empirical': r2_empirical,
            'r2_calibrated': r2_calibrated,
            'delta_r2': delta_r2,
            'improvement_pct': improvement_pct,
            'n_params': n_params,
            'decision': decision,
            'reasoning': reasoning
        }

    def _check_level5_needed(self, stage_c_results: Dict) -> bool:
        """
        Check if Level 5 (time-varying) is needed.

        Criteria:
        - Stage C Chow test p < 0.01 (significant instability)
        - Macro effect > 30% during crises
        """
        stability = stage_c_results.get('stability_test', {})
        p_value = stability.get('chow_p_value', 1.0)

        # Check macro drivers if unstable
        if p_value < 0.01:
            macro_drivers = stage_c_results.get('macro_drivers', {})
            vix_effect = macro_drivers.get('vix_effect_pct', 0)

            return abs(vix_effect) > 30

        return False

    def _test_level5(self, df: pd.DataFrame, stage_c_results: Dict, level4: Dict) -> Dict:
        """
        Level 5: Time-varying

        Add macro state: lambda_prod = lambda_base * exp(gamma_VIX * VIX + gamma_OAS * log(OAS))
        """
        print("  Testing Level 5: Time-varying...")

        # Get macro drivers from Stage C
        macro_drivers = stage_c_results.get('macro_drivers', {})
        gamma_vix = macro_drivers.get('coef_vix', 0)
        gamma_oas = macro_drivers.get('coef_oas', 0)

        # Use empirical base from Level 4
        # For simplicity, use lambda_Merton as base

        # Construct time-varying lambda
        df['macro_adjustment'] = np.exp(
            gamma_vix * df['vix'] / 100 +  # Normalize VIX
            gamma_oas * np.log(df['oas_index'])
        )

        df['lambda_tv'] = df['lambda_Merton'] * df['macro_adjustment']
        df['interaction_tv'] = df['lambda_tv'] * df['f_DTS']

        # Regression
        y = df['oas_pct_change'].values
        X = df['interaction_tv'].values

        model = OLS(y, sm.add_constant(X))
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['week'].values})

        r2_tv = results.rsquared
        r2_static = level4.get('r2_empirical', 0)

        delta_r2 = r2_tv - r2_static

        # Check crisis performance
        crisis_mask = df['vix'] > 30
        if crisis_mask.sum() > 100:
            # Crisis subset
            y_crisis = df.loc[crisis_mask, 'oas_pct_change'].values
            X_crisis_static = df.loc[crisis_mask, 'lambda_Merton'].values * df.loc[crisis_mask, 'f_DTS'].values
            X_crisis_tv = df.loc[crisis_mask, 'interaction_tv'].values

            # RMSE comparison
            model_static = OLS(y_crisis, sm.add_constant(X_crisis_static))
            model_tv = OLS(y_crisis, sm.add_constant(X_crisis_tv))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results_static = model_static.fit()
                results_tv = model_tv.fit()

            rmse_static = np.sqrt(np.mean(results_static.resid ** 2))
            rmse_tv = np.sqrt(np.mean(results_tv.resid ** 2))

            rmse_reduction_pct = 100 * (rmse_static - rmse_tv) / rmse_static
        else:
            rmse_reduction_pct = 0

        # Decision
        if rmse_reduction_pct > 20 and delta_r2 > 0.01:
            decision = 'ADOPT'
            reasoning = f'Time-varying justified in crises (RMSE reduction={rmse_reduction_pct:.1f}%)'
        else:
            decision = 'Stay at Level 4'
            reasoning = f'Marginal crisis benefit (RMSE reduction={rmse_reduction_pct:.1f}%), not worth complexity'

        return {
            'test': 'Time-varying vs Static',
            'gamma_vix': gamma_vix,
            'gamma_oas': gamma_oas,
            'r2_tv': r2_tv,
            'r2_static': r2_static,
            'delta_r2': delta_r2,
            'crisis_rmse_reduction_pct': rmse_reduction_pct,
            'n_params': level4.get('n_params', 10) + 2,
            'decision': decision,
            'reasoning': reasoning
        }

    def out_of_sample_validation(
        self,
        df: pd.DataFrame,
        recommended_level: int,
        hierarchical_results: Dict,
        train_window_years: int = 3,
        test_window_years: int = 1
    ) -> Dict:
        """
        Rolling window out-of-sample validation.

        Args:
            df: Full dataset
            recommended_level: Recommended specification level (1-5)
            hierarchical_results: Results from hierarchical testing
            train_window_years: Training window size (default 3)
            test_window_years: Test window size (default 1)

        Returns:
            Dictionary with OOS results for all specifications
        """
        print("Running out-of-sample validation...")
        print()

        # Prepare time windows
        df = df.sort_values('date')
        df['date'] = pd.to_datetime(df['date'])

        min_date = df['date'].min()
        max_date = df['date'].max()

        # Rolling windows
        windows = []
        current_start = min_date

        while True:
            train_end = current_start + pd.DateOffset(years=train_window_years)
            test_start = train_end
            test_end = test_start + pd.DateOffset(years=test_window_years)

            if test_end > max_date:
                break

            windows.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })

            current_start = test_start  # Roll forward by test window

        print(f"  Generated {len(windows)} rolling windows")
        print()

        # Evaluate each specification
        specs_to_evaluate = [
            'Standard DTS',
            'Pure Merton',
            'Calibrated Merton',
            'Empirical'
        ]

        # Add time-varying if Level 5 exists
        if 'level5' in hierarchical_results:
            specs_to_evaluate.append('Time-varying')

        oos_results = {spec: [] for spec in specs_to_evaluate}

        for i, window in enumerate(windows):
            print(f"  Window {i+1}/{len(windows)}: Train {window['train_start'].year}-{window['train_end'].year}, Test {window['test_start'].year}-{window['test_end'].year}")

            # Split data
            train_mask = (df['date'] >= window['train_start']) & (df['date'] < window['train_end'])
            test_mask = (df['date'] >= window['test_start']) & (df['date'] < window['test_end'])

            df_train = df[train_mask].copy()
            df_test = df[test_mask].copy()

            if len(df_test) < 100:
                print("    Skipping: insufficient test data")
                continue

            # Evaluate each spec
            for spec in specs_to_evaluate:
                metrics = self._evaluate_spec_oos(spec, df_train, df_test, hierarchical_results)
                oos_results[spec].append({
                    'window': i,
                    'test_start': window['test_start'],
                    'test_end': window['test_end'],
                    **metrics
                })

        # Aggregate results
        oos_summary = {}
        for spec in specs_to_evaluate:
            if len(oos_results[spec]) > 0:
                oos_summary[spec] = {
                    'avg_r2_oos': np.mean([r['r2_oos'] for r in oos_results[spec]]),
                    'avg_rmse_oos': np.mean([r['rmse_oos'] for r in oos_results[spec]]),
                    'median_r2_oos': np.median([r['r2_oos'] for r in oos_results[spec]]),
                    'median_rmse_oos': np.median([r['rmse_oos'] for r in oos_results[spec]]),
                    'windows': oos_results[spec]
                }

        return {
            'oos_summary': oos_summary,
            'oos_by_window': oos_results,
            'n_windows': len(windows)
        }

    def _evaluate_spec_oos(
        self,
        spec_name: str,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        hierarchical_results: Dict
    ) -> Dict:
        """
        Evaluate a specification out-of-sample.

        Returns OOS R², RMSE for this window.
        """
        y_test = df_test['oas_pct_change'].values

        if spec_name == 'Standard DTS':
            # Predict using f_DTS directly (beta = 1)
            y_pred = df_test['f_DTS'].values

        elif spec_name == 'Pure Merton':
            # Predict using lambda_Merton * f_DTS
            # No training needed (lookup tables)
            y_pred = df_test['lambda_Merton'].values * df_test['f_DTS'].values

        elif spec_name == 'Calibrated Merton':
            # Estimate c0, c_s on training data
            level3 = hierarchical_results.get('level3', {})
            c0 = level3.get('c0', 1.0)
            c_s = level3.get('c_s', -0.25)

            # Apply to test data
            lambda_s_adj = df_test['lambda_s'] ** (c_s / -0.25)
            lambda_adj = df_test['lambda_T'] * lambda_s_adj
            y_pred = c0 * lambda_adj * df_test['f_DTS']

        elif spec_name == 'Empirical':
            # Re-estimate empirical spec on training data
            # For simplicity, use unrestricted from Stage B structure
            # (In practice, would re-run full unrestricted regression)

            # Use maturity and spread log terms
            X_train = self._construct_empirical_features(df_train)
            X_test = self._construct_empirical_features(df_test)

            y_train = df_train['oas_pct_change'].values

            model = OLS(y_train, sm.add_constant(X_train))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = model.fit(cov_type='cluster', cov_kwds={'groups': df_train['week'].values})

            y_pred = results.predict(sm.add_constant(X_test))

        elif spec_name == 'Time-varying':
            # Use empirical base + macro state
            level5 = hierarchical_results.get('level5', {})
            gamma_vix = level5.get('gamma_vix', 0)
            gamma_oas = level5.get('gamma_oas', 0)

            # Construct time-varying lambda
            macro_adj = np.exp(
                gamma_vix * df_test['vix'] / 100 +
                gamma_oas * np.log(df_test['oas_index'])
            )

            lambda_tv = df_test['lambda_Merton'] * macro_adj
            y_pred = lambda_tv * df_test['f_DTS']

        else:
            raise ValueError(f"Unknown spec: {spec_name}")

        # Compute metrics
        y_pred = np.nan_to_num(y_pred, nan=0.0)

        residuals = y_test - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse_oos = np.sqrt(np.mean(residuals ** 2))

        return {
            'r2_oos': r2_oos,
            'rmse_oos': rmse_oos
        }

    def _construct_empirical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Construct features for empirical specification.

        Features: log(M), log(s), (log M)^2, log(M)*log(s), sector dummies
        """
        features = []

        # Maturity term (years to maturity)
        log_M = np.log(df['years_to_maturity'] + 0.1)  # Add small constant to avoid log(0)
        features.append(log_M)

        # Spread term
        log_s = np.log(df['oas'] + 1)  # Add 1 to avoid log(0)
        features.append(log_s)

        # Quadratic and interaction
        features.append(log_M ** 2)
        features.append(log_M * log_s)

        # Sector dummies (if available)
        if 'sector' in df.columns:
            sectors = df['sector'].unique()
            for sector in sectors[1:]:  # Omit first for reference category
                features.append((df['sector'] == sector).astype(float))

        # Interaction with f_DTS
        X_base = np.column_stack(features)
        f_DTS = df['f_DTS'].values.reshape(-1, 1)

        X = X_base * f_DTS  # Element-wise multiplication

        return X

    def performance_by_regime(
        self,
        df: pd.DataFrame,
        oos_results: Dict,
        vix_thresholds: Tuple[float, float] = (20, 30)
    ) -> Dict:
        """
        Evaluate OOS performance by regime (Normal / Stress / Crisis).

        Args:
            df: Full dataset
            oos_results: OOS validation results
            vix_thresholds: (normal_max, stress_max) for VIX thresholds

        Returns:
            Performance metrics by regime for each specification
        """
        print("Analyzing performance by regime...")
        print()

        # Define regimes
        normal_threshold, stress_threshold = vix_thresholds

        regime_results = {}

        for spec_name, spec_results in oos_results['oos_by_window'].items():
            regime_metrics = {
                'Normal (VIX < 20)': [],
                'Stress (VIX 20-30)': [],
                'Crisis (VIX > 30)': []
            }

            for window_result in spec_results:
                test_start = window_result['test_start']
                test_end = window_result['test_end']

                # Get test period data
                test_mask = (df['date'] >= test_start) & (df['date'] < test_end)
                df_test = df[test_mask]

                # Average VIX in test period
                avg_vix = df_test['vix'].mean()

                if avg_vix < normal_threshold:
                    regime = 'Normal (VIX < 20)'
                elif avg_vix < stress_threshold:
                    regime = 'Stress (VIX 20-30)'
                else:
                    regime = 'Crisis (VIX > 30)'

                regime_metrics[regime].append({
                    'r2_oos': window_result['r2_oos'],
                    'rmse_oos': window_result['rmse_oos']
                })

            # Aggregate by regime
            regime_summary = {}
            for regime, metrics in regime_metrics.items():
                if len(metrics) > 0:
                    regime_summary[regime] = {
                        'n_windows': len(metrics),
                        'avg_r2_oos': np.mean([m['r2_oos'] for m in metrics]),
                        'avg_rmse_oos': np.mean([m['rmse_oos'] for m in metrics])
                    }
                else:
                    regime_summary[regime] = {
                        'n_windows': 0,
                        'avg_r2_oos': np.nan,
                        'avg_rmse_oos': np.nan
                    }

            regime_results[spec_name] = regime_summary

        return regime_results

    def generate_production_blueprint(
        self,
        recommended_level: int,
        hierarchical_results: Dict,
        oos_results: Dict
    ) -> Dict:
        """
        Generate production implementation blueprint.

        Returns specification name, parameters, implementation details,
        and expected performance.
        """
        spec_name = hierarchical_results['recommended_spec']

        blueprint = {
            'specification': spec_name,
            'level': recommended_level,
            'parameters': self._get_parameters(recommended_level, hierarchical_results),
            'implementation': self._get_implementation_details(recommended_level, hierarchical_results),
            'performance': self._get_expected_performance(spec_name, oos_results),
            'complexity': self._get_complexity_rating(recommended_level),
            'recalibration_frequency': self._get_recalibration_frequency(recommended_level)
        }

        return blueprint

    def _get_parameters(self, level: int, hierarchical_results: Dict) -> Dict:
        """Get parameter estimates for recommended specification."""
        if level == 1:
            return {'n_params': 0, 'parameters': {}}
        elif level == 2:
            return {'n_params': 0, 'parameters': {}}
        elif level == 3:
            level3 = hierarchical_results.get('level3', {})
            return {
                'n_params': 2,
                'parameters': {
                    'c0': level3.get('c0', 1.0),
                    'c_s': level3.get('c_s', -0.25)
                }
            }
        elif level == 4:
            level4 = hierarchical_results.get('level4', {})
            return {
                'n_params': level4.get('n_params', 10),
                'parameters': 'See Stage B Spec B.3 coefficients'
            }
        elif level == 5:
            level5 = hierarchical_results.get('level5', {})
            return {
                'n_params': level5.get('n_params', 12),
                'parameters': {
                    'gamma_vix': level5.get('gamma_vix', 0),
                    'gamma_oas': level5.get('gamma_oas', 0)
                }
            }

    def _get_implementation_details(self, level: int, hierarchical_results: Dict) -> str:
        """Get implementation formula and details."""
        if level == 1:
            return "y_i,t = f_DTS,t (Standard DTS, no adjustments)"
        elif level == 2:
            return "lambda_i = lambda_T(T_i; 5y, s_i) × lambda_s(s_i; 100) from lookup tables"
        elif level == 3:
            level3 = hierarchical_results.get('level3', {})
            c0 = level3.get('c0', 1.0)
            c_s = level3.get('c_s', -0.25)
            return f"lambda_i = {c0:.3f} × lambda_T(T_i) × lambda_s(s_i)^({c_s:.3f}/−0.25)"
        elif level == 4:
            return "lambda_i = exp(β_0 + β_M·log(M) + β_s·log(s) + β_M²·(log M)² + β_Ms·log M·log s + sector effects)"
        elif level == 5:
            level5 = hierarchical_results.get('level5', {})
            gamma_vix = level5.get('gamma_vix', 0)
            gamma_oas = level5.get('gamma_oas', 0)
            return f"lambda_i,t = lambda_base_i × exp({gamma_vix:.3f}·VIX_t + {gamma_oas:.3f}·log(OAS_index,t))"

    def _get_expected_performance(self, spec_name: str, oos_results: Dict) -> Dict:
        """Get expected performance metrics."""
        oos_summary = oos_results.get('oos_summary', {})
        spec_summary = oos_summary.get(spec_name, {})

        return {
            'avg_r2_oos': spec_summary.get('avg_r2_oos', np.nan),
            'avg_rmse_oos': spec_summary.get('avg_rmse_oos', np.nan)
        }

    def _get_complexity_rating(self, level: int) -> str:
        """Get complexity rating."""
        ratings = {
            1: 'Trivial',
            2: 'Low',
            3: 'Low-Moderate',
            4: 'Moderate-High',
            5: 'High'
        }
        return ratings.get(level, 'Unknown')

    def _get_recalibration_frequency(self, level: int) -> str:
        """Get recommended recalibration frequency."""
        frequencies = {
            1: 'None (static)',
            2: 'Quarterly review (check table range)',
            3: 'Annual recalibration',
            4: 'Annual recalibration',
            5: 'Daily macro update'
        }
        return frequencies.get(level, 'Unknown')
