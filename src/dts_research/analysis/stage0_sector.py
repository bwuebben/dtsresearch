"""
Stage 0: Sector Interaction Analysis

Implements the third component of evolved Stage 0:
- Tests for sector-specific DTS sensitivities
- Financial sector: Expected β > 0 (amplifies market moves)
- Utility sector: Expected β < 0 (dampens market moves)
- Joint F-test for significance of sector interactions

Based on Specification 0.3 / Equation 4.6 from the paper:
    y_{i,t} = α + β_0 [λ^Merton_{i,t} · f_{DTS,t}] +
              Σ_s β_s · 1{i∈s} · [λ^Merton_{i,t} · f_{DTS,t}] + ε_{i,t}

where y_{i,t} is the percentage spread change, and Industrial is the reference sector.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import statsmodels.api as sm
from scipy import stats

from ..data.sector_classification import SectorClassifier
from ..models.merton import MertonLambdaCalculator, calculate_merton_lambda
from ..utils.statistical_tests import (
    clustered_standard_errors,
    joint_f_test
)


class SectorInteractionAnalysis:
    """
    Sector interaction analysis for Stage 0.

    Regression specification:
    y_{i,t} = α + β_0 [λ^Merton · f_DTS] +
              β_fin · [λ^Merton · f_DTS × Financial] +
              β_util · [λ^Merton · f_DTS × Utility] +
              β_energy · [λ^Merton · f_DTS × Energy] + ε_{i,t}

    Tests:
    1. Joint F-test: H0: β_fin = β_util = β_energy = 0
    2. Financial sector: β_0 + β_fin is their DTS sensitivity
    3. Utility sector: β_0 + β_util is their DTS sensitivity
    """

    def __init__(self):
        """Initialize sector interaction analysis."""
        self.merton_calc = MertonLambdaCalculator()

    def run_sector_analysis(
        self,
        bond_data: pd.DataFrame,
        universe: str = 'IG',
        cluster_by: str = 'week'
    ) -> Dict:
        """
        Run complete sector interaction analysis.

        Args:
            bond_data: DataFrame with bond observations
            universe: 'IG' or 'HY'
            cluster_by: 'week', 'issuer', or 'two-way'

        Returns:
            Dictionary with results:
            - base_regression: Results without sector interactions
            - sector_regression: Results with sector interactions
            - joint_test: F-test for joint significance
            - sector_tests: Individual sector hypothesis tests
            - diagnostics: Quality checks
        """
        # Step 1: Add sector classification
        classifier = SectorClassifier()
        bond_data = classifier.classify_sector(bond_data, bclass_column='sector_classification')
        bond_data = classifier.add_sector_dummies(bond_data)

        # Step 2: Filter to universe
        if universe == 'IG':
            bond_data = bond_data[bond_data['rating'].isin(['AAA', 'AA', 'A', 'BBB'])].copy()
        elif universe == 'HY':
            bond_data = bond_data[bond_data['rating'].isin(['BB', 'B', 'CCC'])].copy()
        else:
            raise ValueError("universe must be 'IG' or 'HY'")

        if len(bond_data) == 0:
            return self._empty_results(universe, "No bonds in universe")

        # Step 3: Compute spread changes
        bond_data = self._compute_spread_changes(bond_data)

        if len(bond_data) == 0:
            return self._empty_results(universe, "No data after computing spread changes")

        # Step 4: Compute Merton lambda and index DTS factor
        bond_data = self._compute_merton_scaled_factor(bond_data)

        if len(bond_data) == 0:
            return self._empty_results(universe, "No data after computing Merton factors")

        # Step 5: Run base regression (no sector interactions)
        base_regression = self._run_base_regression(bond_data, cluster_by)

        # Step 6: Run sector regression (with sector interactions)
        sector_regression = self._run_sector_regression(bond_data, cluster_by)

        # Step 7: Joint F-test for sector interactions
        joint_test = self._test_joint_significance(bond_data, cluster_by)

        # Step 8: Individual sector tests
        sector_tests = self._test_sector_predictions(sector_regression)

        # Step 9: Diagnostics
        diagnostics = self._compute_diagnostics(bond_data, sector_regression)

        return {
            'universe': universe,
            'base_regression': base_regression,
            'sector_regression': sector_regression,
            'joint_test': joint_test,
            'sector_tests': sector_tests,
            'diagnostics': diagnostics
        }

    def _compute_spread_changes(self, bond_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percentage spread changes for each bond.

        Args:
            bond_data: DataFrame with 'oas', 'cusip' (or bond id), 'date'

        Returns:
            DataFrame with 'spread_change' and 'oas_lag' columns
        """
        # Sort by bond and date
        bond_id_col = 'cusip' if 'cusip' in bond_data.columns else 'bond_id'
        bond_data = bond_data.sort_values([bond_id_col, 'date'])

        # Compute lagged spread within each bond
        bond_data['oas_lag'] = bond_data.groupby(bond_id_col)['oas'].shift(1)

        # Compute percentage spread change
        bond_data['spread_change'] = (bond_data['oas'] - bond_data['oas_lag']) / bond_data['oas_lag']

        # Remove NaN (first observation for each bond)
        bond_data = bond_data.dropna(subset=['spread_change', 'oas_lag'])

        # Remove extreme outliers (likely data errors)
        bond_data = bond_data[bond_data['spread_change'].abs() <= 1.0]  # ±100%

        return bond_data

    def _compute_merton_scaled_factor(self, bond_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Merton-scaled DTS factor: λ^Merton × f_DTS

        Args:
            bond_data: DataFrame with spread changes

        Returns:
            DataFrame with lambda_merton, f_dts, and merton_scaled_factor columns
        """
        # Compute Merton lambda for each bond (based on lagged spread and maturity)
        bond_data['lambda_merton'] = calculate_merton_lambda(
            bond_data['time_to_maturity'].values,
            bond_data['oas_lag'].values
        )

        # Compute index-level DTS factor (average percentage spread change per week)
        index_factor = bond_data.groupby('date').agg(
            f_dts=('spread_change', 'mean')
        ).reset_index()

        # Merge back to bond data
        bond_data = bond_data.merge(index_factor, on='date', how='left')

        # Compute Merton-scaled factor
        bond_data['merton_scaled_factor'] = bond_data['lambda_merton'] * bond_data['f_dts']

        return bond_data

    def _run_base_regression(
        self,
        bond_data: pd.DataFrame,
        cluster_by: str
    ) -> Dict:
        """
        Run base regression: y_{i,t} = α + β_0 · [λ^Merton · f_DTS] + ε_{i,t}

        This tests whether Merton-scaled DTS explains spread changes.

        Args:
            bond_data: DataFrame with bond observations and merton_scaled_factor
            cluster_by: Clustering variable

        Returns:
            Dictionary with regression results
        """
        # Prepare data
        required_cols = ['spread_change', 'merton_scaled_factor', 'date']
        if 'issuer_id' in bond_data.columns:
            required_cols.append('issuer_id')

        df = bond_data[required_cols].dropna()

        if len(df) < 100:
            return {'warning': 'Insufficient observations', 'n_obs': len(df)}

        # Dependent variable: percentage spread change
        y = df['spread_change'].values

        # Independent variable: Merton-scaled DTS factor (reshape to 2D)
        X = df['merton_scaled_factor'].values.reshape(-1, 1)

        # Cluster variable
        if cluster_by == 'week':
            cluster_var = df['date'].values
        elif cluster_by == 'issuer' and 'issuer_id' in df.columns:
            cluster_var = df['issuer_id'].values
        else:
            cluster_var = df['date'].values  # Default to week

        # Run regression with clustered SEs
        try:
            results = clustered_standard_errors(
                X=X,
                y=y,
                cluster_var=cluster_var,
                add_constant=True
            )

            return {
                'beta_0': results['params'][1],  # Coefficient on Merton-scaled factor
                'beta_0_se': results['se'][1],
                'beta_0_tstat': results['t_stats'][1],
                'beta_0_pvalue': results['p_values'][1],
                'alpha': results['params'][0],
                'r_squared': results['r_squared'],
                'n_obs': results['n_obs'],
                'n_clusters': results['n_clusters'],
                'interpretation': self._interpret_base_beta(results['params'][1], results['p_values'][1])
            }

        except Exception as e:
            return {'warning': f'Regression failed: {str(e)}'}

    def _interpret_base_beta(self, beta: float, pvalue: float) -> str:
        """Interpret the base regression coefficient."""
        if 0.9 <= beta <= 1.1 and pvalue < 0.05:
            return f"β_0 = {beta:.3f} ≈ 1: Merton-scaled DTS works well for base (Industrial) sector"
        elif beta < 0.9:
            return f"β_0 = {beta:.3f} < 1: Merton over-predicts spread sensitivity"
        elif beta > 1.1:
            return f"β_0 = {beta:.3f} > 1: Merton under-predicts spread sensitivity"
        else:
            return f"β_0 = {beta:.3f}: Merton-scaled DTS significant (p={pvalue:.4f})"

    def _run_sector_regression(
        self,
        bond_data: pd.DataFrame,
        cluster_by: str
    ) -> Dict:
        """
        Run sector regression with interactions:
        y_{i,t} = α + β_0·[λ^Merton · f_DTS] +
                  β_fin·[λ^Merton · f_DTS × Financial] +
                  β_util·[λ^Merton · f_DTS × Utility] +
                  β_energy·[λ^Merton · f_DTS × Energy] + ε_{i,t}

        Args:
            bond_data: DataFrame with bond observations, sector dummies, and merton_scaled_factor
            cluster_by: Clustering variable

        Returns:
            Dictionary with regression results
        """
        # Prepare data
        required_cols = ['spread_change', 'merton_scaled_factor', 'date',
                         'sector_financial', 'sector_utility', 'sector_energy']
        if 'issuer_id' in bond_data.columns:
            required_cols.append('issuer_id')

        df = bond_data[required_cols].dropna()

        if len(df) < 100:
            return {'warning': 'Insufficient observations', 'n_obs': len(df)}

        # Dependent variable: percentage spread change
        y = df['spread_change'].values

        # Independent variables: Merton-scaled factor and sector interactions
        msf = df['merton_scaled_factor'].values
        msf_fin = msf * df['sector_financial'].values
        msf_util = msf * df['sector_utility'].values
        msf_energy = msf * df['sector_energy'].values

        X = np.column_stack([msf, msf_fin, msf_util, msf_energy])

        # Cluster variable
        if cluster_by == 'week':
            cluster_var = df['date'].values
        elif cluster_by == 'issuer' and 'issuer_id' in df.columns:
            cluster_var = df['issuer_id'].values
        else:
            cluster_var = df['date'].values

        # Run regression with clustered SEs
        try:
            results = clustered_standard_errors(
                X=X,
                y=y,
                cluster_var=cluster_var,
                add_constant=True
            )

            # β_0 is sensitivity for Industrial (reference sector)
            # β_0 + β_fin is sensitivity for Financial
            # etc.
            return {
                'alpha': results['params'][0],
                'beta_0': results['params'][1],  # Base DTS sensitivity (Industrial)
                'beta_financial': results['params'][2],  # Financial DEVIATION from base
                'beta_utility': results['params'][3],  # Utility DEVIATION from base
                'beta_energy': results['params'][4],  # Energy DEVIATION from base
                'beta_0_se': results['se'][1],
                'beta_financial_se': results['se'][2],
                'beta_utility_se': results['se'][3],
                'beta_energy_se': results['se'][4],
                'beta_0_tstat': results['t_stats'][1],
                'beta_financial_tstat': results['t_stats'][2],
                'beta_utility_tstat': results['t_stats'][3],
                'beta_energy_tstat': results['t_stats'][4],
                'beta_0_pvalue': results['p_values'][1],
                'beta_financial_pvalue': results['p_values'][2],
                'beta_utility_pvalue': results['p_values'][3],
                'beta_energy_pvalue': results['p_values'][4],
                'r_squared': results['r_squared'],
                'n_obs': results['n_obs'],
                'n_clusters': results['n_clusters'],
                # Derived: total sector sensitivities
                'sensitivity_industrial': results['params'][1],
                'sensitivity_financial': results['params'][1] + results['params'][2],
                'sensitivity_utility': results['params'][1] + results['params'][3],
                'sensitivity_energy': results['params'][1] + results['params'][4]
            }

        except Exception as e:
            return {'warning': f'Regression failed: {str(e)}'}

    def _test_joint_significance(
        self,
        bond_data: pd.DataFrame,
        cluster_by: str
    ) -> Dict:
        """
        Joint F-test: H0: β_fin = β_util = β_energy = 0
        (All sectors have the same DTS sensitivity)

        Args:
            bond_data: DataFrame with bond observations and merton_scaled_factor
            cluster_by: Clustering variable

        Returns:
            Dictionary with test results
        """
        # Prepare data
        required_cols = ['spread_change', 'merton_scaled_factor', 'date',
                         'sector_financial', 'sector_utility', 'sector_energy']
        if 'issuer_id' in bond_data.columns:
            required_cols.append('issuer_id')

        df = bond_data[required_cols].dropna()

        if len(df) < 100:
            return {'warning': 'Insufficient observations'}

        y = df['spread_change'].values
        msf = df['merton_scaled_factor'].values
        msf_fin = msf * df['sector_financial'].values
        msf_util = msf * df['sector_utility'].values
        msf_energy = msf * df['sector_energy'].values

        # Unrestricted model: includes sector interactions
        X_unrestricted = np.column_stack([msf, msf_fin, msf_util, msf_energy])
        X_unrestricted = sm.add_constant(X_unrestricted)

        # Cluster variable
        if cluster_by == 'week':
            cluster_var = df['date'].values
        elif 'issuer_id' in df.columns:
            cluster_var = df['issuer_id'].values
        else:
            cluster_var = df['date'].values

        # Joint F-test on interaction terms (indices 2, 3, 4)
        try:
            test_results = joint_f_test(
                X=X_unrestricted,
                y=y,
                restriction_indices=[2, 3, 4],  # Test β_fin, β_util, β_energy jointly
                cluster_var=cluster_var
            )

            return {
                'test': 'H0: β_fin = β_util = β_energy = 0 (sectors have same DTS sensitivity)',
                'f_statistic': test_results['f_statistic'],
                'p_value': test_results['p_value'],
                'df_numerator': test_results['df_numerator'],
                'df_denominator': test_results['df_denominator'],
                'reject_null': test_results['p_value'] < 0.05,
                'sectors_differ': test_results['p_value'] < 0.05,
                'interpretation': self._interpret_joint_test(test_results['p_value'])
            }

        except Exception as e:
            return {'warning': f'Test failed: {str(e)}'}

    def _interpret_joint_test(self, p_value: float) -> str:
        """Interpret joint F-test."""
        if np.isnan(p_value):
            return "Test failed"

        if p_value < 0.05:
            return f"Sector interactions are jointly significant (p={p_value:.4f}) - sectors differ"
        else:
            return f"Sector interactions not significant (p={p_value:.4f}) - sectors similar"

    def _test_sector_predictions(self, sector_regression: Dict) -> Dict:
        """
        Test sector-specific predictions:
        1. Financial: β_fin > 0 (amplifies market moves beyond Merton baseline)
        2. Utility: β_util < 0 (dampens market moves below Merton baseline)

        Args:
            sector_regression: Results from _run_sector_regression

        Returns:
            Dictionary with sector-specific tests
        """
        if 'warning' in sector_regression:
            return {'warning': sector_regression['warning']}

        # Financial sector test (one-sided: β_fin > 0, meaning Financials move MORE than baseline)
        beta_fin = sector_regression['beta_financial']
        se_fin = sector_regression['beta_financial_se']
        t_fin = beta_fin / se_fin
        p_fin = 1 - stats.norm.cdf(t_fin)  # Right-tail

        # Utility sector test (one-sided: β_util < 0, meaning Utilities move LESS than baseline)
        beta_util = sector_regression['beta_utility']
        se_util = sector_regression['beta_utility_se']
        t_util = beta_util / se_util
        p_util = stats.norm.cdf(t_util)  # Left-tail

        # Energy sector test (two-sided)
        beta_energy = sector_regression['beta_energy']
        se_energy = sector_regression['beta_energy_se']
        t_energy = beta_energy / se_energy
        p_energy = 2 * (1 - stats.norm.cdf(np.abs(t_energy)))

        # Get total sensitivities
        sens_ind = sector_regression.get('sensitivity_industrial', np.nan)
        sens_fin = sector_regression.get('sensitivity_financial', np.nan)
        sens_util = sector_regression.get('sensitivity_utility', np.nan)
        sens_energy = sector_regression.get('sensitivity_energy', np.nan)

        return {
            'financial_test': {
                'hypothesis': 'H0: β_fin ≤ 0 vs H1: β_fin > 0 (Financials amplify)',
                'beta_deviation': beta_fin,
                'se': se_fin,
                't_statistic': t_fin,
                'p_value': p_fin,
                'reject_null': p_fin < 0.05,
                'total_sensitivity': sens_fin,
                'interpretation': (
                    f"Financial deviation = {beta_fin:.3f}: {'amplifies' if beta_fin > 0 else 'dampens'} "
                    f"market moves (total sensitivity = {sens_fin:.3f}, p={p_fin:.4f})"
                )
            },
            'utility_test': {
                'hypothesis': 'H0: β_util ≥ 0 vs H1: β_util < 0 (Utilities dampen)',
                'beta_deviation': beta_util,
                'se': se_util,
                't_statistic': t_util,
                'p_value': p_util,
                'reject_null': p_util < 0.05,
                'total_sensitivity': sens_util,
                'interpretation': (
                    f"Utility deviation = {beta_util:.3f}: {'dampens' if beta_util < 0 else 'amplifies'} "
                    f"market moves (total sensitivity = {sens_util:.3f}, p={p_util:.4f})"
                )
            },
            'energy_test': {
                'hypothesis': 'H0: β_energy = 0 (Energy same as Industrial)',
                'beta_deviation': beta_energy,
                'se': se_energy,
                't_statistic': t_energy,
                'p_value': p_energy,
                'reject_null': p_energy < 0.05,
                'total_sensitivity': sens_energy,
                'interpretation': (
                    f"Energy deviation = {beta_energy:.3f} "
                    f"(total sensitivity = {sens_energy:.3f}, p={p_energy:.4f})"
                )
            },
            'summary': {
                'industrial_baseline': sens_ind,
                'financial_total': sens_fin,
                'utility_total': sens_util,
                'energy_total': sens_energy,
                'need_sector_adjustment': (p_fin < 0.05 or p_util < 0.05 or p_energy < 0.05)
            }
        }

    def _compute_diagnostics(
        self,
        bond_data: pd.DataFrame,
        sector_regression: Dict
    ) -> Dict:
        """
        Compute diagnostic statistics.

        Args:
            bond_data: Bond data
            sector_regression: Regression results

        Returns:
            Dictionary with diagnostics
        """
        # Sector distribution
        sector_dist = bond_data['sector'].value_counts() if 'sector' in bond_data.columns else pd.Series()

        return {
            'n_observations': sector_regression.get('n_obs', 0),
            'n_clusters': sector_regression.get('n_clusters', 0),
            'n_unique_bonds': bond_data['cusip'].nunique() if 'cusip' in bond_data.columns else bond_data['bond_id'].nunique() if 'bond_id' in bond_data.columns else np.nan,
            'n_unique_weeks': bond_data['date'].nunique() if 'date' in bond_data.columns else np.nan,
            'r_squared': sector_regression.get('r_squared', np.nan),
            'sector_distribution': sector_dist.to_dict(),
            'pct_industrial': 100.0 * sector_dist.get('Industrial', 0) / len(bond_data) if len(bond_data) > 0 else 0,
            'pct_financial': 100.0 * sector_dist.get('Financial', 0) / len(bond_data) if len(bond_data) > 0 else 0,
            'pct_utility': 100.0 * sector_dist.get('Utility', 0) / len(bond_data) if len(bond_data) > 0 else 0,
            'pct_energy': 100.0 * sector_dist.get('Energy', 0) / len(bond_data) if len(bond_data) > 0 else 0,
            'mean_merton_scaled_factor': bond_data['merton_scaled_factor'].mean() if 'merton_scaled_factor' in bond_data.columns else np.nan,
            'mean_spread_change': bond_data['spread_change'].mean() if 'spread_change' in bond_data.columns else np.nan
        }

    def _empty_results(self, universe: str, reason: str) -> Dict:
        """Return empty results structure with warning."""
        return {
            'universe': universe,
            'base_regression': {'warning': reason},
            'sector_regression': {'warning': reason},
            'joint_test': {'warning': reason},
            'sector_tests': {'warning': reason},
            'diagnostics': {'warning': reason}
        }

    def compare_ig_hy(
        self,
        ig_results: Dict,
        hy_results: Dict
    ) -> Dict:
        """
        Compare sector results between IG and HY.

        Args:
            ig_results: Results from run_sector_analysis(universe='IG')
            hy_results: Results from run_sector_analysis(universe='HY')

        Returns:
            Dictionary with comparison statistics
        """
        ig_joint = ig_results['joint_test'].get('sectors_differ', False)
        hy_joint = hy_results['joint_test'].get('sectors_differ', False)

        ig_fin = ig_results['sector_tests'].get('financial_test', {}).get('reject_null', False)
        hy_fin = hy_results['sector_tests'].get('financial_test', {}).get('reject_null', False)

        ig_util = ig_results['sector_tests'].get('utility_test', {}).get('reject_null', False)
        hy_util = hy_results['sector_tests'].get('utility_test', {}).get('reject_null', False)

        # Get baseline sensitivities
        ig_base_beta = ig_results['base_regression'].get('beta_0', np.nan)
        hy_base_beta = hy_results['base_regression'].get('beta_0', np.nan)

        return {
            'ig_sectors_differ': ig_joint,
            'hy_sectors_differ': hy_joint,
            'both_sectors_differ': ig_joint and hy_joint,
            'ig_financial_amplifies': ig_fin,
            'hy_financial_amplifies': hy_fin,
            'ig_utility_dampens': ig_util,
            'hy_utility_dampens': hy_util,
            'ig_base_beta': ig_base_beta,
            'hy_base_beta': hy_base_beta,
            'ig_base_near_1': 0.8 <= ig_base_beta <= 1.2 if not np.isnan(ig_base_beta) else False,
            'hy_base_near_1': 0.8 <= hy_base_beta <= 1.2 if not np.isnan(hy_base_beta) else False,
            'interpretation': self._interpret_ig_hy_comparison(
                ig_joint, hy_joint, ig_fin, hy_fin, ig_base_beta, hy_base_beta
            )
        }

    def _interpret_ig_hy_comparison(
        self,
        ig_joint: bool,
        hy_joint: bool,
        ig_fin: bool,
        hy_fin: bool,
        ig_base: float,
        hy_base: float
    ) -> str:
        """Interpret IG vs HY comparison."""
        parts = []

        # Base beta interpretation
        ig_base_ok = 0.8 <= ig_base <= 1.2 if not np.isnan(ig_base) else False
        hy_base_ok = 0.8 <= hy_base <= 1.2 if not np.isnan(hy_base) else False

        if ig_base_ok and hy_base_ok:
            parts.append("Base Merton-scaled DTS works in both universes")
        elif ig_base_ok:
            parts.append(f"Base DTS works for IG (β={ig_base:.2f}) but not HY (β={hy_base:.2f})")
        elif hy_base_ok:
            parts.append(f"Base DTS works for HY (β={hy_base:.2f}) but not IG (β={ig_base:.2f})")
        else:
            parts.append(f"Base DTS needs calibration in both (IG β={ig_base:.2f}, HY β={hy_base:.2f})")

        # Sector effects
        if ig_joint and hy_joint:
            parts.append("Sector adjustments needed in both universes")
        elif ig_joint:
            parts.append("Sector adjustments needed in IG only")
        elif hy_joint:
            parts.append("Sector adjustments needed in HY only")
        else:
            parts.append("No sector adjustments needed")

        return "; ".join(parts)


def run_sector_analysis_both_universes(
    bond_data: pd.DataFrame,
    cluster_by: str = 'week'
) -> Dict:
    """
    Convenience function to run sector analysis for both IG and HY.

    Args:
        bond_data: DataFrame with bond observations
        cluster_by: 'week' or 'issuer'

    Returns:
        Dictionary with results for both universes plus comparison
    """
    analyzer = SectorInteractionAnalysis()

    # Run for IG
    ig_results = analyzer.run_sector_analysis(bond_data, universe='IG', cluster_by=cluster_by)

    # Run for HY
    hy_results = analyzer.run_sector_analysis(bond_data, universe='HY', cluster_by=cluster_by)

    # Compare
    comparison = analyzer.compare_ig_hy(ig_results, hy_results)

    return {
        'IG': ig_results,
        'HY': hy_results,
        'comparison': comparison
    }
