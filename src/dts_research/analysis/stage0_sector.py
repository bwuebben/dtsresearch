"""
Stage 0: Sector Interaction Analysis

Implements the third component of evolved Stage 0:
- Tests for sector-specific maturity effects
- Financial sector: Expected β > 0 (correlation risk, regulatory)
- Utility sector: Expected β < 0 (regulatory protection, stable cash flows)
- Joint F-test for significance of sector interactions

Based on Specification 0.3 from the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import statsmodels.api as sm
from scipy import stats

from ..data.sector_classification import SectorClassifier
from ..utils.statistical_tests import (
    clustered_standard_errors,
    joint_f_test
)


class SectorInteractionAnalysis:
    """
    Sector interaction analysis for Stage 0.

    Regression specification:
    ln(s_it) = α + λ·T_it + β_fin·(T_it × Financial_i) +
               β_util·(T_it × Utility_i) + β_energy·(T_it × Energy_i) + ε_it

    Tests:
    1. Joint F-test: H0: β_fin = β_util = β_energy = 0
    2. Financial sector: H0: β_fin ≤ 0 vs H1: β_fin > 0
    3. Utility sector: H0: β_util ≥ 0 vs H1: β_util < 0
    """

    def __init__(self):
        """Initialize sector interaction analysis."""
        pass

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

        # Step 3: Run base regression (no sector interactions)
        base_regression = self._run_base_regression(bond_data, cluster_by)

        # Step 4: Run sector regression (with sector interactions)
        sector_regression = self._run_sector_regression(bond_data, cluster_by)

        # Step 5: Joint F-test for sector interactions
        joint_test = self._test_joint_significance(bond_data, cluster_by)

        # Step 6: Individual sector tests
        sector_tests = self._test_sector_predictions(sector_regression)

        # Step 7: Diagnostics
        diagnostics = self._compute_diagnostics(bond_data, sector_regression)

        return {
            'universe': universe,
            'base_regression': base_regression,
            'sector_regression': sector_regression,
            'joint_test': joint_test,
            'sector_tests': sector_tests,
            'diagnostics': diagnostics
        }

    def _run_base_regression(
        self,
        bond_data: pd.DataFrame,
        cluster_by: str
    ) -> Dict:
        """
        Run base regression: ln(s_it) = α + λ·T_it + ε_it

        Args:
            bond_data: DataFrame with bond observations
            cluster_by: Clustering variable

        Returns:
            Dictionary with regression results
        """
        # Prepare data
        df = bond_data[['oas', 'time_to_maturity', 'date', 'issuer_id']].dropna()

        if len(df) < 100:
            return {'warning': 'Insufficient observations', 'n_obs': len(df)}

        y = np.log(df['oas'].values)
        X = df['time_to_maturity'].values

        # Cluster variable
        if cluster_by == 'week':
            cluster_var = df['date'].values
        elif cluster_by == 'issuer':
            cluster_var = df['issuer_id'].values
        else:
            raise ValueError("cluster_by must be 'week' or 'issuer'")

        # Run regression with clustered SEs
        try:
            results = clustered_standard_errors(
                X=X,
                y=y,
                cluster_var=cluster_var,
                add_constant=True
            )

            return {
                'lambda': results['params'][1],
                'lambda_se': results['se'][1],
                'lambda_tstat': results['t_stats'][1],
                'lambda_pvalue': results['p_values'][1],
                'alpha': results['params'][0],
                'r_squared': results['r_squared'],
                'n_obs': results['n_obs'],
                'n_clusters': results['n_clusters']
            }

        except Exception as e:
            return {'warning': f'Regression failed: {str(e)}'}

    def _run_sector_regression(
        self,
        bond_data: pd.DataFrame,
        cluster_by: str
    ) -> Dict:
        """
        Run sector regression with interactions:
        ln(s_it) = α + λ·T_it + β_fin·(T × Financial) + β_util·(T × Utility) + β_energy·(T × Energy) + ε_it

        Args:
            bond_data: DataFrame with bond observations and sector dummies
            cluster_by: Clustering variable

        Returns:
            Dictionary with regression results
        """
        # Prepare data
        required_cols = ['oas', 'time_to_maturity', 'date', 'issuer_id',
                         'sector_financial', 'sector_utility', 'sector_energy']
        df = bond_data[required_cols].dropna()

        if len(df) < 100:
            return {'warning': 'Insufficient observations', 'n_obs': len(df)}

        # Dependent variable
        y = np.log(df['oas'].values)

        # Independent variables
        T = df['time_to_maturity'].values
        T_fin = T * df['sector_financial'].values
        T_util = T * df['sector_utility'].values
        T_energy = T * df['sector_energy'].values

        X = np.column_stack([T, T_fin, T_util, T_energy])

        # Cluster variable
        if cluster_by == 'week':
            cluster_var = df['date'].values
        elif cluster_by == 'issuer':
            cluster_var = df['issuer_id'].values
        else:
            raise ValueError("cluster_by must be 'week' or 'issuer'")

        # Run regression with clustered SEs
        try:
            results = clustered_standard_errors(
                X=X,
                y=y,
                cluster_var=cluster_var,
                add_constant=True
            )

            return {
                'alpha': results['params'][0],
                'lambda': results['params'][1],  # Base maturity effect (Industrial)
                'beta_financial': results['params'][2],  # Financial interaction
                'beta_utility': results['params'][3],  # Utility interaction
                'beta_energy': results['params'][4],  # Energy interaction
                'lambda_se': results['se'][1],
                'beta_financial_se': results['se'][2],
                'beta_utility_se': results['se'][3],
                'beta_energy_se': results['se'][4],
                'lambda_tstat': results['t_stats'][1],
                'beta_financial_tstat': results['t_stats'][2],
                'beta_utility_tstat': results['t_stats'][3],
                'beta_energy_tstat': results['t_stats'][4],
                'lambda_pvalue': results['p_values'][1],
                'beta_financial_pvalue': results['p_values'][2],
                'beta_utility_pvalue': results['p_values'][3],
                'beta_energy_pvalue': results['p_values'][4],
                'r_squared': results['r_squared'],
                'n_obs': results['n_obs'],
                'n_clusters': results['n_clusters']
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

        Args:
            bond_data: DataFrame with bond observations
            cluster_by: Clustering variable

        Returns:
            Dictionary with test results
        """
        # Prepare data (same as sector regression)
        required_cols = ['oas', 'time_to_maturity', 'date', 'issuer_id',
                         'sector_financial', 'sector_utility', 'sector_energy']
        df = bond_data[required_cols].dropna()

        if len(df) < 100:
            return {'warning': 'Insufficient observations'}

        y = np.log(df['oas'].values)
        T = df['time_to_maturity'].values
        T_fin = T * df['sector_financial'].values
        T_util = T * df['sector_utility'].values
        T_energy = T * df['sector_energy'].values

        # Unrestricted model: includes sector interactions
        X_unrestricted = np.column_stack([T, T_fin, T_util, T_energy])
        X_unrestricted = sm.add_constant(X_unrestricted)

        # Cluster variable
        if cluster_by == 'week':
            cluster_var = df['date'].values
        else:
            cluster_var = df['issuer_id'].values

        # Joint F-test on interaction terms (indices 2, 3, 4)
        try:
            test_results = joint_f_test(
                X=X_unrestricted,
                y=y,
                restriction_indices=[2, 3, 4],  # Test β_fin, β_util, β_energy jointly
                cluster_var=cluster_var
            )

            return {
                'test': 'H0: β_fin = β_util = β_energy = 0',
                'f_statistic': test_results['f_statistic'],
                'p_value': test_results['p_value'],
                'df_numerator': test_results['df_numerator'],
                'df_denominator': test_results['df_denominator'],
                'reject_null': test_results['p_value'] < 0.05,
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
        1. Financial: H0: β_fin ≤ 0 vs H1: β_fin > 0
        2. Utility: H0: β_util ≥ 0 vs H1: β_util < 0

        Args:
            sector_regression: Results from _run_sector_regression

        Returns:
            Dictionary with sector-specific tests
        """
        if 'warning' in sector_regression:
            return {'warning': sector_regression['warning']}

        # Financial sector test (one-sided: β_fin > 0)
        beta_fin = sector_regression['beta_financial']
        se_fin = sector_regression['beta_financial_se']
        t_fin = beta_fin / se_fin
        p_fin = 1 - stats.norm.cdf(t_fin)  # Right-tail

        # Utility sector test (one-sided: β_util < 0)
        beta_util = sector_regression['beta_utility']
        se_util = sector_regression['beta_utility_se']
        t_util = beta_util / se_util
        p_util = stats.norm.cdf(t_util)  # Left-tail

        return {
            'financial_test': {
                'hypothesis': 'H0: β_fin ≤ 0 vs H1: β_fin > 0',
                'beta': beta_fin,
                'se': se_fin,
                't_statistic': t_fin,
                'p_value': p_fin,
                'reject_null': p_fin < 0.05,
                'interpretation': f"Financial β = {beta_fin:.4f} {'is' if p_fin < 0.05 else 'is not'} significantly positive (p={p_fin:.4f})"
            },
            'utility_test': {
                'hypothesis': 'H0: β_util ≥ 0 vs H1: β_util < 0',
                'beta': beta_util,
                'se': se_util,
                't_statistic': t_util,
                'p_value': p_util,
                'reject_null': p_util < 0.05,
                'interpretation': f"Utility β = {beta_util:.4f} {'is' if p_util < 0.05 else 'is not'} significantly negative (p={p_util:.4f})"
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
        sector_dist = bond_data['sector'].value_counts()

        return {
            'n_observations': sector_regression.get('n_obs', 0),
            'n_clusters': sector_regression.get('n_clusters', 0),
            'sector_distribution': sector_dist.to_dict(),
            'pct_industrial': 100.0 * sector_dist.get('Industrial', 0) / len(bond_data) if len(bond_data) > 0 else 0,
            'pct_financial': 100.0 * sector_dist.get('Financial', 0) / len(bond_data) if len(bond_data) > 0 else 0,
            'pct_utility': 100.0 * sector_dist.get('Utility', 0) / len(bond_data) if len(bond_data) > 0 else 0,
            'pct_energy': 100.0 * sector_dist.get('Energy', 0) / len(bond_data) if len(bond_data) > 0 else 0
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
        ig_joint = ig_results['joint_test'].get('reject_null', False)
        hy_joint = hy_results['joint_test'].get('reject_null', False)

        ig_fin = ig_results['sector_tests'].get('financial_test', {}).get('reject_null', False)
        hy_fin = hy_results['sector_tests'].get('financial_test', {}).get('reject_null', False)

        ig_util = ig_results['sector_tests'].get('utility_test', {}).get('reject_null', False)
        hy_util = hy_results['sector_tests'].get('utility_test', {}).get('reject_null', False)

        return {
            'ig_sectors_significant': ig_joint,
            'hy_sectors_significant': hy_joint,
            'both_sectors_significant': ig_joint and hy_joint,
            'ig_financial_positive': ig_fin,
            'hy_financial_positive': hy_fin,
            'ig_utility_negative': ig_util,
            'hy_utility_negative': hy_util,
            'interpretation': self._interpret_ig_hy_comparison(ig_joint, hy_joint, ig_fin, hy_fin)
        }

    def _interpret_ig_hy_comparison(
        self,
        ig_joint: bool,
        hy_joint: bool,
        ig_fin: bool,
        hy_fin: bool
    ) -> str:
        """Interpret IG vs HY comparison."""
        if ig_joint and hy_joint:
            return "Sector effects significant in both IG and HY - sectors matter"
        elif ig_joint:
            return "Sector effects significant only in IG"
        elif hy_joint:
            return "Sector effects significant only in HY"
        else:
            return "Sector effects not significant in either universe - sectors don't matter"


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
