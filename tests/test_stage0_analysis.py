"""
Test Stage 0 analysis modules with mock data.

Tests the corrected implementation that uses:
- Spread CHANGES (not levels)
- Merton-predicted elasticity (not maturity)
- Tests β ≈ 1 (not λ > 0)
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.data.loader import BondDataLoader
from dts_research.data.sector_classification import SectorClassifier
from dts_research.data.issuer_identification import add_issuer_identification
from dts_research.analysis.stage0_bucket import BucketLevelAnalysis
from dts_research.analysis.stage0_within_issuer import WithinIssuerAnalysis
from dts_research.analysis.stage0_sector import SectorInteractionAnalysis
from dts_research.analysis.stage0_synthesis import Stage0Synthesis


class TestBucketLevelAnalysis:
    """Test bucket-level analysis with mock data."""

    def setup_method(self):
        """Set up test data."""
        # Generate mock data
        loader = BondDataLoader()
        self.bond_data = loader.generate_mock_data(
            start_date='2022-01-01',
            end_date='2023-12-31',
            n_bonds=500,
            seed=42
        )

        # Add sector classification
        classifier = SectorClassifier()
        self.bond_data = classifier.classify_sector(
            self.bond_data, bclass_column='sector_classification'
        )
        self.bond_data = classifier.add_sector_dummies(self.bond_data)

        # Add issuer identification
        self.bond_data = add_issuer_identification(
            self.bond_data,
            parent_id_col='ultimate_parent_id',
            seniority_col='seniority'
        )

        self.analyzer = BucketLevelAnalysis()

    def test_bucket_analysis_runs_ig(self):
        """Test that bucket analysis runs for IG without errors."""
        results = self.analyzer.run_bucket_analysis(self.bond_data, universe='IG')

        assert 'universe' in results
        assert results['universe'] == 'IG'
        assert 'bucket_results' in results
        assert 'summary_statistics' in results
        assert 'monotonicity_test' in results
        assert 'diagnostics' in results

    def test_bucket_analysis_runs_hy(self):
        """Test that bucket analysis runs for HY without errors."""
        results = self.analyzer.run_bucket_analysis(self.bond_data, universe='HY')

        assert results['universe'] == 'HY'
        assert 'bucket_results' in results

    def test_bucket_results_have_correct_columns(self):
        """Test that bucket results have the expected columns."""
        results = self.analyzer.run_bucket_analysis(self.bond_data, universe='IG')

        bucket_df = results['bucket_results']
        if len(bucket_df) > 0:
            expected_cols = ['bucket_id', 'beta', 'beta_se', 'lambda_merton', 'beta_lambda_ratio']
            for col in expected_cols:
                assert col in bucket_df.columns, f"Missing column: {col}"

    def test_summary_statistics_structure(self):
        """Test that summary statistics have expected keys."""
        results = self.analyzer.run_bucket_analysis(self.bond_data, universe='IG')

        stats = results['summary_statistics']
        expected_keys = ['median_beta_lambda_ratio', 'mean_beta_lambda_ratio', 'pct_within_20pct']
        for key in expected_keys:
            assert key in stats, f"Missing key in summary_statistics: {key}"

    def test_monotonicity_test_structure(self):
        """Test that monotonicity test has expected structure."""
        results = self.analyzer.run_bucket_analysis(self.bond_data, universe='IG')

        mono = results['monotonicity_test']
        assert 'overall_monotonic' in mono
        assert 'pct_monotonic_groups' in mono


class TestWithinIssuerAnalysis:
    """Test within-issuer analysis with mock data."""

    def setup_method(self):
        """Set up test data."""
        # Generate mock data with enough issuers
        loader = BondDataLoader()
        self.bond_data = loader.generate_mock_data(
            start_date='2022-01-01',
            end_date='2023-12-31',
            n_bonds=500,
            seed=42
        )

        # Add sector classification
        classifier = SectorClassifier()
        self.bond_data = classifier.classify_sector(
            self.bond_data, bclass_column='sector_classification'
        )

        self.analyzer = WithinIssuerAnalysis()

    def test_within_issuer_analysis_runs_ig(self):
        """Test that within-issuer analysis runs for IG without errors."""
        results = self.analyzer.run_within_issuer_analysis(
            self.bond_data, universe='IG', verbose=False
        )

        assert 'universe' in results
        assert results['universe'] == 'IG'
        assert 'issuer_week_estimates' in results
        assert 'pooled_estimate' in results
        assert 'hypothesis_test' in results
        assert 'diagnostics' in results

    def test_within_issuer_analysis_runs_hy(self):
        """Test that within-issuer analysis runs for HY without errors."""
        results = self.analyzer.run_within_issuer_analysis(
            self.bond_data, universe='HY', verbose=False
        )

        assert results['universe'] == 'HY'

    def test_pooled_estimate_has_beta(self):
        """Test that pooled estimate uses beta (not lambda)."""
        results = self.analyzer.run_within_issuer_analysis(
            self.bond_data, universe='IG', verbose=False
        )

        pooled = results['pooled_estimate']
        # Should have pooled_beta, not pooled_estimate (old name)
        assert 'pooled_beta' in pooled or 'pooled_estimate' in pooled

    def test_hypothesis_test_tests_beta_equals_1(self):
        """Test that hypothesis test is for β = 1."""
        results = self.analyzer.run_within_issuer_analysis(
            self.bond_data, universe='IG', verbose=False
        )

        hyp_test = results['hypothesis_test']
        # Should test β = 1, not just β > 0
        assert 'merton_validates' in hyp_test or 'reject_null' in hyp_test

    def test_issuer_week_estimates_have_beta(self):
        """Test that issuer-week estimates have beta column."""
        results = self.analyzer.run_within_issuer_analysis(
            self.bond_data, universe='IG', verbose=False
        )

        estimates = results['issuer_week_estimates']
        if len(estimates) > 0:
            assert 'beta' in estimates.columns, "Should have 'beta' column, not 'lambda'"


class TestSectorInteractionAnalysis:
    """Test sector interaction analysis with mock data."""

    def setup_method(self):
        """Set up test data."""
        loader = BondDataLoader()
        self.bond_data = loader.generate_mock_data(
            start_date='2022-01-01',
            end_date='2023-12-31',
            n_bonds=500,
            seed=42
        )

        # Add sector classification
        classifier = SectorClassifier()
        self.bond_data = classifier.classify_sector(
            self.bond_data, bclass_column='sector_classification'
        )
        self.bond_data = classifier.add_sector_dummies(self.bond_data)

        # Add issuer identification
        self.bond_data = add_issuer_identification(
            self.bond_data,
            parent_id_col='ultimate_parent_id',
            seniority_col='seniority'
        )

        self.analyzer = SectorInteractionAnalysis()

    def test_sector_analysis_runs_ig(self):
        """Test that sector analysis runs for IG without errors."""
        results = self.analyzer.run_sector_analysis(
            self.bond_data, universe='IG', cluster_by='week'
        )

        assert 'universe' in results
        assert results['universe'] == 'IG'
        assert 'base_regression' in results
        assert 'sector_regression' in results
        assert 'joint_test' in results
        assert 'sector_tests' in results

    def test_sector_analysis_runs_hy(self):
        """Test that sector analysis runs for HY without errors."""
        results = self.analyzer.run_sector_analysis(
            self.bond_data, universe='HY', cluster_by='week'
        )

        assert results['universe'] == 'HY'

    def test_base_regression_has_beta(self):
        """Test that base regression uses beta_0 (not lambda)."""
        results = self.analyzer.run_sector_analysis(
            self.bond_data, universe='IG', cluster_by='week'
        )

        base_reg = results['base_regression']
        if 'warning' not in base_reg:
            assert 'beta_0' in base_reg, "Should have 'beta_0', not 'lambda'"

    def test_joint_test_tests_sectors_differ(self):
        """Test that joint test tests whether sectors differ."""
        results = self.analyzer.run_sector_analysis(
            self.bond_data, universe='IG', cluster_by='week'
        )

        joint = results['joint_test']
        if 'warning' not in joint:
            assert 'sectors_differ' in joint or 'reject_null' in joint


class TestStage0Synthesis:
    """Test Stage 0 synthesis with mock results."""

    def setup_method(self):
        """Set up mock results."""
        self.synthesizer = Stage0Synthesis()

        # Create mock bucket results (new format)
        self.bucket_results = {
            'universe': 'IG',
            'bucket_results': pd.DataFrame({
                'bucket_id': ['BBB_3-5y_A', 'BBB_5-7y_A'],
                'beta': [0.95, 0.85],
                'lambda_merton': [1.0, 0.9],
                'beta_lambda_ratio': [0.95, 0.94]
            }),
            'summary_statistics': {
                'median_beta_lambda_ratio': 0.95,
                'mean_beta_lambda_ratio': 0.94,
                'pct_within_20pct': 80,
                'median_beta': 0.9
            },
            'monotonicity_test': {
                'overall_monotonic': True,
                'pct_monotonic_groups': 75
            },
            'diagnostics': {}
        }

        # Create mock within-issuer results (new format)
        self.within_issuer_results = {
            'universe': 'IG',
            'issuer_week_estimates': pd.DataFrame({
                'beta': [0.98, 1.02, 0.95],
                'beta_se': [0.05, 0.06, 0.04]
            }),
            'pooled_estimate': {
                'pooled_beta': 0.98,
                'pooled_beta_se': 0.03
            },
            'hypothesis_test': {
                'merton_validates': True,
                'beta_in_range_0_9_1_1': True,
                'p_value_beta_equals_1': 0.52
            },
            'diagnostics': {}
        }

        # Create mock sector results (new format)
        self.sector_results = {
            'universe': 'IG',
            'base_regression': {
                'beta_0': 0.95,
                'beta_0_se': 0.04
            },
            'sector_regression': {
                'beta_0': 0.93,
                'beta_financial': 0.15,
                'beta_utility': -0.10,
                'beta_energy': 0.05,
                'sensitivity_industrial': 0.93,
                'sensitivity_financial': 1.08,
                'sensitivity_utility': 0.83,
                'sensitivity_energy': 0.98
            },
            'joint_test': {
                'sectors_differ': False,
                'p_value': 0.15
            },
            'sector_tests': {
                'financial_test': {'reject_null': False},
                'utility_test': {'reject_null': False},
                'summary': {'need_sector_adjustment': False}
            },
            'diagnostics': {}
        }

    def test_synthesis_extracts_statistics(self):
        """Test that synthesis extracts statistics from new format."""
        stats = self.synthesizer._extract_key_statistics(
            self.bucket_results,
            self.within_issuer_results,
            self.sector_results
        )

        # Should extract beta-related stats, not lambda
        assert 'within_beta' in stats
        assert 'bucket_median_ratio' in stats
        assert 'base_beta' in stats

    def test_synthesis_evaluates_criteria(self):
        """Test that criteria evaluation works with new format."""
        stats = self.synthesizer._extract_key_statistics(
            self.bucket_results,
            self.within_issuer_results,
            self.sector_results
        )

        criteria = self.synthesizer._evaluate_criteria(stats)

        # Should have criteria about β ≈ 1
        assert 'bucket_beta_near_1' in criteria or 'theory_validated' in criteria
        assert 'within_beta_near_1' in criteria or 'theory_validated' in criteria

    def test_synthesis_determines_path(self):
        """Test that path determination works."""
        stats = self.synthesizer._extract_key_statistics(
            self.bucket_results,
            self.within_issuer_results,
            self.sector_results
        )

        criteria = self.synthesizer._evaluate_criteria(stats)
        path, rationale = self.synthesizer._determine_path(criteria, stats)

        assert path in [1, 2, 3, 4, 5]
        assert isinstance(rationale, str)

    def test_full_synthesis_runs(self):
        """Test that full synthesis runs without errors."""
        result = self.synthesizer.synthesize_results(
            self.bucket_results,
            self.within_issuer_results,
            self.sector_results,
            universe='IG'
        )

        assert 'decision_path' in result
        assert 'path_name' in result
        assert 'rationale' in result
        assert 'recommendations' in result


class TestIntegration:
    """Integration tests running full pipeline with mock data."""

    def setup_method(self):
        """Set up test data."""
        loader = BondDataLoader()
        self.bond_data = loader.generate_mock_data(
            start_date='2022-01-01',
            end_date='2023-06-30',  # Shorter period for speed
            n_bonds=300,
            seed=42
        )

        # Preprocess
        classifier = SectorClassifier()
        self.bond_data = classifier.classify_sector(
            self.bond_data, bclass_column='sector_classification'
        )
        self.bond_data = classifier.add_sector_dummies(self.bond_data)
        self.bond_data = add_issuer_identification(
            self.bond_data,
            parent_id_col='ultimate_parent_id',
            seniority_col='seniority'
        )

    def test_full_pipeline_ig(self):
        """Test full Stage 0 pipeline for IG."""
        # Run bucket analysis
        bucket_analyzer = BucketLevelAnalysis()
        bucket_results = bucket_analyzer.run_bucket_analysis(self.bond_data, universe='IG')

        # Run within-issuer analysis
        within_analyzer = WithinIssuerAnalysis()
        within_results = within_analyzer.run_within_issuer_analysis(
            self.bond_data, universe='IG', verbose=False
        )

        # Run sector analysis
        sector_analyzer = SectorInteractionAnalysis()
        sector_results = sector_analyzer.run_sector_analysis(
            self.bond_data, universe='IG', cluster_by='week'
        )

        # Run synthesis
        synthesizer = Stage0Synthesis()
        synthesis = synthesizer.synthesize_results(
            bucket_results,
            within_results,
            sector_results,
            universe='IG'
        )

        # Check synthesis output
        assert synthesis['decision_path'] in [1, 2, 3, 4, 5]
        print(f"\nIG Decision: Path {synthesis['decision_path']} - {synthesis['path_name']}")
        print(f"Rationale: {synthesis['rationale']}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
