"""
Stage 0: Synthesis and Decision Framework

Combines results from three analyses to determine best modeling path:
1. Bucket-level analysis → Tests cross-sectional λ and monotonicity
2. Within-issuer analysis → Tests λ > 0 (Merton prediction)
3. Sector analysis → Tests for sector heterogeneity

Decision Framework (5 paths):
- Path 1: Standard DTS (λ > 0, monotonic, no sectors)
- Path 2: Pure Merton (λ consistent with theory, use calibrated λ)
- Path 3: Calibrated Merton (adjust λ to data)
- Path 4: Merton + Sectors (significant sector effects)
- Path 5: Theory Fails (λ ≤ 0 or non-monotonic, use alternative model)

Based on Section 2.4 from the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class Stage0Synthesis:
    """
    Synthesis of Stage 0 results and decision framework.

    Combines bucket-level, within-issuer, and sector analyses to determine
    the optimal modeling path for Stages A-E.
    """

    def __init__(self):
        """Initialize synthesis."""
        self.decision_paths = {
            1: "Standard DTS",
            2: "Pure Merton (calibrated λ)",
            3: "Calibrated Merton (data-driven λ)",
            4: "Merton + Sectors",
            5: "Theory Fails (alternative model)"
        }

    def synthesize_results(
        self,
        bucket_results: Dict,
        within_issuer_results: Dict,
        sector_results: Dict,
        universe: str = 'IG'
    ) -> Dict:
        """
        Synthesize Stage 0 results and determine modeling path.

        Args:
            bucket_results: Results from bucket-level analysis
            within_issuer_results: Results from within-issuer analysis
            sector_results: Results from sector interaction analysis
            universe: 'IG' or 'HY'

        Returns:
            Dictionary with:
            - decision_path: Path number (1-5)
            - path_name: Path description
            - rationale: Explanation of decision
            - key_statistics: Summary statistics for decision
            - recommendations: Next steps for Stages A-E
        """
        # Extract key statistics
        stats = self._extract_key_statistics(
            bucket_results, within_issuer_results, sector_results
        )

        # Evaluate decision criteria
        criteria = self._evaluate_criteria(stats)

        # Determine path
        path, rationale = self._determine_path(criteria, stats)

        # Generate recommendations
        recommendations = self._generate_recommendations(path, stats, universe)

        return {
            'universe': universe,
            'decision_path': path,
            'path_name': self.decision_paths[path],
            'rationale': rationale,
            'key_statistics': stats,
            'decision_criteria': criteria,
            'recommendations': recommendations
        }

    def _extract_key_statistics(
        self,
        bucket_results: Dict,
        within_issuer_results: Dict,
        sector_results: Dict
    ) -> Dict:
        """
        Extract key statistics from all three analyses.

        Args:
            bucket_results: Bucket analysis results
            within_issuer_results: Within-issuer analysis results
            sector_results: Sector analysis results

        Returns:
            Dictionary with key statistics
        """
        # Bucket-level statistics
        bucket_lambda = bucket_results.get('regression_results', {}).get('lambda', np.nan)
        bucket_pvalue = bucket_results.get('regression_results', {}).get('lambda_pvalue', np.nan)
        bucket_r2 = bucket_results.get('regression_results', {}).get('r_squared', np.nan)
        monotonic = bucket_results.get('monotonicity_test', {}).get('overall_monotonic', False)
        pct_monotonic = bucket_results.get('monotonicity_test', {}).get('pct_monotonic_groups', 0)

        # Within-issuer statistics
        within_lambda = within_issuer_results.get('pooled_estimate', {}).get('pooled_estimate', np.nan)
        within_se = within_issuer_results.get('pooled_estimate', {}).get('pooled_se', np.nan)
        within_positive = within_issuer_results.get('hypothesis_test', {}).get('reject_null', False)
        within_pvalue = within_issuer_results.get('hypothesis_test', {}).get('p_value', np.nan)

        # Sector statistics
        base_lambda = sector_results.get('base_regression', {}).get('lambda', np.nan)
        sector_lambda = sector_results.get('sector_regression', {}).get('lambda', np.nan)
        sectors_significant = sector_results.get('joint_test', {}).get('reject_null', False)
        sector_pvalue = sector_results.get('joint_test', {}).get('p_value', np.nan)
        fin_positive = sector_results.get('sector_tests', {}).get('financial_test', {}).get('reject_null', False)
        util_negative = sector_results.get('sector_tests', {}).get('utility_test', {}).get('reject_null', False)

        return {
            'bucket_lambda': bucket_lambda,
            'bucket_pvalue': bucket_pvalue,
            'bucket_r2': bucket_r2,
            'monotonic': monotonic,
            'pct_monotonic': pct_monotonic,
            'within_lambda': within_lambda,
            'within_se': within_se,
            'within_positive': within_positive,
            'within_pvalue': within_pvalue,
            'base_lambda': base_lambda,
            'sector_lambda': sector_lambda,
            'sectors_significant': sectors_significant,
            'sector_pvalue': sector_pvalue,
            'financial_positive': fin_positive,
            'utility_negative': util_negative
        }

    def _evaluate_criteria(self, stats: Dict) -> Dict:
        """
        Evaluate decision criteria based on statistics.

        Criteria:
        1. λ > 0: At least one analysis shows significantly positive λ
        2. Monotonic: Bucket analysis shows monotonic pattern
        3. Consistent: Bucket and within-issuer λ are similar
        4. Sectors matter: Sector interactions are significant

        Args:
            stats: Key statistics dictionary

        Returns:
            Dictionary with boolean criteria
        """
        # Criterion 1: λ > 0
        lambda_positive = (
            stats['within_positive'] or
            (not np.isnan(stats['bucket_lambda']) and stats['bucket_lambda'] > 0 and stats['bucket_pvalue'] < 0.05) or
            (not np.isnan(stats['base_lambda']) and stats['base_lambda'] > 0)
        )

        # Criterion 2: Monotonic
        monotonic = stats['monotonic']

        # Criterion 3: Consistent λ across analyses
        if not np.isnan(stats['bucket_lambda']) and not np.isnan(stats['within_lambda']):
            # Check if estimates are within 2 standard errors
            diff = abs(stats['bucket_lambda'] - stats['within_lambda'])
            threshold = 2 * stats['within_se'] if not np.isnan(stats['within_se']) else 0.1
            consistent = diff < threshold
        else:
            consistent = False  # Can't assess if data missing

        # Criterion 4: Sectors matter
        sectors_matter = stats['sectors_significant']

        return {
            'lambda_positive': lambda_positive,
            'monotonic': monotonic,
            'consistent_lambda': consistent,
            'sectors_matter': sectors_matter
        }

    def _determine_path(self, criteria: Dict, stats: Dict) -> Tuple[int, str]:
        """
        Determine decision path based on criteria.

        Decision tree:
        1. If λ ≤ 0 or non-monotonic → Path 5 (Theory Fails)
        2. If sectors significant → Path 4 (Merton + Sectors)
        3. If consistent and monotonic → Path 1 (Standard DTS)
        4. If positive but inconsistent → Path 3 (Calibrated Merton)
        5. Default → Path 2 (Pure Merton)

        Args:
            criteria: Evaluated criteria
            stats: Key statistics

        Returns:
            Tuple of (path_number, rationale)
        """
        # Path 5: Theory fails
        if not criteria['lambda_positive']:
            return (5, "λ is not significantly positive - Merton model fails. Consider alternative models.")

        if not criteria['monotonic']:
            return (5, "λ is non-monotonic across maturity - Merton model fails. Consider alternative models.")

        # Path 4: Sectors matter
        if criteria['sectors_matter']:
            fin_msg = "Financial sector has positive λ" if stats['financial_positive'] else ""
            util_msg = "Utility sector has negative λ" if stats['utility_negative'] else ""
            sector_msg = " and ".join([m for m in [fin_msg, util_msg] if m])
            return (4, f"Sector interactions are significant. {sector_msg}. Use Merton + Sectors model.")

        # Path 1: Standard DTS
        if criteria['consistent_lambda'] and criteria['monotonic']:
            return (1, f"λ is positive (≈{stats['within_lambda']:.4f}), monotonic, and consistent across analyses. Use standard DTS.")

        # Path 3: Calibrated Merton
        if criteria['lambda_positive'] and not criteria['consistent_lambda']:
            return (3, f"λ is positive but inconsistent between bucket ({stats['bucket_lambda']:.4f}) and within-issuer ({stats['within_lambda']:.4f}). Calibrate λ to data.")

        # Path 2: Pure Merton (default)
        return (2, "λ is positive. Use pure Merton calibration with theoretical λ as starting point.")

    def _generate_recommendations(
        self,
        path: int,
        stats: Dict,
        universe: str
    ) -> Dict:
        """
        Generate recommendations for Stages A-E based on path.

        Args:
            path: Decision path number
            stats: Key statistics
            universe: 'IG' or 'HY'

        Returns:
            Dictionary with recommendations for each stage
        """
        if path == 1:  # Standard DTS
            return {
                'stage_A': f"Use bucket λ = {stats['bucket_lambda']:.4f} as initial estimate",
                'stage_B': "Proceed with standard DTS specification (no sector adjustments)",
                'stage_C': "Test for time-variation in λ",
                'stage_D': "Standard robustness checks",
                'stage_E': "Use Specification 1 (Standard DTS)"
            }

        elif path == 2:  # Pure Merton
            return {
                'stage_A': "Use Merton calibration: λ = f(leverage, volatility, risk-free rate)",
                'stage_B': "Proceed with Merton-calibrated λ (no sector adjustments)",
                'stage_C': "Test for time-variation in leverage/volatility inputs",
                'stage_D': "Robustness to calibration parameters",
                'stage_E': "Use Specification 2 (Pure Merton)"
            }

        elif path == 3:  # Calibrated Merton
            avg_lambda = np.nanmean([stats['bucket_lambda'], stats['within_lambda'], stats['base_lambda']])
            return {
                'stage_A': f"Use data-calibrated λ = {avg_lambda:.4f} (average of analyses)",
                'stage_B': "Proceed with calibrated λ (no sector adjustments)",
                'stage_C': "Test for time-variation in data-calibrated λ",
                'stage_D': "Robustness to alternative calibration methods",
                'stage_E': "Use Specification 3 (Calibrated Merton)"
            }

        elif path == 4:  # Merton + Sectors
            return {
                'stage_A': f"Use sector-specific λ: Base = {stats['sector_lambda']:.4f}, adjust by sector",
                'stage_B': "Use Specification B.3 with sector interactions",
                'stage_C': "Test for sector-specific time-variation",
                'stage_D': "Robustness by sector subsamples",
                'stage_E': "Use sector-specific specifications (2a, 2b, 3a, 4a)"
            }

        else:  # Path 5: Theory Fails
            return {
                'stage_A': "CAUTION: Standard specifications may not apply",
                'stage_B': "Consider alternative factor models (ratings-based, PCA)",
                'stage_C': "Skip (theory-driven tests not applicable)",
                'stage_D': "Focus on model-free robustness",
                'stage_E': "Do not use Merton-based specifications"
            }

    def compare_ig_hy_paths(
        self,
        ig_synthesis: Dict,
        hy_synthesis: Dict
    ) -> Dict:
        """
        Compare decision paths between IG and HY.

        Args:
            ig_synthesis: Synthesis results for IG
            hy_synthesis: Synthesis results for HY

        Returns:
            Dictionary with comparison and interpretation
        """
        ig_path = ig_synthesis['decision_path']
        hy_path = hy_synthesis['decision_path']

        same_path = (ig_path == hy_path)

        interpretation = self._interpret_path_comparison(ig_path, hy_path)

        return {
            'ig_path': ig_path,
            'ig_path_name': ig_synthesis['path_name'],
            'hy_path': hy_path,
            'hy_path_name': hy_synthesis['path_name'],
            'same_path': same_path,
            'interpretation': interpretation,
            'unified_approach': self._suggest_unified_approach(ig_path, hy_path)
        }

    def _interpret_path_comparison(self, ig_path: int, hy_path: int) -> str:
        """Interpret path comparison between IG and HY."""
        if ig_path == hy_path:
            return f"Both IG and HY follow Path {ig_path} ({self.decision_paths[ig_path]}) - use unified approach"

        if ig_path == 4 or hy_path == 4:
            return "Sector effects differ between IG and HY - use universe-specific approaches"

        if ig_path == 5 or hy_path == 5:
            if ig_path == 5 and hy_path == 5:
                return "Theory fails in both universes - major concern, investigate further"
            elif ig_path == 5:
                return "Theory fails in IG but holds in HY - use HY approach for IG with caution"
            else:
                return "Theory fails in HY but holds in IG - use IG approach for HY with caution"

        return f"Paths differ (IG: {ig_path}, HY: {hy_path}) but both are theory-consistent - use separate approaches"

    def _suggest_unified_approach(self, ig_path: int, hy_path: int) -> str:
        """Suggest unified approach across universes."""
        if ig_path == hy_path:
            return f"Use Path {ig_path} for both IG and HY"

        # If one uses sectors, use sectors for both
        if ig_path == 4 or hy_path == 4:
            return "Use Path 4 (Merton + Sectors) for both, even if sectors not significant in one universe"

        # If one fails, investigate the other
        if ig_path == 5 or hy_path == 5:
            working_path = hy_path if ig_path == 5 else ig_path
            return f"Use Path {working_path} with caution, investigate why theory fails in other universe"

        # Otherwise use more sophisticated approach
        max_path = max(ig_path, hy_path)
        return f"Use Path {max_path} for both (more sophisticated approach)"


def run_stage0_synthesis(
    bucket_results_ig: Dict,
    bucket_results_hy: Dict,
    within_issuer_results_ig: Dict,
    within_issuer_results_hy: Dict,
    sector_results_ig: Dict,
    sector_results_hy: Dict
) -> Dict:
    """
    Convenience function to run complete Stage 0 synthesis for both universes.

    Args:
        bucket_results_ig: Bucket analysis results for IG
        bucket_results_hy: Bucket analysis results for HY
        within_issuer_results_ig: Within-issuer results for IG
        within_issuer_results_hy: Within-issuer results for HY
        sector_results_ig: Sector analysis results for IG
        sector_results_hy: Sector analysis results for HY

    Returns:
        Dictionary with synthesis for both universes plus comparison
    """
    synthesizer = Stage0Synthesis()

    # Synthesize IG
    ig_synthesis = synthesizer.synthesize_results(
        bucket_results_ig,
        within_issuer_results_ig,
        sector_results_ig,
        universe='IG'
    )

    # Synthesize HY
    hy_synthesis = synthesizer.synthesize_results(
        bucket_results_hy,
        within_issuer_results_hy,
        sector_results_hy,
        universe='HY'
    )

    # Compare paths
    comparison = synthesizer.compare_ig_hy_paths(ig_synthesis, hy_synthesis)

    return {
        'IG': ig_synthesis,
        'HY': hy_synthesis,
        'comparison': comparison
    }
