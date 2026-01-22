"""
Stage 0: Synthesis and Decision Framework

Combines results from three analyses to determine best modeling path:
1. Bucket-level analysis → Tests if empirical β ≈ theoretical λ^Merton
2. Within-issuer analysis → Tests if β ≈ 1 (Merton cross-maturity predictions)
3. Sector analysis → Tests for sector-specific DTS sensitivities

Decision Framework (5 paths based on whether β ≈ 1):
- Path 1: Standard DTS (β ≈ 1 everywhere, no sectors)
- Path 2: Pure Merton (β ≈ 1, use theoretical λ tables)
- Path 3: Calibrated Merton (β consistent but ≠ 1, need scaling)
- Path 4: Merton + Sectors (significant sector deviations)
- Path 5: Theory Fails (β not positive or non-monotonic patterns)

Based on Section 4.5 from the paper.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class Stage0Synthesis:
    """
    Synthesis of Stage 0 results and decision framework.

    Combines bucket-level, within-issuer, and sector analyses to determine
    the optimal modeling path for Stages A-E.

    Key test: Does β ≈ 1? (Not just: Is λ > 0?)
    """

    def __init__(self):
        """Initialize synthesis."""
        self.decision_paths = {
            1: "Standard DTS (theory works well)",
            2: "Pure Merton (use theoretical λ tables)",
            3: "Calibrated Merton (scale β to data)",
            4: "Merton + Sectors (sector-specific adjustments)",
            5: "Theory Fails (consider alternative models)"
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
        # Bucket-level statistics (β/λ ratio should be ≈ 1)
        summary_stats = bucket_results.get('summary_statistics', {})
        bucket_median_ratio = summary_stats.get('median_beta_lambda_ratio', np.nan)
        bucket_pct_within_20 = summary_stats.get('pct_within_20pct', 0)
        bucket_median_beta = summary_stats.get('median_beta', np.nan)
        monotonic = bucket_results.get('monotonicity_test', {}).get('overall_monotonic', False)
        pct_monotonic = bucket_results.get('monotonicity_test', {}).get('pct_monotonic_groups', 0)

        # Within-issuer statistics (β should be ≈ 1)
        pooled = within_issuer_results.get('pooled_estimate', {})
        within_beta = pooled.get('pooled_beta', np.nan)
        within_beta_se = pooled.get('pooled_beta_se', np.nan)
        hyp_test = within_issuer_results.get('hypothesis_test', {})
        within_validates = hyp_test.get('merton_validates', False)
        within_beta_in_range = hyp_test.get('beta_in_range_0_9_1_1', False)
        within_pvalue_eq_1 = hyp_test.get('p_value_beta_equals_1', np.nan)

        # Sector statistics
        base_beta = sector_results.get('base_regression', {}).get('beta_0', np.nan)
        sectors_differ = sector_results.get('joint_test', {}).get('sectors_differ', False)
        sector_pvalue = sector_results.get('joint_test', {}).get('p_value', np.nan)
        sector_tests = sector_results.get('sector_tests', {})
        fin_amplifies = sector_tests.get('financial_test', {}).get('reject_null', False)
        util_dampens = sector_tests.get('utility_test', {}).get('reject_null', False)
        need_sector_adj = sector_tests.get('summary', {}).get('need_sector_adjustment', False)

        return {
            # Bucket-level
            'bucket_median_ratio': bucket_median_ratio,
            'bucket_pct_within_20': bucket_pct_within_20,
            'bucket_median_beta': bucket_median_beta,
            'monotonic': monotonic,
            'pct_monotonic': pct_monotonic,
            # Within-issuer
            'within_beta': within_beta,
            'within_beta_se': within_beta_se,
            'within_validates': within_validates,
            'within_beta_in_range': within_beta_in_range,
            'within_pvalue_eq_1': within_pvalue_eq_1,
            # Sector
            'base_beta': base_beta,
            'sectors_differ': sectors_differ,
            'sector_pvalue': sector_pvalue,
            'financial_amplifies': fin_amplifies,
            'utility_dampens': util_dampens,
            'need_sector_adjustment': need_sector_adj
        }

    def _evaluate_criteria(self, stats: Dict) -> Dict:
        """
        Evaluate decision criteria based on statistics.

        Criteria (based on testing β ≈ 1, not λ > 0):
        1. β ≈ 1 at bucket level: Median β/λ ratio in [0.9, 1.1]
        2. β ≈ 1 at within-issuer level: Pooled β in [0.9, 1.1]
        3. Monotonic: β decreases with maturity as Merton predicts
        4. Sectors matter: Sector interactions are significant
        5. Consistent: Bucket and within-issuer analyses agree

        Args:
            stats: Key statistics dictionary

        Returns:
            Dictionary with boolean criteria
        """
        # Criterion 1: β ≈ 1 at bucket level (ratio of empirical β to theoretical λ)
        bucket_ratio = stats['bucket_median_ratio']
        bucket_beta_near_1 = (
            not np.isnan(bucket_ratio) and
            0.9 <= bucket_ratio <= 1.1 and
            stats['bucket_pct_within_20'] >= 60  # At least 60% of buckets fit
        )

        # Criterion 2: β ≈ 1 at within-issuer level
        within_beta = stats['within_beta']
        within_beta_near_1 = (
            not np.isnan(within_beta) and
            0.9 <= within_beta <= 1.1 and
            (np.isnan(stats['within_pvalue_eq_1']) or stats['within_pvalue_eq_1'] > 0.10)
        )

        # Criterion 3: Monotonic (β decreases with maturity)
        monotonic = stats['monotonic']

        # Criterion 4: Sectors matter
        sectors_matter = stats['sectors_differ'] or stats['need_sector_adjustment']

        # Criterion 5: Analyses are consistent
        if not np.isnan(bucket_ratio) and not np.isnan(within_beta):
            # Both should be near 1, or both should deviate in same direction
            bucket_ok = 0.8 <= bucket_ratio <= 1.2
            within_ok = 0.8 <= within_beta <= 1.2
            consistent = bucket_ok and within_ok
        else:
            consistent = False

        # Criterion 6: Base sector β ≈ 1
        base_beta = stats['base_beta']
        base_beta_near_1 = not np.isnan(base_beta) and 0.8 <= base_beta <= 1.2

        return {
            'bucket_beta_near_1': bucket_beta_near_1,
            'within_beta_near_1': within_beta_near_1,
            'monotonic': monotonic,
            'sectors_matter': sectors_matter,
            'consistent': consistent,
            'base_beta_near_1': base_beta_near_1,
            'theory_validated': bucket_beta_near_1 and within_beta_near_1 and monotonic
        }

    def _determine_path(self, criteria: Dict, stats: Dict) -> Tuple[int, str]:
        """
        Determine decision path based on criteria.

        Decision tree (based on β ≈ 1, not λ > 0):
        1. If patterns wrong (non-monotonic or β far from 1) → Path 5 (Theory Fails)
        2. If sectors significant → Path 4 (Merton + Sectors)
        3. If β ≈ 1 everywhere → Path 1 (Standard DTS) or Path 2 (Pure Merton)
        4. If β consistent but ≠ 1 → Path 3 (Calibrated Merton)
        5. Default → Path 2 (Pure Merton)

        Args:
            criteria: Evaluated criteria
            stats: Key statistics

        Returns:
            Tuple of (path_number, rationale)
        """
        within_beta = stats['within_beta']
        bucket_ratio = stats['bucket_median_ratio']

        # Path 5: Theory fails
        if not criteria['monotonic']:
            return (5, "β does NOT decrease with maturity as Merton predicts - theory fails")

        # Check if β is wildly off (not just ≠ 1, but totally wrong direction or magnitude)
        if not np.isnan(within_beta) and (within_beta < 0 or within_beta > 2.0):
            return (5, f"Within-issuer β = {within_beta:.2f} is far from 1 - theory substantially fails")

        if not np.isnan(bucket_ratio) and (bucket_ratio < 0.5 or bucket_ratio > 2.0):
            return (5, f"Bucket β/λ ratio = {bucket_ratio:.2f} is far from 1 - theory substantially fails")

        # Path 4: Sectors matter
        if criteria['sectors_matter']:
            fin_msg = "Financials amplify" if stats['financial_amplifies'] else ""
            util_msg = "Utilities dampen" if stats['utility_dampens'] else ""
            sector_msg = ", ".join([m for m in [fin_msg, util_msg] if m]) or "sectors differ"
            return (4, f"Sector effects significant: {sector_msg}. Use Merton + Sectors.")

        # Path 1 or 2: Theory works well
        if criteria['theory_validated']:
            if criteria['within_beta_near_1'] and criteria['bucket_beta_near_1']:
                return (1, f"β ≈ 1 validated: within-issuer β = {within_beta:.2f}, bucket ratio = {bucket_ratio:.2f}. Use standard DTS.")
            else:
                return (2, f"Theory works but not perfectly: within-issuer β = {within_beta:.2f}. Use Pure Merton tables.")

        # Path 3: Theory works but needs calibration
        if criteria['consistent'] and not criteria['theory_validated']:
            return (3, f"β consistent but ≠ 1 (within-issuer: {within_beta:.2f}, bucket ratio: {bucket_ratio:.2f}). Calibrate β to data.")

        # Path 2: Default - use Merton with caution
        return (2, f"Partial validation. Use Pure Merton as starting point (within-issuer β = {within_beta:.2f}).")

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
        within_beta = stats['within_beta']
        bucket_ratio = stats['bucket_median_ratio']
        base_beta = stats['base_beta']

        if path == 1:  # Standard DTS (theory validated)
            return {
                'stage_A': f"Use Merton λ tables directly (β ≈ 1 validated: {within_beta:.2f})",
                'stage_B': "Proceed with standard DTS specification (no adjustments needed)",
                'stage_C': "Test for time-variation in base sensitivity",
                'stage_D': "Standard robustness checks",
                'stage_E': "Use Specification 1 (Standard Merton DTS)"
            }

        elif path == 2:  # Pure Merton
            return {
                'stage_A': f"Use Merton λ tables (β = {within_beta:.2f}, close enough)",
                'stage_B': "Proceed with Merton-based λ (minor scaling may help)",
                'stage_C': "Test for time-variation in sensitivity",
                'stage_D': "Robustness to λ table interpolation",
                'stage_E': "Use Specification 2 (Pure Merton)"
            }

        elif path == 3:  # Calibrated Merton
            # Compute calibration scaling factor
            scaling = within_beta if not np.isnan(within_beta) else bucket_ratio if not np.isnan(bucket_ratio) else 1.0
            return {
                'stage_A': f"Use λ_prod = λ_Merton × {scaling:.2f} (calibrated scaling)",
                'stage_B': f"Apply scaling factor β = {scaling:.2f} to Merton predictions",
                'stage_C': "Test for time-variation in calibration factor",
                'stage_D': "Robustness to alternative calibration methods",
                'stage_E': "Use Specification 3 (Calibrated Merton)"
            }

        elif path == 4:  # Merton + Sectors
            return {
                'stage_A': f"Use sector-specific λ: Base = {base_beta:.2f} × λ_Merton",
                'stage_B': "Use Specification B.3 with sector interactions",
                'stage_C': "Test for sector-specific time-variation",
                'stage_D': "Robustness by sector subsamples",
                'stage_E': "Use sector-adjusted specifications"
            }

        else:  # Path 5: Theory Fails
            return {
                'stage_A': "CAUTION: Merton theory does not match data patterns",
                'stage_B': "Consider alternative factor models (empirical PCA, ratings-based)",
                'stage_C': "Focus on empirical patterns rather than theory-driven tests",
                'stage_D': "Extensive robustness checks required",
                'stage_E': "Use empirical specifications rather than Merton-based"
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
