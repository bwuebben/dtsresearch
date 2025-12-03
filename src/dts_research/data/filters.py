"""
Enhanced data filtering utilities for Stage 0 analyses.

Implements sophisticated filters for:
- Within-issuer analysis (maturity dispersion, pull-to-par exclusion)
- Outlier detection and removal
- Sample quality validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from ..config import (
    MIN_BONDS_PER_ISSUER_WEEK,
    MIN_MATURITY_DISPERSION_YEARS,
    PULL_TO_PAR_EXCLUSION_YEARS,
    MAX_SPREAD_CHANGE_PCT
)


class DataFilters:
    """
    Sophisticated data filtering for Stage 0 and beyond.
    """

    def __init__(self):
        """Initialize data filters with configuration."""
        self.min_bonds_per_issuer_week = MIN_BONDS_PER_ISSUER_WEEK
        self.min_maturity_dispersion = MIN_MATURITY_DISPERSION_YEARS
        self.pull_to_par_exclusion = PULL_TO_PAR_EXCLUSION_YEARS
        self.max_spread_change = MAX_SPREAD_CHANGE_PCT

    def filter_for_within_issuer_analysis(
        self,
        bond_data: pd.DataFrame,
        maturity_column: str = 'time_to_maturity',
        spread_change_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply all within-issuer analysis filters.

        Filters applied:
        1. ≥ min_bonds bonds per issuer per week
        2. ≥ min_dispersion years maturity dispersion per issuer-week
        3. Exclude bonds within pull_to_par years of maturity
        4. Exclude spread changes > max_spread_change %

        Args:
            bond_data: DataFrame with bond observations
            maturity_column: Column name for time to maturity
            spread_change_column: Optional column for spread changes

        Returns:
            Tuple of (filtered_data, filter_statistics)
        """
        initial_count = len(bond_data)
        stats = {'initial_observations': initial_count}

        # Ensure required columns exist
        if 'issuer_id' not in bond_data.columns:
            raise ValueError("issuer_id column required. Run issuer identification first.")
        if 'date' not in bond_data.columns:
            raise ValueError("date column required for within-issuer analysis.")

        # Filter 1: Min bonds per issuer-week
        bond_data, filter1_stats = self._filter_min_bonds_per_issuer_week(bond_data)
        stats['filter1_min_bonds'] = filter1_stats

        # Filter 2: Min maturity dispersion
        bond_data, filter2_stats = self._filter_maturity_dispersion(
            bond_data, maturity_column
        )
        stats['filter2_maturity_dispersion'] = filter2_stats

        # Filter 3: Pull-to-par exclusion
        bond_data, filter3_stats = self._filter_pull_to_par(
            bond_data, maturity_column
        )
        stats['filter3_pull_to_par'] = filter3_stats

        # Filter 4: Spread change outliers (if column provided)
        if spread_change_column and spread_change_column in bond_data.columns:
            bond_data, filter4_stats = self._filter_spread_change_outliers(
                bond_data, spread_change_column
            )
            stats['filter4_spread_outliers'] = filter4_stats
        else:
            stats['filter4_spread_outliers'] = {'skipped': True}

        stats['final_observations'] = len(bond_data)
        stats['pct_retained'] = 100.0 * len(bond_data) / initial_count if initial_count > 0 else 0

        return bond_data, stats

    def _filter_min_bonds_per_issuer_week(
        self,
        bond_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Keep only issuer-weeks with ≥ min_bonds bonds.

        Args:
            bond_data: DataFrame with issuer_id and date

        Returns:
            Tuple of (filtered_data, statistics)
        """
        initial_count = len(bond_data)

        # Count bonds per issuer-week
        issuer_week_counts = bond_data.groupby(['issuer_id', 'date']).size()

        # Keep only issuer-weeks with enough bonds
        valid_issuer_weeks = issuer_week_counts[
            issuer_week_counts >= self.min_bonds_per_issuer_week
        ].index

        # Filter bond data
        bond_data['issuer_week'] = list(zip(bond_data['issuer_id'], bond_data['date']))
        filtered = bond_data[
            bond_data['issuer_week'].isin(valid_issuer_weeks)
        ].copy()
        filtered = filtered.drop('issuer_week', axis=1)

        stats = {
            'initial_obs': initial_count,
            'final_obs': len(filtered),
            'removed': initial_count - len(filtered),
            'pct_removed': 100.0 * (initial_count - len(filtered)) / initial_count if initial_count > 0 else 0,
            'threshold': self.min_bonds_per_issuer_week
        }

        return filtered, stats

    def _filter_maturity_dispersion(
        self,
        bond_data: pd.DataFrame,
        maturity_column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Keep only issuer-weeks with ≥ min_dispersion years maturity range.

        Args:
            bond_data: DataFrame with issuer_id, date, and maturity
            maturity_column: Column name for time to maturity

        Returns:
            Tuple of (filtered_data, statistics)
        """
        initial_count = len(bond_data)

        # Compute maturity range per issuer-week
        def maturity_range(group):
            return group[maturity_column].max() - group[maturity_column].min()

        issuer_week_dispersion = bond_data.groupby(
            ['issuer_id', 'date']
        ).apply(maturity_range)

        # Keep only issuer-weeks with sufficient dispersion
        valid_issuer_weeks = issuer_week_dispersion[
            issuer_week_dispersion >= self.min_maturity_dispersion
        ].index

        # Filter bond data
        bond_data['issuer_week'] = list(zip(bond_data['issuer_id'], bond_data['date']))
        filtered = bond_data[
            bond_data['issuer_week'].isin(valid_issuer_weeks)
        ].copy()
        filtered = filtered.drop('issuer_week', axis=1)

        stats = {
            'initial_obs': initial_count,
            'final_obs': len(filtered),
            'removed': initial_count - len(filtered),
            'pct_removed': 100.0 * (initial_count - len(filtered)) / initial_count if initial_count > 0 else 0,
            'threshold_years': self.min_maturity_dispersion
        }

        return filtered, stats

    def _filter_pull_to_par(
        self,
        bond_data: pd.DataFrame,
        maturity_column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Exclude bonds within pull_to_par years of maturity.

        Near maturity, spread changes are dominated by pull-to-par
        mechanical effects, not credit fundamentals.

        Args:
            bond_data: DataFrame with maturity
            maturity_column: Column name for time to maturity

        Returns:
            Tuple of (filtered_data, statistics)
        """
        initial_count = len(bond_data)

        # Keep bonds with maturity > threshold
        filtered = bond_data[
            bond_data[maturity_column] > self.pull_to_par_exclusion
        ].copy()

        stats = {
            'initial_obs': initial_count,
            'final_obs': len(filtered),
            'removed': initial_count - len(filtered),
            'pct_removed': 100.0 * (initial_count - len(filtered)) / initial_count if initial_count > 0 else 0,
            'threshold_years': self.pull_to_par_exclusion
        }

        return filtered, stats

    def _filter_spread_change_outliers(
        self,
        bond_data: pd.DataFrame,
        spread_change_column: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Exclude observations with extreme spread changes.

        Extreme spread changes (> max_spread_change %) are likely
        data errors or special situations (defaults, restructurings).

        Args:
            bond_data: DataFrame with spread changes
            spread_change_column: Column name for spread percentage changes

        Returns:
            Tuple of (filtered_data, statistics)
        """
        initial_count = len(bond_data)

        # Filter absolute spread changes > threshold
        filtered = bond_data[
            bond_data[spread_change_column].abs() <= self.max_spread_change
        ].copy()

        stats = {
            'initial_obs': initial_count,
            'final_obs': len(filtered),
            'removed': initial_count - len(filtered),
            'pct_removed': 100.0 * (initial_count - len(filtered)) / initial_count if initial_count > 0 else 0,
            'threshold_pct': self.max_spread_change
        }

        return filtered, stats

    def print_filter_summary(self, filter_stats: Dict):
        """
        Print human-readable summary of filter application.

        Args:
            filter_stats: Statistics dictionary from filter_for_within_issuer_analysis
        """
        print("=" * 80)
        print("WITHIN-ISSUER ANALYSIS FILTER SUMMARY")
        print("=" * 80)
        print()
        print(f"Initial observations: {filter_stats['initial_observations']:,}")
        print()

        # Filter 1: Min bonds
        if 'filter1_min_bonds' in filter_stats and not filter_stats['filter1_min_bonds'].get('skipped'):
            f1 = filter_stats['filter1_min_bonds']
            print(f"Filter 1: Min bonds per issuer-week (≥{f1['threshold']})")
            print(f"  Removed: {f1['removed']:,} ({f1['pct_removed']:.1f}%)")
            print(f"  Remaining: {f1['final_obs']:,}")
            print()

        # Filter 2: Maturity dispersion
        if 'filter2_maturity_dispersion' in filter_stats and not filter_stats['filter2_maturity_dispersion'].get('skipped'):
            f2 = filter_stats['filter2_maturity_dispersion']
            print(f"Filter 2: Maturity dispersion (≥{f2['threshold_years']:.1f} years)")
            print(f"  Removed: {f2['removed']:,} ({f2['pct_removed']:.1f}%)")
            print(f"  Remaining: {f2['final_obs']:,}")
            print()

        # Filter 3: Pull-to-par
        if 'filter3_pull_to_par' in filter_stats and not filter_stats['filter3_pull_to_par'].get('skipped'):
            f3 = filter_stats['filter3_pull_to_par']
            print(f"Filter 3: Pull-to-par exclusion (>{f3['threshold_years']:.1f} years)")
            print(f"  Removed: {f3['removed']:,} ({f3['pct_removed']:.1f}%)")
            print(f"  Remaining: {f3['final_obs']:,}")
            print()

        # Filter 4: Spread outliers
        if 'filter4_spread_outliers' in filter_stats and not filter_stats['filter4_spread_outliers'].get('skipped'):
            f4 = filter_stats['filter4_spread_outliers']
            print(f"Filter 4: Spread change outliers (≤{f4['threshold_pct']:.0f}%)")
            print(f"  Removed: {f4['removed']:,} ({f4['pct_removed']:.1f}%)")
            print(f"  Remaining: {f4['final_obs']:,}")
            print()

        print(f"FINAL observations: {filter_stats['final_observations']:,}")
        print(f"TOTAL retention rate: {filter_stats['pct_retained']:.1f}%")
        print("=" * 80)


def apply_within_issuer_filters(
    bond_data: pd.DataFrame,
    maturity_column: str = 'time_to_maturity',
    spread_change_column: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to apply within-issuer filters.

    Args:
        bond_data: DataFrame with bond observations
        maturity_column: Column name for time to maturity
        spread_change_column: Optional column for spread changes
        verbose: Print filter summary

    Returns:
        Tuple of (filtered_data, filter_statistics)
    """
    filters = DataFilters()

    filtered_data, stats = filters.filter_for_within_issuer_analysis(
        bond_data,
        maturity_column=maturity_column,
        spread_change_column=spread_change_column
    )

    if verbose:
        filters.print_filter_summary(stats)

    return filtered_data, stats


def validate_sample_quality(
    bond_data: pd.DataFrame,
    min_observations: int = 1000,
    min_issuers: int = 50,
    min_weeks: int = 100
) -> Dict:
    """
    Validate that filtered sample has sufficient quality for analysis.

    Args:
        bond_data: Filtered DataFrame
        min_observations: Minimum total observations
        min_issuers: Minimum number of unique issuers
        min_weeks: Minimum number of unique weeks

    Returns:
        Dictionary with quality checks and warnings
    """
    checks = {}

    # Count observations
    n_obs = len(bond_data)
    checks['n_observations'] = n_obs
    checks['pass_observations'] = n_obs >= min_observations

    # Count unique issuers
    if 'issuer_id' in bond_data.columns:
        n_issuers = bond_data['issuer_id'].nunique()
        checks['n_issuers'] = n_issuers
        checks['pass_issuers'] = n_issuers >= min_issuers
    else:
        checks['n_issuers'] = None
        checks['pass_issuers'] = None

    # Count unique weeks
    if 'date' in bond_data.columns:
        n_weeks = bond_data['date'].nunique()
        checks['n_weeks'] = n_weeks
        checks['pass_weeks'] = n_weeks >= min_weeks
    else:
        checks['n_weeks'] = None
        checks['pass_weeks'] = None

    # Overall pass/fail
    checks['all_checks_pass'] = all([
        checks['pass_observations'],
        checks.get('pass_issuers', True),
        checks.get('pass_weeks', True)
    ])

    # Generate warnings
    warnings = []
    if not checks['pass_observations']:
        warnings.append(f"Insufficient observations: {n_obs} < {min_observations}")
    if checks.get('pass_issuers') is not None and not checks['pass_issuers']:
        warnings.append(f"Insufficient issuers: {checks['n_issuers']} < {min_issuers}")
    if checks.get('pass_weeks') is not None and not checks['pass_weeks']:
        warnings.append(f"Insufficient weeks: {checks['n_weeks']} < {min_weeks}")

    checks['warnings'] = warnings

    return checks
