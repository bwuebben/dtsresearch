"""
Bucket definition module for Stage 0 bucket-level analysis.

Creates 72 buckets per universe (IG/HY) based on:
- Rating (6 levels)
- Maturity (6 buckets)
- Sector (2 major: Industrial+Energy vs Financial+Utility)

Computes representative characteristics (s̄, T̄) for each bucket.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..config import (
    RATING_BUCKETS_IG,
    RATING_BUCKETS_HY,
    MATURITY_BUCKETS,
    MATURITY_BUCKET_BOUNDARIES,
    MIN_OBSERVATIONS_PER_BUCKET
)


class BucketDefinitions:
    """
    Defines and manages bucket classifications for Stage 0 analysis.

    Creates 72 buckets: 6 ratings × 6 maturities × 2 sectors = 72
    """

    def __init__(self):
        """Initialize bucket definitions."""
        self.rating_buckets_ig = RATING_BUCKETS_IG
        self.rating_buckets_hy = RATING_BUCKETS_HY
        self.maturity_buckets = MATURITY_BUCKETS
        self.maturity_boundaries = MATURITY_BUCKET_BOUNDARIES
        self.min_observations = MIN_OBSERVATIONS_PER_BUCKET

    def assign_rating_bucket(
        self,
        bond_data: pd.DataFrame,
        rating_column: str = 'rating'
    ) -> pd.DataFrame:
        """
        Assign bonds to rating buckets.

        Args:
            bond_data: DataFrame with bond ratings
            rating_column: Column name containing ratings

        Returns:
            DataFrame with 'rating_bucket' column added
        """
        bond_data = bond_data.copy()

        if rating_column not in bond_data.columns:
            raise ValueError(f"Rating column '{rating_column}' not found")

        bond_data['rating_bucket'] = bond_data[rating_column].apply(
            self._map_rating_to_bucket
        )

        return bond_data

    def _map_rating_to_bucket(self, rating: str) -> str:
        """
        Map individual rating to rating bucket.

        Args:
            rating: Bond rating (e.g., 'AA', 'BBB+', 'B')

        Returns:
            Rating bucket (e.g., 'AAA/AA', 'BBB', 'B')
        """
        if pd.isna(rating):
            return 'BBB'  # Default for missing

        rating_str = str(rating).strip().upper()

        # Remove +/- modifiers
        rating_clean = rating_str.replace('+', '').replace('-', '')

        # IG buckets
        if rating_clean in ['AAA', 'AA']:
            return 'AAA/AA'
        elif rating_clean in ['A']:
            return 'A'
        elif rating_clean in ['BBB']:
            return 'BBB'
        # HY buckets
        elif rating_clean in ['BB']:
            return 'BB'
        elif rating_clean in ['B']:
            return 'B'
        elif rating_clean in ['CCC', 'CC', 'C', 'D']:
            return 'CCC'
        else:
            # Default based on likely category
            if any(x in rating_clean for x in ['A', 'BBB']):
                return 'BBB'
            else:
                return 'B'

    def assign_maturity_bucket(
        self,
        bond_data: pd.DataFrame,
        maturity_column: str = 'time_to_maturity'
    ) -> pd.DataFrame:
        """
        Assign bonds to maturity buckets.

        Args:
            bond_data: DataFrame with time to maturity
            maturity_column: Column name containing years to maturity

        Returns:
            DataFrame with 'maturity_bucket' column added
        """
        bond_data = bond_data.copy()

        if maturity_column not in bond_data.columns:
            raise ValueError(f"Maturity column '{maturity_column}' not found")

        # Use pd.cut to bin into maturity buckets
        bond_data['maturity_bucket'] = pd.cut(
            bond_data[maturity_column],
            bins=self.maturity_boundaries,
            labels=self.maturity_buckets,
            right=False  # [lower, upper)
        )

        # Handle any that fell outside (shouldn't happen with inf upper bound)
        bond_data['maturity_bucket'] = bond_data['maturity_bucket'].fillna('10y+')

        return bond_data

    def assign_sector_group(
        self,
        bond_data: pd.DataFrame,
        sector_column: str = 'sector'
    ) -> pd.DataFrame:
        """
        Assign bonds to sector groups for bucketing.

        For Stage 0, we use 2 major sector groups:
        - Group A: Industrial + Energy (more homogeneous)
        - Group B: Financial + Utility (regulatory/stable cash flows)

        Args:
            bond_data: DataFrame with sector classification
            sector_column: Column name containing sector

        Returns:
            DataFrame with 'sector_group' column added
        """
        bond_data = bond_data.copy()

        if sector_column not in bond_data.columns:
            raise ValueError(f"Sector column '{sector_column}' not found")

        def map_to_group(sector):
            if pd.isna(sector):
                return 'A'  # Default
            sector_str = str(sector).strip()
            if sector_str in ['Financial', 'Utility']:
                return 'B'
            else:  # Industrial, Energy
                return 'A'

        bond_data['sector_group'] = bond_data[sector_column].apply(map_to_group)

        return bond_data

    def create_bucket_id(
        self,
        bond_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create composite bucket ID from rating, maturity, and sector group.

        Bucket ID format: "{rating}_{maturity}_{sector_group}"
        Example: "BBB_3-5y_A"

        Args:
            bond_data: DataFrame with rating_bucket, maturity_bucket, sector_group

        Returns:
            DataFrame with 'bucket_id' column added
        """
        required_cols = ['rating_bucket', 'maturity_bucket', 'sector_group']
        missing = [col for col in required_cols if col not in bond_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        bond_data = bond_data.copy()
        bond_data['bucket_id'] = (
            bond_data['rating_bucket'].astype(str) + '_' +
            bond_data['maturity_bucket'].astype(str) + '_' +
            bond_data['sector_group'].astype(str)
        )

        return bond_data

    def compute_bucket_characteristics(
        self,
        bond_data: pd.DataFrame,
        spread_column: str = 'oas',
        maturity_column: str = 'time_to_maturity'
    ) -> pd.DataFrame:
        """
        Compute representative characteristics (s̄, T̄) for each bucket.

        Args:
            bond_data: DataFrame with bucket_id and bond characteristics
            spread_column: Column name for spread
            maturity_column: Column name for maturity

        Returns:
            DataFrame with bucket-level statistics
        """
        if 'bucket_id' not in bond_data.columns:
            raise ValueError("bucket_id column not found. Run create_bucket_id first.")

        required_cols = [spread_column, maturity_column]
        missing = [col for col in required_cols if col not in bond_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Aggregate by bucket
        bucket_stats = bond_data.groupby('bucket_id').agg({
            spread_column: ['mean', 'median', 'std', 'count'],
            maturity_column: ['mean', 'median', 'std'],
            'rating_bucket': 'first',
            'maturity_bucket': 'first',
            'sector_group': 'first'
        }).reset_index()

        # Flatten column names
        bucket_stats.columns = [
            'bucket_id',
            'spread_mean', 'spread_median', 'spread_std', 'n_observations',
            'maturity_mean', 'maturity_median', 'maturity_std',
            'rating_bucket', 'maturity_bucket', 'sector_group'
        ]

        # Use median as representative values (more robust to outliers)
        bucket_stats['s_bar'] = bucket_stats['spread_median']
        bucket_stats['T_bar'] = bucket_stats['maturity_median']

        # Filter out buckets with too few observations
        bucket_stats = bucket_stats[
            bucket_stats['n_observations'] >= self.min_observations
        ].copy()

        return bucket_stats

    def validate_bucket_coverage(
        self,
        bond_data: pd.DataFrame
    ) -> Dict:
        """
        Validate bucket coverage and identify sparse buckets.

        Args:
            bond_data: DataFrame with bucket_id

        Returns:
            Dictionary with coverage statistics
        """
        if 'bucket_id' not in bond_data.columns:
            raise ValueError("bucket_id column not found.")

        bucket_counts = bond_data['bucket_id'].value_counts()

        # Expected buckets (not all may exist in data)
        expected_ig = len(self.rating_buckets_ig) * len(self.maturity_buckets) * 2
        expected_hy = len(self.rating_buckets_hy) * len(self.maturity_buckets) * 2
        expected_total = expected_ig + expected_hy

        stats = {
            'total_bonds': len(bond_data),
            'n_buckets_populated': len(bucket_counts),
            'n_buckets_expected': expected_total,
            'n_buckets_with_min_obs': (bucket_counts >= self.min_observations).sum(),
            'pct_coverage': 100.0 * len(bucket_counts) / expected_total,
            'mean_observations_per_bucket': bucket_counts.mean(),
            'median_observations_per_bucket': bucket_counts.median(),
            'min_observations_in_bucket': bucket_counts.min(),
            'max_observations_in_bucket': bucket_counts.max(),
            'buckets_below_threshold': bucket_counts[
                bucket_counts < self.min_observations
            ].to_dict()
        }

        return stats

    def get_bucket_grid(
        self,
        universe: str = 'both'
    ) -> pd.DataFrame:
        """
        Get the complete bucket grid definition.

        Args:
            universe: 'IG', 'HY', or 'both'

        Returns:
            DataFrame with all bucket combinations
        """
        if universe not in ['IG', 'HY', 'both']:
            raise ValueError("universe must be 'IG', 'HY', or 'both'")

        bucket_grid = []

        # IG buckets
        if universe in ['IG', 'both']:
            for rating in self.rating_buckets_ig:
                for maturity in self.maturity_buckets:
                    for sector_group in ['A', 'B']:
                        bucket_grid.append({
                            'bucket_id': f"{rating}_{maturity}_{sector_group}",
                            'rating_bucket': rating,
                            'maturity_bucket': maturity,
                            'sector_group': sector_group,
                            'universe': 'IG'
                        })

        # HY buckets
        if universe in ['HY', 'both']:
            for rating in self.rating_buckets_hy:
                for maturity in self.maturity_buckets:
                    for sector_group in ['A', 'B']:
                        bucket_grid.append({
                            'bucket_id': f"{rating}_{maturity}_{sector_group}",
                            'rating_bucket': rating,
                            'maturity_bucket': maturity,
                            'sector_group': sector_group,
                            'universe': 'HY'
                        })

        return pd.DataFrame(bucket_grid)


def classify_bonds_into_buckets(
    bond_data: pd.DataFrame,
    rating_column: str = 'rating',
    maturity_column: str = 'time_to_maturity',
    sector_column: str = 'sector',
    compute_characteristics: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convenience function to classify bonds into buckets.

    Args:
        bond_data: DataFrame with bond information
        rating_column: Column name for rating
        maturity_column: Column name for maturity
        sector_column: Column name for sector
        compute_characteristics: Whether to compute bucket characteristics

    Returns:
        Tuple of (bond_data with bucket_id, bucket_characteristics)
    """
    classifier = BucketDefinitions()

    # Assign to buckets
    bond_data = classifier.assign_rating_bucket(bond_data, rating_column)
    bond_data = classifier.assign_maturity_bucket(bond_data, maturity_column)
    bond_data = classifier.assign_sector_group(bond_data, sector_column)
    bond_data = classifier.create_bucket_id(bond_data)

    # Compute characteristics if requested
    bucket_chars = None
    if compute_characteristics:
        bucket_chars = classifier.compute_bucket_characteristics(
            bond_data,
            spread_column='oas',
            maturity_column=maturity_column
        )

    return bond_data, bucket_chars
