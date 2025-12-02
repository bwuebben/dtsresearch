"""
Bucket classification system for Stage 0 analysis.

Defines buckets by:
- Rating: AAA/AA, A, BBB for IG; BB, B, CCC for HY
- Maturity: 1-2y, 2-3y, 3-5y, 5-7y, 7-10y, 10y+
- Sector: Bloomberg Class 3
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from ..models.merton import MertonLambdaCalculator


class BucketClassifier:
    """
    Classifies bonds into buckets for Stage 0 analysis.
    """

    # Rating categories
    RATING_BUCKETS_IG = ['AAA/AA', 'A', 'BBB']
    RATING_BUCKETS_HY = ['BB', 'B', 'CCC']
    ALL_RATING_BUCKETS = RATING_BUCKETS_IG + RATING_BUCKETS_HY

    # Maturity buckets (in years)
    MATURITY_BUCKETS = [
        ('1-2y', 1, 2),
        ('2-3y', 2, 3),
        ('3-5y', 3, 5),
        ('5-7y', 5, 7),
        ('7-10y', 7, 10),
        ('10y+', 10, 100)
    ]

    # Rating to bucket mapping
    RATING_MAP = {
        'AAA': 'AAA/AA',
        'AA+': 'AAA/AA',
        'AA': 'AAA/AA',
        'AA-': 'AAA/AA',
        'A+': 'A',
        'A': 'A',
        'A-': 'A',
        'BBB+': 'BBB',
        'BBB': 'BBB',
        'BBB-': 'BBB',
        'BB+': 'BB',
        'BB': 'BB',
        'BB-': 'BB',
        'B+': 'B',
        'B': 'B',
        'B-': 'B',
        'CCC+': 'CCC',
        'CCC': 'CCC',
        'CCC-': 'CCC',
        'CC': 'CCC',
        'C': 'CCC',
        'D': 'CCC'
    }

    def __init__(self):
        self.merton_calc = MertonLambdaCalculator()

    def classify_rating(self, rating: str) -> str:
        """
        Map specific rating to rating bucket.

        Args:
            rating: Specific credit rating (e.g., 'AA', 'BBB+')

        Returns:
            Rating bucket (e.g., 'AAA/AA', 'BBB')
        """
        rating_clean = rating.strip().upper()
        return self.RATING_MAP.get(rating_clean, 'Unknown')

    def classify_maturity(self, years: float) -> str:
        """
        Map time to maturity to maturity bucket.

        Args:
            years: Time to maturity in years

        Returns:
            Maturity bucket label (e.g., '3-5y')
        """
        for label, lower, upper in self.MATURITY_BUCKETS:
            if lower <= years < upper:
                return label
        return 'Unknown'

    def get_maturity_midpoint(self, maturity_bucket: str) -> float:
        """Get representative maturity for a maturity bucket."""
        midpoints = {
            '1-2y': 1.5,
            '2-3y': 2.5,
            '3-5y': 4.0,
            '5-7y': 6.0,
            '7-10y': 8.5,
            '10y+': 12.0
        }
        return midpoints.get(maturity_bucket, 5.0)

    def classify_bonds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add bucket classifications to bond dataframe.

        Args:
            df: DataFrame with columns: rating, time_to_maturity, sector

        Returns:
            DataFrame with added columns: rating_bucket, maturity_bucket, bucket_id
        """
        df = df.copy()

        # Classify ratings and maturities
        df['rating_bucket'] = df['rating'].apply(self.classify_rating)
        df['maturity_bucket'] = df['time_to_maturity'].apply(self.classify_maturity)

        # Create composite bucket ID
        df['bucket_id'] = (
            df['rating_bucket'] + ' | ' +
            df['maturity_bucket'] + ' | ' +
            df['sector'].astype(str)
        )

        # Identify IG vs HY
        df['is_ig'] = df['rating_bucket'].isin(self.RATING_BUCKETS_IG)

        return df

    def compute_bucket_characteristics(
        self,
        df: pd.DataFrame,
        min_observations: int = 50
    ) -> pd.DataFrame:
        """
        Compute representative characteristics for each bucket.

        Args:
            df: Classified bond dataframe
            min_observations: Minimum number of observations for valid bucket

        Returns:
            DataFrame with bucket-level statistics
        """
        bucket_stats = []

        for bucket_id in df['bucket_id'].unique():
            bucket_data = df[df['bucket_id'] == bucket_id]

            if len(bucket_data) < min_observations:
                continue

            # Parse bucket components
            parts = bucket_id.split(' | ')
            rating_bucket = parts[0]
            maturity_bucket = parts[1]
            sector = parts[2]

            # Compute statistics
            median_maturity = bucket_data['time_to_maturity'].median()
            median_spread = bucket_data['oas'].median()
            n_obs = len(bucket_data)
            n_bonds = bucket_data['bond_id'].nunique()
            n_weeks = bucket_data['date'].nunique() if 'date' in bucket_data.columns else None

            # Calculate theoretical Merton lambda
            lambda_T = self.merton_calc.lambda_T(median_maturity, median_spread)
            lambda_s = self.merton_calc.lambda_s(median_spread)
            lambda_merton = float(lambda_T * lambda_s)

            bucket_stats.append({
                'bucket_id': bucket_id,
                'rating_bucket': rating_bucket,
                'maturity_bucket': maturity_bucket,
                'sector': sector,
                'median_maturity': median_maturity,
                'median_spread': median_spread,
                'lambda_merton': lambda_merton,
                'lambda_T': float(lambda_T),
                'lambda_s': float(lambda_s),
                'n_observations': n_obs,
                'n_bonds': n_bonds,
                'n_weeks': n_weeks,
                'is_ig': rating_bucket in self.RATING_BUCKETS_IG
            })

        return pd.DataFrame(bucket_stats)

    def get_cross_maturity_buckets(
        self,
        bucket_stats: pd.DataFrame,
        rating: str,
        sector: str
    ) -> pd.DataFrame:
        """
        Get all maturity buckets for a given rating and sector.

        Useful for testing cross-maturity patterns.

        Args:
            bucket_stats: Bucket statistics dataframe
            rating: Rating bucket (e.g., 'BBB')
            sector: Sector name

        Returns:
            DataFrame filtered to specified rating and sector
        """
        mask = (
            (bucket_stats['rating_bucket'] == rating) &
            (bucket_stats['sector'] == sector)
        )
        result = bucket_stats[mask].copy()

        # Sort by maturity
        maturity_order = {label: i for i, (label, _, _) in enumerate(self.MATURITY_BUCKETS)}
        result['maturity_order'] = result['maturity_bucket'].map(maturity_order)
        result = result.sort_values('maturity_order')

        return result

    def get_same_maturity_buckets(
        self,
        bucket_stats: pd.DataFrame,
        maturity: str,
        sector: str
    ) -> pd.DataFrame:
        """
        Get all rating buckets for a given maturity and sector.

        Useful for testing same-maturity credit quality patterns.

        Args:
            bucket_stats: Bucket statistics dataframe
            maturity: Maturity bucket (e.g., '3-5y')
            sector: Sector name

        Returns:
            DataFrame filtered to specified maturity and sector
        """
        mask = (
            (bucket_stats['maturity_bucket'] == maturity) &
            (bucket_stats['sector'] == sector)
        )
        result = bucket_stats[mask].copy()

        # Sort by rating quality (AAA best to CCC worst)
        rating_order = {r: i for i, r in enumerate(self.ALL_RATING_BUCKETS)}
        result['rating_order'] = result['rating_bucket'].map(rating_order)
        result = result.sort_values('rating_order')

        return result

    def summarize_bucket_coverage(self, bucket_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary table of bucket coverage.

        Args:
            bucket_stats: Bucket statistics dataframe

        Returns:
            Pivot table showing number of observations by rating and maturity
        """
        summary = bucket_stats.pivot_table(
            values='n_observations',
            index='rating_bucket',
            columns='maturity_bucket',
            aggfunc='sum',
            fill_value=0
        )

        # Reorder columns by maturity
        maturity_cols = [label for label, _, _ in self.MATURITY_BUCKETS]
        maturity_cols = [c for c in maturity_cols if c in summary.columns]
        summary = summary[maturity_cols]

        # Reorder rows by rating
        rating_rows = [r for r in self.ALL_RATING_BUCKETS if r in summary.index]
        summary = summary.loc[rating_rows]

        return summary
