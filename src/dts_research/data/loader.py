"""
Data loading module with database connectivity and mock data generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class BondDataLoader:
    """
    Loads bond data from database or generates mock data for testing.

    Expected database schema:
    - bond_id: unique identifier
    - date: observation date
    - oas: option-adjusted spread (bps)
    - rating: credit rating
    - maturity_date: bond maturity date
    - sector: Bloomberg Class 3 sector
    - issuer_id: issuer identifier
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize data loader.

        Args:
            connection_string: Database connection string (to be filled in by user)
        """
        self.connection_string = connection_string
        self.connection = None

    def connect(self):
        """Establish database connection."""
        if self.connection_string:
            # TODO: User fills in connection logic
            # Example: self.connection = psycopg2.connect(self.connection_string)
            raise NotImplementedError("User must implement database connection")

    def load_bond_data(
        self,
        start_date: str,
        end_date: str,
        query: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load bond data from database.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            query: Optional custom SQL query (user fills in)

        Returns:
            DataFrame with columns: bond_id, date, oas, rating, maturity_date,
            sector, issuer_id
        """
        if query is None:
            # TODO: User fills in SQL query
            query = """
            -- User should customize this query based on their database schema
            SELECT
                bond_id,
                date,
                oas,
                rating,
                maturity_date,
                sector,
                issuer_id
            FROM bond_table
            WHERE date BETWEEN %(start_date)s AND %(end_date)s
            """

        if self.connection:
            # TODO: User implements actual query execution
            # Example: return pd.read_sql(query, self.connection, params={'start_date': start_date, 'end_date': end_date})
            raise NotImplementedError("User must implement query execution")
        else:
            # Generate mock data for testing
            return self.generate_mock_data(start_date, end_date)

    def generate_mock_data(
        self,
        start_date: str,
        end_date: str,
        n_bonds: int = 500,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate realistic mock bond data for testing.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            n_bonds: Number of bonds to generate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with mock bond data
        """
        np.random.seed(seed)

        # Date range (weekly observations)
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='W-FRI')
        n_weeks = len(dates)

        # Define rating categories
        ig_ratings = ['AAA', 'AA', 'A', 'BBB']
        hy_ratings = ['BB', 'B', 'CCC']
        all_ratings = ig_ratings + hy_ratings

        # Define sectors
        sectors = ['Industrial', 'Financial', 'Utility', 'Technology']

        # Generate bond characteristics
        bonds = []
        for i in range(n_bonds):
            rating = np.random.choice(all_ratings, p=[0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05])
            sector = np.random.choice(sectors)

            # Maturity between 1 and 15 years from start_date
            years_to_maturity = np.random.uniform(1, 15)
            maturity_date = start + timedelta(days=int(years_to_maturity * 365))

            # Base spread depends on rating
            if rating in ['AAA', 'AA']:
                base_spread = np.random.uniform(50, 150)
            elif rating == 'A':
                base_spread = np.random.uniform(100, 200)
            elif rating == 'BBB':
                base_spread = np.random.uniform(150, 300)
            elif rating == 'BB':
                base_spread = np.random.uniform(300, 500)
            elif rating == 'B':
                base_spread = np.random.uniform(500, 800)
            else:  # CCC
                base_spread = np.random.uniform(800, 1500)

            bonds.append({
                'bond_id': f'BOND_{i:04d}',
                'rating': rating,
                'sector': sector,
                'maturity_date': maturity_date,
                'base_spread': base_spread,
                'issuer_id': f'ISSUER_{i // 5:03d}'  # ~5 bonds per issuer
            })

        # Generate time series data
        records = []

        # Create market factor (common shock to all bonds)
        market_returns = np.random.normal(0, 0.02, n_weeks)  # 2% weekly vol
        market_cumulative = np.cumsum(market_returns)

        for bond in bonds:
            bond_vol = 0.03  # Individual bond volatility
            idiosyncratic_returns = np.random.normal(0, bond_vol, n_weeks)

            # Spread evolution: base + market component + idiosyncratic
            # Beta depends on maturity (shorter bonds more sensitive)
            years_to_mat = (bond['maturity_date'] - start).days / 365
            theoretical_beta = 1.5 if years_to_mat < 3 else (1.2 if years_to_mat < 7 else 0.9)

            spreads = bond['base_spread'] * np.exp(
                theoretical_beta * market_cumulative +
                np.cumsum(idiosyncratic_returns)
            )

            # Clip to reasonable range
            spreads = np.clip(spreads, 10, 3000)

            for date_idx, date in enumerate(dates):
                # Calculate time to maturity at this date
                ttm = (bond['maturity_date'] - date).days / 365

                if ttm > 0:  # Only include if not matured
                    records.append({
                        'bond_id': bond['bond_id'],
                        'date': date,
                        'oas': spreads[date_idx],
                        'rating': bond['rating'],
                        'maturity_date': bond['maturity_date'],
                        'time_to_maturity': ttm,
                        'sector': bond['sector'],
                        'issuer_id': bond['issuer_id']
                    })

        df = pd.DataFrame(records)
        return df.sort_values(['date', 'bond_id']).reset_index(drop=True)

    def load_index_data(
        self,
        start_date: str,
        end_date: str,
        index_type: str = 'IG'
    ) -> pd.DataFrame:
        """
        Load index-level spread data (for DTS factor).

        Args:
            start_date: Start date
            end_date: End date
            index_type: 'IG' or 'HY'

        Returns:
            DataFrame with columns: date, oas
        """
        if self.connection:
            # TODO: User implements index data query
            raise NotImplementedError("User must implement index data loading")
        else:
            # Generate mock index data
            return self.generate_mock_index_data(start_date, end_date, index_type)

    def generate_mock_index_data(
        self,
        start_date: str,
        end_date: str,
        index_type: str = 'IG',
        seed: int = 42
    ) -> pd.DataFrame:
        """Generate mock index-level spread data."""
        np.random.seed(seed)

        dates = pd.date_range(start_date, end_date, freq='W-FRI')
        n_weeks = len(dates)

        # Base spread level
        base = 150 if index_type == 'IG' else 500

        # Generate spread series with mean reversion
        returns = np.random.normal(0, 0.015, n_weeks)  # 1.5% weekly vol
        spreads = base * np.exp(np.cumsum(returns))
        spreads = np.clip(spreads, 50, 1000 if index_type == 'IG' else 2000)

        return pd.DataFrame({
            'date': dates,
            'oas': spreads,
            'index_type': index_type
        })

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
