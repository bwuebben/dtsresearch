"""
Data loading module with database connectivity and mock data generation.

Updated to include fields needed for evolved Stage 0:
- ultimate_parent_id: Ultimate parent company ID
- BCLASS3/BCLASS4: Bloomberg industry classification
- seniority: Bond seniority (Senior/Subordinated)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ..config import (
    BLOOMBERG_CLASS_LEVEL,
    MOCK_N_BONDS,
    MOCK_N_ISSUERS,
    MOCK_SECTOR_DISTRIBUTION,
    MOCK_RATING_DISTRIBUTION
)


class BondDataLoader:
    """
    Loads bond data from database or generates mock data for testing.

    Expected database schema (evolved for Stage 0):
    - bond_id: unique identifier
    - date: observation date
    - oas: option-adjusted spread (bps)
    - rating: credit rating
    - maturity_date: bond maturity date
    - ultimate_parent_id: ultimate parent company identifier
    - sector_classification: Bloomberg BCLASS3 or BCLASS4 value
    - seniority: bond seniority (Senior/Subordinated)
    - security_type: security type (Corp/MTN/Note/Bond/Debenture)

    Note: The sector_classification field will be mapped to research sectors
    (Industrial/Financial/Utility/Energy) using the SectorClassifier module.
    The issuer_id will be created as a composite of ultimate_parent_id + seniority.
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
            ultimate_parent_id, BCLASS3 (or BCLASS4), seniority, security_type
        """
        if query is None:
            # TODO: User fills in SQL query
            # Configure BCLASS level in config.py (BLOOMBERG_CLASS_LEVEL)
            bclass_field = BLOOMBERG_CLASS_LEVEL  # 'BCLASS3' or 'BCLASS4'
            query = f"""
            -- User should customize this query based on their database schema
            SELECT
                bond_id,
                date,
                oas,
                rating,
                maturity_date,
                ultimate_parent_id,
                {bclass_field} as sector_classification,
                seniority,
                security_type
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
        n_bonds: int = None,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate realistic mock bond data for testing with new Stage 0 fields.

        Creates multi-bond issuers with realistic maturity dispersion to enable
        within-issuer analysis.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            n_bonds: Number of bonds (if None, uses MOCK_N_BONDS from config)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with mock bond data including:
            - bond_id, date, oas, rating, maturity_date, time_to_maturity
            - ultimate_parent_id, sector_classification (BCLASS3), seniority, security_type
        """
        np.random.seed(seed)

        if n_bonds is None:
            n_bonds = MOCK_N_BONDS

        # Date range (weekly observations)
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='W-FRI')
        n_weeks = len(dates)

        # Define rating categories
        ig_ratings = ['AAA', 'AA', 'A', 'BBB']
        hy_ratings = ['BB', 'B', 'CCC']
        all_ratings = ig_ratings + hy_ratings

        # Sample BCLASS3 values for sectors
        bclass3_by_sector = {
            'Industrial': ['INDUSTRIAL', 'MANUFACTURING', 'CONSUMER', 'TECHNOLOGY', 'TELECOM'],
            'Financial': ['BANKING', 'INSURANCE', 'FINANCIAL_SERVICES', 'REITS', 'BROKER_DEALER'],
            'Utility': ['ELECTRIC', 'GAS_DISTRIBUTION', 'WATER', 'INDEPENDENT_POWER'],
            'Energy': ['OIL_GAS_PROD', 'OIL_REFINING', 'COAL', 'OIL_SERVICES']
        }

        # Security types
        security_types = ['Corp', 'MTN', 'Note', 'Bond', 'Debenture']

        # Create issuers first (from config)
        n_issuers = MOCK_N_ISSUERS
        issuers = []

        for i in range(n_issuers):
            # Sample sector based on distribution
            sector = np.random.choice(
                list(MOCK_SECTOR_DISTRIBUTION.keys()),
                p=list(MOCK_SECTOR_DISTRIBUTION.values())
            )

            # Sample rating based on distribution
            rating = np.random.choice(
                list(MOCK_RATING_DISTRIBUTION.keys()),
                p=list(MOCK_RATING_DISTRIBUTION.values())
            )

            # Sample BCLASS3 value for this sector
            bclass3 = np.random.choice(bclass3_by_sector[sector])

            issuers.append({
                'ultimate_parent_id': f'PARENT_{i:04d}',
                'sector': sector,
                'bclass3': bclass3,
                'rating': rating
            })

        # Generate bonds from issuers
        # Each issuer has 2-5 bonds with different maturities and seniorities
        bonds = []
        bond_counter = 0

        for issuer in issuers:
            # Number of bonds for this issuer (2-5 to enable within-issuer analysis)
            n_bonds_issuer = np.random.randint(2, 6)

            # Base spread for this issuer (depends on rating)
            rating = issuer['rating']
            if rating in ['AAA', 'AA']:
                issuer_base_spread = np.random.uniform(50, 150)
            elif rating == 'A':
                issuer_base_spread = np.random.uniform(100, 200)
            elif rating == 'BBB':
                issuer_base_spread = np.random.uniform(150, 300)
            elif rating == 'BB':
                issuer_base_spread = np.random.uniform(300, 500)
            elif rating == 'B':
                issuer_base_spread = np.random.uniform(500, 800)
            else:  # CCC
                issuer_base_spread = np.random.uniform(800, 1500)

            for _ in range(n_bonds_issuer):
                if bond_counter >= n_bonds:
                    break

                # Seniority: 80% Senior, 20% Subordinated
                seniority = np.random.choice(['Senior', 'Subordinated'], p=[0.8, 0.2])

                # Subordinated bonds have higher spread
                seniority_premium = 1.3 if seniority == 'Subordinated' else 1.0

                # Maturity between 1 and 15 years, ensuring dispersion within issuer
                years_to_maturity = np.random.uniform(1, 15)
                maturity_date = start + timedelta(days=int(years_to_maturity * 365))

                # Security type
                security_type = np.random.choice(security_types)

                bonds.append({
                    'bond_id': f'BOND_{bond_counter:05d}',
                    'ultimate_parent_id': issuer['ultimate_parent_id'],
                    'rating': rating,
                    'sector': issuer['sector'],
                    'sector_classification': issuer['bclass3'],
                    'seniority': seniority,
                    'security_type': security_type,
                    'maturity_date': maturity_date,
                    'base_spread': issuer_base_spread * seniority_premium
                })

                bond_counter += 1

            if bond_counter >= n_bonds:
                break

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
                        'sector_classification': bond['sector_classification'],
                        'ultimate_parent_id': bond['ultimate_parent_id'],
                        'seniority': bond['seniority'],
                        'security_type': bond['security_type']
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
