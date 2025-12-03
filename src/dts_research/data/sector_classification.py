"""
Sector classification module using Bloomberg Industry Classification.

Maps Bloomberg BCLASS3 or BCLASS4 codes to the 4 research sectors:
- Industrial
- Financial
- Utility
- Energy

This mapping is critical for sector interaction analysis in Stage 0.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from ..config import BLOOMBERG_CLASS_LEVEL, SECTOR_MAPPING


class SectorClassifier:
    """
    Classifies bonds into research sectors using Bloomberg classification.

    Uses BCLASS3 or BCLASS4 (configurable) to map to 4 canonical sectors.
    """

    def __init__(self, bclass_level: str = None):
        """
        Initialize sector classifier.

        Args:
            bclass_level: 'BCLASS3' or 'BCLASS4'. If None, uses config default.
        """
        self.bclass_level = bclass_level or BLOOMBERG_CLASS_LEVEL
        if self.bclass_level not in ['BCLASS3', 'BCLASS4']:
            raise ValueError(f"Invalid bclass_level: {self.bclass_level}. Must be 'BCLASS3' or 'BCLASS4'")

        # Initialize sector mapping
        # Note: These mappings should be customized based on actual Bloomberg data
        self._init_sector_mappings()

    def _init_sector_mappings(self):
        """
        Initialize mappings from Bloomberg classification to research sectors.

        These mappings should be customized based on actual Bloomberg
        BCLASS3/BCLASS4 values in your dataset.
        """
        # BCLASS3 mappings (Level 3 - Industry Group)
        # Format: BCLASS3_value → Research Sector
        self.bclass3_mapping = {
            # Financial sector keywords
            'Banks': 'Financial',
            'Insurance': 'Financial',
            'Diversified Financial Services': 'Financial',
            'Real Estate': 'Financial',
            'Capital Markets': 'Financial',
            'Consumer Finance': 'Financial',
            'Thrifts & Mortgage Finance': 'Financial',

            # Utility sector keywords
            'Electric Utilities': 'Utility',
            'Gas Utilities': 'Utility',
            'Water Utilities': 'Utility',
            'Multi-Utilities': 'Utility',
            'Independent Power Producers': 'Utility',
            'Renewable Electricity': 'Utility',

            # Energy sector keywords
            'Oil & Gas Exploration & Production': 'Energy',
            'Oil & Gas Refining & Marketing': 'Energy',
            'Oil & Gas Storage & Transportation': 'Energy',
            'Integrated Oil & Gas': 'Energy',
            'Oil & Gas Equipment & Services': 'Energy',
            'Coal & Consumable Fuels': 'Energy',

            # Default everything else to Industrial
        }

        # BCLASS4 mappings (Level 4 - Industry)
        # More granular than BCLASS3
        self.bclass4_mapping = {
            # Financial - Banks
            'Money Center Banks': 'Financial',
            'Regional Banks': 'Financial',
            'Diversified Banks': 'Financial',
            'Asset Management & Custody Banks': 'Financial',
            'Investment Banking & Brokerage': 'Financial',
            'Diversified Capital Markets': 'Financial',

            # Financial - Insurance
            'Life & Health Insurance': 'Financial',
            'Property & Casualty Insurance': 'Financial',
            'Reinsurance': 'Financial',
            'Insurance Brokers': 'Financial',

            # Utility
            'Electric Utilities': 'Utility',
            'Gas Utilities': 'Utility',
            'Water Utilities': 'Utility',
            'Multi-Utilities & Unregulated Power': 'Utility',
            'Independent Power Producers & Energy Traders': 'Utility',

            # Energy
            'Oil & Gas Exploration & Production': 'Energy',
            'Integrated Oil & Gas': 'Energy',
            'Oil & Gas Refining & Marketing & Transportation': 'Energy',
            'Oil & Gas Equipment & Services': 'Energy',
            'Coal & Consumable Fuels': 'Energy',
        }

    def classify_sector(
        self,
        bond_data: pd.DataFrame,
        bclass_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Classify bonds into research sectors.

        Args:
            bond_data: DataFrame with bond information
            bclass_column: Column name containing Bloomberg classification.
                          If None, uses self.bclass_level as column name.

        Returns:
            DataFrame with added 'sector' column
        """
        bond_data = bond_data.copy()

        # Determine which column to use
        if bclass_column is None:
            bclass_column = self.bclass_level

        if bclass_column not in bond_data.columns:
            raise ValueError(f"Column '{bclass_column}' not found in bond_data")

        # Get appropriate mapping
        if self.bclass_level == 'BCLASS3':
            mapping = self.bclass3_mapping
        else:
            mapping = self.bclass4_mapping

        # Apply mapping
        bond_data['sector'] = bond_data[bclass_column].apply(
            lambda x: self._map_to_sector(x, mapping)
        )

        return bond_data

    def _map_to_sector(self, bclass_value: str, mapping: Dict[str, str]) -> str:
        """
        Map a Bloomberg classification value to research sector.

        Args:
            bclass_value: Bloomberg classification string
            mapping: Dictionary mapping BCLASS to sector

        Returns:
            Research sector ('Industrial', 'Financial', 'Utility', 'Energy')
        """
        if pd.isna(bclass_value):
            return 'Industrial'  # Default for missing

        bclass_str = str(bclass_value).strip()

        # Direct match
        if bclass_str in mapping:
            return mapping[bclass_str]

        # Fuzzy match using keywords
        bclass_upper = bclass_str.upper()

        # Check Financial keywords
        financial_keywords = ['BANK', 'INSURANCE', 'FINANC', 'CAPITAL', 'INVEST',
                             'BROKERAGE', 'MORTGAGE', 'REAL ESTATE', 'REIT']
        if any(kw in bclass_upper for kw in financial_keywords):
            return 'Financial'

        # Check Utility keywords
        utility_keywords = ['UTILITY', 'UTILITIES', 'ELECTRIC', 'GAS', 'WATER',
                          'POWER', 'RENEWABLE']
        if any(kw in bclass_upper for kw in utility_keywords):
            return 'Utility'

        # Check Energy keywords
        energy_keywords = ['OIL', 'GAS', 'ENERGY', 'PETROLEUM', 'COAL',
                          'EXPLORATION', 'PRODUCTION', 'REFIN', 'DRILLING']
        if any(kw in bclass_upper for kw in energy_keywords):
            return 'Energy'

        # Default to Industrial for everything else
        return 'Industrial'

    def validate_sector_coverage(
        self,
        bond_data: pd.DataFrame
    ) -> Dict:
        """
        Validate sector classification coverage and distribution.

        Args:
            bond_data: DataFrame with 'sector' column

        Returns:
            Dictionary with coverage statistics
        """
        if 'sector' not in bond_data.columns:
            raise ValueError("'sector' column not found. Run classify_sector first.")

        sector_counts = bond_data['sector'].value_counts()
        total = len(bond_data)

        stats = {
            'total_bonds': total,
            'sector_counts': sector_counts.to_dict(),
            'sector_percentages': (100.0 * sector_counts / total).to_dict(),
            'n_sectors': len(sector_counts),
            'most_common_sector': sector_counts.index[0] if len(sector_counts) > 0 else None,
            'least_common_sector': sector_counts.index[-1] if len(sector_counts) > 0 else None,
        }

        # Check for expected 4 sectors
        expected_sectors = {'Industrial', 'Financial', 'Utility', 'Energy'}
        missing_sectors = expected_sectors - set(sector_counts.index)
        if missing_sectors:
            stats['warning'] = f"Missing sectors: {missing_sectors}"

        return stats

    def add_sector_dummies(
        self,
        bond_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add sector dummy variables for regression analysis.

        Creates binary indicators for Financial, Utility, Energy
        (Industrial is the reference category).

        Args:
            bond_data: DataFrame with 'sector' column

        Returns:
            DataFrame with added dummy columns
        """
        if 'sector' not in bond_data.columns:
            raise ValueError("'sector' column not found.")

        bond_data = bond_data.copy()

        # Create dummies (Industrial is reference, so excluded)
        for sector in ['Financial', 'Utility', 'Energy']:
            bond_data[f'sector_{sector.lower()}'] = (
                bond_data['sector'] == sector
            ).astype(int)

        return bond_data

    def get_sector_summary(
        self,
        bond_data: pd.DataFrame,
        by_rating: bool = True,
        by_maturity: bool = True
    ) -> pd.DataFrame:
        """
        Get summary statistics of sector distribution.

        Args:
            bond_data: DataFrame with 'sector' column
            by_rating: Include breakdown by rating
            by_maturity: Include breakdown by maturity bucket

        Returns:
            DataFrame with sector summary statistics
        """
        if 'sector' not in bond_data.columns:
            raise ValueError("'sector' column not found.")

        summary_data = []

        # Overall distribution
        sector_counts = bond_data['sector'].value_counts()
        for sector in ['Industrial', 'Financial', 'Utility', 'Energy']:
            count = sector_counts.get(sector, 0)
            pct = 100.0 * count / len(bond_data) if len(bond_data) > 0 else 0
            summary_data.append({
                'Breakdown': 'Overall',
                'Category': sector,
                'Count': count,
                'Percentage': pct
            })

        # By rating if requested
        if by_rating and 'rating' in bond_data.columns:
            for rating in bond_data['rating'].unique():
                rating_data = bond_data[bond_data['rating'] == rating]
                sector_counts = rating_data['sector'].value_counts()
                for sector in ['Industrial', 'Financial', 'Utility', 'Energy']:
                    count = sector_counts.get(sector, 0)
                    pct = 100.0 * count / len(rating_data) if len(rating_data) > 0 else 0
                    summary_data.append({
                        'Breakdown': f'Rating: {rating}',
                        'Category': sector,
                        'Count': count,
                        'Percentage': pct
                    })

        # By maturity if requested
        if by_maturity and 'maturity_bucket' in bond_data.columns:
            for mat_bucket in bond_data['maturity_bucket'].unique():
                mat_data = bond_data[bond_data['maturity_bucket'] == mat_bucket]
                sector_counts = mat_data['sector'].value_counts()
                for sector in ['Industrial', 'Financial', 'Utility', 'Energy']:
                    count = sector_counts.get(sector, 0)
                    pct = 100.0 * count / len(mat_data) if len(mat_data) > 0 else 0
                    summary_data.append({
                        'Breakdown': f'Maturity: {mat_bucket}',
                        'Category': sector,
                        'Count': count,
                        'Percentage': pct
                    })

        return pd.DataFrame(summary_data)


def classify_bonds_by_sector(
    bond_data: pd.DataFrame,
    bclass_column: Optional[str] = None,
    bclass_level: str = None
) -> pd.DataFrame:
    """
    Convenience function to classify bonds by sector.

    Args:
        bond_data: DataFrame with bond information
        bclass_column: Column name containing Bloomberg classification
        bclass_level: 'BCLASS3' or 'BCLASS4'

    Returns:
        DataFrame with 'sector' column added
    """
    classifier = SectorClassifier(bclass_level=bclass_level)
    return classifier.classify_sector(bond_data, bclass_column=bclass_column)
