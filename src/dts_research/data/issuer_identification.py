"""
Issuer identification and classification module.

Maps bonds to Ultimate Parent + Seniority for within-issuer analysis.
This ensures that bonds from the same issuer but different maturities
can be properly grouped and analyzed.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class IssuerIdentifier:
    """
    Handles issuer identification and classification.

    Creates composite issuer IDs from Ultimate Parent + Seniority
    to ensure proper grouping for within-issuer analysis.
    """

    def __init__(self):
        """Initialize issuer identifier."""
        self.issuer_mapping = {}

    def create_composite_issuer_id(
        self,
        bond_data: pd.DataFrame,
        parent_id_col: str = 'ultimate_parent_id',
        seniority_col: str = 'seniority'
    ) -> pd.DataFrame:
        """
        Create composite issuer ID from Ultimate Parent + Seniority.

        Within-issuer analysis requires bonds from same ultimate issuer
        AND same seniority level (senior bonds vs subordinated differ
        systematically in DTS sensitivity).

        Args:
            bond_data: DataFrame with bond information
            parent_id_col: Column name for ultimate parent ID
            seniority_col: Column name for seniority classification

        Returns:
            DataFrame with added 'issuer_id' column
        """
        # Validate required columns exist
        required_cols = [parent_id_col, seniority_col]
        missing = [col for col in required_cols if col not in bond_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Create composite ID: parent_id + '_' + seniority
        # E.g., "IBM_Senior", "IBM_Subordinated"
        bond_data = bond_data.copy()
        bond_data['issuer_id'] = (
            bond_data[parent_id_col].astype(str) + '_' +
            bond_data[seniority_col].astype(str)
        )

        return bond_data

    def classify_seniority(
        self,
        bond_data: pd.DataFrame,
        seniority_field: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Classify bonds by seniority level.

        If seniority field exists, standardize it to 'Senior' or 'Subordinated'.
        If missing, attempt to infer from security type or other fields.

        Args:
            bond_data: DataFrame with bond information
            seniority_field: Optional field containing seniority info

        Returns:
            DataFrame with standardized 'seniority' column
        """
        bond_data = bond_data.copy()

        if seniority_field and seniority_field in bond_data.columns:
            # Standardize existing seniority field
            bond_data['seniority'] = bond_data[seniority_field].apply(
                self._standardize_seniority
            )
        elif 'security_type' in bond_data.columns:
            # Infer from security type
            bond_data['seniority'] = bond_data['security_type'].apply(
                self._infer_seniority_from_security_type
            )
        else:
            # Default to 'Senior' if no information available
            # (most corporate bonds are senior unsecured)
            bond_data['seniority'] = 'Senior'

        return bond_data

    def _standardize_seniority(self, seniority_value: str) -> str:
        """
        Standardize seniority classification to 'Senior' or 'Subordinated'.

        Args:
            seniority_value: Raw seniority value

        Returns:
            Standardized seniority ('Senior' or 'Subordinated')
        """
        if pd.isna(seniority_value):
            return 'Senior'  # Default

        seniority_str = str(seniority_value).upper()

        # Check for subordinated indicators
        subordinated_keywords = ['SUB', 'JR', 'JUNIOR', 'MEZZANINE']
        if any(kw in seniority_str for kw in subordinated_keywords):
            return 'Subordinated'

        # Default to senior
        return 'Senior'

    def _infer_seniority_from_security_type(self, security_type: str) -> str:
        """
        Infer seniority from security type field.

        Args:
            security_type: Security type descriptor

        Returns:
            Inferred seniority ('Senior' or 'Subordinated')
        """
        if pd.isna(security_type):
            return 'Senior'

        sec_type_str = str(security_type).upper()

        # Subordinated indicators in security type
        if any(kw in sec_type_str for kw in ['SUBORDINATED', 'SUB', 'JR']):
            return 'Subordinated'

        # Senior indicators (explicit)
        if any(kw in sec_type_str for kw in ['SENIOR', 'SR', 'SNR']):
            return 'Senior'

        # Default to senior for standard corporate bonds
        return 'Senior'

    def validate_issuer_coverage(
        self,
        bond_data: pd.DataFrame,
        min_bonds_per_issuer: int = 2
    ) -> Dict:
        """
        Validate issuer identification coverage.

        Args:
            bond_data: DataFrame with issuer_id
            min_bonds_per_issuer: Minimum bonds required per issuer

        Returns:
            Dictionary with coverage statistics
        """
        if 'issuer_id' not in bond_data.columns:
            raise ValueError("issuer_id column not found. Run create_composite_issuer_id first.")

        # Count bonds per issuer
        issuer_counts = bond_data.groupby('issuer_id').size()

        stats = {
            'total_issuers': len(issuer_counts),
            'total_bonds': len(bond_data),
            'issuers_with_multiple_bonds': (issuer_counts >= min_bonds_per_issuer).sum(),
            'bonds_in_multi_bond_issuers': bond_data[
                bond_data['issuer_id'].isin(
                    issuer_counts[issuer_counts >= min_bonds_per_issuer].index
                )
            ].shape[0],
            'pct_bonds_in_multi_bond_issuers': 100.0 * bond_data[
                bond_data['issuer_id'].isin(
                    issuer_counts[issuer_counts >= min_bonds_per_issuer].index
                )
            ].shape[0] / len(bond_data) if len(bond_data) > 0 else 0,
            'mean_bonds_per_issuer': issuer_counts.mean(),
            'median_bonds_per_issuer': issuer_counts.median(),
            'max_bonds_per_issuer': issuer_counts.max()
        }

        return stats

    def filter_for_within_issuer_analysis(
        self,
        bond_data: pd.DataFrame,
        min_bonds: int = 3
    ) -> pd.DataFrame:
        """
        Filter bonds to those suitable for within-issuer analysis.

        Keeps only issuers with at least min_bonds outstanding bonds.
        This ensures sufficient within-issuer variation for analysis.

        Args:
            bond_data: DataFrame with issuer_id
            min_bonds: Minimum number of bonds per issuer

        Returns:
            Filtered DataFrame
        """
        if 'issuer_id' not in bond_data.columns:
            raise ValueError("issuer_id column not found.")

        # Count bonds per issuer
        issuer_counts = bond_data.groupby('issuer_id').size()

        # Keep only issuers with >= min_bonds
        valid_issuers = issuer_counts[issuer_counts >= min_bonds].index

        filtered = bond_data[bond_data['issuer_id'].isin(valid_issuers)].copy()

        print(f"  Filtered for within-issuer analysis:")
        print(f"    Issuers: {len(issuer_counts)} → {len(valid_issuers)}")
        print(f"    Bonds: {len(bond_data):,} → {len(filtered):,}")
        print(f"    Required: ≥{min_bonds} bonds per issuer")

        return filtered


def add_issuer_identification(
    bond_data: pd.DataFrame,
    parent_id_col: str = 'ultimate_parent_id',
    seniority_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to add issuer identification to bond data.

    Args:
        bond_data: DataFrame with bond information
        parent_id_col: Column name for ultimate parent ID
        seniority_col: Optional column name for seniority

    Returns:
        DataFrame with issuer_id and standardized seniority
    """
    identifier = IssuerIdentifier()

    # Classify seniority if not already standardized
    if 'seniority' not in bond_data.columns or seniority_col:
        bond_data = identifier.classify_seniority(bond_data, seniority_col)

    # Create composite issuer ID
    bond_data = identifier.create_composite_issuer_id(
        bond_data,
        parent_id_col=parent_id_col,
        seniority_col='seniority'
    )

    return bond_data
