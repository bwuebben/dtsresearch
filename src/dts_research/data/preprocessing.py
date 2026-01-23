"""
Centralized data preprocessing for OAS spread changes.

This module provides the single source of truth for computing:
- Bond-level OAS percentage changes
- Index-level DTS factor (average OAS percentage change)

All stage analysis modules should use these functions rather than
implementing their own spread change calculations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def compute_spread_changes(
    df: pd.DataFrame,
    bond_id_col: str = 'bond_id',
    oas_col: str = 'oas',
    date_col: str = 'date',
    max_change_pct: float = 1.0
) -> pd.DataFrame:
    """
    Compute percentage spread changes for each bond.

    Calculates daily relative spread changes: (oas_t - oas_{t-1}) / oas_{t-1}

    Args:
        df: DataFrame with bond observations
        bond_id_col: Column name for bond identifier (default: 'bond_id')
        oas_col: Column name for OAS values (default: 'oas')
        date_col: Column name for dates (default: 'date')
        max_change_pct: Maximum absolute percentage change to keep (default: 1.0 = ±100%)
                        Values exceeding this are treated as outliers and removed.

    Returns:
        DataFrame with added columns:
        - oas_lag: Previous period OAS
        - oas_pct_change: Percentage change in OAS (bond-level)

    Note:
        First observation for each bond will have NaN spread change and is dropped.
    """
    df = df.copy()

    # Determine bond ID column
    if bond_id_col not in df.columns:
        if 'cusip' in df.columns:
            bond_id_col = 'cusip'
        else:
            raise ValueError(f"Bond ID column '{bond_id_col}' not found. Available: {df.columns.tolist()}")

    # Sort by bond and date
    df = df.sort_values([bond_id_col, date_col])

    # Compute lagged spread within each bond
    df['oas_lag'] = df.groupby(bond_id_col)[oas_col].shift(1)

    # Compute percentage spread change
    df['oas_pct_change'] = (df[oas_col] - df['oas_lag']) / df['oas_lag']

    # Remove NaN (first observation for each bond)
    df = df.dropna(subset=['oas_pct_change'])

    # Remove extreme outliers (likely data errors)
    if max_change_pct is not None:
        df = df[df['oas_pct_change'].abs() <= max_change_pct]

    return df


def compute_index_dts_factor(
    df: pd.DataFrame,
    oas_col: str = 'oas',
    date_col: str = 'date',
    output_col: str = 'oas_index_pct_change'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute index-level DTS factor: average percentage spread change per date.

    The index factor represents the market-wide spread movement and is used
    as the independent variable in DTS regressions.

    Args:
        df: DataFrame with bond observations (must have oas_col and date_col)
        oas_col: Column name for OAS values (default: 'oas')
        date_col: Column name for dates (default: 'date')
        output_col: Column name for output index factor (default: 'oas_index_pct_change')

    Returns:
        Tuple of:
        - df: DataFrame with index factor merged in as output_col
        - index_factor: DataFrame with date and index-level statistics

    Note:
        Uses equal-weighted average of OAS levels, then computes percentage change.
    """
    df = df.copy()

    # Compute index-level average spread per date
    index_factor = df.groupby(date_col)[oas_col].mean()

    # Compute percentage change of index
    index_factor_pct = index_factor.pct_change()

    # Create index factor DataFrame
    index_df = index_factor_pct.reset_index()
    index_df.columns = [date_col, output_col]

    # Merge back to bond data
    df = df.merge(index_df, on=date_col, how='left')

    # Also create detailed index factor DataFrame for reference
    index_factor_full = pd.DataFrame({
        date_col: index_factor.index,
        'oas_index': index_factor.values,
        output_col: index_factor_pct.values
    })

    return df, index_factor_full


def prepare_spread_change_data(
    df: pd.DataFrame,
    bond_id_col: str = 'bond_id',
    oas_col: str = 'oas',
    date_col: str = 'date',
    max_change_pct: float = 1.0,
    add_week_identifier: bool = True,
    add_spread_regime: bool = True,
    ig_threshold: float = 300.0
) -> pd.DataFrame:
    """
    Prepare bond data with spread changes and index DTS factor.

    This is the main preprocessing function that combines bond-level spread
    changes with the index-level DTS factor. Use this function in run_stage*.py
    scripts and analysis modules.

    Args:
        df: DataFrame with bond observations
        bond_id_col: Column name for bond identifier (default: 'bond_id')
        oas_col: Column name for OAS values (default: 'oas')
        date_col: Column name for dates (default: 'date')
        max_change_pct: Maximum absolute percentage change to keep (default: 1.0 = ±100%)
        add_week_identifier: Add 'week' column for clustering (default: True)
        add_spread_regime: Add 'spread_regime' column (IG/HY) (default: True)
        ig_threshold: OAS threshold for IG/HY classification (default: 300 bps)

    Returns:
        DataFrame with added columns:
        - oas_lag: Previous period OAS
        - oas_pct_change: Percentage change in OAS (bond-level)
        - oas_index_pct_change: Index-level DTS factor
        - f_DTS: Alias for oas_index_pct_change (for compatibility)
        - week: Week identifier for clustering (if add_week_identifier=True)
        - spread_regime: 'IG' or 'HY' based on spread level (if add_spread_regime=True)

    Example:
        >>> from dts_research.data.preprocessing import prepare_spread_change_data
        >>> df = prepare_spread_change_data(bond_data)
        >>> # Now df has oas_pct_change and oas_index_pct_change columns
    """
    df = df.copy()

    # Step 1: Compute bond-level spread changes
    df = compute_spread_changes(
        df,
        bond_id_col=bond_id_col,
        oas_col=oas_col,
        date_col=date_col,
        max_change_pct=max_change_pct
    )

    # Step 2: Compute index-level DTS factor
    df, _ = compute_index_dts_factor(
        df,
        oas_col=oas_col,
        date_col=date_col,
        output_col='oas_index_pct_change'
    )

    # Add f_DTS alias for compatibility with some modules
    df['f_DTS'] = df['oas_index_pct_change']

    # Step 3: Add optional columns
    if add_week_identifier:
        df['week'] = (
            df[date_col].dt.isocalendar().week.astype(str) + '_' +
            df[date_col].dt.year.astype(str)
        )

    if add_spread_regime:
        df['spread_regime'] = np.where(df[oas_col] < ig_threshold, 'IG', 'HY')

    # Step 4: Drop rows with NaN in key columns
    df = df.dropna(subset=['oas_pct_change', 'oas_index_pct_change'])

    return df


def compute_spread_changes_legacy(
    df: pd.DataFrame,
    bond_id_col: str = 'bond_id',
    output_col: str = 'spread_change'
) -> pd.DataFrame:
    """
    Compute spread changes with legacy column name for Stage 0 compatibility.

    Stage 0 modules use 'spread_change' instead of 'oas_pct_change'.
    This function provides backwards compatibility.

    Args:
        df: DataFrame with bond observations
        bond_id_col: Column name for bond identifier
        output_col: Column name for output (default: 'spread_change')

    Returns:
        DataFrame with 'spread_change' and 'oas_lag' columns
    """
    df = compute_spread_changes(df, bond_id_col=bond_id_col)

    # Rename to legacy column name
    if output_col != 'oas_pct_change':
        df[output_col] = df['oas_pct_change']

    return df
