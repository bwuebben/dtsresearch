"""
Configuration file for DTS Research project.

Users should create a config_local.py file to override these settings
with their specific database credentials and paths.
"""

# Database configuration
DATABASE_CONFIG = {
    'connection_string': None,  # Fill in with your connection string
    'query_template': """
        -- Customize this query based on your database schema
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
    """,
}

# Analysis parameters
ANALYSIS_CONFIG = {
    'start_date': '2010-01-01',
    'end_date': '2024-12-31',
    'min_bucket_observations': 50,  # Minimum observations per bucket
    'use_mock_data': True,  # Set to False for real data
    'mock_n_bonds': 500,
    'mock_seed': 42,
}

# Regression parameters
REGRESSION_CONFIG = {
    'cluster_by': 'date',  # Cluster standard errors by week
    'min_sample_size': 30,  # Minimum sample for regression
}

# Statistical test parameters
TEST_CONFIG = {
    'alpha': 0.05,  # Significance level
    'outlier_threshold': 1.5,  # For β/λ ratio outliers
    'bootstrap_iterations': 1000,
}

# Output parameters
OUTPUT_CONFIG = {
    'figures_dir': './output/figures',
    'reports_dir': './output/reports',
    'save_figures': True,
    'figure_dpi': 300,
    'figure_format': 'png',
}

# Merton model parameters
MERTON_CONFIG = {
    'use_power_law': False,  # Use exact Merton vs power law approximation
    'reference_maturity': 5.0,  # years
    'reference_spread': 100.0,  # bps
}

# Try to import local overrides
try:
    from config_local import *
    print("Loaded local configuration from config_local.py")
except ImportError:
    pass
