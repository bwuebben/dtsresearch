"""
Configuration parameters for DTS Research Pipeline.

This file contains configurable parameters that can be adjusted
without modifying core analysis code.
"""

# =============================================================================
# SECTOR CLASSIFICATION
# =============================================================================

# Bloomberg classification level to use for sector mapping
# Options: 'BCLASS3' or 'BCLASS4'
BLOOMBERG_CLASS_LEVEL = 'BCLASS3'  # Default to BCLASS3

# Mapping from Bloomberg classification to 4 research sectors
# This will need to be populated based on actual BCLASS3/4 values
SECTOR_MAPPING = {
    # BCLASS3 examples (update with actual mappings):
    'Industrial': 'Industrial',
    'Financial': 'Financial',
    'Utility': 'Utility',
    'Energy': 'Energy',
    # Add more mappings as needed
}

# =============================================================================
# WITHIN-ISSUER ANALYSIS FILTERS
# =============================================================================

# Minimum number of bonds per issuer per week for within-issuer analysis
MIN_BONDS_PER_ISSUER_WEEK = 3

# Minimum maturity dispersion (years) required for within-issuer analysis
MIN_MATURITY_DISPERSION_YEARS = 2.0

# Exclude bonds within this many years of maturity (pull-to-par effect)
PULL_TO_PAR_EXCLUSION_YEARS = 1.0

# Maximum spread change (percentage) to exclude outliers
MAX_SPREAD_CHANGE_PCT = 200.0

# =============================================================================
# BUCKET DEFINITIONS
# =============================================================================

# Rating buckets for IG universe
RATING_BUCKETS_IG = ['AAA/AA', 'A', 'BBB']

# Rating buckets for HY universe
RATING_BUCKETS_HY = ['BB', 'B', 'CCC']

# Maturity buckets (years)
MATURITY_BUCKETS = [
    '1-2y',   # [1, 2)
    '2-3y',   # [2, 3)
    '3-5y',   # [3, 5)
    '5-7y',   # [5, 7)
    '7-10y',  # [7, 10)
    '10y+'    # [10, inf)
]

# Maturity bucket boundaries (for binning)
MATURITY_BUCKET_BOUNDARIES = [1, 2, 3, 5, 7, 10, float('inf')]

# Sectors for bucket analysis
BUCKET_SECTORS = ['Industrial', 'Financial', 'Utility', 'Energy']

# Minimum observations per bucket for reliable estimation
MIN_OBSERVATIONS_PER_BUCKET = 50

# =============================================================================
# STAGE 0 ANALYSIS PARAMETERS
# =============================================================================

# Significance level for hypothesis tests
ALPHA_LEVEL = 0.05

# Confidence level for confidence intervals
CONFIDENCE_LEVEL = 0.95

# Bootstrap iterations for robust standard errors
BOOTSTRAP_ITERATIONS = 1000

# Clustering level for standard errors
CLUSTER_LEVEL = 'week'  # Options: 'week', 'issuer', 'two_way'

# =============================================================================
# REGIME DEFINITIONS
# =============================================================================

# VIX thresholds for regime classification
VIX_NORMAL_THRESHOLD = 20
VIX_STRESS_THRESHOLD = 40

# Spread thresholds (bps) for regime classification
SPREAD_IG_THRESHOLD = 300
SPREAD_HY_THRESHOLD = 600

# =============================================================================
# MERTON MODEL PARAMETERS
# =============================================================================

# Risk-free rate assumptions (basis points)
RISK_FREE_RATE_DEFAULT = 250  # 2.5%

# Recovery rate assumptions
RECOVERY_RATE_DEFAULT = 0.40  # 40% for senior unsecured

# Asset volatility assumptions by rating
ASSET_VOL_BY_RATING = {
    'AAA': 0.15,
    'AA': 0.18,
    'A': 0.22,
    'BBB': 0.28,
    'BB': 0.35,
    'B': 0.45,
    'CCC': 0.60
}

# =============================================================================
# OUT-OF-SAMPLE VALIDATION PARAMETERS
# =============================================================================

# Training window size (years) for OOS validation
OOS_TRAIN_YEARS = 3

# Testing window size (years) for OOS validation
OOS_TEST_YEARS = 1

# Maximum OOS degradation allowed (%)
MAX_OOS_DEGRADATION_PCT = 30.0

# Minimum RMSE improvement required to justify complexity (%)
MIN_RMSE_IMPROVEMENT_PCT = 5.0

# =============================================================================
# PRODUCTION SPECIFICATION SELECTION
# =============================================================================

# Thresholds for Level 2 (Pure Merton)
PURE_MERTON_BETA_LOWER = 0.9
PURE_MERTON_BETA_UPPER = 1.1

# Minimum R² for statistical fit (Level 1)
MIN_R_SQUARED = 0.05

# Minimum out-of-sample R² (Level 3)
MIN_OOS_R_SQUARED = 0.03

# Maximum calculation time for production (milliseconds)
MAX_CALC_TIME_MS = 100

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Number of decimal places for table output
TABLE_DECIMALS = 3

# Figure DPI for publication quality
FIGURE_DPI = 300

# Figure size defaults (width, height in inches)
FIGURE_SIZE_SINGLE = (10, 6)
FIGURE_SIZE_MULTI = (16, 10)

# =============================================================================
# MOCK DATA GENERATION
# =============================================================================

# Number of bonds for mock data
MOCK_N_BONDS = 500

# Number of issuers for mock data
MOCK_N_ISSUERS = 150

# Date range for mock data
MOCK_START_DATE = '2010-01-01'
MOCK_END_DATE = '2024-12-31'

# Distribution of bonds across sectors (must sum to 1.0)
MOCK_SECTOR_DISTRIBUTION = {
    'Industrial': 0.50,
    'Financial': 0.30,
    'Utility': 0.10,
    'Energy': 0.10
}

# Distribution of bonds across ratings (must sum to 1.0)
MOCK_RATING_DISTRIBUTION = {
    'AAA': 0.05,
    'AA': 0.10,
    'A': 0.25,
    'BBB': 0.30,
    'BB': 0.15,
    'B': 0.10,
    'CCC': 0.05
}
