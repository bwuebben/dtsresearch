# Stage 0: Evolved DTS Foundation Analysis

## Overview

Stage 0 is the foundational analysis that determines the appropriate modeling approach for subsequent stages. It implements three complementary analyses from the evolved paper methodology to assess whether the Merton structural model provides an adequate framework for Duration Times Spread (DTS) effects.

**Key Question**: Does the Merton model adequately describe maturity-spread relationships in corporate bonds?

**Output**: Decision path (1-5) that guides specification choices in Stages A-E

## Three-Pronged Analysis

### 1. Bucket-Level Cross-Sectional Analysis (Specification 0.1)

**Purpose**: Test basic Merton prediction across rating-maturity buckets

**Specification**:
```
ln(s_b) = α + λ·T_b + ε_b
```

where:
- `s_b` = average OAS in bucket b
- `T_b` = average maturity in bucket b
- Buckets = 8 rating groups × 9 maturity groups = 72 buckets

**Implementation**: `src/dts_research/analysis/stage0_bucket.py`

**Key Tests**:
- H0: λ ≤ 0 vs H1: λ > 0 (Merton predicts λ > 0)
- Monotonicity: Are spreads increasing in maturity within rating classes?
- Bucket population: Are we adequately covering the rating-maturity space?

### 2. Within-Issuer Fixed Effects Analysis (Specification 0.2)

**Purpose**: Control for common credit risk by using same issuer, different maturities

**Specification** (for each issuer-week):
```
ln(s_it) = α_issuer-week + λ·T_it + ε_it
```

where:
- Fixed effect `α_issuer-week` absorbs all common credit risk factors
- Only variation is maturity across bonds from same issuer
- Pooled estimate uses inverse-variance weighting across issuer-weeks

**Implementation**: `src/dts_research/analysis/stage0_within_issuer.py`

**Requirements**:
- ≥3 bonds per issuer-week
- ≥2 years maturity dispersion
- Exclude bonds within 1 year of maturity (pull-to-par)

**Key Tests**:
- H0: λ ≤ 0 vs H1: λ > 0 (one-sided test)
- Distribution of issuer-week estimates
- Consistency across time

### 3. Sector Interaction Analysis (Specification 0.3)

**Purpose**: Test whether DTS effects vary by industry sector

**Specification**:
```
ln(s_it) = α + λ·T_it + β_F·(T_it × Financial_i) + β_U·(T_it × Utility_i)
           + β_E·(T_it × Energy_i) + ε_it
```

where sectors are:
- Financial (banks, insurance, REITs)
- Utility (electric, gas, water)
- Energy (oil & gas, coal, renewable)
- Industrial (baseline - all other firms)

**Implementation**: `src/dts_research/analysis/stage0_sector.py`

**Key Tests**:
- Joint test: H0: β_F = β_U = β_E = 0 (sectors don't matter)
- Individual sector effects
- Sector-specific λ estimates

## Decision Framework

Stage 0 synthesizes the three analyses into one of five decision paths:

### Path 1: Perfect Alignment
- **Criteria**: Bucket λ > 0 (p < 0.05), Within λ > 0 (p < 0.05), Monotonic, Sectors don't matter
- **Interpretation**: Merton model works perfectly
- **Action**: Proceed with standard DTS specifications in all stages

### Path 2: Works with Sector Heterogeneity
- **Criteria**: Base effects positive and significant, but sectors matter
- **Interpretation**: Merton model works but needs sector adjustments
- **Action**: Add sector interactions throughout analysis

### Path 3: Weak but Present
- **Criteria**: Effects in right direction but weak (p < 0.10) or inconsistent
- **Interpretation**: Merton model weakly supported
- **Action**: Proceed cautiously, report robustness extensively

### Path 4: Mixed Evidence
- **Criteria**: Strong evidence from one approach, weak from another
- **Interpretation**: Partial support for Merton
- **Action**: Use supported specifications, avoid unsupported ones

### Path 5: Theory Fails
- **Criteria**: Non-monotonic, wrong sign, or no significance
- **Interpretation**: Merton model inadequate
- **Action**: Consider alternative models (reduced-form, rating-based factors)

**Implementation**: `src/dts_research/analysis/stage0_synthesis.py`

## Outputs

### Figures (10 total)

1. **Bucket Characteristics Scatter**: Spread vs maturity by rating class
2. **Cross-Sectional Regression Fit**: Bucket regression with confidence bands
3. **Maturity Monotonicity**: Box plots of spreads by maturity bucket
4. **Within-Issuer λ Distribution**: Histogram of issuer-week estimates
5. **Within-Issuer λ Time Series**: Evolution of pooled estimate over time
6. **Sector Interaction Coefficients**: Bar chart of sector effects
7. **Sector-Specific λ Comparison**: Forest plot of sector-specific estimates
8. **Decision Path Comparison**: IG vs HY decision paths
9. **λ Estimates Comparison**: Three methods side-by-side
10. **Diagnostic Dashboard**: 2×3 grid with all key diagnostics

### Tables (17 total)

**Bucket Analysis** (6 tables):
- Table 1-2: Bucket characteristics (IG, HY)
- Table 3-4: Bucket regression results (IG, HY)
- Table 5-6: Monotonicity tests (IG, HY)

**Within-Issuer Analysis** (4 tables):
- Table 7-8: Within-issuer summary statistics (IG, HY)
- Table 9-10: Pooled estimates and hypothesis tests (IG, HY)

**Sector Analysis** (4 tables):
- Table 11-12: Sector regression results (IG, HY)
- Table 13-14: Sector hypothesis tests (IG, HY)

**Synthesis** (3 tables):
- Table 15-16: Decision paths and recommendations (IG, HY)
- Table 17: Cross-universe comparison

### Summary Report

`STAGE_0_SUMMARY.txt` contains:
- Decision paths for IG and HY
- Key statistics from all three analyses
- Rationale for each decision
- Recommendations for Stages A-E

## Usage

### Command Line

```bash
python run_stage0.py \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --output-dir output/stage0 \
    --verbose
```

### Python API

```python
from dts_research.data.loader import BondDataLoader
from dts_research.analysis.stage0_bucket import run_bucket_analysis_both_universes
from dts_research.analysis.stage0_within_issuer import run_within_issuer_analysis_both_universes
from dts_research.analysis.stage0_sector import run_sector_analysis_both_universes
from dts_research.analysis.stage0_synthesis import run_stage0_synthesis

# Load data
loader = BondDataLoader()
bond_data = loader.load_bond_data('2020-01-01', '2023-12-31')

# Add sector classification and issuer identification
from dts_research.data.sector_classification import SectorClassifier
from dts_research.data.issuer_identification import add_issuer_identification

classifier = SectorClassifier()
bond_data = classifier.classify_sector(bond_data, bclass_column='sector_classification')
bond_data = classifier.add_sector_dummies(bond_data)
bond_data = add_issuer_identification(bond_data)

# Run analyses
bucket_results = run_bucket_analysis_both_universes(bond_data)
within_results = run_within_issuer_analysis_both_universes(bond_data)
sector_results = run_sector_analysis_both_universes(bond_data)

# Synthesize
synthesis = run_stage0_synthesis(
    bucket_results['IG'], bucket_results['HY'],
    within_results['IG'], within_results['HY'],
    sector_results['IG'], sector_results['HY']
)

# Check decision path
print(f"IG Decision: Path {synthesis['IG']['decision_path']}")
print(f"HY Decision: Path {synthesis['HY']['decision_path']}")
```

## Data Requirements

### Required Fields
- `bond_id`: Unique bond identifier
- `date`: Observation date
- `oas`: Option-adjusted spread (bps)
- `rating`: Credit rating (AAA, AA, A, BBB, BB, B, CCC)
- `time_to_maturity`: Years to maturity
- `ultimate_parent_id`: Ultimate parent company ID
- `seniority`: Bond seniority (Senior/Subordinated)
- `sector_classification`: Bloomberg BCLASS3 sector code

### Filtering

**Bucket Analysis**:
- Spreads > 0
- Time to maturity > 0
- Valid ratings

**Within-Issuer Analysis**:
- All bucket filters
- ≥3 bonds per issuer-week
- ≥2 years maturity dispersion within issuer-week
- Exclude bonds with < 1 year to maturity (pull-to-par)

**Sector Analysis**:
- All bucket filters
- Valid sector classification

## Configuration

Key parameters in `src/dts_research/config.py`:

```python
# Bucket definitions
RATING_BUCKETS = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Default']
MATURITY_BUCKETS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 100]

# Within-issuer filters
MIN_BONDS_PER_ISSUER_WEEK = 3
MIN_MATURITY_DISPERSION_YEARS = 2.0
PULL_TO_PAR_EXCLUSION_YEARS = 1.0

# Sector classification
SECTOR_MAPPING = {
    'Financial': ['Banks', 'Insurance', 'REITs'],
    'Utility': ['Electric', 'Gas', 'Water'],
    'Energy': ['Oil & Gas', 'Coal', 'Renewable'],
    'Industrial': ['all others']
}
```

## Interpretation Guide

### When to Trust Stage 0 Results

**Strong Evidence** (Paths 1-2):
- Consistent signs across all three methods
- Statistical significance (p < 0.05) in multiple tests
- Monotonicity clearly present
- Large number of populated buckets and issuer-weeks

**Weak Evidence** (Path 3):
- Right direction but weak significance (p < 0.10)
- Some monotonicity but not consistent
- Limited sample in some buckets/issuers

**Contradictory Evidence** (Path 4):
- Strong in one method, weak/wrong in another
- Need to understand why methods disagree
- May indicate specific specification issues

**Theory Failure** (Path 5):
- Wrong signs (λ < 0)
- Non-monotonic patterns
- No statistical significance
- Time to consider alternative models

### Common Issues and Solutions

**Issue 1: Sparse buckets**
- **Symptom**: Many empty buckets, especially in tails
- **Solution**: Use coarser bucket grid or focus on populated regions
- **Impact**: Reduces precision of bucket-level λ estimate

**Issue 2: Few multi-bond issuers**
- **Symptom**: Low issuer-week count in within-issuer analysis
- **Solution**: Relax min_bonds requirement or use longer time windows
- **Impact**: Within-issuer estimate less reliable

**Issue 3: Sector imbalance**
- **Symptom**: One sector dominates (e.g., 80% Industrial)
- **Solution**: Use sector fixed effects or analyze sectors separately
- **Impact**: Sector interactions may be underpowered

**Issue 4: Pull-to-par effects**
- **Symptom**: Short-maturity bonds have compressed spreads
- **Solution**: Exclude bonds < 1 year to maturity
- **Impact**: Already handled by filters

## Next Steps

After completing Stage 0:

1. **Review Summary**: Check `STAGE_0_SUMMARY.txt` for decision paths
2. **Examine Diagnostics**: Look at Figure 10 (diagnostic dashboard)
3. **Validate Results**: Ensure populated buckets and adequate sample
4. **Determine Path**: Use decision path to guide Stages A-E

### Decision Path Implications

**Path 1 → Stages A-E**: Standard specifications throughout

**Path 2 → Stages A-E**: Add sector interactions in all specifications

**Path 3 → Stages A-D**: Use standard specs but extensive robustness, skip E

**Path 4 → Stages A-C**: Use supported specifications only, limited Stage D, skip E

**Path 5 → Alternative Analysis**: Merton model fails, consider:
- Reduced-form models
- Rating-based factor models
- PCA/factor analysis
- Non-parametric approaches

## References

See Section 3 ("Evolved Stage 0") in the main paper for detailed methodology and theoretical justification.

## Module Structure

```
src/dts_research/
├── config.py                           # Configuration parameters
├── data/
│   ├── bucket_definitions.py           # 72-bucket grid
│   ├── issuer_identification.py        # Composite issuer IDs
│   ├── sector_classification.py        # BCLASS → 4 sectors
│   └── filters.py                      # Data filtering
├── analysis/
│   ├── stage0_bucket.py               # Bucket-level analysis
│   ├── stage0_within_issuer.py        # Within-issuer analysis
│   ├── stage0_sector.py               # Sector interaction analysis
│   └── stage0_synthesis.py            # Decision framework
├── plotting/
│   └── stage0_plots.py                # 10 visualizations
└── utils/
    ├── statistical_tests.py           # Inference tools
    └── reporting0.py                  # 17 tables + summary
```

## Testing

Run Stage 0 with mock data:

```bash
python run_stage0.py \
    --start-date 2022-01-01 \
    --end-date 2022-02-28 \
    --output-dir output/stage0_test \
    --verbose
```

This generates 2 months of synthetic data with 500 bonds across 211 multi-bond issuers.
