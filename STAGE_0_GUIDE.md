# Stage 0: Evolved DTS Foundation Analysis

## Overview

Stage 0 is the foundational analysis that determines the appropriate modeling approach for subsequent stages. It implements three complementary analyses from the evolved paper methodology to assess whether the Merton structural model provides an adequate framework for Duration Times Spread (DTS) effects.

**Key Question**: Does the Merton model adequately describe spread sensitivity to market-wide credit shocks?

**Output**: Decision path (1-5) that guides specification choices in Stages A-E

## Three-Pronged Analysis

### 1. Bucket-Level Time-Series Analysis (Specification 0.1)

**Purpose**: Test whether empirical spread sensitivities match Merton-predicted elasticities across rating-maturity buckets

**Specification** (for each bucket k):
```
y_{i,t} = α^(k) + β^(k) · f_{DTS,t} + ε_{i,t}
```

where:
- `y_{i,t} = Δs_{i,t} / s_{i,t-1}`: percentage spread change for bond i
- `f_{DTS,t} = ΔS^index_t / S^index_t-1`: index-level percentage spread change (DTS factor)
- `β^(k)`: empirical DTS sensitivity for bucket k
- Buckets = 8 rating groups × 9 maturity groups = 72 buckets

**Implementation**: `src/dts_research/analysis/stage0_bucket.py`

**Key Tests**:
- Compare empirical β^(k) to theoretical λ^Merton for each bucket
- Test whether β/λ ratio is close to 1.0 (median within 0.8-1.2)
- Monotonicity: Does β decrease with maturity within rating classes (as Merton predicts)?
- Bucket population: Are we adequately covering the rating-maturity space?

### 2. Within-Issuer Analysis (Specification 0.2)

**Purpose**: Test Merton prediction that spread sensitivity equals theoretical λ, using within-issuer variation to control for credit quality

**Specification** (for each issuer-week):
```
Δs_{i,t} / s_{i,t-1} = α_{j,t} + β · λ^Merton_{i,t} + ε_{i,t}
```

where:
- LHS: percentage spread change for bond i from issuer j at time t
- `α_{j,t}`: issuer-week fixed effect (absorbs all common credit risk factors)
- `λ^Merton_{i,t}`: Merton-predicted elasticity for bond i (function of maturity and spread)
- `β`: coefficient on Merton lambda (should equal 1 if theory is correct)

**Implementation**: `src/dts_research/analysis/stage0_within_issuer.py`

**Requirements**:
- ≥3 bonds per issuer-week
- ≥2 years maturity dispersion
- Exclude bonds within 1 year of maturity (pull-to-par)

**Key Tests**:
- H0: β = 1 vs H1: β ≠ 1 (Merton predicts β = 1)
- Is pooled β within [0.9, 1.1]?
- Distribution of issuer-week estimates
- Consistency across time

### 3. Sector Interaction Analysis (Specification 0.3)

**Purpose**: Test whether spread sensitivities vary by industry sector when using Merton-scaled DTS factor

**Specification**:
```
y_{i,t} = α + β_0 · (λ^Merton_i × f_{DTS,t})
          + β_F · (Financial_i × λ^Merton_i × f_{DTS,t})
          + β_U · (Utility_i × λ^Merton_i × f_{DTS,t})
          + β_E · (Energy_i × λ^Merton_i × f_{DTS,t}) + ε_{i,t}
```

where:
- `y_{i,t}`: percentage spread change
- `λ^Merton_i × f_{DTS,t}`: Merton-scaled DTS factor
- Sectors: Financial, Utility, Energy vs Industrial (baseline)
- `β_0`: baseline sensitivity (Industrial sector)
- `β_F, β_U, β_E`: sector-specific deviations from baseline

**Implementation**: `src/dts_research/analysis/stage0_sector.py`

**Key Tests**:
- Joint test: H0: β_F = β_U = β_E = 0 (sectors don't differ)
- Individual sector tests: Does any sector significantly deviate?
- Sector-specific total sensitivities: β_0 + β_sector ≈ 1?

## Decision Framework

Stage 0 synthesizes the three analyses into one of five decision paths:

### Path 1: Perfect Alignment
- **Criteria**:
  - Bucket β/λ ratio median in [0.8, 1.2]
  - Within-issuer β in [0.9, 1.1], p-value for β=1 > 0.05
  - Monotonicity holds
  - Sectors don't differ significantly
- **Interpretation**: Merton model works well
- **Action**: Proceed with standard DTS specifications in all stages

### Path 2: Works with Sector Heterogeneity
- **Criteria**: Base effects validate theory (β ≈ 1), but sectors differ significantly
- **Interpretation**: Merton model works but needs sector adjustments
- **Action**: Add sector interactions throughout analysis

### Path 3: Weak but Present
- **Criteria**: Effects in right direction but weak (β in [0.7, 1.3] but outside [0.9, 1.1])
- **Interpretation**: Merton model weakly supported
- **Action**: Proceed cautiously, report robustness extensively

### Path 4: Mixed Evidence
- **Criteria**: Strong evidence from one approach, weak from another
- **Interpretation**: Partial support for Merton
- **Action**: Use supported specifications, avoid unsupported ones

### Path 5: Theory Fails
- **Criteria**: β significantly different from 1, non-monotonic, or wrong patterns
- **Interpretation**: Merton model inadequate
- **Action**: Consider alternative models (reduced-form, rating-based factors)

**Implementation**: `src/dts_research/analysis/stage0_synthesis.py`

## Outputs

### Figures (10 total)

1. **Bucket Beta vs Lambda Scatter**: Empirical β vs theoretical λ by bucket
2. **Beta Heatmap**: Heatmap of β across rating-maturity grid
3. **Beta Distribution**: Distribution of β/λ ratios
4. **Within-Issuer Coefficients**: Maturity coefficient estimates
5. **Within-Issuer Weights**: Inverse-variance weights by issuer-week
6. **Within-Issuer Distribution**: Distribution of issuer-week β estimates
7. **Sector by Maturity**: Sector effects across maturity buckets
8. **Sector Effects**: Bar chart of sector-specific sensitivities
9. **Sector F-test**: Visualization of joint test results
10. **Decision Summary**: Decision framework visualization

### Tables (17 total)

**Bucket Analysis** (6 tables):
- Table 1-2: Bucket characteristics and β estimates (IG, HY)
- Table 3-4: β vs λ comparison and ratio analysis (IG, HY)
- Table 5-6: Monotonicity tests (IG, HY)

**Within-Issuer Analysis** (4 tables):
- Table 7-8: Within-issuer summary statistics (IG, HY)
- Table 9-10: Pooled β estimates and hypothesis tests for β=1 (IG, HY)

**Sector Analysis** (4 tables):
- Table 11-12: Sector regression results (IG, HY)
- Table 13-14: Sector hypothesis tests and total sensitivities (IG, HY)

**Synthesis** (3 tables):
- Table 15-16: Decision paths and recommendations (IG, HY)
- Table 17: Cross-universe comparison

### Summary Report

`stage0_summary.txt` contains:
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
from dts_research.analysis.stage0_bucket import BucketLevelAnalysis
from dts_research.analysis.stage0_within_issuer import WithinIssuerAnalysis
from dts_research.analysis.stage0_sector import SectorInteractionAnalysis
from dts_research.analysis.stage0_synthesis import Stage0Synthesis

# Load data
loader = BondDataLoader()
bond_data = loader.generate_mock_data('2020-01-01', '2023-12-31', n_bonds=500)
index_data = loader.generate_mock_index_data('2020-01-01', '2023-12-31')

# Add sector classification and issuer identification
from dts_research.data.sector_classification import SectorClassifier
from dts_research.data.issuer_identification import add_issuer_identification

classifier = SectorClassifier()
bond_data = classifier.classify_sector(bond_data, bclass_column='sector_classification')
bond_data = classifier.add_sector_dummies(bond_data)
bond_data = add_issuer_identification(bond_data)

# Run analyses
bucket_analyzer = BucketLevelAnalysis()
bucket_results = bucket_analyzer.run_bucket_analysis(bond_data, universe='IG')

within_analyzer = WithinIssuerAnalysis()
within_results = within_analyzer.run_within_issuer_analysis(bond_data, universe='IG')

sector_analyzer = SectorInteractionAnalysis()
sector_results = sector_analyzer.run_sector_analysis(bond_data, universe='IG')

# Synthesize results
synthesizer = Stage0Synthesis()
synthesis = synthesizer.synthesize_results(
    bucket_results, within_results, sector_results, universe='IG'
)

# Check decision path
print(f"Decision: Path {synthesis['decision_path']} - {synthesis['path_name']}")
print(f"Rationale: {synthesis['rationale']}")
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
- β/λ ratio median close to 1.0 across methods
- Statistical tests fail to reject β = 1
- Monotonicity clearly present (β decreasing with maturity)
- Large number of populated buckets and issuer-weeks

**Weak Evidence** (Path 3):
- Right direction but β outside [0.9, 1.1]
- Some monotonicity but not consistent
- Limited sample in some buckets/issuers

**Contradictory Evidence** (Path 4):
- Strong in one method, weak/wrong in another
- Need to understand why methods disagree
- May indicate specific specification issues

**Theory Failure** (Path 5):
- β significantly different from 1
- Non-monotonic patterns (β increasing with maturity)
- No statistical relationship
- Time to consider alternative models

### Common Issues and Solutions

**Issue 1: Sparse buckets**
- **Symptom**: Many empty buckets, especially in tails
- **Solution**: Use coarser bucket grid or focus on populated regions
- **Impact**: Reduces precision of bucket-level estimates

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

1. **Review Summary**: Check `stage0_summary.txt` for decision paths
2. **Examine Diagnostics**: Look at Figure 10 (decision summary)
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

Run Stage 0 tests:

```bash
pytest tests/test_stage0_analysis.py -v
```

Run Stage 0 with mock data:

```bash
python run_stage0.py \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --output-dir output/stage0_test \
    --verbose
```

This generates synthetic data with 500 bonds across multi-bond issuers and runs the complete three-pronged analysis.
