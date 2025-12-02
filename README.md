# DTS Research: Corporate Bond Spread Sensitivity Analysis

Implementation of the research program for testing structural model predictions in corporate bond spread sensitivities, as outlined in the accompanying paper.

## Project Overview

This project implements a multi-stage empirical research program to test whether Merton structural model predictions explain cross-sectional variation in corporate bond spread sensitivities.

## Implemented Stages

### ✅ Stage 0: Raw Validation Using Bucket-Level Analysis

Provides an assumption-free test of Merton predictions:

1. **Raw validation**: Do bonds in similar buckets exhibit spread sensitivities consistent with structural theory?
2. **Bucket-level testing**: Pooled regressions with strong statistical power
3. **Decision point**: Determine whether Merton provides adequate baseline

**Run**: `python run_stage0.py` (~10 seconds)

### ✅ Stage A: Establish Cross-Sectional Variation

Establishes that DTS betas differ across bonds BEFORE testing whether Merton explains why:

1. **Specification A.1**: Bucket-level betas with F-tests for equality
2. **Specification A.2**: Continuous characteristics (rolling window estimation)
3. **Critical decision**: If no variation, standard DTS adequate → STOP

**Run**: `python run_stageA.py` (~15 seconds without Spec A.2, ~3 minutes with)

**See**: `STAGE_A_GUIDE.md` for detailed documentation

### ✅ Stage B: Does Merton Explain the Variation?

Tests whether Merton's structural model explains the variation documented in Stage A:

1. **Specification B.1**: Merton as offset (constrained) - tests if β_Merton = 1
2. **Specification B.2**: Decomposed components - tests β_T and β_s separately
3. **Specification B.3**: Unrestricted - fully flexible comparison
4. **Theory vs Reality**: Direct comparison of empirical betas to Merton predictions
5. **Decision paths**: 4 outcomes (theory works / needs calibration / partial / fails)

**Run**: `python run_stageB.py` (~20 seconds)

**Prerequisite**: Stage A finds variation (F-test p < 0.10)

## Project Structure

```
dtsresearch/
├── src/
│   └── dts_research/
│       ├── data/
│       │   └── loader.py              # Data loading and mock data generation
│       ├── models/
│       │   └── merton.py              # Merton lambda calculations
│       ├── analysis/
│       │   ├── buckets.py             # Bucket classification system
│       │   ├── stage0.py              # Stage 0 regression analysis
│       │   ├── stageA.py              # Stage A analysis
│       │   └── stageB.py              # Stage B analysis ✨ NEW
│       ├── visualization/
│       │   ├── stage0_plots.py        # Figures 0.1-0.3
│       │   ├── stageA_plots.py        # Figures A.1-A.2
│       │   └── stageB_plots.py        # Figures B.1-B.3 ✨ NEW
│       └── utils/
│           ├── reporting.py           # Stage 0 reports
│           ├── reportingA.py          # Stage A reports
│           └── reportingB.py          # Stage B reports ✨ NEW
├── tests/                             # Unit tests
├── output/
│   ├── figures/                       # Generated plots
│   └── reports/                       # Generated tables and summaries
├── run_stage0.py                      # Stage 0 orchestration
├── run_stageA.py                      # Stage A orchestration
├── run_stageB.py                      # Stage B orchestration ✨ NEW
├── STAGE_A_GUIDE.md                   # Stage A documentation
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Mock Data

Run Stage 0 analysis using synthetic data for testing:

```bash
python run_stage0.py
```

This will:
- Generate mock bond data (500 bonds, 2010-2024, weekly observations)
- Classify bonds into 72+ buckets (rating × maturity × sector)
- Run pooled regressions for each bucket
- Perform statistical tests (level, cross-maturity, regime)
- Generate 3 figures and 4 reports in `output/`

### Using Real Database Data

To use your own bond database:

1. **Configure database connection** in `src/dts_research/data/loader.py`:
   - Update the `connect()` method with your connection logic
   - Uncomment and modify the SQL query in `load_bond_data()`
   - Install appropriate database driver (see `requirements.txt`)

2. **Expected database schema**:
   ```sql
   Required columns:
   - bond_id: unique identifier
   - date: observation date (weekly frequency recommended)
   - oas: option-adjusted spread (basis points)
   - rating: credit rating (e.g., 'AAA', 'BBB+', 'B')
   - maturity_date: bond maturity date
   - sector: Bloomberg Class 3 sector or equivalent
   - issuer_id: issuer identifier
   ```

3. **Run with real data**:
   - Edit `run_stage0.py` and set `use_mock_data = False`
   - Provide connection string
   - Run: `python run_stage0.py`

## Methodology

### Bucket Classification

Bonds are classified into buckets defined by:

- **Rating**: AAA/AA, A, BBB, BB, B, CCC
- **Maturity**: 1-2y, 2-3y, 3-5y, 5-7y, 7-10y, 10y+
- **Sector**: Industry classification (e.g., Industrial, Financial, Utility)

This creates ~72 IG buckets and ~72 HY buckets.

### Pooled Regression

For each bucket k, estimate:

```
y_i,t = α^(k) + β^(k) · f_DTS,t + ε_i,t
```

where:
- `y_i,t = ΔS_i,t / S_i,t-1`: percentage spread change for bond i
- `f_DTS,t = ΔS^index_t / S^index_t-1`: index-level percentage spread change
- `β^(k)`: empirical DTS sensitivity for bucket k

Standard errors are clustered by week.

### Theoretical Benchmarks

For each bucket, calculate theoretical Merton lambda:

```
λ^Merton = λ_T(T; s) × λ_s(s)
```

where:
- `λ_T`: maturity adjustment factor (reference: 5y bond)
- `λ_s`: credit quality adjustment factor (reference: 100 bps spread)

Compare empirical β to theoretical λ.

### Statistical Tests

1. **Level test**: H₀: β^(k) = λ^Merton for each bucket
2. **Cross-maturity pattern**: Do short bonds have higher β than long bonds?
3. **Regime pattern**: Does dispersion decline as spreads widen?

## Output Deliverables

### Tables

- **Table 0.1**: Bucket-level results showing β, λ, ratio, t-stat, sample size
- **Table 0.2**: Cross-maturity patterns by rating class
- **Full results CSV**: Complete regression output for all buckets

### Figures

- **Figure 0.1**: Scatter plot of empirical β vs theoretical λ
- **Figure 0.2**: Cross-maturity patterns by rating (6 panels)
- **Figure 0.3**: Regime patterns showing dispersion vs spread level

### Written Summary

2-3 page report addressing:
1. Does Merton predict bucket-level sensitivities?
2. Is cross-maturity pattern correct?
3. Does pattern differ by spread level?
4. Where do largest deviations occur?
5. Practical implication for model selection

### Decision Recommendation

Based on results, provides guidance on whether to:
- Use Merton as baseline for Stages A-C
- Calibrate Merton with scaling factor
- Run parallel theory-constrained and unrestricted tracks
- Emphasize regime-differentiated modeling

## Key Features

### Theoretical Foundation

- **Merton lambda tables**: Pre-computed adjustment factors based on structural theory
- **Regime classification**: 5 regimes from IG narrow maturity to distressed
- **Theory-guided testing**: Statistical tests motivated by theoretical predictions

### Robust Implementation

- **Clustered standard errors**: Account for cross-bond correlation within weeks
- **Minimum sample requirements**: Filter buckets with <50 observations
- **Outlier identification**: Flag buckets where β/λ ratio is extreme

### Extensibility

- **Modular design**: Clean separation of data, models, analysis, visualization
- **Database-agnostic**: Easy to adapt to any SQL database
- **Mock data generator**: Realistic synthetic data for testing and development
- **Type hints**: Full type annotations for better IDE support

## Dependencies

Core packages:
- `numpy`, `pandas`: Data manipulation
- `scipy`: Statistical functions
- `statsmodels`: Regression with clustered standard errors
- `matplotlib`, `seaborn`: Visualization

See `requirements.txt` for complete list with versions.

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src/dts_research tests/
```

## Future Stages

The research program includes additional stages (to be implemented):

- **Stage A**: Establish cross-sectional variation using issuer-week fixed effects
- **Stage B**: Test whether Merton explains the variation
- **Stage C**: Test stability over time
- **Stage D**: Diagnose where theory fails
- **Stage E**: Production specification selection

## Citation

If you use this code for research, please cite the accompanying paper:

```
[Paper citation to be added]
```

## License

[License to be specified]

## Contact

[Contact information to be added]

## Acknowledgments

This implementation is based on the theoretical framework developed in Wuebben (2025) and follows the empirical methodology outlined in the accompanying research paper.
