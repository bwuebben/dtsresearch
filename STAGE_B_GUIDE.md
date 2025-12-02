# Stage B: Does Merton Explain the Variation?

## Overview

Stage B is the **CORE EMPIRICAL TEST** of the research program. It tests whether Merton's structural model explains the cross-sectional variation documented in Stage A.

**Prerequisite**: Stage A must have found variation (F-test p < 0.10)

**Critical Question**: Does theory explain the variation?

## What Stage B Does

### Three Specifications

**Specification B.1: Merton as Offset (Constrained)**
```
y_i,t = α + β_Merton · [λ^Merton_i,t · f_DTS,t] + ε
```
- Single parameter: β_Merton
- Theory prediction: β_Merton = 1 if Merton is exactly correct
- Test: Wald test H₀: β = 1

**Specification B.2: Decomposed Components**
```
y_i,t = α + β_T·[λ_T · f_DTS] + β_s·[λ_s · f_DTS] + ε
```
- Separate maturity (β_T) and credit quality (β_s) effects
- Theory prediction: β_T ≈ 1 AND β_s ≈ 1
- Diagnostic: Which component works?

**Specification B.3: Unrestricted (Comparison)**
```
λ = β₀ + β_M·M + β_M²·M² + β_s·s + β_s²·s² + β_Ms·M·s + dummies
```
- Fully flexible functional form
- Comparison baseline
- Tests if theory misses important patterns

### Theory vs Reality Table

Direct comparison of:
- Empirical β^(k) from Stage A
- Theoretical λ^Merton from tables
- Ratio (β/λ) and deviations

## Quick Start

```bash
# Run Stage B analysis
python run_stageB.py
```

**Runtime**: ~20 seconds with mock data

## What Gets Generated

```
output/
├── figures/
│   ├── stageB_fig1_scatter.png           # Empirical vs theoretical
│   ├── stageB_fig2_residuals.png         # Residual analysis (3 panels)
│   ├── stageB_fig3_surfaces_contour.png  # Lambda surface comparison
│   └── stageB_fig3_surfaces_3d.png       # Lambda surface (3D)
└── reports/
    ├── stageB_table_b1_specifications.csv     # B.1 and B.2 results
    ├── stageB_table_b2_model_comparison.csv   # Model performance
    ├── stageB_table_b3_theory_vs_reality.csv  # Bucket comparison
    ├── stageB_theory_vs_reality_full.csv      # Complete data
    └── stageB_summary.txt                     # 3-4 page analysis
```

## Four Decision Paths

### ✅ PATH 1: Theory Works Well

**Conditions**:
- β_Merton ∈ [0.9, 1.1]
- p-value (H₀: β=1) > 0.05
- R² ratio (Merton/Buckets) > 85%

**Recommendation**:
- Use pure Merton tables (simplest)
- Proceed to Stage C to test time-variation
- High confidence in theoretical foundation
- Production systems can rely on Merton

### ⚠ PATH 2: Theory Needs Calibration

**Conditions**:
- β_Merton ∈ [0.8, 1.2] but outside [0.9, 1.1]
- Patterns match (high R²)
- R² ratio > 80%

**Recommendation**:
- Use calibrated Merton: λ^prod = β_Merton × λ^Merton
- Proceed to Stage C to test stability of β_Merton over time
- Theory has right structure, needs scaling
- Simple one-parameter calibration suffices

### ⚠ PATH 3: Theory Captures Structure But Misses Details

**Conditions**:
- R² ratio between 60-85%
- Some systematic residuals
- β_Merton reasonable but incomplete

**Recommendation**:
- Proceed to Stage C with BOTH tracks:
  1. Theory-guided (calibrated Merton)
  2. Unrestricted empirical
- Compare performance
- Theory useful as prior but incomplete

### ✗ PATH 4: Theory Fundamentally Fails

**Conditions**:
- R² ratio < 50%
- Wrong patterns (e.g., long bonds more sensitive than short)
- β_Merton far from 1

**Recommendation**:
- SKIP Stage C (no point testing time-variation of failed model)
- Proceed to Stage D (robustness) to diagnose WHY
- Then Stage E with unrestricted only
- Report: Structural models inadequate for this market

## Key Outputs

### Table B.1: Constrained Specifications

Shows results for B.1 and B.2:
- β_Merton and standard errors
- Test statistics (H₀: β=1)
- R², RMSE
- Separate for Combined, IG, HY

**Look for**:
- Is β_Merton close to 1?
- Do β_T and β_s both equal 1?
- Which regime performs best?

### Table B.2: Model Comparison

Compares all specifications:
- Stage A (buckets - upper bound)
- Spec B.1 (Merton constrained)
- Spec B.2 (Decomposed)
- Spec B.3 (Unrestricted)

**Metrics**: R², RMSE, AIC, parameters, ΔR²

**Look for**:
- How close is Merton R² to Stage A R²?
- Does unrestricted add much?
- Parameter efficiency (R² per parameter)

### Table B.3: Theory vs Reality

Bucket-by-bucket comparison:
- Empirical β from Stage A
- Theoretical λ from Merton
- Ratio (β/λ)
- Deviation and % deviation
- Outlier flag

**Look for**:
- % of buckets with ratio in [0.8, 1.2]
- Systematic patterns (all ratios > 1 or < 1)
- Which buckets are outliers?

### Figure B.1: Scatter Plot

Empirical β (y-axis) vs Theoretical λ (x-axis)

**Features**:
- 45° line = perfect agreement
- Color-coded by spread regime
- Point size = sample size
- Outlier annotations

**Interpretation**:
- Points on 45° line: theory works
- Systematic deviation: need calibration
- High dispersion: heterogeneity beyond theory

### Figure B.2: Residual Analysis

Three panels showing β - λ by:
- Panel A: Maturity (1y, 3y, 5y, 7y, 10y)
- Panel B: Spread level (<100, 100-200, etc.)
- Panel C: Sector

**Interpretation**:
- Zero line = perfect prediction
- Systematic patterns = model deficiency
- Which dimension has largest residuals?

### Figure B.3: Lambda Surface Comparison

Side-by-side surfaces:
- Merton prediction
- Unrestricted (Spec B.3)

**Interpretation**:
- Where do they differ most?
- Is unrestricted smoother or more complex?
- Does unrestricted reveal new patterns?

## Interpretation Guide

### Specification B.1 Results

**β_Merton = 0.95 (p=0.23 for H₀: β=1)**
→ Theory works! Use pure Merton

**β_Merton = 1.15 (p=0.03)**
→ Systematic upward bias, use calibrated Merton

**β_Merton = 0.72 (p<0.001)**
→ Large deviation, theory may be fundamentally off

### Specification B.2 Results

**β_T = 1.02, β_s = 0.98**
→ Both components work perfectly

**β_T = 0.95, β_s = 1.25**
→ Maturity works, credit quality needs recalibration

**β_T = 1.30, β_s = 1.05**
→ Credit works, maturity functional form wrong

### Theory vs Reality Assessment

**90%+ in range [0.8, 1.2]**
→ Excellent baseline

**Median ratio = 1.15, all ratios > 1.05**
→ Systematic upward bias (recalibrate)

**High dispersion (std = 0.35), median ≈ 1**
→ Theory captures mean, misses heterogeneity

**Long bonds have higher β than short**
→ Theory fundamentally wrong (wrong patterns)

## Configuration

Edit `run_stageB.py`:

```python
# Line 35-36: Data source
use_mock_data = True  # False for real database

# Line 38-39: Date range
start_date = '2010-01-01'
end_date = '2024-12-31'
```

## Expected Runtime

With mock data (500 bonds, 2010-2024):
- **Complete pipeline**: ~20 seconds
- Includes Stage A prerequisite: ~15 seconds
- Stage B specifications: ~5 seconds

With real data (5000 bonds):
- **Complete pipeline**: ~1-2 minutes

## Common Issues

### Stage A Not Run

**Symptom**: Script runs Stage A first

**Cause**: Stage B requires Stage A results

**Solution**: This is expected - Stage B automatically runs Stage A

### Low R² for All Models

**Symptom**: All specifications have R² < 0.10

**Cause**: Weak relationship, noisy data, or insufficient variation

**Solution**:
- Check Stage A F-test (should be significant)
- Review data quality
- Consider longer time period

### β_Merton Very Far from 1

**Symptom**: β_Merton < 0.5 or > 2.0

**Cause**: Theory-data mismatch, data issues, or wrong λ calculation

**Solution**:
- Check Stage 0 results (is Merton baseline reasonable?)
- Verify bond characteristics (maturity, spread)
- Review Merton lambda calculations

### Unrestricted (B.3) Fails

**Symptom**: Specification B.3 produces errors

**Cause**: Not enough variation in characteristics, multicollinearity

**Solution**:
- B.3 is optional comparison
- B.1 and B.2 are sufficient for decision
- Can proceed without B.3

## Next Steps

### If PATH 1 or PATH 2 (Theory Works)

1. **Review** `output/reports/stageB_summary.txt`
2. **Check** theory vs reality table
3. **Verify** interpretation makes sense
4. **Proceed** to Stage C: `python run_stageC.py` (coming soon)

### If PATH 3 (Partial)

1. **Identify** where theory works (which regimes, buckets)
2. **Examine** residual patterns
3. **Consider** regime-specific models
4. **Proceed** to Stage C with both tracks

### If PATH 4 (Failure)

1. **Diagnose** why theory fails (Figure B.2)
2. **Document** specific failure modes
3. **Skip** Stage C
4. **Prepare** for unrestricted-only approach

## Technical Details

### Clustering

- **B.1 and B.2**: Clustered by week × issuer
- Accounts for cross-bond correlation and time-series dependence

### Wald Tests

**Single parameter**: (β - 1) / SE(β)

**Joint test** (B.2): Wald test with restriction matrix

### Model Comparison

- **R²**: Proportion of variance explained
- **RMSE**: Root mean squared error of predictions
- **AIC**: Akaike information criterion (penalizes parameters)
- **ΔR²**: Relative to Stage A upper bound

### Theory vs Reality

- **Ratio**: β/λ (target = 1.0)
- **Deviation**: β - λ (target = 0)
- **Acceptable range**: [0.8, 1.2] (±20%)
- **Outlier threshold**: Ratio < 0.8 or > 1.2

## References

- Paper Section: Stage B (page 792+)
- Stage 0: Merton lambda tables
- Stage A: Empirical beta estimates
- Wuebben (2025): Theoretical foundation

## Questions?

- Check main `README.md` for project overview
- See `ARCHITECTURE.md` for code structure
- Review `run_stageB.py` for implementation
- Examine `src/dts_research/analysis/stageB.py` for methodology
