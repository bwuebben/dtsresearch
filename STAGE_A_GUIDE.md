# Stage A: Establish Cross-Sectional Variation

## Overview

Stage A answers the critical question: **Do DTS betas differ across bonds?**

This is a prerequisite before testing whether Merton explains the variation:
- If **NO variation** → Standard DTS is adequate, stop here
- If **variation exists** → Proceed to Stage B to test if Merton explains it

## What Stage A Does

### Specification A.1: Bucket-Level Betas

For each bucket k (rating × maturity × sector), estimate:
```
y_i,t = α^(k) + β^(k) · f_DTS,t + ε_i,t
```

Then test: **H₀: All β^(k) are equal**

### Specification A.2: Continuous Characteristics

Two-step procedure:

**Step 1:** Estimate bond-specific betas using 2-year rolling windows

**Step 2:** Cross-sectional regression:
```
β_hat_i,τ = γ₀ + γ_M·M + γ_s·s + γ_M²·M² + γ_Ms·M·s + u
```

## Quick Start

```bash
# Run Stage A analysis
python run_stageA.py
```

**Note**: Specification A.2 is time-intensive (rolling window estimation). Set `run_spec_a2 = False` in the script to skip it for faster testing.

## What Gets Generated

```
output/
├── figures/
│   ├── stageA_fig1_heatmap.png              # Beta heatmap (rating × maturity)
│   ├── stageA_fig2_surface_contour.png      # Beta surface (contour plot)
│   ├── stageA_fig2_surface_3d.png           # Beta surface (3D plot)
│   └── stageA_fig3_distributions.png        # Beta distribution diagnostics
└── reports/
    ├── stageA_table_a1_bucket_betas.csv     # Bucket-level estimates
    ├── stageA_table_a2_equality_tests.csv   # F-test results
    ├── stageA_table_a3_continuous_spec.csv  # Spec A.2 coefficients
    ├── stageA_summary.txt                   # 2-page analysis
    └── stageA_full_results.csv              # Complete output
```

## Key Outputs

### 1. F-Test for Overall Equality

**Critical test**: H₀: All betas are equal

- **p > 0.10**: No significant variation → **STOP, use standard DTS**
- **p < 0.10**: Variation exists → Proceed to Stage B

### 2. Economic Significance

- Beta range (min to max)
- Ratio: max/min
- IQR and standard deviation

### 3. Specification A.2 Results

- R² of cross-sectional fit
- Coefficients on maturity, spread, interactions
- Separate for IG and HY

## Decision Criteria

### ✅ Strong Variation → Proceed to Stage B

Conditions:
- F-test p < 0.01
- R² > 0.15
- Ratio (max/min) > 2.0

**Action**: High confidence that adjustments are needed. Proceed to Stage B to test if Merton explains patterns.

### ⚠ Marginal Variation → Proceed with Caution

Conditions:
- F-test 0.01 < p < 0.10
- Moderate economic significance

**Action**: Some evidence of variation. Stage B may find simple theory sufficient.

### ❌ No Variation → STOP

Conditions:
- F-test p > 0.10
- R² < 0.05
- Ratio < 1.5

**Action**: Standard DTS adequate. Report this as primary finding. No need for Stages B-E.

## Interpretation Guide

### Table A.1: Bucket-Level Betas

Shows β^(k) for each rating × maturity combination.

**Look for**:
- Large differences across maturities (short vs long bonds)
- Differences across ratings (AAA vs CCC)
- Pattern: Do short/low-spread bonds have higher betas?

### Table A.2: Equality Tests

F-tests for different dimensions.

**Key tests**:
- Overall: All buckets equal?
- Across maturities (holding rating fixed)
- Across ratings (holding maturity fixed)
- Across sectors

### Table A.3: Continuous Specification

Regression coefficients showing how beta varies with characteristics.

**Interpretation**:
- **γ_M < 0**: Shorter maturity → higher beta
- **γ_s < 0**: Lower spread → higher beta
- **R² > 0.20**: Characteristics explain substantial variation

### Figure A.1: Beta Heatmap

Visual representation of Table A.1.

**Green cells**: Beta > 1 (more sensitive than average)
**Red cells**: Beta < 1 (less sensitive than average)

**Look for gradients**:
- Across rows (maturity effect)
- Down columns (rating effect)

### Figure A.2: Beta Surface

3D/contour plot showing predicted beta as function of maturity and spread.

**Interpretation**:
- Peak location: Which bonds are most sensitive?
- Gradient direction: What drives variation more (maturity or spread)?
- IG vs HY: Different patterns?

## Configuration

Edit `run_stageA.py`:

```python
# Line 35-37: Data source
use_mock_data = True  # False for real database

# Line 38: Time-intensive step
run_spec_a2 = True    # False to skip rolling window estimation

# Line 41-42: Date range
start_date = '2010-01-01'
end_date = '2024-12-31'
```

### Skip Spec A.2 for Faster Testing

Specification A.2 (rolling window estimation) can take 10-30 minutes with real data.

For quick testing, set:
```python
run_spec_a2 = False
```

You still get:
- Spec A.1 results (bucket-level betas)
- All F-tests
- Economic significance analysis
- Decision recommendation

Missing:
- Continuous characteristic regression (Table A.3)
- Beta surface plots (Figure A.2)

## Expected Runtime

With mock data (500 bonds, 2010-2024):
- **Spec A.1 only**: ~15 seconds
- **With Spec A.2**: ~2-3 minutes (rolling windows)

With real data (5000 bonds):
- **Spec A.1 only**: ~30 seconds
- **With Spec A.2**: ~10-30 minutes

## Common Issues

### Low Statistical Power

**Symptom**: All F-tests have high p-values but beta range is large

**Cause**: Small sample sizes, high noise

**Solution**:
- Increase date range
- Lower `min_observations` in bucket classification
- Pool across sectors if needed

### Spec A.2 Fails

**Symptom**: "Insufficient data" error for Step 1

**Cause**: Not enough bonds with 104 consecutive weeks

**Solution**:
- Reduce `window_weeks` parameter (e.g., 52 = 1 year)
- Increase date range
- Check for gaps in data

### No HY Data

**Symptom**: HY panel blank in heatmap

**Cause**: Mock data or real data has few HY bonds

**Solution**: Generate more mock bonds or ensure database includes HY bonds

## Next Steps

### If Variation Found

1. **Review** `output/reports/stageA_summary.txt`
2. **Examine** heatmap to identify patterns
3. **Note** economic magnitude (max/min ratio)
4. **Proceed** to Stage B: `python run_stageB.py` (coming soon)

### If No Variation

1. **Document** finding: Standard DTS is adequate
2. **Report** F-test results (p > 0.10)
3. **Stop** analysis: No need for Stages B-E
4. **Production**: Use β = 1 for all bonds

## Technical Details

### F-Test Implementation

Uses Wald test with inverse variance weighting:
```
χ² = Σ w_k (β^(k) - β_bar)²

where w_k = 1/SE(β^(k))²
```

Converted to F-statistic: F = χ²/df

### Rolling Window Details

- Window size: 104 weeks (2 years)
- Stride: 1 week (overlapping windows)
- Minimum observations: 80% of window size
- Standard errors: Clustered by bond

### Clustering

- Spec A.1: Cluster by week (cross-bond correlation)
- Spec A.2: Cluster by bond (multiple windows per bond)

## References

- Paper Section: Stage A (page 655+)
- Theoretical foundation: Stage 0 establishes Merton predictions
- Next stage: Stage B tests if Merton explains variation

## Questions?

- Check main `README.md` for project overview
- See `ARCHITECTURE.md` for code structure
- Review `run_stageA.py` for implementation details
- Examine `src/dts_research/analysis/stageA.py` for methodology
