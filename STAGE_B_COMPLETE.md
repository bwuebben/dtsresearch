# Stage B Implementation Complete

## Summary

Stage B implementation is complete with **~1,818 lines** of production-ready Python code implementing three specifications to test whether Merton's structural model explains cross-sectional variation in bond spread sensitivities.

## What Was Built

### Core Analysis Module: `stageB.py` (597 lines)

The `StageBAnalysis` class implements:

1. **Specification B.1: Merton as Offset (Constrained)**
   - Single parameter test: β_Merton = 1
   - Pooled regression: `y_i,t = α + β_Merton · [λ^Merton_i,t · f_DTS,t] + ε`
   - Wald test for H₀: β = 1
   - Clustered SEs by week × issuer
   - By-regime analysis (Combined, IG, HY)

2. **Specification B.2: Decomposed Components**
   - Separate maturity (β_T) and credit quality (β_s) effects
   - Regression: `y_i,t = α + β_T·[λ_T · f_DTS] + β_s·[λ_s · f_DTS] + ε`
   - Joint test: H₀: (β_T, β_s) = (1, 1)
   - Identifies which component drives fit
   - By-regime analysis

3. **Specification B.3: Unrestricted (Comparison)**
   - Fully flexible functional form
   - Two-stage approach:
     - Stage 1: Estimate λ̂_i,t from characteristics
     - Stage 2: Regression on λ̂_i,t · f_DTS
   - Polynomials and interactions in maturity and spread
   - Comparison baseline to assess if theory misses patterns

4. **Model Comparison Framework**
   - Compares all specifications plus Stage A upper bound
   - Metrics: R², RMSE, AIC, parameter count
   - ΔR² calculation relative to Stage A
   - R² per parameter efficiency

5. **Theory vs Reality Analysis**
   - Bucket-by-bucket comparison of empirical β vs theoretical λ
   - Ratio calculation (β/λ) with target = 1.0
   - Deviation calculation (β - λ) with target = 0
   - Outlier identification (ratio < 0.8 or > 1.2)
   - Statistical assessment of theory performance

6. **Decision Framework**
   - Four decision paths:
     - PATH 1: Theory works well (β ∈ [0.9, 1.1])
     - PATH 2: Theory needs calibration (β ∈ [0.8, 1.2])
     - PATH 3: Theory captures structure but incomplete
     - PATH 4: Theory fundamentally fails
   - Automated recommendation generation
   - Clear next steps for each path

### Visualization Module: `stageB_plots.py` (374 lines)

The `StageBVisualizer` class creates:

1. **Figure B.1: Empirical vs Theoretical Scatter**
   - Empirical β (y-axis) vs Theoretical λ (x-axis)
   - 45° line showing perfect agreement
   - Color-coded by regime (IG/HY/Distressed)
   - Different markers for each regime
   - Outlier annotations
   - Point size proportional to sample size

2. **Figure B.2: Residual Analysis (3 Panels)**
   - Panel A: Residuals by maturity (1y, 3y, 5y, 7y, 10y)
   - Panel B: Residuals by spread level (<100, 100-200, etc.)
   - Panel C: Residuals by sector
   - Boxplots showing systematic patterns
   - Zero line indicating perfect prediction
   - Identifies where theory deviates most

3. **Figure B.3: Lambda Surface Comparison**
   - Side-by-side surfaces: Merton vs Unrestricted
   - Contour plot version (default)
   - 3D surface plot version (alternative)
   - Grid: maturity (1-10y) × spread (50-1000 bps)
   - Shows where surfaces diverge most
   - Reveals if unrestricted adds complexity or smoothness

### Reporting Module: `reportingB.py` (560 lines)

The `StageBReporter` class generates:

1. **Table B.1: Constrained Specifications**
   - Results for Spec B.1 and B.2
   - Separate rows for Combined, IG, HY
   - Columns: β estimates, SEs, t-stats, test statistics, R², RMSE
   - H₀: β=1 test results with p-values
   - Interpretation guide

2. **Table B.2: Model Comparison**
   - Rows: Stage A (buckets), B.1, B.2, B.3
   - Columns: R², RMSE, AIC, parameters, ΔR²
   - R² per parameter efficiency
   - Performance ranking

3. **Table B.3: Theory vs Reality (Summary)**
   - Top 20 buckets by sample size
   - Columns: bucket, β, λ, ratio, deviation, % deviation, outlier flag
   - Quick assessment of theory performance
   - Highlights problematic buckets

4. **Full Theory vs Reality CSV**
   - Complete bucket-by-bucket comparison
   - All rating × maturity × sector combinations
   - Sample sizes, median characteristics
   - For deep-dive analysis

5. **Written Summary (3-4 pages)**
   - Executive summary
   - Does Merton explain variation?
   - Specification B.1 interpretation
   - Specification B.2 component analysis
   - Theory vs reality assessment
   - Where theory succeeds
   - Where theory fails
   - Is unrestricted necessary?
   - Practical recommendations
   - Implications for Stage C
   - Decision path recommendation

### Orchestration Script: `run_stageB.py` (287 lines)

Complete 9-step pipeline:

1. Load data and run prerequisites (Stage 0 + Stage A)
2. Run Specification B.1 (Merton constrained)
3. Run Specification B.2 (Decomposed components)
4. Run Specification B.3 (Unrestricted)
5. Compare all models
6. Create theory vs reality comparison
7. Generate decision recommendation
8. Generate visualizations (4 figures)
9. Generate reports (5 outputs)

Progress reporting at each step with key statistics.

## Statistics

- **Total Stage B code**: ~1,818 lines
  - Analysis: 597 lines
  - Visualization: 374 lines
  - Reporting: 560 lines
  - Orchestration: 287 lines

- **Outputs generated**:
  - 4 figures (scatter, residuals, surfaces contour, surfaces 3D)
  - 3 tables (B.1, B.2, B.3)
  - 1 full CSV (complete theory vs reality)
  - 1 written summary (3-4 pages)

- **Three specifications** implemented with full statistical testing
- **Four decision paths** with automated recommendations
- **By-regime analysis** for all specifications

## Key Features

### Theoretical Foundation

1. **Merton Lambda Integration**
   - Uses lambda_combined() for joint maturity × credit effect
   - Separate lambda_T and lambda_s for decomposition
   - Bilinear interpolation for continuous values

2. **Statistical Rigor**
   - Clustered standard errors (week × issuer)
   - Wald tests for parameter constraints
   - Joint hypothesis tests for B.2
   - Model comparison with information criteria

3. **Comprehensive Comparison**
   - Direct theory vs reality at bucket level
   - Ratio and deviation metrics
   - Systematic bias detection
   - Outlier identification

### Decision Framework

**PATH 1: Theory Works Well**
- β_Merton ∈ [0.9, 1.1]
- p-value (H₀: β=1) > 0.05
- R² ratio (Merton/Buckets) > 85%
- **Recommendation**: Use pure Merton tables, proceed to Stage C

**PATH 2: Theory Needs Calibration**
- β_Merton ∈ [0.8, 1.2] but outside [0.9, 1.1]
- Patterns match (high R²)
- R² ratio > 80%
- **Recommendation**: Use calibrated Merton (β_Merton × λ^Merton), test stability in Stage C

**PATH 3: Theory Captures Structure But Incomplete**
- R² ratio between 60-85%
- Some systematic residuals
- β_Merton reasonable but incomplete
- **Recommendation**: Stage C with BOTH theory-guided and unrestricted tracks

**PATH 4: Theory Fundamentally Fails**
- R² ratio < 50%
- Wrong patterns (e.g., long bonds more sensitive than short)
- β_Merton far from 1
- **Recommendation**: SKIP Stage C, proceed to Stage D (diagnostics), then Stage E unrestricted-only

### Diagnostic Capabilities

1. **Component Diagnosis** (Spec B.2)
   - Which component works? (maturity vs credit)
   - β_T ≈ 1 and β_s ≈ 1: both work
   - β_T ≈ 1, β_s ≠ 1: credit quality needs recalibration
   - β_T ≠ 1, β_s ≈ 1: maturity functional form wrong

2. **Residual Analysis** (Figure B.2)
   - By maturity: Do residuals increase with maturity?
   - By spread: Do residuals vary by credit quality?
   - By sector: Are there sector-specific deviations?

3. **Surface Comparison** (Figure B.3)
   - Where do Merton and unrestricted diverge?
   - Is unrestricted smoother or more complex?
   - Does unrestricted reveal new patterns?

## Integration with Previous Stages

### Requires Stage A Results
- Empirical bucket betas (β^(k)) for comparison
- F-test results to confirm variation exists
- Sample sizes and characteristics by bucket

### Uses Stage 0 Infrastructure
- Bucket classification system
- Merton lambda calculator
- Regression data preparation
- Clustered standard errors

### Provides for Stage C
- β_Merton estimate for calibration
- Decision on which specification to use
- Baseline R² for time-variation tests
- Component performance (β_T, β_s)

## How to Use

### Quick Start with Mock Data

```bash
python run_stageB.py
```

Runtime: ~20 seconds with 500 bonds (2010-2024)

### Configuration

Edit `run_stageB.py` lines 54-56:

```python
start_date = '2010-01-01'
end_date = '2024-12-31'
use_mock_data = True  # False for real database
```

### Prerequisites

Stage B automatically runs Stage A if needed (adds ~15 seconds).

**Critical**: Stage A must find variation (F-test p < 0.10) for Stage B to be meaningful.

### Outputs

```
output/
├── figures/
│   ├── stageB_fig1_scatter.png
│   ├── stageB_fig2_residuals.png
│   ├── stageB_fig3_surfaces_contour.png
│   └── stageB_fig3_surfaces_3d.png
└── reports/
    ├── stageB_table_b1_specifications.csv
    ├── stageB_table_b2_model_comparison.csv
    ├── stageB_table_b3_theory_vs_reality.csv
    ├── stageB_theory_vs_reality_full.csv
    └── stageB_summary.txt
```

## What Gets Tested

### Specification B.1 Tests
- H₀: β_Merton = 1 (theory exactly correct)
- Wald test with clustered SEs
- By regime: Combined, IG, HY

### Specification B.2 Tests
- H₀: β_T = 1 (maturity component correct)
- H₀: β_s = 1 (credit component correct)
- Joint test: H₀: (β_T, β_s) = (1, 1)
- By regime analysis

### Theory vs Reality Assessment
- % buckets in acceptable range [0.8, 1.2]
- Median ratio (target = 1.0)
- Systematic bias detection (all ratios > 1 or < 1)
- Outlier identification

### Model Comparison
- R² relative to Stage A upper bound
- RMSE comparison
- AIC penalizing parameters
- R² per parameter efficiency

## Example Output Interpretation

### Example 1: Theory Works (PATH 1)

```
β_Merton = 0.95 (SE = 0.08)
Test H0: β=1, p-value = 0.5231
R² = 0.82 (Stage A R² = 0.85)

Theory vs Reality:
  95% of buckets in range [0.8, 1.2]
  Median ratio = 0.98
  Systematic bias: None

→ DECISION: PATH 1 - Theory works well
→ RECOMMENDATION: Use pure Merton tables, proceed to Stage C
```

### Example 2: Needs Calibration (PATH 2)

```
β_Merton = 1.18 (SE = 0.09)
Test H0: β=1, p-value = 0.0467
R² = 0.79 (Stage A R² = 0.85)

β_T = 1.03, β_s = 1.22 (joint p = 0.021)

Theory vs Reality:
  87% of buckets in range [0.8, 1.2]
  Median ratio = 1.15
  Systematic bias: Upward (all ratios > 1.05)

→ DECISION: PATH 2 - Theory needs calibration
→ RECOMMENDATION: Use calibrated Merton (1.18 × λ^Merton)
→ Credit quality component drives deviation
→ Proceed to Stage C to test stability of β_Merton
```

### Example 3: Partial Success (PATH 3)

```
β_Merton = 0.85 (SE = 0.11)
R² = 0.58 (Stage A R² = 0.85)
R² ratio = 68%

Unrestricted (B.3): R² = 0.72

Theory vs Reality:
  71% of buckets in range [0.8, 1.2]
  Median ratio = 0.91
  High dispersion (std = 0.29)

→ DECISION: PATH 3 - Theory captures structure but incomplete
→ RECOMMENDATION: Stage C with BOTH tracks
→ Theory useful as prior but doesn't capture all heterogeneity
```

### Example 4: Theory Fails (PATH 4)

```
β_Merton = 0.52 (SE = 0.15)
R² = 0.31 (Stage A R² = 0.85)
R² ratio = 36%

Theory vs Reality:
  48% of buckets in range [0.8, 1.2]
  Long bonds show higher β than short bonds
  (Opposite of theoretical prediction)

→ DECISION: PATH 4 - Theory fundamentally fails
→ RECOMMENDATION: SKIP Stage C
→ Proceed to Stage D to diagnose WHY theory fails
→ Then Stage E with unrestricted-only approach
```

## Technical Details

### Clustering Strategy

**Specifications B.1 and B.2**:
- Two-way clustering: week × issuer
- Accounts for cross-bond correlation within weeks
- Accounts for time-series dependence within issuer

**Specification B.3**:
- Same clustering approach
- Applied to both stages of two-stage estimation

### Wald Tests

**Single parameter** (B.1):
```
t = (β_Merton - 1) / SE(β_Merton)
p-value from t-distribution
```

**Joint test** (B.2):
```
Restriction matrix: R = [[1, 0], [0, 1]]
Hypothesis: R·β = [1, 1]
Wald statistic = (R·β̂ - r)' [R·V·R']^(-1) (R·β̂ - r)
p-value from χ² distribution (2 df)
```

### Theory vs Reality Metrics

- **Ratio**: β/λ (target = 1.0)
- **Deviation**: β - λ (target = 0)
- **% Deviation**: 100 × (β - λ) / λ
- **Acceptable range**: [0.8, 1.2] (±20%)
- **Outlier threshold**: Ratio < 0.8 or > 1.2

### Model Comparison Metrics

- **R²**: Proportion of variance explained
- **RMSE**: √(SSR/n) - root mean squared error
- **AIC**: -2·log(L) + 2·k - Akaike information criterion
- **ΔR²**: (R²_model - R²_stageA) / R²_stageA

## Files Modified/Created

### New Files
- `src/dts_research/analysis/stageB.py` (597 lines)
- `src/dts_research/visualization/stageB_plots.py` (374 lines)
- `src/dts_research/utils/reportingB.py` (560 lines)
- `run_stageB.py` (287 lines)
- `STAGE_B_GUIDE.md` (400+ lines of documentation)
- `STAGE_B_COMPLETE.md` (this file)

### Modified Files
- `README.md` (added Stage B section)

## Dependencies

No new dependencies beyond Stage 0 and Stage A:
- `numpy`, `pandas` for data manipulation
- `scipy` for statistical functions
- `statsmodels` for regression with clustered SEs
- `matplotlib`, `seaborn` for visualization

## Testing

Run with mock data to verify:

```bash
# Quick test (~20 seconds)
python run_stageB.py

# Check outputs
ls output/figures/stageB_*.png
ls output/reports/stageB_*.csv
cat output/reports/stageB_summary.txt
```

## Next Steps

Based on Stage B decision path:

1. **If PATH 1 or PATH 2**: Proceed to Stage C to test time-variation
2. **If PATH 3**: Prepare for dual-track Stage C (theory + unrestricted)
3. **If PATH 4**: Skip to Stage D for diagnostics

## Common Issues and Solutions

### Issue: Stage A runs automatically
**Solution**: This is expected - Stage B needs Stage A results as input

### Issue: β_Merton very far from 1
**Diagnosis**:
- Check Stage 0 results (is Merton baseline reasonable?)
- Verify bond characteristics (maturity, spread)
- Review lambda calculations

### Issue: Unrestricted (B.3) fails
**Solution**: B.3 is optional comparison baseline. B.1 and B.2 are sufficient for decision making.

### Issue: Low R² for all models
**Diagnosis**:
- Check Stage A F-test (should be significant)
- Review data quality and sample period
- Consider if relationship is inherently weak

## Documentation

- `STAGE_B_GUIDE.md`: Comprehensive usage guide
- `README.md`: Project overview with Stage B section
- This file (`STAGE_B_COMPLETE.md`): Implementation summary

## Summary

Stage B is fully implemented and tested. The code provides:

1. ✅ Three complete specifications (B.1, B.2, B.3)
2. ✅ Comprehensive statistical testing (Wald tests, joint tests)
3. ✅ Theory vs reality comparison framework
4. ✅ Four-path decision framework with clear recommendations
5. ✅ Publication-quality visualizations (4 figures)
6. ✅ Comprehensive reporting (3 tables + full CSV + written summary)
7. ✅ By-regime analysis (Combined, IG, HY)
8. ✅ Diagnostic capabilities (components, residuals, surfaces)
9. ✅ Integration with Stage A and Stage 0
10. ✅ Clear next steps for Stage C

The implementation follows the paper specifications exactly and provides all tools needed to assess whether Merton's structural model explains the cross-sectional variation in bond spread sensitivities.
