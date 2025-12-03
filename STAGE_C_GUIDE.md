# Stage C: Time-Variation Analysis - Implementation Guide

## Overview

**Critical Question**: Is the relationship between lambda and (spread, maturity) stable over time, or do macro variables induce time-variation?

**Key Principle**: Don't add time-variation until you've proven the simple static model fails.

**Decision Tree**:
- If Chow test p > 0.10 → Static lambda sufficient (PATH 1)
- If Chow test p < 0.10 AND macro R² > 0.05 → Time-varying lambda needed (PATH 2)
- If Chow test p < 0.10 BUT macro R² < 0.05 → Regime switches exist but no clear driver (PATH 3)

---

## Theoretical Foundation

### The Question

Stage B found that Merton explains cross-sectional variation. But does this relationship **change over time**?

Two possibilities:
1. **Static**: λ(s, T) is constant → Simple lookup table
2. **Time-varying**: λ(s, T, VIX, OAS_spread, rates, ...) → Need macro state variables

### Why This Matters

**Production implications**:
- Static → Simple, robust, easy to implement
- Time-varying → More complex, need real-time macro data, risk of overfitting

**Don't add complexity without evidence!**

---

## Stage C Specifications

### C.1: Rolling Window Stability Test

**Goal**: Test if β_Merton is stable over time

**Method**: Rolling 1-year windows
```
For each window w:
    y_i,t = α_w + β_Merton,w · [λ^Merton · f_DTS] + ε

Then Chow test: H0: β_1 = β_2 = ... = β_W
```

**Interpretation**:
- p > 0.10 → Static model sufficient
- p < 0.10 → Time-variation detected

---

### C.2: Macro Driver Analysis

**Goal**: If time-variation exists, can we explain it with macro variables?

**Macro state variables**:
1. **VIX** - Market volatility/fear
2. **Credit spreads** (IG OAS, HY OAS) - Credit conditions
3. **Term spread** - Slope of yield curve
4. **Fed funds rate** - Monetary policy stance

**Method**:
```
λ_i,t = f(s_i,t, T_i,t, VIX_t, OAS_t, TermSpread_t, ...)
```

Fit flexible functional form and test if macro R² improvement is substantial.

**Interpretation**:
- High R² (> 5%) → Macro variables useful
- Low R² (< 5%) → Regime switches but no clear driver

---

### C.3: Maturity-Specific Stability

**Goal**: Does time-variation differ by maturity bucket?

**Method**: Run C.1 separately for each maturity bucket

**Why**: Short-term bonds might be more stable than long-term bonds

---

## Implementation Steps

### Step 1: Prepare Data

```python
from dts_research.data.loader import BondDataLoader
from dts_research.analysis.stageC import StageCAnalysis

# Load bond data and macro data
loader = BondDataLoader()
bond_data = loader.load_bond_data(start_date, end_date)
macro_data = loader.load_macro_data(start_date, end_date)

# Prepare regression data (from Stage 0)
stage0 = Stage0Analysis()
regression_data = stage0.prepare_regression_data(bond_data, index_data)

# Add VIX and spread_regime
regression_data = regression_data.merge(macro_data[['date', 'vix']], on='date', how='left')
regression_data['vix'] = regression_data['vix'].fillna(15)
regression_data['spread_regime'] = regression_data['oas'].apply(lambda x: 'IG' if x < 300 else 'HY')
```

### Step 2: Run Rolling Window Stability Test

```python
stageC = StageCAnalysis()

stability_results = stageC.rolling_window_stability_test(
    regression_data,
    window_years=1,      # 1-year rolling windows
    by_regime=True,      # Separate IG/HY
    by_maturity=True     # Separate maturity buckets
)

# Examine results
print(stability_results['chow_test_combined'])
# {'statistic': 2.45, 'p_value': 0.032, 'interpretation': 'Time-variation detected'}
```

### Step 3: Macro Driver Analysis (if unstable)

```python
# Only run if Chow test p < 0.10
if stability_results['chow_test_combined']['p_value'] < 0.10:
    macro_results = stageC.macro_driver_analysis(
        regression_data,
        macro_vars=['vix', 'oas_index', 'term_spread']
    )

    print(macro_results['r_squared_improvement'])
    # 0.087 → 8.7% improvement, worth including
```

### Step 4: Generate Decision

```python
decision = stageC.generate_stage_c_decision(
    stability_results,
    macro_results  # Can be None if stability test passed
)

print(decision)
# "PATH 1: STATIC LAMBDA SUFFICIENT..."
```

---

## Outputs

### Tables

1. **Table C.1**: Rolling window stability test results
   - Window-by-window β_Merton estimates
   - Chow test statistics and p-values
   - By regime and maturity

2. **Table C.2**: Macro driver analysis (if applicable)
   - Coefficients on macro variables
   - R² improvement
   - Statistical significance

### Figures

1. **Figure C.1**: Beta time series
   - Rolling window β_Merton over time
   - 95% confidence bands
   - Highlight crisis periods

2. **Figure C.2**: Beta vs macro variables
   - Scatter plots showing β_Merton ~ VIX, OAS, etc.
   - Identify relationships

3. **Figure C.3**: Implied lambda over time
   - How λ_Merton changes with macro conditions
   - Compare static vs time-varying

4. **Figure C.4**: Crisis vs normal periods
   - Compare β_Merton in different regimes
   - Test structural breaks

### Written Summary

3-4 page summary covering:
- Stability test results
- Macro driver findings (if applicable)
- Decision and rationale
- Implementation recommendations

---

## Decision Paths

### PATH 1: Static Lambda Sufficient

**Condition**: Chow test p > 0.10

**Interpretation**: The relationship between lambda and (spread, maturity) is stable over time.

**Recommendation**:
- Use static λ^Merton from Stage B
- No need for time-varying adjustments
- Simple lookup table in production

**Production spec**:
```python
lambda_i = lambda_combined(s_i, T_i)  # Static function
```

---

### PATH 2: Time-Varying Lambda Needed

**Condition**: Chow test p < 0.10 AND macro R² improvement > 5%

**Interpretation**: Relationship changes over time AND we can explain it with macro variables.

**Recommendation**:
- Include macro state variables in production system
- Use fitted λ(s, T, VIX, OAS, ...)

**Production spec**:
```python
lambda_i,t = lambda_combined(s_i,t, T_i,t, VIX_t, OAS_t, ...)  # Time-varying
```

**Tradeoffs**:
- (+) More accurate during regime shifts
- (-) More complex, requires real-time macro data
- (-) Risk of overfitting

---

### PATH 3: Unexplained Time-Variation

**Condition**: Chow test p < 0.10 BUT macro R² improvement < 5%

**Interpretation**: Relationship changes but we can't explain it with standard macro variables.

**Options**:
1. **Use regime-specific lambdas**: Estimate separate λ for Normal/Stress/Crisis
2. **Use conservative static lambda**: Accept some model error
3. **Investigate further**: Look for other drivers (liquidity, sentiment, etc.)

**Recommendation**: Depends on risk tolerance and production constraints.

---

## Common Pitfalls

### 1. Too Short Windows

**Problem**: 1-month windows have too few observations

**Solution**: Use minimum 6-month windows (preferably 1 year)

### 2. Look-Ahead Bias

**Problem**: Using future macro data to predict current betas

**Solution**: Ensure macro variables are lagged or contemporaneous only

### 3. Overfitting Macro Models

**Problem**: Including 20 macro variables with no economic justification

**Solution**: Stick to 3-5 economically meaningful variables (VIX, OAS, term spread, rates)

### 4. Ignoring Regime Heterogeneity

**Problem**: Pooling IG and HY when stability differs

**Solution**: Always run `by_regime=True` to check

---

## Validation Checklist

- [ ] Rolling window has sufficient observations (> 500 per window)
- [ ] Chow test properly accounts for overlapping windows
- [ ] Macro variables are not forward-looking
- [ ] Results are robust to window size (6mo, 1yr, 2yr)
- [ ] Interpretation aligns with economic intuition
- [ ] Decision is conservative (prefer simpler model when uncertain)

---

## Example Workflow

```python
# 1. Load data
bond_data = loader.load_bond_data('2010-01-01', '2024-12-31')
macro_data = loader.load_macro_data('2010-01-01', '2024-12-31')

# 2. Prepare
regression_data = stage0.prepare_regression_data(bond_data, index_data)
regression_data = add_macro_vars(regression_data, macro_data)

# 3. Test stability
stability = stageC.rolling_window_stability_test(regression_data)

# 4. If unstable, test macro drivers
if stability['chow_test_combined']['p_value'] < 0.10:
    macro = stageC.macro_driver_analysis(regression_data)
else:
    macro = None

# 5. Generate decision
decision = stageC.generate_stage_c_decision(stability, macro)

# 6. Create outputs
visualizer = StageCVisualizer()
figures = visualizer.create_all_stageC_figures(stability, macro)

reporter = StageCReporter()
reporter.save_all_reports(stability, macro, decision)

print(decision)
```

---

## Next Steps

**If PATH 1 (Static)**:
- Proceed to Stage D (robustness tests)
- Use simple λ(s, T) in production

**If PATH 2 (Time-varying)**:
- Proceed to Stage D to test robustness of time-varying spec
- Build macro data pipeline for production

**If PATH 3 (Unexplained)**:
- Consider regime-specific models in Stage E
- May need additional data or different approach

---

## References

- Chow, G. C. (1960). "Tests of Equality Between Sets of Coefficients in Two Linear Regressions"
- Andrews, D. W. K. (1993). "Tests for Parameter Instability and Structural Change"
- Stock, J. H., & Watson, M. W. (2002). "Forecasting Using Principal Components from a Large Number of Predictors"

---

## Quick Reference

**Main script**: `run_stageC.py`

**Key files**:
- `src/dts_research/analysis/stageC.py` - Analysis logic
- `src/dts_research/visualization/stageC_plots.py` - Visualization
- `src/dts_research/utils/reportingC.py` - Tables and summaries

**Runtime**: ~25-30 seconds with mock data, ~5-10 minutes with real data

**Prerequisites**: Stage B results (need β_Merton baseline)

**Critical outputs**:
- Chow test p-value (stability decision)
- Macro R² improvement (if time-varying)
- Decision recommendation

**Decision threshold**: p = 0.10 for stability test, R² > 5% for macro usefulness
