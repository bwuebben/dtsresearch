# Stage E: Production Specification Selection - Implementation Guide

## Overview

**Ultimate Goal**: Select the SINGLE production specification for DTS calculation

**Input**: Results from Stages A-D
**Output**: Implementation blueprint with concrete code and decision rationale

**Key Principle**: Parsimony wins. Choose the simplest specification that passes validation.

---

## Theoretical Foundation

### The Selection Problem

By Stage E, you have multiple candidate specifications:
1. **Pure Merton**: λ(s, T) from theory, β = 1 forced
2. **Calibrated Merton**: λ(s, T) from theory, β estimated
3. **Unrestricted**: Fully flexible polynomials
4. **Time-varying**: Include macro state variables
5. **Adjusted**: Tail/liquidity/shock adjustments

**Which one do you deploy in production?**

### Selection Criteria

**Level 1**: Statistical fit
- Does it explain variation? (R²)
- Is it statistically significant?

**Level 2**: Economic interpretability
- Do coefficients make sense?
- Is it theoretically grounded?

**Level 3**: Out-of-sample performance
- Does it work on held-out data?
- Robustness across time periods?

**Level 4**: Operational feasibility
- Can you implement it in production?
- Data requirements reasonable?

**Level 5**: Parsimony
- Simplest model that passes Levels 1-4

---

## Stage E Specifications

### E.1: Specification Comparison Table

**Goal**: Compare all candidates side-by-side

**Metrics**:
- **In-sample R²**: Fit on training data
- **Out-of-sample R²**: Fit on validation data (3yr train, 1yr test)
- **Parameters**: Number of coefficients
- **Data requirements**: What inputs needed?
- **Complexity score**: Operational difficulty (1-10)

**Method**:
```python
for spec in [pure_merton, calibrated_merton, unrestricted, time_varying, ...]:
    r2_in = fit_on_train(spec)
    r2_out = predict_on_test(spec)
    params = count_parameters(spec)
    complexity = assess_complexity(spec)
```

---

### E.2: Out-of-Sample Validation

**Goal**: Test if specifications work on unseen data

**Method**: Rolling window cross-validation
```
For each 4-year window:
    Train on years 1-3
    Test on year 4
    Measure: R², RMSE, MAE

Average across windows
```

**Key test**: Does out-of-sample performance degrade severely?
- < 10% drop → Robust
- 10-30% drop → Some overfitting, monitor
- > 30% drop → Severe overfitting, reject

---

### E.3: Regime Robustness

**Goal**: Test if specification works across market regimes

**Regimes**:
1. **Normal**: VIX < 20, spreads < 300 bps
2. **Stress**: VIX 20-40, spreads 300-600 bps
3. **Crisis**: VIX > 40, spreads > 600 bps

**Method**:
```python
for regime in [normal, stress, crisis]:
    data_regime = filter_data(regime)
    r2 = fit_and_evaluate(spec, data_regime)
```

**Interpretation**:
- Similar R² across regimes → Robust
- Breakdown in crisis → Need crisis-specific model or adjustments

---

### E.4: Sensitivity Analysis

**Goal**: Test sensitivity to key parameters

**Tests**:
1. **Maturity sensitivity**: How does fit vary by maturity bucket?
2. **Rating sensitivity**: How does fit vary by rating?
3. **Time period**: Pre-crisis vs post-crisis?
4. **Data vintage**: Does it work with stale data (1-week lag)?

---

### E.5: Implementation Blueprint

**Goal**: Produce production-ready specification

**Output**:
1. **Mathematical formula**: Exact equation with coefficients
2. **Pseudocode**: Algorithm for calculation
3. **Data requirements**: List of inputs and frequencies
4. **Validation rules**: Bounds, sanity checks, error handling
5. **Monitoring**: KPIs to track in production

---

## Implementation Steps

### Step 1: Define Candidate Specifications

```python
from dts_research.analysis.stageE import StageEAnalysis

stageE = StageEAnalysis()

# Define candidates based on Stages A-D results
candidates = {
    'pure_merton': {
        'formula': 'lambda_combined(s, T)',
        'parameters': 0,  # Pure theory, no estimation
        'data_requirements': ['oas', 'time_to_maturity']
    },
    'calibrated_merton': {
        'formula': 'beta * lambda_combined(s, T)',
        'parameters': 1,  # Estimate beta
        'data_requirements': ['oas', 'time_to_maturity', 'oas_index_pct_change']
    },
    'unrestricted': {
        'formula': 'poly(s, T, s*T, s^2, T^2, ...)',
        'parameters': 12,  # Many coefficients
        'data_requirements': ['oas', 'time_to_maturity', 'oas_index_pct_change']
    },
    # Add time-varying if Stage C indicated
    # Add adjustments if Stage D indicated
}
```

### Step 2: Run Specification Comparison

```python
comparison = stageE.compare_specifications(
    regression_data,
    candidates,
    metrics=['r2', 'aic', 'bic', 'params', 'complexity']
)

print(comparison)
# Shows side-by-side comparison of all specs
```

### Step 3: Out-of-Sample Validation

```python
oos_results = stageE.out_of_sample_validation(
    regression_data,
    candidates,
    train_years=3,
    test_years=1,
    rolling=True
)

# Check degradation
for spec_name, results in oos_results.items():
    degradation = (results['r2_in'] - results['r2_out']) / results['r2_in']
    print(f"{spec_name}: {degradation:.1%} degradation")
```

### Step 4: Regime Robustness

```python
regime_results = stageE.regime_robustness_test(
    regression_data,
    candidates,
    regimes=['normal', 'stress', 'crisis']
)

# Check consistency
for spec_name, results in regime_results.items():
    r2_range = max(results.values()) - min(results.values())
    print(f"{spec_name}: R² range = {r2_range:.3f}")
```

### Step 5: Select Winner

```python
selection = stageE.select_production_specification(
    comparison,
    oos_results,
    regime_results,
    parsimony_weight=0.3  # Penalize complexity
)

print(selection['winner'])
# "calibrated_merton"

print(selection['rationale'])
# "Balances fit and simplicity. Passes OOS validation..."
```

### Step 6: Generate Implementation Blueprint

```python
blueprint = stageE.create_implementation_blueprint(
    selection['winner'],
    regression_data
)

# Produces 5-7 page document with:
# - Mathematical specification
# - Python/SQL code
# - Data requirements
# - Validation rules
# - Monitoring plan
```

---

## Outputs

### Tables

1. **Table E.1**: Specification comparison matrix
   - All candidates with R², params, complexity

2. **Table E.2**: Out-of-sample performance
   - Rolling window R² for each specification

3. **Table E.3**: Regime robustness
   - R² by regime (normal/stress/crisis)

4. **Table E.4**: Sensitivity analysis
   - By maturity, rating, time period

5. **Table E.5**: Final specification details
   - Coefficients, standard errors, significance

### Figures

1. **Figure E.1**: Out-of-sample R² comparison
   - Bar chart: in-sample vs out-of-sample for each spec

2. **Figure E.2**: Prediction error distribution
   - Histogram of residuals for selected spec

3. **Figure E.3**: Predicted vs actual (out-of-sample)
   - Scatter plot with 45-degree line

4. **Figure E.4**: Specification comparison dashboard
   - Multi-panel summary (R², complexity, OOS, regime)

### Implementation Blueprint

**5-7 page document** covering:

**Section 1: Selected Specification**
- Mathematical formula
- Estimated coefficients
- Rationale for selection

**Section 2: Implementation Code**
- Python example
- SQL example (if database-driven)
- Error handling

**Section 3: Data Requirements**
- Input variables and frequencies
- Data quality checks
- Fallback values

**Section 4: Validation Rules**
- Bounds on lambda (e.g., 0 < λ < 10)
- Sanity checks (e.g., λ increases with spread)
- Outlier handling

**Section 5: Monitoring Plan**
- KPIs to track (R², prediction error, coverage)
- Alert thresholds
- Recalibration schedule

**Section 6: Rollback Plan**
- Conditions for reverting to previous spec
- Fallback specification

**Section 7: Testing Protocol**
- Unit tests
- Integration tests
- Shadow running period

---

## Decision Framework

### Hierarchical Testing

**Level 1: Statistical Fit** (Must pass)
- R² > 0.05 (explains at least 5% of variation)
- β statistically significant (p < 0.05)

**Level 2: Economic Interpretability** (Must pass)
- Coefficients have correct signs
- Magnitudes economically reasonable
- Theoretically grounded

**Level 3: Out-of-Sample** (Must pass)
- R²_out > 0.03 (at least 3% on test data)
- Degradation < 30% from in-sample

**Level 4: Operational Feasibility** (Must pass)
- Data available in production
- Can be calculated in real-time (< 100ms)
- Complexity manageable

**Level 5: Parsimony** (Tiebreaker)
- Among specs that pass Levels 1-4, choose simplest
- Prefer theory-based over black-box
- Prefer fewer parameters

---

## Common Specifications and Selection Logic

### Scenario 1: Merton Validated (Stages A-C)

**Stage results**:
- Stage A: F-test p < 0.001 (variation exists)
- Stage B: β_Merton ≈ 1, p(H0: β=1) = 0.43 (theory works)
- Stage C: Chow p = 0.21 (stable over time)
- Stage D: No tail/liquidity adjustments needed

**Winner**: **Pure Merton** (no calibration needed)
```python
lambda_i = lambda_combined(oas_i, time_to_maturity_i)
```

**Rationale**:
- Theory perfectly calibrated (β ≈ 1)
- Stable over time
- Simplest possible specification
- No estimation risk

---

### Scenario 2: Merton with Calibration

**Stage results**:
- Stage A: F-test p < 0.001
- Stage B: β_Merton = 0.78, p(H0: β=1) = 0.002 (theory off by 22%)
- Stage C: Chow p = 0.18 (stable)
- Stage D: No adjustments needed

**Winner**: **Calibrated Merton**
```python
lambda_i = 0.78 * lambda_combined(oas_i, time_to_maturity_i)
```

**Rationale**:
- Theory structure correct but scale wrong
- Single calibration parameter (β = 0.78)
- Still parsimonious (1 parameter vs 12+ for unrestricted)

---

### Scenario 3: Time-Varying Needed

**Stage results**:
- Stage A: F-test p < 0.001
- Stage B: β_Merton = 0.82
- Stage C: Chow p = 0.03 (unstable), Macro R² = 9% (VIX explains variation)
- Stage D: No adjustments

**Winner**: **Time-Varying Calibrated Merton**
```python
beta_t = 0.65 + 0.008 * VIX_t
lambda_i_t = beta_t * lambda_combined(oas_i, time_to_maturity_i)
```

**Rationale**:
- Time-variation detected and explained
- VIX available in real-time
- Modest complexity increase (2 params instead of 1)

---

### Scenario 4: Merton Fails

**Stage results**:
- Stage A: F-test p < 0.001
- Stage B: β_Merton = 0.23, R² = 0.008 (theory explains < 1%)
- Stage C: Unstable
- Stage D: Liquidity R² improvement = 15%

**Winner**: **Unrestricted** or **Liquidity-Adjusted**
```python
# Option 1: Unrestricted
lambda_i = f_flexible(oas_i, maturity_i)

# Option 2: Liquidity-adjusted
oas_default_i = oas_i - liquidity_component_i
lambda_i = lambda_combined(oas_default_i, maturity_i)
```

**Rationale**:
- Theory fundamentally fails
- Need data-driven approach
- Liquidity adjustment may salvage theory

---

## Validation Checklist

- [ ] All Level 1-4 criteria passed
- [ ] Out-of-sample R² > 0.03
- [ ] OOS degradation < 30%
- [ ] R² consistent across regimes (range < 0.05)
- [ ] Coefficients economically interpretable
- [ ] Data available in production
- [ ] Calculation time < 100ms
- [ ] Implementation blueprint complete
- [ ] Monitoring plan defined
- [ ] Rollback plan documented

---

## Example Workflow

```python
# 1. Load all prior stage results
stage_a_results = load_stage_a_results()
stage_b_results = load_stage_b_results()
stage_c_results = load_stage_c_results()
stage_d_results = load_stage_d_results()

# 2. Define candidates based on prior findings
candidates = stageE.define_candidates(
    stage_a_results,
    stage_b_results,
    stage_c_results,
    stage_d_results
)

# 3. Compare specifications
comparison = stageE.compare_specifications(regression_data, candidates)

# 4. Out-of-sample validation
oos = stageE.out_of_sample_validation(regression_data, candidates)

# 5. Regime robustness
regime = stageE.regime_robustness_test(regression_data, candidates)

# 6. Sensitivity analysis
sensitivity = stageE.sensitivity_analysis(regression_data, candidates)

# 7. Select winner
selection = stageE.select_production_specification(
    comparison, oos, regime, sensitivity
)

# 8. Generate blueprint
blueprint = stageE.create_implementation_blueprint(
    selection['winner'],
    regression_data,
    stage_a_results,
    stage_b_results,
    stage_c_results,
    stage_d_results
)

# 9. Save outputs
visualizer = StageEVisualizer()
figures = visualizer.create_all_stageE_figures(
    comparison, oos, regime, selection
)

reporter = StageEReporter()
reporter.save_all_reports(
    comparison, oos, regime, sensitivity, selection, blueprint
)

print("=" * 80)
print("SELECTED SPECIFICATION:", selection['winner'])
print("=" * 80)
print(blueprint)
```

---

## Common Pitfalls

### 1. Overfitting to In-Sample

**Problem**: Choosing spec with highest in-sample R² without OOS validation

**Solution**: Always use Level 3 (OOS validation) as gatekeeper

### 2. Complexity Creep

**Problem**: Adding features because they incrementally improve R² by 0.1%

**Solution**: Use parsimony (Level 5) to penalize complexity. 1% R² gain not worth 10 additional parameters.

### 3. Ignoring Operational Constraints

**Problem**: Selecting spec that requires data not available in production

**Solution**: Level 4 (operational feasibility) must pass. If data not available, spec disqualified.

### 4. Regime Overfitting

**Problem**: Spec works great in normal times, fails in crisis

**Solution**: Test regime robustness (E.3). If breakdown in crisis, need crisis-specific adjustments.

### 5. Forgetting the Goal

**Problem**: Getting lost in statistical optimization, forgetting production use case

**Solution**: Remember - this is for DTS calculation in production. Simplicity, robustness, interpretability matter more than 0.5% R² gain.

---

## Quick Reference

**Main script**: `run_stageE.py`

**Key files**:
- `src/dts_research/analysis/stageE.py` - Selection logic
- `src/dts_research/visualization/stageE_plots.py` - Visualization
- `src/dts_research/utils/reportingE.py` - Blueprint and reports

**Runtime**: ~45-60 seconds with mock data

**Prerequisites**: Results from Stages A, B, C, D

**Critical output**: **Implementation blueprint** (5-7 pages)

**Decision criteria**:
- Level 1: R² > 0.05
- Level 2: Coefficients make sense
- Level 3: OOS degradation < 30%
- Level 4: Operationally feasible
- Level 5: Simplest among passers

---

## References

- Akaike, H. (1974). "A New Look at the Statistical Model Identification" - AIC
- Schwarz, G. (1978). "Estimating the Dimension of a Model" - BIC
- Stone, M. (1974). "Cross-Validatory Choice and Assessment of Statistical Predictions"
- Hansen, P. R., & Lunde, A. (2005). "A Forecast Comparison of Volatility Models"

---

## Summary

Stage E is the **final decision point** - select ONE specification for production.

**Hierarchical framework** ensures rigorous vetting while maintaining parsimony bias.

**Implementation blueprint** provides everything needed for deployment.

**Conservative approach**: When in doubt, simpler is better.
