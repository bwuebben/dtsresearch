# Stage D: Robustness and Extensions - Implementation Guide

## Overview

**Objective**: Test robustness of Merton model across:
1. **Tail events** (quantile regression)
2. **Shock types** (systematic vs idiosyncratic)
3. **Spread components** (default vs liquidity)

**Key Framing**: These are SECONDARY tests
- If Stages A-C validated Merton → Confirm it's not just a mean effect
- If Stages A-C showed failure → Diagnose WHY (tails? liquidity? etc.)

**Not a standalone stage**: Results inform production spec decisions in Stage E

---

## Theoretical Foundation

### The Question

Stages A-C tested Merton on mean spread changes. But production systems care about:
- **Tails**: Does it work for large moves (VaR/stress testing)?
- **Shock heterogeneity**: Does it handle different types of shocks?
- **Liquidity**: Is the model capturing default risk or just liquidity?

### Why This Matters

**Production implications**:
- **Tail behavior**: Risk management needs to understand extreme events
- **Shock decomposition**: Different shocks may require different treatments
- **Liquidity adjustment**: May need to strip out non-default components

**Goal**: Understand where Merton works well and where it needs adjustments

---

## Stage D Specifications

### D.1: Tail Behavior (Quantile Regression)

**Goal**: Test if Merton holds across the entire distribution, not just the mean

**Method**: Quantile regression at τ = {0.05, 0.25, 0.50, 0.75, 0.95}
```
Q_τ(y_i,t) = α_τ + β_τ · [λ^Merton · f_DTS] + ε

Compare: β_0.05, β_0.50, β_0.95
```

**Interpretation**:
- β_τ constant across quantiles → Symmetric, standard linear model OK
- β_τ increasing → Tail amplification (need adjustments for VaR)
- β_τ decreasing → Tail dampening (model overestimates extreme risk)

**Tail amplification factor**:
```
Left tail: β_0.05 / β_0.50
Right tail: β_0.95 / β_0.50
```

---

### D.2: Shock Decomposition

**Goal**: Does Merton work equally well for different types of shocks?

**Decomposition**:
```
Spread change = Global shock + Sector shock + Issuer shock + Residual

Where:
- Global shock: Market-wide (VIX spike, Fed action)
- Sector shock: Industry-specific (tech selloff, energy boom)
- Issuer shock: Firm-specific (earnings miss, management change)
```

**Method**:
1. Decompose variance using hierarchical model
2. Estimate β separately for each shock type
3. Compare shock-specific elasticities

**Interpretation**:
- β^(Global) > β^(Issuer) → Merton better for systematic risk
- β^(Issuer) > β^(Global) → Merton better for idiosyncratic risk
- All similar → Shock-agnostic (good for production)

---

### D.3: Liquidity Adjustment

**Goal**: Is Merton capturing default risk or just liquidity?

**Problem**: OAS contains both:
```
OAS = OAS_default + OAS_liquidity
```

**Method**:
1. Estimate liquidity component using bid-ask spread, volume, etc.
2. Construct OAS_default = OAS - OAS_liquidity
3. Re-estimate Merton using OAS_default
4. Compare R² improvement

**Interpretation**:
- Large R² improvement → Liquidity was confounding, use adjusted
- Small R² improvement → Total OAS is fine (simpler)

**Liquidity model**:
```
OAS_liquidity ~ f(BidAsk, Volume, Amihud, Roll, ...)
```

---

## Implementation Steps

### Step 1: Prepare Data

```python
from dts_research.data.loader import BondDataLoader
from dts_research.analysis.stageD import StageDAnalysis

# Load bond data
loader = BondDataLoader()
bond_data = loader.load_bond_data(start_date, end_date)
index_data = loader.load_index_data(start_date, end_date, index_type='IG')

# Prepare regression data (from Stage 0)
stage0 = Stage0Analysis()
regression_data = stage0.prepare_regression_data(bond_data, index_data)

# Add spread_regime
regression_data['spread_regime'] = regression_data['oas'].apply(
    lambda x: 'IG' if x < 300 else 'HY'
)
```

### Step 2: Run D.1 - Quantile Regression

```python
stageD = StageDAnalysis()

quantile_results = stageD.quantile_regression_analysis(
    regression_data,
    quantiles=[0.05, 0.25, 0.50, 0.75, 0.95],
    by_regime=True  # Separate IG/HY
)

# Examine tail amplification
print(quantile_results['tail_factors'])
# {'left_tail_amplification': 1.82, 'right_tail_amplification': 1.95}
# → Right tail has 95% higher elasticity than median
```

### Step 3: Run D.2 - Shock Decomposition

```python
shock_results = stageD.shock_decomposition_analysis(
    regression_data
)

# Variance decomposition
print(shock_results['variance_decomposition'])
# {'global': 0.15, 'sector': 0.35, 'issuer': 0.45, 'residual': 0.05}

# Shock-specific betas
print(shock_results['shock_betas'])
# {'beta_global': 0.92, 'beta_sector': 0.88, 'beta_issuer': 0.74}
# → Merton works best for global shocks
```

### Step 4: Run D.3 - Liquidity Adjustment

```python
# Note: Requires liquidity data (bid-ask, volume, etc.)
liquidity_results = stageD.liquidity_adjustment_analysis(
    regression_data  # Must have bid_ask, volume columns
)

# R² comparison
print(liquidity_results['comparison'])
# {'r2_total': 0.023, 'r2_default_only': 0.031, 'improvement': 0.008}
# → 0.8% R² improvement, probably not worth complexity
```

### Step 5: Generate Recommendations

```python
recommendations = stageD.generate_production_recommendations(
    quantile_results,
    shock_results,
    liquidity_results
)

print(recommendations)
# "TAIL ADJUSTMENTS: Standard Merton adequate for VaR/ES..."
```

---

## Outputs

### Tables

1. **Table D.1**: Quantile-specific betas
   - β_τ for each quantile (0.05, 0.25, 0.50, 0.75, 0.95)
   - By regime (IG/HY)

2. **Table D.2**: Tail amplification factors
   - Left tail vs median
   - Right tail vs median
   - Statistical tests

3. **Table D.3**: Variance decomposition
   - % explained by global, sector, issuer shocks
   - By regime

4. **Table D.4**: Shock-specific elasticities
   - β^(Global), β^(Sector), β^(Issuer)
   - Comparison tests

5. **Table D.5**: Liquidity model estimates
   - Coefficients on bid-ask, volume, etc.
   - R² of liquidity model

6. **Table D.6**: Merton fit comparison (total vs default)
   - R² using total OAS
   - R² using default-only OAS
   - Improvement

7. **Table D.7**: Liquidity quartile analysis
   - How does fit vary by liquidity level?

### Figures

1. **Figure D.1**: Quantile regression
   - β_τ across quantiles with confidence intervals
   - By regime

2. **Figure D.2**: Shock-specific elasticities
   - Bar chart comparing β^(G), β^(S), β^(I)

3. **Figure D.3**: Liquidity adjustment impact
   - Scatter: total OAS vs default OAS fit
   - R² improvement by liquidity quartile

4. **Figure D.4**: Variance decomposition (supplementary)
   - Pie chart or stacked bar

### Written Summary

3-4 page summary covering:
- Tail behavior findings
- Shock heterogeneity
- Liquidity decomposition results
- Production recommendations

---

## Production Recommendations

### Tail Adjustments

**If symmetric (β_τ similar across quantiles)**:
- No adjustments needed
- Standard Merton adequate for VaR/ES

**If tail amplification (β_0.05 > 1.5 × β_0.50)**:
```python
# Adjust for tail events
if abs(spread_change) > threshold:
    lambda_adjusted = lambda_base * tail_amplification_factor
```

---

### Shock-Type Adjustments

**If shock-agnostic (all β similar)**:
- No adjustments needed
- Use standard Merton for all shocks

**If shock-heterogeneous**:
```python
# Example: Global shocks have higher sensitivity
if shock_type == 'global':
    lambda_i = lambda_base * 1.15  # 15% amplification
elif shock_type == 'issuer':
    lambda_i = lambda_base * 0.85  # 15% dampening
```

---

### Liquidity Decomposition

**If small R² improvement (< 2%)**:
- Use total OAS (simpler)
- Liquidity decomposition not worth complexity

**If large R² improvement (> 5%)**:
```python
# Strip out liquidity component
oas_default = oas_total - estimate_liquidity_component(
    bid_ask, volume, amihud
)
lambda_i = lambda_combined(oas_default, maturity)
```

---

## Common Pitfalls

### 1. Quantile Regression Convergence

**Problem**: Quantile regression may not converge with sparse data

**Solution**:
- Pool across buckets if needed
- Use regularization
- Check residual patterns

### 2. Shock Decomposition Identification

**Problem**: Hard to separate global vs sector vs issuer shocks

**Solution**:
- Use hierarchical/nested models
- Leverage industry classifications
- Validate with known events (e.g., 2008 crisis = global)

### 3. Liquidity Proxy Quality

**Problem**: Bid-ask spread is noisy proxy for liquidity

**Solution**:
- Use multiple liquidity measures (Amihud, Roll, volume)
- Principal components to combine
- Validate against transaction costs if available

### 4. Overfitting on Tail Events

**Problem**: Few observations in tails, easy to overfit

**Solution**:
- Use conservative thresholds (τ = 0.05/0.95, not 0.01/0.99)
- Cross-validate tail adjustments
- Prefer simple parametric adjustments

---

## Validation Checklist

- [ ] Quantile regression: Sufficient observations in each quantile (> 1000)
- [ ] Tail amplification: Robust to quantile choice (0.05 vs 0.10)
- [ ] Shock decomposition: Variance sums to ~100%
- [ ] Shock betas: Economically interpretable
- [ ] Liquidity model: R² > 0.20 (otherwise not useful)
- [ ] Liquidity adjustment: Improvement > 2% to justify complexity
- [ ] All results: Robust across IG/HY regimes

---

## Example Workflow

```python
# 1. Load data
bond_data = loader.load_bond_data('2010-01-01', '2024-12-31')
regression_data = stage0.prepare_regression_data(bond_data, index_data)

# 2. Run D.1
quantile = stageD.quantile_regression_analysis(regression_data)

# 3. Run D.2
shock = stageD.shock_decomposition_analysis(regression_data)

# 4. Run D.3 (if liquidity data available)
liquidity = stageD.liquidity_adjustment_analysis(regression_data)

# 5. Generate recommendations
recs = stageD.generate_production_recommendations(quantile, shock, liquidity)

# 6. Create outputs
visualizer = StageDVisualizer()
figures = visualizer.create_all_stageD_figures(quantile, shock, liquidity)

reporter = StageDReporter()
reporter.save_all_reports(quantile, shock, liquidity, recs)

print(recs)
```

---

## Interpretation Examples

### Example 1: Well-Behaved Model
```
Quantile betas: [0.021, 0.022, 0.023, 0.022, 0.024]
Tail amplification: Left = 0.91x, Right = 1.04x
→ SYMMETRIC, no tail adjustments needed

Shock betas: Global = 0.89, Sector = 0.87, Issuer = 0.88
→ SHOCK-AGNOSTIC, no adjustments needed

Liquidity ΔR²: 0.8%
→ USE TOTAL OAS, not worth decomposing
```

**Production decision**: Use standard Merton, no adjustments

---

### Example 2: Tail Amplification
```
Quantile betas: [0.015, 0.020, 0.023, 0.028, 0.037]
Tail amplification: Left = 0.65x, Right = 1.61x
→ RIGHT TAIL AMPLIFICATION (61% higher than median)

Shock betas: Similar
Liquidity ΔR²: Small
```

**Production decision**:
- Apply tail adjustment for risk management
- Use standard Merton for mean forecasts
```python
if spread_change > 95th_percentile:
    lambda_adjusted = lambda_base * 1.61
```

---

### Example 3: Liquidity Dominance
```
Quantile betas: Symmetric
Shock betas: Similar

Liquidity model R²: 0.35
Liquidity ΔR²: 12%
→ LIQUIDITY IMPORTANT, explains 12% additional variance
```

**Production decision**:
- Strip liquidity from OAS before applying Merton
- Need liquidity data pipeline in production

---

## Next Steps

**After Stage D**:
- Incorporate findings into Stage E production specification
- Document any tail/shock/liquidity adjustments needed
- Validate recommendations with out-of-sample tests

---

## References

- Koenker, R., & Bassett, G. (1978). "Regression Quantiles"
- Angrist, J., & Pischke, J. S. (2008). "Mostly Harmless Econometrics" (Ch. 7 on quantile)
- Amihud, Y. (2002). "Illiquidity and Stock Returns"
- Dick-Nielsen, J., Feldhütter, P., & Lando, D. (2012). "Corporate Bond Liquidity"

---

## Quick Reference

**Main script**: `run_stageD.py`

**Key files**:
- `src/dts_research/analysis/stageD.py` - Analysis logic
- `src/dts_research/visualization/stageD_plots.py` - Visualization
- `src/dts_research/utils/reportingD.py` - Tables and summaries

**Runtime**: ~40 seconds with mock data, ~10-15 minutes with real data

**Prerequisites**: Stage 0 (regression data), optional liquidity data

**Critical outputs**:
- Tail amplification factors (VaR adjustments)
- Shock-specific betas (heterogeneity)
- Liquidity R² improvement (decomposition decision)

**Decision thresholds**:
- Tail amplification > 1.5x → Consider adjustments
- Liquidity ΔR² > 5% → Consider decomposition
- Shock beta differences > 20% → Consider shock-specific models
