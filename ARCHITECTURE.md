# Architecture Documentation

## Overview

This project implements a multi-stage empirical research program using a modular, extensible architecture. The design emphasizes:

1. **Separation of concerns** - data, models, analysis, visualization
2. **Theory-guided implementation** - Merton structural model as foundation
3. **Database agnosticism** - easy to adapt to any SQL database
4. **Testability** - mock data generation for development
5. **Reproducibility** - clear pipeline from data to results

## Module Structure

### 1. Data Layer (`src/dts_research/data/`)

**Purpose**: Load and prepare bond data

**Key classes**:
- `BondDataLoader`: Handles database connectivity and data retrieval

**Design patterns**:
- **Abstraction**: Single interface for both real and mock data
- **Lazy loading**: Connection established only when needed
- **User customization**: Clear TODOs for database-specific code

**Data flow**:
```
Database/Mock → BondDataLoader → Pandas DataFrame → Analysis modules
```

**Schema**:
```python
Required columns:
- bond_id: str
- date: datetime
- oas: float (basis points)
- rating: str
- maturity_date: datetime
- time_to_maturity: float (years)
- sector: str
- issuer_id: str
```

### 2. Models Layer (`src/dts_research/models/`)

**Purpose**: Implement theoretical predictions

**Key classes**:
- `MertonLambdaCalculator`: Compute theoretical adjustment factors

**Theory implementation**:
- Lambda tables from Wuebben (2025) hard-coded
- Bilinear interpolation for continuous values
- Regime classification (5 regimes)
- Both exact and power-law approximations

**Key methods**:
```python
lambda_T(maturity, spread)      # Maturity adjustment
lambda_s(spread)                 # Credit quality adjustment
lambda_combined(maturity, spread) # Total adjustment
classify_regime(spread, mat_range) # Regime identification
```

**Design decisions**:
- Vectorized operations (numpy)
- Bounded extrapolation for extreme values
- Support both scalar and array inputs

### 3. Analysis Layer (`src/dts_research/analysis/`)

**Purpose**: Implement empirical tests

#### 3.1 Bucket Classification (`buckets.py`)

**Key class**: `BucketClassifier`

**Functionality**:
- Classify bonds into buckets (rating × maturity × sector)
- Compute bucket-level statistics
- Calculate theoretical Merton lambdas per bucket
- Support cross-maturity and same-maturity queries

**Bucket definition**:
```python
Ratings: ['AAA/AA', 'A', 'BBB', 'BB', 'B', 'CCC']
Maturities: ['1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10y+']
Sectors: Any (user-defined)

Total buckets: 6 ratings × 6 maturities × N sectors
```

#### 3.2 Stage 0 Analysis (`stage0.py`)

**Key class**: `Stage0Analysis`

**Pipeline**:
1. `prepare_regression_data()` - Compute percentage changes
2. `run_bucket_regression()` - Pooled OLS per bucket
3. `run_all_bucket_regressions()` - Iterate over all buckets
4. Statistical tests:
   - `test_level_hypothesis()` - H₀: β = λ
   - `test_cross_maturity_pattern()` - Monotonicity
   - `test_regime_pattern()` - Dispersion vs spread
5. `identify_outliers()` - Flag extreme deviations
6. `generate_decision_recommendation()` - Next steps

**Regression specification**:
```python
y_i,t = α + β · f_DTS,t + ε_i,t

where:
- y_i,t: percentage spread change for bond i
- f_DTS,t: index-level percentage spread change
- Clustered standard errors by week
- OLS estimation via statsmodels
```

### 4. Visualization Layer (`src/dts_research/visualization/`)

**Purpose**: Generate publication-quality figures

**Key class**: `Stage0Visualizer`

**Figures**:
1. **Figure 0.1**: Scatter (β vs λ) with 45° line
2. **Figure 0.2**: Cross-maturity patterns (6 panels by rating)
3. **Figure 0.3**: Regime patterns (dispersion vs spread)

**Design features**:
- Seaborn styling for consistency
- Color coding by spread regime (IG/HY/Distressed)
- Point sizes proportional to sample size
- Outlier annotation
- 300 DPI for publication quality

### 5. Utilities Layer (`src/dts_research/utils/`)

**Purpose**: Reporting and output

**Key class**: `Stage0Reporter`

**Deliverables**:
- **Table 0.1**: Bucket-level results (top N by sample size)
- **Table 0.2**: Cross-maturity patterns (pivot table)
- **Written summary**: 2-3 page report with 5 sections
- **Full results CSV**: Complete output

**Report structure**:
```
1. Executive summary with decision
2. Aggregate level test results
3. Cross-maturity pattern analysis
4. Regime pattern analysis
5. Outlier identification
6. Practical implications
7. Next steps
```

## Data Flow

### Complete Pipeline

```
┌─────────────────┐
│  Database or    │
│  Mock Generator │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BondDataLoader  │
│  - Load data    │
│  - Validate     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│BucketClassifier │
│  - Classify     │
│  - Aggregate    │
│  - Merton λ     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stage0Analysis  │
│  - Regressions  │
│  - Stat tests   │
│  - Outliers     │
└────────┬────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│Visualizer    │ │Reporter  │ │CSV output│
│ - 3 figures  │ │- 2 tables│ │- Raw data│
└──────────────┘ └──────────┘ └──────────┘
```

### Regression Workflow

```
Bond data + Index data
    ↓
prepare_regression_data()
    ↓ (Compute percentage changes)
Regression-ready DataFrame
    ↓
For each bucket:
    ↓
run_bucket_regression()
    ↓ (OLS with clustered SE)
    β, SE(β), t-stat, R²
    ↓
Merge with bucket stats
    ↓ (Add theoretical λ)
    β, λ, β/λ ratio, deviation
    ↓
Statistical tests
    ↓
Results DataFrame
```

## Extension Points

### Adding New Stages

To implement Stage A, B, etc.:

1. Create `src/dts_research/analysis/stageA.py`
2. Define new analysis class (follow `Stage0Analysis` pattern)
3. Create visualizer in `src/dts_research/visualization/`
4. Create reporter in `src/dts_research/utils/`
5. Add orchestration script `run_stageA.py`

**Example**:
```python
class StageAAnalysis:
    def __init__(self):
        self.results = None

    def run_issuer_week_regression(self, df):
        # Implement issuer-week fixed effects
        pass

    def test_cross_sectional_variation(self, results):
        # Statistical tests
        pass
```

### Adding New Models

To add alternative structural models:

1. Create `src/dts_research/models/new_model.py`
2. Implement calculator class with same interface:
   ```python
   class NewModelCalculator:
       def lambda_combined(self, maturity, spread):
           # Return adjustment factor
           pass
   ```
3. Use in bucket classification or analysis

### Custom Database Schema

To adapt to your schema:

1. Edit `src/dts_research/data/loader.py`
2. Modify `load_bond_data()` query
3. Add transformation logic if needed:
   ```python
   def transform_data(self, df):
       # Map your columns to expected schema
       df['time_to_maturity'] = ...
       return df
   ```

## Configuration

### Centralized Settings (`config.py`)

All parameters in one place:
- Database credentials
- Analysis parameters (dates, thresholds)
- Regression settings
- Output formats

### Local Overrides (`config_local.py`)

User-specific settings without git tracking:
```python
# config_local.py (not in git)
DATABASE_CONFIG['connection_string'] = 'postgresql://...'
ANALYSIS_CONFIG['use_mock_data'] = False
```

## Testing Strategy

### Unit Tests (`tests/`)

**Coverage**:
- Merton lambda calculations (table values, interpolation)
- Edge cases (extreme spreads, maturities)
- Vectorization correctness

**Run tests**:
```bash
pytest tests/ -v --cov=src/dts_research
```

### Integration Test (Mock Pipeline)

Run complete pipeline with mock data:
```bash
python run_stage0.py  # Should complete in ~10 seconds
```

### Validation Checks

Built into analysis:
- Minimum sample size enforcement
- NaN handling
- Outlier flagging
- Bucket coverage reporting

## Dependencies

### Core Stack

| Package | Purpose | Why |
|---------|---------|-----|
| pandas | Data manipulation | Industry standard |
| numpy | Numerical computing | Fast vectorization |
| scipy | Interpolation, stats | Scientific computing |
| statsmodels | Regression w/ clustering | Econometric standard |
| matplotlib | Base plotting | Publication quality |
| seaborn | Enhanced visuals | Beautiful defaults |

### Optional

- Database drivers (psycopg2, pymssql, etc.)
- Jupyter (interactive analysis)
- pytest (testing)

## Performance Considerations

### Bottlenecks

1. **Data loading**: Database query time
2. **Regressions**: O(n_buckets × bucket_size)
3. **Clustering**: Standard error calculation

### Optimizations

- Vectorized operations (numpy/pandas)
- Parallel bucket regressions (can add multiprocessing)
- Lazy evaluation where possible
- Efficient data structures (avoid copies)

### Scaling

Current implementation handles:
- **5,000 bonds** × **52 weeks** × **15 years** = **3.9M observations**
- Runtime: ~1-2 minutes on modern hardware

For larger datasets:
- Add multiprocessing for bucket regressions
- Use dask for out-of-core processing
- Consider database aggregation for bucketing

## Code Quality

### Standards

- Type hints throughout
- Docstrings for all public methods
- PEP 8 style compliance
- DRY principle (shared utilities)

### Documentation Levels

1. **Code**: Inline comments for complex logic
2. **Module**: Docstrings with examples
3. **API**: README with usage
4. **Architecture**: This document

### Maintainability

- Clear module boundaries
- Minimal inter-module dependencies
- Consistent naming conventions
- Version control friendly (no binary files in repo)

## Future Enhancements

### Planned Features

1. **Stage A-E implementations**
2. **Panel regression methods**
3. **Time-varying parameter estimation**
4. **Robustness tests**
5. **Alternative structural models**

### Potential Improvements

1. **Performance**: Parallel processing, caching
2. **Visualization**: Interactive plots (plotly)
3. **Reporting**: LaTeX table generation
4. **Database**: Connection pooling, async queries
5. **Testing**: Property-based testing, fuzzing
6. **CI/CD**: Automated testing, documentation

## Best Practices

### For Users

1. Always work in virtual environment
2. Keep `config_local.py` out of version control
3. Run tests before committing changes
4. Check bucket coverage before interpreting results
5. Review outliers carefully

### For Developers

1. Add tests for new features
2. Update documentation
3. Follow existing code patterns
4. Use type hints
5. Profile before optimizing

### For Researchers

1. Document data sources
2. Record configuration used
3. Save intermediate results
4. Version control analysis scripts
5. Report software versions in papers

## References

- Wuebben (2025): Theoretical foundation for Merton lambdas
- Accompanying paper: Full methodology documentation
- statsmodels docs: Regression with clustering
- Merton (1974): Original structural model
