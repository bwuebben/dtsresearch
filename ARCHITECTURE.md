# Architecture Documentation

## Overview

This project implements a complete multi-stage empirical research program (Stages 0-E) using a modular, extensible architecture. The design emphasizes:

1. **Separation of concerns** - data, models, analysis, visualization
2. **Theory-guided implementation** - Merton structural model as foundation
3. **Database agnosticism** - easy to adapt to any SQL database
4. **Testability** - mock data generation for development
5. **Reproducibility** - clear pipeline from data to results
6. **Extensibility** - Each stage builds on previous foundations

**Implementation Status**: ALL 6 STAGES COMPLETE (0, A, B, C, D, E)

## Module Structure

### 1. Data Layer (`src/dts_research/data/`)

**Purpose**: Load, classify, and prepare bond data

**Key classes**:
- `BondDataLoader`: Handles database connectivity and data retrieval
- `SectorClassifier`: Maps Bloomberg BCLASS3 to research sectors (Financial, Utility, Energy, Industrial)
- `issuer_identification`: Creates composite issuer ID (Ultimate Parent + Seniority)

**Design patterns**:
- **Abstraction**: Single interface for both real and mock data
- **Lazy loading**: Connection established only when needed
- **User customization**: Clear TODOs for database-specific code
- **Sector standardization**: Consistent sector classification across all stages

**Data flow**:
```
Database/Mock → BondDataLoader → SectorClassifier → Issuer ID → Pandas DataFrame → Analysis modules
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
- sector: str (or sector_classification for BCLASS3)
- issuer_id: str (composite: ultimate_parent_id + seniority)
- ultimate_parent_id: str (for within-issuer analysis)
- seniority: str (for within-issuer analysis)

Added by classification:
- sector_research: str ('Financial', 'Utility', 'Energy', 'Industrial')
- is_financial: bool
- is_utility: bool
- is_energy: bool
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

#### 3.2 Stage 0 Analysis (Evolved DTS Foundation)

**Purpose**: Three-pronged theoretical validation framework to determine if Merton adequately describes spread sensitivities to market shocks

**Key modules**:
- `stage0_bucket.py`: Spec 0.1 - Time-series regression of spread changes on DTS factor
- `stage0_within_issuer.py`: Spec 0.2 - Regression of spread changes on λ^Merton, tests β = 1
- `stage0_sector.py`: Spec 0.3 - Sector interactions using Merton-scaled DTS factor
- `stage0_synthesis.py`: Five-path decision framework based on β ≈ 1 criterion

**Specification 0.1 (Bucket-Level Time-Series)**:
```python
y_{i,t} = α^(k) + β^(k) · f_{DTS,t} + ε_{i,t}

where:
- y_{i,t} = Δs/s: percentage spread change for bond i
- f_{DTS,t}: index-level percentage spread change (DTS factor)
- Clustered standard errors by week
- Test: Compare β^(k) to λ^Merton (β/λ ratio ≈ 1?)
- Test: Monotonicity (β decreases with maturity as Merton predicts)
```

**Specification 0.2 (Within-Issuer)**:
```python
Δs_{i,t} / s_{i,t-1} = α_{j,t} + β · λ^Merton_{i,t} + ε_{i,t}

where:
- α_{j,t}: issuer-week fixed effects (absorbs credit quality)
- λ^Merton_{i,t}: Merton-predicted elasticity for bond i
- β: coefficient on Merton lambda (should equal 1 if theory correct)
- Test: H0: β = 1 (not just β > 0)
- Inverse-variance weighted pooling across issuer-weeks
- Requires: ≥3 bonds per issuer-week, ≥2 years TTM dispersion
```

**Specification 0.3 (Sector Interaction)**:
```python
y_{i,t} = α + β_0 · (λ^Merton_i × f_{DTS,t})
          + β_F · (Financial_i × λ^Merton_i × f_{DTS,t})
          + β_U · (Utility_i × λ^Merton_i × f_{DTS,t})
          + β_E · (Energy_i × λ^Merton_i × f_{DTS,t}) + ε_{i,t}

where:
- λ^Merton_i × f_{DTS,t}: Merton-scaled DTS factor
- β_0: baseline sensitivity (Industrial sector)
- β_F, β_U, β_E: sector-specific deviations
- Test: Joint F-test (sectors don't differ), individual sector tests
- Test: Total sensitivity per sector (β_0 + β_sector ≈ 1?)
- Clustered by issuer or week
```

**Five Decision Paths** (based on β ≈ 1 criterion):
1. **Path 1 (Perfect Alignment)**: β/λ ratio ≈ 1 across all methods → Use standard Merton throughout
2. **Path 2 (Sector Heterogeneity)**: Base β ≈ 1 but sectors differ significantly → Add sector adjustments
3. **Path 3 (Weak Evidence)**: β in [0.7, 1.3] but outside [0.9, 1.1] → Proceed cautiously
4. **Path 4 (Mixed Evidence)**: Conflicting results across methods → Use model-free approaches
5. **Path 5 (Theory Fails)**: β far from 1, non-monotonic → Focus on purely empirical models

#### 3.3 Stage A-E Analysis Modules (Stage 0 Integrated)

**Stage A** (`stageA.py` - ~890 lines including Stage 0 integration):
- Spec A.1: Bucket-level betas with F-tests for equality
- Spec A.2: Continuous characteristics with rolling windows
- **Stage 0 Integration**: Skips if Path 5, can reuse buckets if Path 1-2
- Decision: Proceed if variation exists (F-test p < 0.10) OR Stage 0 Path 1-4

**Stage B** (`stageB.py` - ~880 lines including Stage 0 integration):
- Spec B.1: Merton as offset (constrained β=1)
- Spec B.2: Decomposed components (β_T, β_s)
- Spec B.3: Unrestricted empirical model (includes sector dummies)
- **Stage 0 Integration**: Skips if Path 5
- Decision: 4 paths based on theory fit

**Stage C** (`stageC.py` - ~835 lines including Stage 0 integration):
- Rolling window stability tests (Chow test)
- Macro driver analysis (VIX, OAS interaction)
- Maturity-specific time-variation
- **Stage 0 Integration**: Skips if Path 4 or 5 (time-variation tests require working theory)
- Decision: Static vs time-varying needed

**Stage D** (`stageD.py` - ~920 lines including Stage 0 integration):
- D.1: Tail behavior (quantile regression)
- D.2: Shock decomposition (global, sector, issuer)
- D.3: Liquidity adjustment (default + liquidity)
- **Stage 0 Integration**: Path 5 focuses on model-free robustness only
- Production recommendations for each extension

**Stage E** (`stageE.py` - ~890 lines including Stage 0 integration):
- Hierarchical testing framework (5 levels)
- Out-of-sample validation (rolling windows)
- Performance by regime (Normal/Stress/Crisis)
- **Stage 0 Integration**: Path 5 only tests levels 1 and 4 (skips Merton-based levels)
- Production blueprint generation

**Common patterns across all stages**:
- Accept optional `stage0_results` parameter in constructor
- Check decision path and conditionally skip/modify analyses
- Use Stage0's `prepare_regression_data()` as foundation
- Cluster standard errors (by week or issuer)
- Statistical tests with clear decision criteria
- Integration with visualization and reporting modules

### 4. Visualization Layer (`src/dts_research/visualization/`)

**Purpose**: Generate publication-quality figures for all stages

**Visualizer classes**:
- `Stage0Visualizer` (~650 lines): 10 figures for evolved DTS foundation (3 per spec + decision viz)
- `StageAVisualizer` (~390 lines): 3 figures for cross-sectional variation
- `StageBVisualizer` (~530 lines): 4 figures for Merton explanation
- `StageCVisualizer` (~580 lines): 4 figures for stability/time-variation
- `StageDVisualizer` (~530 lines): 4 figures for robustness extensions
- `StageEVisualizer` (~510 lines): 4 figures for production selection

**Total**: 30 publication-quality figures across all stages

**Common design features**:
- Seaborn styling for consistency
- Color coding by spread regime (IG/HY/Distressed)
- Point sizes proportional to sample size
- Outlier annotation where relevant
- 300 DPI for publication quality
- Crisis period shading (2008-2009, 2020)
- Consistent color palettes across stages

### 5. Utilities Layer (`src/dts_research/utils/`)

**Purpose**: Reporting and output for all stages

**Reporter classes**:
- `Stage0Reporter` (~1,050 lines): 17 tables + 3-5 page executive summary
- `StageAReporter` (~360 lines): 3+ tables + 2 page summary
- `StageBReporter` (~570 lines): 4 tables + 3-4 page summary
- `StageCReporter` (~520 lines): 3+ tables + 3-4 page summary
- `StageDReporter` (~700 lines): 7 tables + 3-4 page summary
- `StageEReporter` (~780 lines): 4+ tables + 5-7 page implementation blueprint

**Total**: 38+ tables + 6 written summaries + 1 implementation blueprint

**Common report structure**:
1. Executive summary with decision
2. Statistical test results
3. Model comparison (where applicable)
4. Detailed analysis tables
5. Practical implications
6. Next steps

**Stage E blueprint** (production deployment):
- Algorithmic steps and pseudo-code
- Recalibration protocols
- Edge case handling
- Performance monitoring framework
- Economic value examples

## Data Flow

### Complete Pipeline

```
┌─────────────────┐
│  Database or    │
│  Mock Generator │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ BondDataLoader   │
│  - Load data     │
│  - Validate      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│SectorClassifier  │
│  - BCLASS3 map   │
│  - Add dummies   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│Issuer ID         │
│  - Composite ID  │
│  - Parent+Senior │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│BucketClassifier  │
│  - Classify      │
│  - Aggregate     │
│  - Merton λ      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stage0Analysis   │
│  - Spec 0.1      │
│  - Spec 0.2      │
│  - Spec 0.3      │
│  - Decision Path │
└────────┬─────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
│Visualizer    │ │Reporter  │ │CSV output│ │Stage A-E   │
│ - 10 figures │ │-17 tables│ │- Raw data│ │(integrated)│
└──────────────┘ └──────────┘ └──────────┘ └────────────┘
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

## Complete Stage Implementation

### All Stages Now Complete

**Stages 0-E** are fully implemented following consistent architecture:

**Each stage includes**:
1. Analysis module: `src/dts_research/analysis/stage*.py`
2. Visualizer: `src/dts_research/visualization/stage*_plots.py`
3. Reporter: `src/dts_research/utils/reporting*.py`
4. Orchestration script: `run_stage*.py`
5. Documentation: `STAGE_*_GUIDE.md` and `STAGE_*_COMPLETE.md`

**Common patterns across all stages**:
```python
class Stage*Analysis:
    def __init__(self):
        self.results = None

    def main_analysis_method(self, df, prerequisites):
        # Core analysis logic
        # Statistical tests
        # Return structured results dict
        pass

    def _helper_methods(self):
        # Internal computation
        pass
```

### Extension Points for Future Research

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
python run_stage0.py  # ~3 minutes (evolved with 3 specs)
python run_stageA.py  # ~15 seconds (without Spec A.2)
python run_stageB.py  # ~20 seconds
python run_stageC.py  # ~25-30 seconds
python run_stageD.py  # ~30-40 seconds
python run_stageE.py  # ~45-60 seconds

# Or run all sequentially
for script in run_stage*.py; do python $script; done  # ~5-6 minutes total
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
