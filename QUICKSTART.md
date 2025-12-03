# Quick Start Guide

Get the complete DTS research program (Stages 0-E) running in 5 minutes!

## Installation (2 minutes)

```bash
# Clone/download the project
cd dtsresearch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Your First Stage (3 minutes)

### Stage 0: Evolved DTS Foundation

```bash
python run_stage0.py
```

**Runtime**: ~3 minutes with mock data

**Output**:
- `output/figures/` - 10 publication-quality figures (3 specs + decision viz)
- `output/reports/` - 17 tables and executive summary

## Check Results

```bash
# View summary
cat output/reports/stage0_summary.txt

# View tables
open output/reports/stage0_table01_bucket_results.csv

# View figures
open output/figures/stage0_fig1_scatter.png
```

## Run Complete Pipeline (5-6 minutes)

```bash
# Run all stages sequentially
python run_stage0.py  # ~3 minutes (evolved with 3 specs)
python run_stageA.py  # ~15 seconds
python run_stageB.py  # ~20 seconds
python run_stageC.py  # ~25-30 seconds
python run_stageD.py  # ~30-40 seconds
python run_stageE.py  # ~45-60 seconds

# Total: ~5-6 minutes
```

**Or run all at once**:
```bash
for script in run_stage0.py run_stageA.py run_stageB.py run_stageC.py run_stageD.py run_stageE.py; do
    python $script
done
```

## What Gets Generated

### Figures (30 total)
```bash
output/figures/
├── stage0_fig*.png      # 10 figures (3 specs + decision viz)
├── stageA_fig*.png      # 3 figures (heatmap, 3D surface, contour)
├── stageB_fig*.png      # 4 figures (scatter, residuals, 2 surfaces)
├── stageC_fig*.png      # 4 figures (time series, macro, lambda, crisis)
├── stageD_fig*.png      # 4 figures (quantiles, shocks, liquidity, variance)
└── stageE_fig*.png      # 4 figures (OOS R², errors, predictions, comparison)
```

### Tables (38+ CSV files)
```bash
output/reports/
├── stage0_*.csv         # 17 tables (3 specs + decision framework)
├── stageA_*.csv         # 3+ tables
├── stageB_*.csv         # 4 tables
├── stageC_*.csv         # 3+ tables
├── stageD_*.csv         # 7 tables
└── stageE_*.csv         # 4+ tables
```

### Written Reports (7 text files)
```bash
output/reports/
├── stage0_summary.txt                      # 3-5 pages (executive summary)
├── stageA_summary.txt                      # 2 pages
├── stageB_summary.txt                      # 3-4 pages
├── stageC_summary.txt                      # 3-4 pages
├── stageD_summary.txt                      # 3-4 pages
├── stageE_summary.txt                      # Brief summary
└── stageE_implementation_blueprint.txt     # 5-7 pages (production guide)
```

## Quick Examples

### Example 1: Calculate Merton Lambda

```python
from src.dts_research.models.merton import MertonLambdaCalculator

calc = MertonLambdaCalculator()

# For a 5-year BBB bond with 200 bps spread
lambda_T = calc.lambda_T(maturity=5.0, spread=200)  # Maturity adjustment
lambda_s = calc.lambda_s(spread=200)                # Credit quality adjustment
lambda_total = calc.lambda_combined(5.0, 200)       # Combined

print(f"Maturity adjustment: {lambda_T:.3f}")
print(f"Spread adjustment: {lambda_s:.3f}")
print(f"Total lambda: {lambda_total:.3f}")
```

### Example 2: Classify Bonds into Buckets

```python
import pandas as pd
from src.dts_research.analysis.buckets import BucketClassifier

classifier = BucketClassifier()

# Sample bond data
bonds = pd.DataFrame({
    'rating': ['A', 'BBB', 'BB'],
    'time_to_maturity': [3.5, 7.2, 2.1],
    'sector': ['Financials', 'Industrials', 'Financials'],
    'oas': [100, 180, 350]
})

# Classify
bonds_with_buckets = classifier.classify_bonds(bonds)
print(bonds_with_buckets[['rating', 'rating_bucket', 'maturity_bucket']])
```

### Example 3: Run Custom Analysis

```python
from src.dts_research.data.loader import BondDataLoader
from src.dts_research.analysis.stage0 import Stage0Analysis

# Generate mock data
loader = BondDataLoader()
bond_data = loader.generate_mock_data('2020-01-01', '2024-12-31', n_bonds=100)
index_data = loader.generate_mock_index_data('2020-01-01', '2024-12-31', index_type='IG')

# Run analysis
stage0 = Stage0Analysis()
regression_data = stage0.prepare_regression_data(bond_data, index_data)
results = stage0.run_all_bucket_regressions(regression_data)

print(f"Analyzed {len(results)} buckets")
print(results[['bucket', 'beta', 'lambda_Merton', 'ratio']].head())
```

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/dts_research

# Specific test
pytest tests/test_merton.py -v
```

## Understand the Research Flow

```
Stage 0: Five-Path Decision Framework
   Three specs: Bucket-level, Within-issuer, Sector interaction
   ↓
   Five Paths: 1=Perfect, 2=Sector, 3=Weak, 4=Mixed, 5=Fails
   ↓
Stage A: Does cross-sectional variation exist?
   (Skips if Path 5, can reuse Stage 0 buckets if Path 1-2)
   ↓ YES
Stage B: Does Merton explain the variation?
   (Skips if Path 5)
   ↓ PATH 1-3 (theory works)
Stage C: Static or time-varying needed?
   (Skips if Path 4-5, theory-driven tests need working theory)
   ↓
Stage D: Robustness checks (tail, shocks, liquidity)
   (Path 5: model-free only)
   ↓
Stage E: Production specification selection
   (Path 5: only tests levels 1 & 4)
   ↓
FINAL: Production blueprint + recommended model
```

## Next Steps

### 1. Review Stage 0 Results

```bash
cat output/reports/stage0_summary.txt
```

**Look for**:
- Median β/λ ratio (should be 0.8-1.2 for theory to work)
- Percentage in acceptable range (>70% is good)
- Cross-maturity pattern (should match theory)

### 2. If Stage 0 Looks Good → Run Stage A

```bash
python run_stageA.py
cat output/reports/stageA_summary.txt
```

**Decision**: If F-test shows no variation → Standard DTS is adequate, STOP

### 3. Continue Through Pipeline

Run each stage sequentially, reviewing summaries after each.

### 4. Final Production Specification

After completing all stages:

```bash
# Review hierarchical test results
cat output/reports/stageE_table_e1_hierarchical_tests.csv

# Read production blueprint
cat output/reports/stageE_implementation_blueprint.txt

# Check recommended specification
cat output/reports/stageE_table_e4_production_spec.csv
```

## Common Use Cases

### Use Case 1: Quick Validation (3 minutes)

"Does Merton theory provide a reasonable baseline?"

```bash
python run_stage0.py
cat output/reports/stage0_summary.txt
```

### Use Case 2: Complete Research Pipeline (5-6 minutes)

"Run the full research program and get production recommendation"

```bash
for script in run_stage*.py; do python $script; done
cat output/reports/stageE_implementation_blueprint.txt
```

### Use Case 3: Theory Testing (1 minute)

"Does Merton explain cross-sectional variation?"

```bash
python run_stage0.py
python run_stageA.py
python run_stageB.py
cat output/reports/stageB_summary.txt  # Check Path 1-4
```

### Use Case 4: Production Deployment

"Get deployment-ready specification"

```bash
# Run all stages
for script in run_stage*.py; do python $script; done

# Review blueprint
cat output/reports/stageE_implementation_blueprint.txt

# Validate on your hold-out data (modify run_stageE.py)
# Deploy following blueprint guidance
```

## Customization

### Use Your Own Data

Edit `src/dts_research/data/loader.py`:

```python
def connect(self):
    # Add your database connection
    self.connection = psycopg2.connect(
        host='your_host',
        database='your_db',
        user='your_user',
        password='your_password'
    )

def load_bond_data(self, start_date, end_date):
    # Modify SQL query for your schema
    query = """
        SELECT
            bond_id,
            date,
            oas,
            rating,
            maturity_date,
            sector,
            issuer_id
        FROM your_bond_table
        WHERE date BETWEEN %s AND %s
    """
    return pd.read_sql(query, self.connection, params=[start_date, end_date])
```

Then in `run_stage0.py` (and other run scripts):

```python
use_mock_data = False  # Change to False
connection_string = "your_connection_string_here"
```

### Modify Analysis Parameters

Edit the run scripts to change:
- Date ranges: `start_date`, `end_date`
- Sample size: `n_bonds` for mock data
- Thresholds: F-test p-value, β ratio ranges, etc.

### Add Custom Sectors

In `src/dts_research/analysis/buckets.py`, modify sector list:

```python
# Default: ['Financials', 'Industrials', 'Utilities', 'Consumer']
# Add yours:
sectors = ['Energy', 'Technology', 'Healthcare', ...]
```

## Troubleshooting

### Import Error

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Re-install dependencies
pip install -r requirements.txt
```

### No Output Directory

```bash
# Create manually if needed
mkdir -p output/figures output/reports
```

### Slow Performance

Mock data (~500 bonds, 2010-2024):
- Expected: ~5-6 minutes total (Stage 0 now ~3 min)
- If slower: Check CPU, reduce n_bonds in mock data

Real data (1M observations):
- Expected: ~10-15 minutes total (excluding Spec A.2)
- Spec A.2 rolling windows: Add ~30-40 minutes

### Database Connection Issues

- Check connection string format
- Verify database credentials
- Test connection independently before running stages
- Start with mock data to verify code works

## Documentation

### Quick References
- `START_HERE.md`: Main entry point (you are here!)
- `README.md`: Project overview and methodology
- `ARCHITECTURE.md`: Technical architecture

### Stage-Specific Guides
- `STAGE_A_GUIDE.md`: How to use Stage A
- `STAGE_B_GUIDE.md`: How to use Stage B
- `STAGE_C_GUIDE.md`: How to use Stage C
- `STAGE_D_GUIDE.md`: How to use Stage D
- `STAGE_E_GUIDE.md`: How to use Stage E

### Implementation Details
- `STAGE_*_COMPLETE.md`: 6 files with line counts, method breakdowns, technical decisions

## Summary

**You now have**:
- ✅ Complete DTS research program (Stages 0-E)
- ✅ ~17,514 lines of production Python code
- ✅ Evolved Stage 0 with three-pronged validation
- ✅ Five-path decision framework guiding all analysis
- ✅ Mock data generator for testing
- ✅ 30 publication-quality figures
- ✅ 38+ comprehensive tables
- ✅ Production deployment blueprint

**Time investment**:
- Installation: 2 minutes
- First run (Stage 0): 3 minutes
- Complete pipeline: 5-6 minutes
- Understanding results: 10-15 minutes

**Total**: ~20-25 minutes from installation to production recommendation!

**Ready to deploy!** 🎉
