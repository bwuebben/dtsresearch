# 🚀 START HERE

Welcome to the DTS Research project! This guide will get you up and running in 5 minutes.

## What You Have

A complete implementation of **Stage 0** from your paper, ready to run on either:
- Mock data (for testing) ✅ Works immediately
- Your database (fill in connection details)

## Quick Start (3 Steps)

### 1. Install Dependencies (2 minutes)
```bash
cd dtsresearch
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Stage 0 (30 seconds)
```bash
python run_stage0.py
```

### 3. Check Results
```bash
# View summary report
cat output/reports/stage0_summary.txt

# View figures
open output/figures/

# View tables
open output/reports/stage0_table01_bucket_results.csv
```

## What Gets Generated

```
output/
├── figures/
│   ├── stage0_fig1_scatter.png          # β vs λ scatter plot
│   ├── stage0_fig2_crossmaturity.png    # Cross-maturity patterns
│   └── stage0_fig3_regimes.png          # Regime patterns
└── reports/
    ├── stage0_summary.txt               # 2-3 page analysis
    ├── stage0_table01_bucket_results.csv
    ├── stage0_table02_cross_maturity.csv
    └── stage0_full_results.csv
```

## Using Your Own Data

### Step 1: Configure Database
Edit `src/dts_research/data/loader.py`:

```python
# Line ~25: Add your connection logic
def connect(self):
    import psycopg2  # or your database driver
    self.connection = psycopg2.connect(self.connection_string)

# Line ~45: Customize SQL query for your schema
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
    WHERE date BETWEEN %(start_date)s AND %(end_date)s
"""
```

### Step 2: Update Main Script
Edit `run_stage0.py`:

```python
# Line ~35: Change to use real data
use_mock_data = False

# Line ~50: Add your connection string
connection_string = "postgresql://user:pass@host:port/db"
```

### Step 3: Run
```bash
python run_stage0.py
```

## Project Structure

```
dtsresearch/
├── run_stage0.py              ← Main script (start here)
├── config.py                  ← Configuration settings
├── requirements.txt           ← Python dependencies
│
├── src/dts_research/          ← Source code
│   ├── data/                  ← Data loading
│   ├── models/                ← Merton lambdas
│   ├── analysis/              ← Stage 0 analysis
│   ├── visualization/         ← Plotting
│   └── utils/                 ← Reporting
│
├── examples/                  ← Example scripts
├── tests/                     ← Unit tests
└── output/                    ← Generated (after running)
```

## What Stage 0 Does

From your paper, Stage 0:

1. **Classifies bonds into buckets**
   - Rating: AAA/AA, A, BBB, BB, B, CCC
   - Maturity: 1-2y, 2-3y, 3-5y, 5-7y, 7-10y, 10y+
   - Sector: Your classification

2. **Runs pooled regressions per bucket**
   - y_i,t = α + β·f_DTS,t + ε
   - Clustered standard errors by week

3. **Compares to Merton theory**
   - Calculate theoretical λ^Merton for each bucket
   - Test whether β ≈ λ

4. **Performs statistical tests**
   - Level test: H₀: β = λ
   - Cross-maturity patterns
   - Regime patterns

5. **Generates deliverables**
   - 3 figures (Figures 0.1-0.3)
   - 2 tables (Tables 0.1-0.2)
   - Written summary
   - Decision recommendation

## Expected Output

```
================================================================================
STAGE 0: RAW VALIDATION USING BUCKET-LEVEL ANALYSIS
================================================================================

Step 1: Loading bond data...
  Loaded 26,000 bond-week observations
  Bonds: 500
  Date range: 2010-01-01 to 2024-12-31

Step 2: Classifying bonds into buckets...
  Total buckets: 72
  IG buckets: 36
  HY buckets: 36

Step 3: Running pooled regressions for each bucket...
  Successfully estimated 72 bucket regressions

Step 4: Running statistical tests...
  Test 1: Mean deviation = 0.023, p-value = 0.147
  Test 2: Cross-maturity pattern confirmed
  Test 3: IG dispersion > HY dispersion ✓

Step 5: Generating visualizations...
  Created 3 figures

Step 6: Generating reports...
  Created reports

================================================================================
DECISION: ✓ Merton provides good baseline
================================================================================
```

## Documentation

| File | What It Covers |
|------|----------------|
| `README.md` | Full methodology and usage |
| `QUICKSTART.md` | 5-minute quick start |
| `ARCHITECTURE.md` | Code structure and design |
| `PROJECT_SUMMARY.md` | Complete feature list |
| `START_HERE.md` | This file |

## Common Tasks

### Run example script
```bash
python examples/example_merton_lambda.py
```

### Run tests
```bash
pytest tests/ -v
```

### Customize configuration
```bash
cp config.py config_local.py
# Edit config_local.py with your settings
```

### See project statistics
```bash
find src -name "*.py" | xargs wc -l
```

## Next Steps

1. ✅ Run with mock data (verify installation)
2. 📊 Review output/reports/stage0_summary.txt
3. 🔍 Examine figures in output/figures/
4. 🗄️ Connect your database (edit loader.py)
5. 🚀 Run with real data
6. 📈 Interpret decision recommendation
7. ➡️ Proceed to Stage A

## Need Help?

- **Installation issues**: Check `requirements.txt` installed correctly
- **Database errors**: Verify connection string in `loader.py`
- **Import errors**: Ensure you're in virtual environment
- **Output missing**: Check `output/` directory created automatically

## Key Features

✨ **Complete Stage 0 implementation** from your paper
🎯 **Theory-guided** - Merton model foundation
📊 **Publication-ready** - All figures and tables
🧪 **Tested** - Unit tests and mock data
📚 **Well-documented** - 5 guide documents
🔧 **Extensible** - Ready for Stages A-E
⚡ **Fast** - ~10 seconds with mock data

## Questions?

1. Read `README.md` for methodology
2. Check `ARCHITECTURE.md` for code structure
3. See `PROJECT_SUMMARY.md` for complete feature list
4. Review example scripts in `examples/`

---

**Ready?** Run `python run_stage0.py` and check `output/reports/stage0_summary.txt`! 🎉
