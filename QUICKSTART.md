# Quick Start Guide

## Installation (5 minutes)

```bash
# Clone/download the project
cd dtsresearch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Stage 0 Analysis (2 minutes)

```bash
# Run with mock data
python run_stage0.py
```

This generates:
- `output/figures/` - 3 publication-quality figures
- `output/reports/` - 4 tables and summary report

## Check Results

```bash
# View summary
cat output/reports/stage0_summary.txt

# View tables
open output/reports/stage0_table01_bucket_results.csv

# View figures
open output/figures/stage0_fig1_scatter.png
```

## Example: Calculate Merton Lambda

```bash
python examples/example_merton_lambda.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Use Your Own Data

1. Edit `src/dts_research/data/loader.py`:
   - Fill in `connect()` method
   - Update SQL query in `load_bond_data()`

2. Install database driver:
   ```bash
   pip install psycopg2-binary  # PostgreSQL
   # or other driver from requirements.txt
   ```

3. Edit `run_stage0.py`:
   ```python
   use_mock_data = False
   connection_string = "your_connection_here"
   ```

4. Run:
   ```bash
   python run_stage0.py
   ```

## Expected Runtime

With mock data (500 bonds, 2010-2024):
- Data generation: ~1 second
- Bucket classification: ~1 second
- Regressions (72 buckets): ~5 seconds
- Visualizations: ~2 seconds
- **Total: ~10 seconds**

With real data (5000 bonds):
- Expect ~1-2 minutes total

## Key Files

| File | Purpose |
|------|---------|
| `run_stage0.py` | Main script - run this |
| `config.py` | Configuration settings |
| `src/dts_research/models/merton.py` | Lambda calculations |
| `src/dts_research/data/loader.py` | Data loading (customize here) |
| `output/reports/stage0_summary.txt` | Main results |

## Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**No output directory?**
- Script creates it automatically
- Check permissions

**Database connection fails?**
- Verify connection string
- Check database driver installed
- Test connection separately

**Regression fails for some buckets?**
- Normal if bucket has <30 observations
- Check bucket coverage in output

## Next Steps

1. Review `output/reports/stage0_summary.txt`
2. Examine figures in `output/figures/`
3. Check decision recommendation
4. Proceed to Stage A (coming soon)

## Support

- Check `README.md` for full documentation
- See `examples/` for more examples
- Review test files in `tests/` for usage patterns

## Project Structure

```
dtsresearch/
├── run_stage0.py           ← Start here
├── config.py               ← Settings
├── requirements.txt        ← Dependencies
├── src/dts_research/       ← Source code
│   ├── data/              ← Data loading
│   ├── models/            ← Merton lambdas
│   ├── analysis/          ← Stage 0 analysis
│   ├── visualization/     ← Plotting
│   └── utils/             ← Reporting
├── examples/              ← Example scripts
├── tests/                 ← Unit tests
└── output/                ← Generated (after running)
    ├── figures/
    └── reports/
```
