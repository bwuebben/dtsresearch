# 🚀 START HERE

Welcome to the DTS Research project! This guide will get you up and running in 5 minutes.

## What You Have

A complete implementation of **Stages 0, A, and B** from your paper, ready to run on either:
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

### 2. Run a Stage (30 seconds to 3 minutes)

**Stage 0: Raw Validation (~10 seconds)**
```bash
python run_stage0.py
```

**Stage A: Establish Cross-Sectional Variation (~15 seconds without Spec A.2)**
```bash
python run_stageA.py
```

**Stage B: Does Merton Explain Variation? (~20 seconds)**
```bash
python run_stageB.py
```

### 3. Check Results
```bash
# View summary reports
cat output/reports/stage0_summary.txt
cat output/reports/stageA_summary.txt
cat output/reports/stageB_summary.txt

# View figures
open output/figures/

# View tables
ls output/reports/*.csv
```

## What Gets Generated

### Stage 0 Outputs
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

### Stage A Outputs
```
output/
├── figures/
│   ├── stageA_fig1_heatmap.png          # Beta heatmap (rating × maturity)
│   ├── stageA_fig2_surface_3d.png       # Beta surface (3D)
│   └── stageA_fig2_surface_contour.png  # Beta surface (contour)
└── reports/
    ├── stageA_summary.txt               # 2-page analysis
    ├── stageA_table_a1_bucket_betas.csv
    ├── stageA_table_a2_equality_tests.csv
    └── stageA_table_a3_a2_results.csv   # If Spec A.2 run
```

### Stage B Outputs
```
output/
├── figures/
│   ├── stageB_fig1_scatter.png          # Empirical vs theoretical
│   ├── stageB_fig2_residuals.png        # Residual analysis (3 panels)
│   ├── stageB_fig3_surfaces_contour.png # Lambda surface comparison
│   └── stageB_fig3_surfaces_3d.png      # Lambda surface (3D)
└── reports/
    ├── stageB_summary.txt               # 3-4 page analysis
    ├── stageB_table_b1_specifications.csv
    ├── stageB_table_b2_model_comparison.csv
    ├── stageB_table_b3_theory_vs_reality.csv
    └── stageB_theory_vs_reality_full.csv
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
Edit `run_stage0.py`, `run_stageA.py`, or `run_stageB.py`:

```python
# Change to use real data
use_mock_data = False

# Add your connection string
connection_string = "postgresql://user:pass@host:port/db"
```

### Step 3: Run
```bash
python run_stage0.py  # or run_stageA.py or run_stageB.py
```

## Project Structure

```
dtsresearch/
├── run_stage0.py              ← Stage 0 orchestration
├── run_stageA.py              ← Stage A orchestration
├── run_stageB.py              ← Stage B orchestration
├── requirements.txt           ← Python dependencies
│
├── src/dts_research/          ← Source code
│   ├── data/
│   │   └── loader.py          ← Data loading and mock data
│   ├── models/
│   │   └── merton.py          ← Merton lambda calculations
│   ├── analysis/
│   │   ├── buckets.py         ← Bucket classification
│   │   ├── stage0.py          ← Stage 0 analysis
│   │   ├── stageA.py          ← Stage A analysis
│   │   └── stageB.py          ← Stage B analysis
│   ├── visualization/
│   │   ├── stage0_plots.py    ← Figures 0.1-0.3
│   │   ├── stageA_plots.py    ← Figures A.1-A.2
│   │   └── stageB_plots.py    ← Figures B.1-B.3
│   └── utils/
│       ├── reporting.py       ← Stage 0 reports
│       ├── reportingA.py      ← Stage A reports
│       └── reportingB.py      ← Stage B reports
│
├── tests/                     ← Unit tests
└── output/                    ← Generated outputs (after running)
```

## What Each Stage Does

### Stage 0: Raw Validation Using Bucket-Level Analysis

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

4. **Decision**: Does Merton provide adequate baseline?

### Stage A: Establish Cross-Sectional Variation

**Prerequisite for Stage B**: Must establish that variation exists

1. **Specification A.1: Bucket-level betas**
   - Estimate β^(k) for each bucket
   - F-tests for equality across dimensions
   - Critical: If no variation, standard DTS adequate → STOP

2. **Specification A.2: Continuous characteristics** (optional)
   - Rolling 2-year windows for bond-specific betas
   - Cross-sectional regression on maturity and spread
   - More granular but slower (~3 minutes)

3. **Decision**: Is there cross-sectional variation?
   - F-test p < 0.10 → Proceed to Stage B
   - F-test p ≥ 0.10 → Standard DTS adequate, STOP

### Stage B: Does Merton Explain the Variation?

**Prerequisite**: Stage A found variation (F-test p < 0.10)

**Critical Question**: Does theory explain the variation?

1. **Specification B.1: Merton as offset (constrained)**
   - Single parameter: β_Merton
   - Test H₀: β_Merton = 1

2. **Specification B.2: Decomposed components**
   - Separate β_T (maturity) and β_s (credit quality)
   - Test H₀: β_T = 1 and β_s = 1

3. **Specification B.3: Unrestricted**
   - Fully flexible functional form
   - Comparison baseline

4. **Theory vs Reality Table**
   - Direct comparison of β^(k) vs λ^Merton
   - Bucket-by-bucket assessment

5. **Decision**: Four paths
   - PATH 1: Theory works well → Use pure Merton
   - PATH 2: Theory needs calibration → Use β_Merton × λ^Merton
   - PATH 3: Theory captures structure but incomplete → Dual tracks
   - PATH 4: Theory fundamentally fails → Skip Stage C

## Expected Output

### Stage 0
```
================================================================================
STAGE 0: RAW VALIDATION USING BUCKET-LEVEL ANALYSIS
================================================================================

Step 1: Loading bond data...
  Loaded 26,000 bond-week observations
  Bonds: 500

Step 2: Classifying bonds into buckets...
  Total buckets: 72

Step 3: Running pooled regressions...
  Successfully estimated 72 bucket regressions

Step 4: Running statistical tests...
  Test 1: Mean deviation = 0.023, p-value = 0.147
  Test 2: Cross-maturity pattern confirmed
  Test 3: IG dispersion > HY dispersion ✓

================================================================================
DECISION: ✓ Merton provides good baseline
================================================================================
```

### Stage A
```
================================================================================
STAGE A: ESTABLISH CROSS-SECTIONAL VARIATION
================================================================================

Critical Question: Do DTS betas differ significantly across bonds?

Step 3: Running Specification A.1 (bucket-level)...
  Estimated 72 bucket betas

Step 4: Running equality tests...
  F-test (all buckets): F = 4.52, p = 0.0001 ***
  F-test (by rating):   F = 3.84, p = 0.0023 **
  F-test (by maturity): F = 2.91, p = 0.0142 *

================================================================================
STAGE A DECISION
================================================================================
PROCEED TO STAGE B
Significant cross-sectional variation exists (F-test p < 0.001)
================================================================================
```

### Stage B
```
================================================================================
STAGE B: DOES MERTON EXPLAIN THE VARIATION?
================================================================================

Step 2: Running Specification B.1 (Merton constrained)...
  β_Merton = 0.952 (SE = 0.078)
  Test H0: β=1, p-value = 0.5389
  R² = 0.823
  → Theory prediction validated

Step 3: Running Specification B.2 (decomposed)...
  β_T (maturity) = 0.985 (SE = 0.091)
  β_s (credit) = 0.973 (SE = 0.084)
  Joint test p-value = 0.8472
  → Both components work well

================================================================================
STAGE B DECISION
================================================================================
PATH 1: Theory Works Well

Conditions met:
✓ β_Merton ∈ [0.9, 1.1]
✓ p-value (H₀: β=1) > 0.05
✓ R² ratio (Merton/Buckets) > 85%

Recommendation:
→ Use pure Merton tables (simplest approach)
→ Proceed to Stage C to test time-variation
→ High confidence in theoretical foundation
================================================================================
```

## Documentation

| File | What It Covers | Lines |
|------|----------------|-------|
| `README.md` | Full methodology and usage | 297 |
| `START_HERE.md` | This file - quick start | 300+ |
| `STAGE_A_GUIDE.md` | Stage A detailed documentation | 318 |
| `STAGE_B_GUIDE.md` | Stage B detailed documentation | 400+ |
| `STAGE_A_COMPLETE.md` | Stage A implementation summary | 400+ |
| `STAGE_B_COMPLETE.md` | Stage B implementation summary | 500+ |

## Common Tasks

### Run all stages sequentially
```bash
python run_stage0.py
python run_stageA.py
python run_stageB.py
```

### Run only Stage A (fastest for cross-sectional test)
```bash
python run_stageA.py  # ~15 seconds without Spec A.2
```

### Run with Spec A.2 (more detailed but slower)
```bash
# Edit run_stageA.py, set run_spec_a2 = True
python run_stageA.py  # ~3 minutes
```

### Run tests
```bash
pytest tests/ -v
```

### See project statistics
```bash
find src -name "*.py" | xargs wc -l
# Total: ~6,000 lines of production code
```

## Typical Workflow

1. **Start with Stage 0**
   ```bash
   python run_stage0.py
   cat output/reports/stage0_summary.txt
   ```
   - Does Merton provide adequate baseline?
   - If YES: Proceed to Stage A
   - If NO: Consider calibration or alternative theory

2. **Run Stage A**
   ```bash
   python run_stageA.py
   cat output/reports/stageA_summary.txt
   ```
   - Is there cross-sectional variation? (F-test p < 0.10)
   - If YES: Proceed to Stage B
   - If NO: Standard DTS adequate, STOP

3. **Run Stage B**
   ```bash
   python run_stageB.py
   cat output/reports/stageB_summary.txt
   ```
   - Does Merton explain the variation?
   - Follow one of four decision paths
   - Proceed to Stage C (or skip if PATH 4)

## Next Steps After Stage B

Based on your Stage B decision path:

- **PATH 1 or PATH 2**: Proceed to Stage C to test time-variation
- **PATH 3**: Stage C with dual tracks (theory + unrestricted)
- **PATH 4**: Skip Stage C, proceed to Stage D (diagnostics)

## Key Features

✨ **Complete Stages 0, A, B** from your paper
🎯 **Theory-guided** - Merton model foundation
📊 **Publication-ready** - All figures and tables
🧪 **Tested** - Unit tests and mock data
📚 **Well-documented** - 6 comprehensive guides
🔧 **Extensible** - Ready for Stages C-E
⚡ **Fast** - 10-20 seconds per stage with mock data
🔄 **Integrated** - Each stage builds on previous

## Implementation Statistics

- **Stage 0**: ~2,427 lines of code
- **Stage A**: ~1,714 lines of code
- **Stage B**: ~1,818 lines of code
- **Total**: ~6,000 lines of production Python code
- **Runtime**: ~45 seconds total with mock data
- **Outputs**: 11 figures + 10 tables + 3 written summaries

## Need Help?

- **Installation issues**: Check `requirements.txt` installed correctly
- **Database errors**: Verify connection string in `loader.py`
- **Import errors**: Ensure you're in virtual environment
- **Output missing**: Check `output/` directory created automatically
- **Stage prerequisites**: Each stage automatically runs prerequisites if needed

## Questions?

1. Read `README.md` for methodology
2. Check stage-specific guides:
   - `STAGE_A_GUIDE.md` for Stage A details
   - `STAGE_B_GUIDE.md` for Stage B details
3. See complete implementation summaries:
   - `STAGE_A_COMPLETE.md` for Stage A code
   - `STAGE_B_COMPLETE.md` for Stage B code
4. Review example scripts in `examples/` (if created)

---

**Ready?**

Start with Stage 0:
```bash
python run_stage0.py
```

Then check the decision recommendation:
```bash
cat output/reports/stage0_summary.txt
```

If Merton looks good, proceed to Stage A to test for cross-sectional variation! 🎉
