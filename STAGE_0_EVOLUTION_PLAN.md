# Stage 0 Evolution Implementation Plan

## Status: IN PROGRESS - Phase 1

This document tracks the implementation of the evolved Stage 0 based on the new paper (./docs/paper.tex).

---

## Overview

The new paper fundamentally transforms Stage 0 from simple data preparation into a comprehensive 3-pronged theoretical validation framework with:
1. **Bucket-Level Analysis**: Test Merton at aggregate level (72 buckets per universe)
2. **Within-Issuer Analysis**: Clean test using same issuer, different maturities
3. **Sector Interaction Analysis**: Formal test if sectors differ systematically

---

## Configuration Decisions

### Data Fields
- **Ultimate Parent ID**: Available in database ✅
- **Sector Classification**: Use Bloomberg `BCLASS3` (configurable to `BCLASS4`)
- **Seniority**: Infer from security type or use existing field

### Within-Issuer Filters (Configurable in `config.py`)
- **Minimum bonds per issuer per week**: 3 (was 2 in paper, upgraded per user)
- **Minimum maturity dispersion**: 2 years
- **Pull-to-par exclusion**: 1 year from maturity
- **Max spread change**: 200%

### Fallback Strategy
- Old Stage 0 (simple data prep) remains available as fallback
- New Stage 0 comprehensive analysis is the default

---

## Implementation Progress

### ✅ Completed

#### Phase 1: Core Infrastructure
1. ✅ **config.py created** - All configurable parameters centralized
   - Bloomberg classification level (BCLASS3/4)
   - Within-issuer filter parameters (≥3 bonds, ≥2yr dispersion)
   - Bucket definitions, regime thresholds, etc.

2. ✅ **issuer_identification.py created** - Issuer ID module
   - `create_composite_issuer_id()` - Parent + Seniority composite
   - `classify_seniority()` - Standardize to Senior/Subordinated
   - `validate_issuer_coverage()` - Coverage statistics
   - `filter_for_within_issuer_analysis()` - Apply min bonds filter

### 🔄 In Progress

#### Phase 1: Core Infrastructure (Continued)
3. **sector_classification.py** - Next to create
   - Map BCLASS3/4 to 4 research sectors (Industrial, Financial, Utility, Energy)
   - Validation and coverage checks

4. **bucket_definitions.py** - To create
   - Define 72 buckets: 6 ratings × 6 maturities × 2 major sectors
   - Compute representative (s̄, T̄) for each bucket

5. **statistical_tests.py** - To create
   - Clustered standard errors (week clustering)
   - Chow test, joint F-test, Wald test
   - Inverse-variance weighted pooling

6. **Update loader.py** - To update
   - Add fields: ultimate_parent_id, BCLASS3/4, seniority
   - Enhanced mock data generation with new fields

---

## Next Steps

### Immediate (Today)
1. Create `sector_classification.py` module
2. Create `bucket_definitions.py` module
3. Create `statistical_tests.py` module
4. Update `loader.py` for new fields
5. Update mock data generation

### This Week
6. Create `stage0_bucket.py` - Bucket-level analysis
7. Create `stage0_within_issuer.py` - Within-issuer analysis
8. Create `stage0_sector.py` - Sector interaction analysis
9. Create `stage0_synthesis.py` - Decision framework
10. Create `stage0_plots.py` - 10 figures
11. Create `reporting0.py` - 17 tables + summary
12. Create `run_stage0.py` - Runner script

### Next Week
13. Test Stage 0 pipeline end-to-end
14. Rewrite STAGE_0_GUIDE.md
15. Rewrite STAGE_0_COMPLETE.md

---

## File Structure (New)

```
src/dts_research/
├── config.py                        ✅ NEW - Configuration parameters
├── data/
│   ├── issuer_identification.py      ✅ NEW - Issuer ID + seniority
│   ├── sector_classification.py      🔄 TODO - BCLASS → 4 sectors
│   ├── bucket_definitions.py         🔄 TODO - 72 bucket definitions
│   ├── loader.py                      🔄 UPDATE - Add new fields
│   └── ...
├── analysis/
│   ├── stage0_bucket.py               🔄 TODO - Bucket-level analysis
│   ├── stage0_within_issuer.py        🔄 TODO - Within-issuer analysis
│   ├── stage0_sector.py               🔄 TODO - Sector interactions
│   ├── stage0_synthesis.py            🔄 TODO - Decision framework
│   ├── stage0.py                      ✅ KEEP - Legacy fallback
│   └── ...
├── visualization/
│   ├── stage0_plots.py                🔄 TODO - 10 figures
│   └── ...
├── utils/
│   ├── statistical_tests.py           🔄 TODO - Clustered SE, tests
│   ├── reporting0.py                  🔄 TODO - 17 tables + summary
│   └── ...
```

---

## Deliverables: Stage 0

### Tables (17 total)
1. **Table 0.1**: Bucket-level β̂ vs λᴹᵉʳᵗᵒⁿ (all 72 buckets)
2. **Table 0.2**: Cross-maturity patterns (Spearman correlations)
3. **Table 0.3**: Within-issuer pooled results (by sample splits)
4. **Table 0.4**: Diagnostic regressions (spread level, dispersion, crisis)
5. **Table 0.5**: Sector interaction estimates (β₀, βFinancial, βUtility, βEnergy)
6. **Table 0.6**: Pairwise sector comparisons
7. **Table 0.7**: Stage 0 synthesis summary

*Plus 10 supplementary tables for different regimes/specifications*

### Figures (10 total)
1. **Figure 0.1**: Bucket scatter (β̂ vs λᴹᵉʳᵗᵒⁿ)
2. **Figure 0.2**: Patterns by rating class
3. **Figure 0.3**: Regime pattern (convergence as spreads widen)
4. **Figure 0.4**: Within-issuer β̂ distribution
5. **Figure 0.5**: Issuer case studies (3-5 examples)
6. **Figure 0.6**: Diagnostic - spread level effect
7. **Figure 0.7**: Sector pattern comparison
8. **Figure 0.8**: Sector interaction effects
9. **Figure 0.9**: Sector heterogeneity by regime
10. **Figure 0.10**: Decision flow diagram

### Written Summary (5-7 pages)
- Executive summary of 3 questions (bucket/within-issuer/sector)
- Detailed findings for each component
- Synthesis and path recommendation
- Data quality diagnostics
- Implementation guidance for downstream stages

---

## Decision Framework (5 Paths)

Based on Stage 0 synthesis, determines which path to follow:

### Path 1: STANDARD DTS ADEQUATE
- Condition: No evidence Merton improves over standard DTS
- Action: Stop pipeline, use standard approach

### Path 2: PURE MERTON
- Condition: β ≈ 1, no sector effects
- Action: Use theory without calibration

### Path 3: CALIBRATED MERTON
- Condition: β ≠ 1 but theory structure correct, no sectors
- Action: Estimate β, use scaled Merton

### Path 4: MERTON + SECTOR ADJUSTMENTS
- Condition: Significant sector heterogeneity
- Action: Use Merton with sector-specific adjustments

### Path 5: THEORY FAILS
- Condition: Merton doesn't explain variation
- Action: Proceed to unrestricted empirical approach

---

## Timeline Estimate

- **Phase 1 (Infrastructure)**: 2-3 days ⏳ IN PROGRESS
- **Phase 2 (Stage 0 Core)**: 5-7 days
- **Phase 2 (Visualization & Reporting)**: 3-4 days
- **Phase 2 (Testing & Documentation)**: 2-3 days

**Total for Stage 0**: ~2-3 weeks

---

## Notes

- Stage 0 is largest component - more code than Stages A-E combined
- Represents shift from data prep to comprehensive theory testing
- Creates foundation for all downstream analyses
- Sector adjustments propagate through all stages

---

## Questions/Issues

1. ✅ **RESOLVED**: Bloomberg classification level → Use BCLASS3 (configurable)
2. ✅ **RESOLVED**: Within-issuer min bonds → 3 (user preference)
3. **PENDING**: Exact BCLASS3 → Sector mapping (need Bloomberg reference)
4. **PENDING**: Mock data generation strategy for realistic multi-bond issuers

---

## References

- New paper: `./docs/paper.tex`
- Old paper: `./docs/paper_old.tex`
- Config: `src/dts_research/config.py`
- Current todo list: See TodoWrite tool output

---

*Last Updated: 2025-12-03*
*Status: Phase 1 - Infrastructure (20% complete)*
