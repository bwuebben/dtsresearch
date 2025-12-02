"""
Example: Using the Merton lambda calculator.

This demonstrates how to calculate theoretical adjustment factors
for different bond characteristics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.models.merton import MertonLambdaCalculator
import numpy as np


def main():
    print("="*70)
    print("MERTON LAMBDA CALCULATOR - EXAMPLE")
    print("="*70)
    print()

    # Initialize calculator
    calc = MertonLambdaCalculator(use_power_law=False)

    # Example 1: Single bond
    print("Example 1: Calculate lambda for a single bond")
    print("-" * 70)

    maturity = 3.0  # years
    spread = 150    # bps

    lambda_T = calc.lambda_T(maturity, spread)
    lambda_s = calc.lambda_s(spread)
    lambda_total = calc.lambda_combined(maturity, spread)

    print(f"Bond characteristics:")
    print(f"  Maturity: {maturity} years")
    print(f"  Spread: {spread} bps")
    print()
    print(f"Adjustment factors (relative to 5y, 100bps reference):")
    print(f"  λ_T (maturity adjustment): {lambda_T:.3f}")
    print(f"  λ_s (credit adjustment):   {lambda_s:.3f}")
    print(f"  λ_total (combined):        {lambda_total:.3f}")
    print()
    print(f"Interpretation:")
    if lambda_total > 1:
        print(f"  This bond is {lambda_total:.1%} MORE sensitive than reference bond")
    else:
        print(f"  This bond is {(1-lambda_total):.1%} LESS sensitive than reference bond")
    print()

    # Example 2: Cross-maturity comparison
    print("Example 2: Cross-maturity comparison for BBB bonds (200 bps)")
    print("-" * 70)

    maturities = np.array([1, 3, 5, 7, 10])
    spread = 200  # BBB spread

    lambdas = calc.lambda_combined(maturities, spread)

    print(f"Spread level: {spread} bps (BBB rating)")
    print()
    print("Maturity  λ_total  Relative Sensitivity")
    print("-" * 40)
    for mat, lam in zip(maturities, lambdas):
        pct = lam * 100
        print(f"{mat:>3}y      {lam:>5.2f}    {pct:>5.1f}%")
    print()
    print(f"Key insight: 1y bonds are {lambdas[0]/lambdas[-1]:.2f}x more sensitive than 10y bonds")
    print()

    # Example 3: Regime classification
    print("Example 3: Regime classification")
    print("-" * 70)

    test_cases = [
        (150, 1.5, "IG portfolio, narrow maturity range"),
        (150, 5.0, "IG portfolio, wide maturity range"),
        (600, 4.0, "HY portfolio, moderate maturity range"),
        (1200, 3.0, "Distressed portfolio"),
    ]

    for spread, mat_range, description in test_cases:
        regime = calc.classify_regime(spread, mat_range)
        print(f"\n{description}")
        print(f"  Spread: {spread} bps, Maturity range: {mat_range}y")
        print(f"  Regime: {regime}")
        print(f"  {calc.get_regime_description(regime)}")

    print()
    print("="*70)


if __name__ == '__main__':
    main()
