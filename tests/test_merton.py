"""
Unit tests for Merton lambda calculations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dts_research.models.merton import MertonLambdaCalculator


class TestMertonLambdaCalculator:
    """Test Merton lambda calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calc = MertonLambdaCalculator(use_power_law=False)

    def test_lambda_T_reference_maturity(self):
        """Lambda_T should be 1.0 at reference maturity (5y)."""
        result = self.calc.lambda_T(5.0, 100)
        assert abs(result - 1.0) < 1e-6

    def test_lambda_s_reference_spread(self):
        """Lambda_s should be 1.0 at reference spread (100 bps)."""
        result = self.calc.lambda_s(100)
        assert abs(result - 1.0) < 1e-6

    def test_lambda_T_short_vs_long(self):
        """Short maturity bonds should have higher lambda than long maturity."""
        lambda_1y = self.calc.lambda_T(1.0, 100)
        lambda_10y = self.calc.lambda_T(10.0, 100)
        assert lambda_1y > lambda_10y

    def test_lambda_T_table_values(self):
        """Test specific table values for lambda_T."""
        # From Table: 50 bps, 1y should be 3.62
        result = self.calc.lambda_T(1.0, 50)
        assert abs(result - 3.62) < 0.01

        # From Table: 100 bps, 10y should be 0.64
        result = self.calc.lambda_T(10.0, 100)
        assert abs(result - 0.64) < 0.01

    def test_lambda_s_table_values(self):
        """Test specific table values for lambda_s."""
        # From Table: 50 bps should be 1.145
        result = self.calc.lambda_s(50)
        assert abs(result - 1.145) < 0.01

        # From Table: 300 bps should be 0.746
        result = self.calc.lambda_s(300)
        assert abs(result - 0.746) < 0.01

    def test_lambda_combined(self):
        """Test combined lambda calculation."""
        maturity = 3.0
        spread = 200

        lambda_T = self.calc.lambda_T(maturity, spread)
        lambda_s = self.calc.lambda_s(spread)
        lambda_combined = self.calc.lambda_combined(maturity, spread)

        expected = lambda_T * lambda_s
        assert abs(lambda_combined - expected) < 1e-6

    def test_vectorized_calculation(self):
        """Test that vectorized calculations work."""
        maturities = np.array([1, 3, 5, 7, 10])
        spread = 100

        results = self.calc.lambda_T(maturities, spread)

        assert len(results) == len(maturities)
        assert results[2] == 1.0  # 5y reference

    def test_classify_regime(self):
        """Test regime classification."""
        # IG narrow maturity
        regime = self.calc.classify_regime(150, 1.5)
        assert "IG narrow maturity" in regime

        # IG wide maturity (PRIMARY FAILURE MODE)
        regime = self.calc.classify_regime(150, 5.0)
        assert "PRIMARY FAILURE MODE" in regime

        # HY
        regime = self.calc.classify_regime(600, 4.0)
        assert "HY" in regime

        # Distressed
        regime = self.calc.classify_regime(1200, 3.0)
        assert "Distressed" in regime

    def test_power_law_approximation(self):
        """Test power law approximation mode."""
        calc_power = MertonLambdaCalculator(use_power_law=True)

        # Power law: (s/100)^-0.25
        result = calc_power.lambda_s(200)
        expected = (200/100)**(-0.25)

        assert abs(result - expected) < 1e-6


class TestMertonEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        self.calc = MertonLambdaCalculator()

    def test_very_short_maturity(self):
        """Test maturity < 1 year."""
        result = self.calc.lambda_T(0.5, 100)
        # Should extrapolate using 1y values
        assert result > 1.0

    def test_very_long_maturity(self):
        """Test maturity > 10 years."""
        result = self.calc.lambda_T(15.0, 100)
        # Should extrapolate using 10y values
        assert result < 1.0

    def test_extreme_spreads(self):
        """Test very low and very high spreads."""
        # Very tight spread
        result_low = self.calc.lambda_s(10)
        assert result_low > 1.0

        # Very wide spread
        result_high = self.calc.lambda_s(5000)
        assert result_high < 1.0

    def test_scalar_vs_array_consistency(self):
        """Test that scalar and array inputs give consistent results."""
        maturity_scalar = 3.0
        maturity_array = np.array([3.0])
        spread = 100

        result_scalar = self.calc.lambda_T(maturity_scalar, spread)
        result_array = self.calc.lambda_T(maturity_array, spread)

        assert abs(result_scalar - result_array[0]) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
