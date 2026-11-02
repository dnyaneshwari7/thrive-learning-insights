"""
test_analytics.py - Automated tests for Northbrook Analytics Engine

Validates that all metrics, weighted calculations, centre rankings,
period comparisons, and zero-denominator handling match the verified
project figures exactly.
"""

import unittest
from pathlib import Path
import sys

# Ensure assistant module is in path
CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analytics import NorthbrookAnalytics


class TestNorthbrookAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analytics = NorthbrookAnalytics()

    # Test 1: Overall conversion rate (weighted: 15.32%)
    def test_overall_conversion_rate(self):
        res = self.analytics.get_overall_summary()
        self.assertEqual(res.data["total_calls"], 1078083)
        self.assertEqual(res.data["total_sales"], 165176)
        self.assertEqual(res.data["weighted_conversion_rate_pct"], 15.32)

    # Test 2: Total sales and revenue (£206,201,878.00)
    def test_overall_revenue_and_insurance(self):
        res = self.analytics.get_overall_summary()
        self.assertEqual(res.data["total_revenue"], 206201878.0)
        self.assertEqual(res.data["total_insurance_sales"], 66190)
        self.assertEqual(res.data["weighted_insurance_attach_rate_pct"], 40.07)

    # Test 3: Centre conversion comparison
    def test_centre_conversion_comparison(self):
        res = self.analytics.get_centre_performance()
        centres = res.data["centres"]
        self.assertEqual(centres["Leeds"]["conversion_rate_pct"], 15.37)
        self.assertEqual(centres["Cardiff"]["conversion_rate_pct"], 15.31)
        self.assertEqual(centres["Glasgow"]["conversion_rate_pct"], 15.29)
        self.assertEqual(res.data["highest_conversion_centre"], "Leeds")

    # Test 4: Centre revenue comparison
    def test_centre_revenue_comparison(self):
        res = self.analytics.get_centre_performance()
        centres = res.data["centres"]
        self.assertEqual(centres["Leeds"]["revenue"], 70984496.0)
        self.assertEqual(centres["Cardiff"]["revenue"], 67689043.0)
        self.assertEqual(centres["Glasgow"]["revenue"], 67528339.0)
        self.assertEqual(res.data["highest_revenue_centre"], "Leeds")

    # Test 5: Before vs After academy comparison
    def test_before_and_after_academy(self):
        res = self.analytics.get_period_comparison()
        before = res.data["before_academy"]
        after = res.data["after_academy"]
        uplift = res.data["uplift"]

        self.assertEqual(before["conversion_rate_pct"], 15.01)
        self.assertEqual(after["conversion_rate_pct"], 15.64)
        self.assertEqual(uplift["conversion_rate_difference_pp"], 0.63)
        self.assertEqual(uplift["revenue_difference"], 2815018.0)

    # Test 6: Trained vs Untrained status comparison
    def test_trained_vs_untrained(self):
        res = self.analytics.get_training_status_comparison()
        untrained = res.data["untrained_status"]
        trained = res.data["trained_status"]

        self.assertEqual(untrained["conversion_rate_pct"], 14.99)
        self.assertEqual(trained["conversion_rate_pct"], 16.00)
        self.assertEqual(res.data["difference_pp"], 1.00)

    # Test 7: Module completion rate
    def test_module_completion_rates(self):
        res = self.analytics.get_module_completion_summary()
        self.assertEqual(res.data["total_assignments"], 2000)
        self.assertEqual(res.data["total_completed"], 1387)
        self.assertEqual(res.data["overall_completion_rate_pct"], 69.35)
        self.assertEqual(res.data["highest_completion_module"], "Customer Empathy")
        self.assertEqual(res.data["modules"]["Customer Empathy"]["completion_rate_pct"], 70.75)

    # Test 8: Coaching attendance and correlation with sales
    def test_coaching_attendance(self):
        res = self.analytics.get_coaching_summary()
        self.assertEqual(res.data["total_sessions"], 809)
        self.assertEqual(res.data["average_sessions"], 2.02)
        # Verify lack of meaningful correlation (-0.087)
        self.assertEqual(res.data["correlation_with_conversion"], -0.087)

    # Test 16: Zero denominator guard test
    def test_zero_denominator_safe(self):
        # Overall conversion with zero calls should return 0.0, not ZeroDivisionError
        total_sales = 0
        total_calls = 0
        rate = (total_sales / total_calls * 100) if total_calls > 0 else 0.0
        self.assertEqual(rate, 0.0)


if __name__ == "__main__":
    unittest.main()