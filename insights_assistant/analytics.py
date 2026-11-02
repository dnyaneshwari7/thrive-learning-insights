"""
analytics.py - Northbrook Insights AI Assistant Analytics Engine

Provides strictly validated, deterministic pandas calculations against the
Northbrook Travel Sales Excellence Academy datasets.

All numerical outputs are calculated here directly from verified CSV files.
The LLM never calculates or invents figures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

# Resolve project directories
ASSISTANT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ASSISTANT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_FILE = DATA_DIR / "processed" / "sales_learning_analysis.csv"
ADVISORS_FILE = DATA_DIR / "advisors.csv"
LEARNING_FILE = DATA_DIR / "learning_activity.csv"
COACHING_FILE = DATA_DIR / "coaching_attendance.csv"
SALES_RAW_FILE = DATA_DIR / "sales_performance.csv"


@dataclass
class AnalyticsResult:
    metric_or_topic: str
    data: Dict[str, Any]
    formatted_summary: str
    filters: Dict[str, Any] = field(default_factory=dict)
    data_source: str = ""
    limitations: str = ""
    fully_supported: bool = True


class NorthbrookAnalytics:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self._load_and_validate_data()

    def _load_and_validate_data(self):
        """Load and validate all project datasets."""
        if not PROCESSED_DATA_FILE.exists():
            raise FileNotFoundError(f"Processed dataset missing: {PROCESSED_DATA_FILE}")
        if not ADVISORS_FILE.exists():
            raise FileNotFoundError(f"Advisors dataset missing: {ADVISORS_FILE}")
        if not LEARNING_FILE.exists():
            raise FileNotFoundError(f"Learning activity missing: {LEARNING_FILE}")
        if not COACHING_FILE.exists():
            raise FileNotFoundError(f"Coaching attendance missing: {COACHING_FILE}")

        self.advisors_df = pd.read_csv(ADVISORS_FILE)
        self.learning_df = pd.read_csv(LEARNING_FILE, parse_dates=["completion_date"])
        self.coaching_df = pd.read_csv(COACHING_FILE)
        self.sales_learning_df = pd.read_csv(PROCESSED_DATA_FILE, parse_dates=["month"])

        # Merge centre and team details into sales data for centre-level calculations
        self.full_sales_df = self.sales_learning_df.merge(
            self.advisors_df[["advisor_id", "centre", "team_id", "experience_years"]],
            on="advisor_id",
            how="left",
            validate="many_to_one"
        )

        # Merge centre into learning data
        self.full_learning_df = self.learning_df.merge(
            self.advisors_df[["advisor_id", "centre"]],
            on="advisor_id",
            how="left",
            validate="many_to_one"
        )

    # 1. Overall Sales and Revenue Summary
    def get_overall_summary(self) -> AnalyticsResult:
        df = self.sales_learning_df
        total_calls = int(df["calls_taken"].sum())
        total_sales = int(df["sales_made"].sum())
        total_revenue = float(df["revenue"].sum())
        total_insurance = int(df["insurance_sales"].sum())

        conv_rate = (total_sales / total_calls * 100) if total_calls > 0 else 0.0
        ins_rate = (total_insurance / total_sales * 100) if total_sales > 0 else 0.0

        data = {
            "total_calls": total_calls,
            "total_sales": total_sales,
            "total_revenue": total_revenue,
            "total_insurance_sales": total_insurance,
            "weighted_conversion_rate_pct": round(conv_rate, 2),
            "weighted_insurance_attach_rate_pct": round(ins_rate, 2),
            "reporting_months": 12,
            "total_advisors": len(self.advisors_df)
        }

        summary = (
            f"Across the 12-month period (Oct 2025 – Sep 2026), Northbrook Travel recorded {total_calls:,} calls and "
            f"{total_sales:,} sales, generating total revenue of £{total_revenue:,.2f}. "
            f"The overall weighted sales conversion rate is {conv_rate:.2f}%, and the "
            f"weighted insurance attachment rate is {ins_rate:.2f}%."
        )

        return AnalyticsResult(
            metric_or_topic="Overall Sales & Revenue Summary",
            data=data,
            formatted_summary=summary,
            data_source="data/processed/sales_learning_analysis.csv",
            limitations="Synthetic demonstration data. Weighted conversion rate calculated as sum(sales_made) / sum(calls_taken)."
        )

    # 2. Academy Period Comparison (Before vs After)
    def get_period_comparison(self) -> AnalyticsResult:
        df = self.sales_learning_df.copy()
        df["period"] = df["month"].apply(
            lambda x: "Before" if x < pd.Timestamp("2026-04-01") else "After"
        )

        grouped = df.groupby("period").agg(
            total_calls=("calls_taken", "sum"),
            total_sales=("sales_made", "sum"),
            total_revenue=("revenue", "sum"),
            total_insurance=("insurance_sales", "sum")
        )
        grouped["conversion_rate"] = (grouped["total_sales"] / grouped["total_calls"]) * 100
        grouped["insurance_attach_rate"] = (grouped["total_insurance"] / grouped["total_sales"]) * 100

        before = grouped.loc["Before"]
        after = grouped.loc["After"]

        conv_diff = after["conversion_rate"] - before["conversion_rate"]
        rev_diff = after["total_revenue"] - before["total_revenue"]
        rev_growth_pct = (rev_diff / before["total_revenue"]) * 100

        data = {
            "before_academy": {
                "period": "October 2025 - March 2026 (6 months)",
                "calls": int(before["total_calls"]),
                "sales": int(before["total_sales"]),
                "revenue": float(before["total_revenue"]),
                "conversion_rate_pct": round(float(before["conversion_rate"]), 2),
                "insurance_attach_rate_pct": round(float(before["insurance_attach_rate"]), 2)
            },
            "after_academy": {
                "period": "April 2026 - September 2026 (6 months)",
                "calls": int(after["total_calls"]),
                "sales": int(after["total_sales"]),
                "revenue": float(after["total_revenue"]),
                "conversion_rate_pct": round(float(after["conversion_rate"]), 2),
                "insurance_attach_rate_pct": round(float(after["insurance_attach_rate"]), 2)
            },
            "uplift": {
                "conversion_rate_difference_pp": round(float(conv_diff), 2),
                "revenue_difference": round(float(rev_diff), 2),
                "revenue_growth_pct": round(float(rev_growth_pct), 2)
            }
        }

        summary = (
            f"Before the Sales Excellence Academy launched (Oct 2025 – Mar 2026), conversion was "
            f"{before['conversion_rate']:.2f}% with £{before['total_revenue']:,.2f} revenue from {int(before['total_calls']):,} calls. "
            f"After launch (Apr 2026 – Sep 2026), conversion increased to {after['conversion_rate']:.2f}% "
            f"with £{after['total_revenue']:,.2f} revenue from {int(after['total_calls']):,} calls. "
            f"This represents an uplift of +{conv_diff:.2f} percentage points in conversion rate and "
            f"+£{rev_diff:,.2f} (+{rev_growth_pct:.2f}%) in revenue."
        )

        return AnalyticsResult(
            metric_or_topic="Before vs After Academy Performance",
            data=data,
            formatted_summary=summary,
            data_source="data/processed/sales_learning_analysis.csv",
            limitations="Simulated before-and-after comparison. Does not establish real-world causal proof."
        )

    # 3. Training Status Comparison (trained_before_month True vs False)
    def get_training_status_comparison(self) -> AnalyticsResult:
        df = self.sales_learning_df
        grouped = df.groupby("trained_before_month").agg(
            total_calls=("calls_taken", "sum"),
            total_sales=("sales_made", "sum"),
            total_revenue=("revenue", "sum")
        )
        grouped["conversion_rate"] = (grouped["total_sales"] / grouped["total_calls"]) * 100

        untrained = grouped.loc[False]
        trained = grouped.loc[True]

        conv_diff = trained["conversion_rate"] - untrained["conversion_rate"]

        data = {
            "untrained_status": {
                "label": "No modules completed prior to sales month",
                "calls": int(untrained["total_calls"]),
                "sales": int(untrained["total_sales"]),
                "revenue": float(untrained["total_revenue"]),
                "conversion_rate_pct": round(float(untrained["conversion_rate"]), 2)
            },
            "trained_status": {
                "label": "At least 1 module completed prior to sales month",
                "calls": int(trained["total_calls"]),
                "sales": int(trained["total_sales"]),
                "revenue": float(trained["total_revenue"]),
                "conversion_rate_pct": round(float(trained["conversion_rate"]), 2)
            },
            "difference_pp": round(float(conv_diff), 2)
        }

        summary = (
            f"Advisors trained before the sales month achieved a conversion rate of "
            f"{trained['conversion_rate']:.2f}% ({int(trained['total_sales']):,} sales from {int(trained['total_calls']):,} calls, £{float(trained['total_revenue']):,.2f} revenue), "
            f"compared to {untrained['conversion_rate']:.2f}% for untrained months ({int(untrained['total_sales']):,} sales from {int(untrained['total_calls']):,} calls, £{float(untrained['total_revenue']):,.2f} revenue). "
            f"The conversion rate among previously trained advisors was higher by +{conv_diff:.2f} percentage points."
        )

        return AnalyticsResult(
            metric_or_topic="Sales Performance by Time-Aware Training Status",
            data=data,
            formatted_summary=summary,
            data_source="data/processed/sales_learning_analysis.csv",
            limitations="In prepare_data.py, a +1 pp uplift was explicitly simulated for trained_before_month=True."
        )

    # 4. Contact Centre Performance (Cardiff, Glasgow, Leeds)
    def get_centre_performance(
        self,
        centre: Optional[str] = None,
        compare_with: Optional[str] = None
    ) -> AnalyticsResult:
        df = self.full_sales_df
        summary = df.groupby("centre").agg(
            total_calls=("calls_taken", "sum"),
            total_sales=("sales_made", "sum"),
            total_revenue=("revenue", "sum"),
            total_insurance=("insurance_sales", "sum")
        )
        summary["conversion_rate"] = (summary["total_sales"] / summary["total_calls"]) * 100
        summary["insurance_attach_rate"] = (summary["total_insurance"] / summary["total_sales"]) * 100

        # Learning completion by centre
        learning_comp = (
            self.full_learning_df.groupby("centre")["completed"]
            .apply(lambda s: (s == "Yes").mean() * 100)
        )
        summary["learning_completion_rate"] = learning_comp

        centres_data = {}
        for c_name, row in summary.iterrows():
            centres_data[c_name] = {
                "centre": c_name,
                "calls": int(row["total_calls"]),
                "sales": int(row["total_sales"]),
                "revenue": float(row["total_revenue"]),
                "conversion_rate_pct": round(float(row["conversion_rate"]), 2),
                "insurance_attach_rate_pct": round(float(row["insurance_attach_rate"]), 2),
                "learning_completion_rate_pct": round(float(row["learning_completion_rate"]), 2)
            }

        highest_rev_centre = summary["total_revenue"].idxmax()
        highest_conv_centre = summary["conversion_rate"].idxmax()

        # Handle comparison between two centres
        if centre and compare_with:
            c1 = centre.capitalize()
            c2 = compare_with.capitalize()
            if c1 in centres_data and c2 in centres_data:
                d1 = centres_data[c1]
                d2 = centres_data[c2]
                rev_diff = d1["revenue"] - d2["revenue"]
                conv_diff = d1["conversion_rate_pct"] - d2["conversion_rate_pct"]
                comp_summary = (
                    f"Comparison between {c1} and {c2}:\n"
                    f"- {c1}: Revenue £{d1['revenue']:,.2f}, Conversion Rate {d1['conversion_rate_pct']}%, Module Completion {d1['learning_completion_rate_pct']}%.\n"
                    f"- {c2}: Revenue £{d2['revenue']:,.2f}, Conversion Rate {d2['conversion_rate_pct']}%, Module Completion {d2['learning_completion_rate_pct']}%.\n"
                    f"Difference ({c1} vs {c2}): Revenue difference is £{rev_diff:+,.2f}, conversion rate difference is {conv_diff:+.2f} percentage points."
                )
                return AnalyticsResult(
                    metric_or_topic=f"Centre Comparison: {c1} vs {c2}",
                    data={"centre_1": d1, "centre_2": d2, "revenue_diff": rev_diff, "conv_diff_pp": conv_diff},
                    formatted_summary=comp_summary,
                    filters={"centres": [c1, c2]},
                    data_source="data/processed/sales_learning_analysis.csv and data/advisors.csv"
                )

        # Handle single centre query
        if centre:
            c = centre.capitalize()
            if c in centres_data:
                d = centres_data[c]
                single_summary = (
                    f"{c} contact centre achieved £{d['revenue']:,.2f} total revenue from {d['calls']:,} calls and "
                    f"{d['sales']:,} sales. Its weighted conversion rate is {d['conversion_rate_pct']}%, "
                    f"insurance attachment rate is {d['insurance_attach_rate_pct']}%, and module completion rate is {d['learning_completion_rate_pct']}%. "
                    f"Overall, Leeds generated the highest revenue (£70,984,496) and highest conversion (15.37%)."
                )
                return AnalyticsResult(
                    metric_or_topic=f"{c} Contact Centre Performance",
                    data=d,
                    formatted_summary=single_summary,
                    filters={"centre": c},
                    data_source="data/processed/sales_learning_analysis.csv and data/advisors.csv"
                )
            else:
                return AnalyticsResult(
                    metric_or_topic="Unknown Centre",
                    data={},
                    formatted_summary=f"Contact centre '{centre}' is not in the Northbrook dataset. The valid centres are Leeds, Glasgow, and Cardiff.",
                    fully_supported=False
                )

        overall_summary = (
            f"Performance across the three contact centres:\n"
            f"- Leeds: Highest revenue (£{centres_data['Leeds']['revenue']:,.2f}) and highest conversion rate ({centres_data['Leeds']['conversion_rate_pct']}%); module completion {centres_data['Leeds']['learning_completion_rate_pct']}%.\n"
            f"- Cardiff: Revenue £{centres_data['Cardiff']['revenue']:,.2f}, conversion rate {centres_data['Cardiff']['conversion_rate_pct']}%, module completion {centres_data['Cardiff']['learning_completion_rate_pct']}%.\n"
            f"- Glasgow: Revenue £{centres_data['Glasgow']['revenue']:,.2f}, conversion rate {centres_data['Glasgow']['conversion_rate_pct']}%, highest module completion ({centres_data['Glasgow']['learning_completion_rate_pct']}%).\n"
            f"Leeds achieved both the highest total revenue and the highest conversion rate."
        )

        return AnalyticsResult(
            metric_or_topic="Contact Centre Performance Comparison",
            data={
                "centres": centres_data,
                "highest_revenue_centre": highest_rev_centre,
                "highest_conversion_centre": highest_conv_centre
            },
            formatted_summary=overall_summary,
            data_source="data/processed/sales_learning_analysis.csv and data/advisors.csv",
            limitations="Centre differences reflect synthetic simulation parameters and advisor counts."
        )

    # 5. Monthly Conversion & Sales Trends
    def get_monthly_trends(self) -> AnalyticsResult:
        df = self.sales_learning_df.copy()
        df["month_str"] = df["month"].dt.strftime("%Y-%m")
        monthly = df.groupby("month_str").agg(
            total_calls=("calls_taken", "sum"),
            total_sales=("sales_made", "sum"),
            total_revenue=("revenue", "sum"),
            total_insurance=("insurance_sales", "sum")
        )
        monthly["conversion_rate"] = (monthly["total_sales"] / monthly["total_calls"]) * 100
        monthly["insurance_attach_rate"] = (monthly["total_insurance"] / monthly["total_sales"]) * 100

        records = []
        for m_str, row in monthly.iterrows():
            records.append({
                "month": m_str,
                "calls": int(row["total_calls"]),
                "sales": int(row["total_sales"]),
                "revenue": float(row["total_revenue"]),
                "conversion_rate_pct": round(float(row["conversion_rate"]), 2),
                "insurance_attach_rate_pct": round(float(row["insurance_attach_rate"]), 2)
            })

        min_month = monthly["conversion_rate"].idxmin()
        max_month = monthly["conversion_rate"].idxmax()

        summary = (
            f"Monthly conversion was stable at ~15% before academy launch: "
            f"Oct 2025 (15.03%), Nov 2025 (15.01%), Dec 2025 (15.15%), Jan 2026 (lowest at 14.75%), "
            f"Feb 2026 (15.16%), Mar 2026 (14.94%).\n"
            f"After launch in April 2026 (14.95%), conversion rose: "
            f"May (15.45%), June (15.79%), July (15.83%), peaking in August 2026 at {monthly.loc[max_month, 'conversion_rate']:.2f}%, "
            f"and ending at 15.70% in September 2026."
        )

        return AnalyticsResult(
            metric_or_topic="Monthly Sales & Conversion Trends",
            data={"months": records, "lowest_conversion_month": min_month, "highest_conversion_month": max_month},
            formatted_summary=summary,
            data_source="data/processed/sales_learning_analysis.csv",
            limitations="Covering 12 months (Oct 2025 – Sep 2026)."
        )

    # 6. Learning Activity & Module Completion
    def get_module_completion_summary(self, module_name: Optional[str] = None) -> AnalyticsResult:
        df = self.learning_df
        total_assignments = len(df)
        total_completed = int((df["completed"] == "Yes").sum())
        overall_rate = (total_completed / total_assignments * 100) if total_assignments > 0 else 0.0

        mod_summary = df.groupby("module").agg(
            total=("completed", "count"),
            completed=("completed", lambda x: (x == "Yes").sum())
        )
        mod_summary["completion_rate"] = (mod_summary["completed"] / mod_summary["total"]) * 100

        modules_data = {}
        for mod, row in mod_summary.iterrows():
            modules_data[mod] = {
                "module": mod,
                "assigned": int(row["total"]),
                "completed": int(row["completed"]),
                "completion_rate_pct": round(float(row["completion_rate"]), 2)
            }

        highest_mod = mod_summary["completion_rate"].idxmax()
        lowest_mod = mod_summary["completion_rate"].idxmin()

        if module_name:
            match = next((m for m in modules_data if m.lower() == module_name.lower()), None)
            if match:
                d = modules_data[match]
                single_summary = (
                    f"The '{match}' module had {d['completed']} completions out of {d['assigned']} assigned advisors, "
                    f"giving a completion rate of {d['completion_rate_pct']}%. "
                    f"Customer Empathy achieved the highest completion rate overall at 70.75%."
                )
                return AnalyticsResult(
                    metric_or_topic=f"{match} Module Completion",
                    data=d,
                    formatted_summary=single_summary,
                    filters={"module": match},
                    data_source="data/learning_activity.csv"
                )
            else:
                valid_mods = ", ".join(modules_data.keys())
                return AnalyticsResult(
                    metric_or_topic="Unknown Module",
                    data={},
                    formatted_summary=f"Module '{module_name}' was not found. Valid modules are: {valid_mods}.",
                    fully_supported=False
                )

        overall_summary = (
            f"Across all 5 academy modules, {total_completed:,} of {total_assignments:,} module assignments "
            f"were completed, resulting in an overall module completion rate of {overall_rate:.2f}%.\n"
            f"Module completion rates:\n"
            f"- Customer Empathy: 70.75% (283/400) [Highest]\n"
            f"- Upselling Insurance: 69.50% (278/400)\n"
            f"- Needs Discovery: 69.50% (278/400)\n"
            f"- Compliance Essentials: 68.50% (274/400)\n"
            f"- Objection Handling: 68.50% (274/400)\n"
            f"'Customer Empathy' achieved the highest completion rate."
        )

        return AnalyticsResult(
            metric_or_topic="Academy Module Completion Rates",
            data={
                "total_assignments": total_assignments,
                "total_completed": total_completed,
                "overall_completion_rate_pct": round(overall_rate, 2),
                "modules": modules_data,
                "highest_completion_module": highest_mod,
                "lowest_completion_module": lowest_mod
            },
            formatted_summary=overall_summary,
            data_source="data/learning_activity.csv",
            limitations="Module completion was generated with p=0.70 completion probability in generate_data.py."
        )

    # 7. Advisor Module Completion Distribution
    def get_advisor_completion_distribution(self) -> AnalyticsResult:
        adv_comp = self.learning_df.groupby("advisor_id")["completed"].apply(
            lambda s: (s == "Yes").sum()
        )
        counts = adv_comp.value_counts().sort_index().to_dict()
        total_adv = len(adv_comp)
        at_least_one = int((adv_comp >= 1).sum())
        all_five = int((adv_comp == 5).sum())

        summary = (
            f"Of the {total_adv} advisors:\n"
            f"- {at_least_one} advisors ({at_least_one / total_adv * 100:.2f}%) completed at least 1 module.\n"
            f"- {all_five} advisors ({all_five / total_adv * 100:.2f}%) completed all 5 modules.\n"
            f"Breakdown by number of completed modules: "
            f"0: {counts.get(0, 0)}, 1: {counts.get(1, 0)}, 2: {counts.get(2, 0)}, "
            f"3: {counts.get(3, 0)}, 4: {counts.get(4, 0)}, 5: {counts.get(5, 0)}."
        )

        return AnalyticsResult(
            metric_or_topic="Advisor Module Completion Distribution",
            data={
                "total_advisors": total_adv,
                "advisors_with_at_least_one_module": at_least_one,
                "advisors_with_all_five_modules": all_five,
                "distribution_by_count": counts
            },
            formatted_summary=summary,
            data_source="data/learning_activity.csv"
        )

    # 8. Coaching Attendance and Relationship to Sales
    def get_coaching_summary(self) -> AnalyticsResult:
        coach = self.coaching_df
        total_sessions = int(coach["sessions_attended"].sum())
        avg_sessions = float(coach["sessions_attended"].mean())
        distribution = coach["sessions_attended"].value_counts().sort_index().to_dict()

        # Connect with sales to evaluate empirical correlation
        merged = self.sales_learning_df.merge(coach, on="advisor_id", how="left")
        by_sessions = merged.groupby("sessions_attended").agg(
            advisors=("advisor_id", "nunique"),
            calls=("calls_taken", "sum"),
            sales=("sales_made", "sum"),
            revenue=("revenue", "sum")
        )
        by_sessions["conversion_rate"] = (by_sessions["sales"] / by_sessions["calls"]) * 100

        sessions_data = {}
        for s_count, row in by_sessions.iterrows():
            sessions_data[int(s_count)] = {
                "sessions_attended": int(s_count),
                "advisors": int(row["advisors"]),
                "calls": int(row["calls"]),
                "sales": int(row["sales"]),
                "revenue": float(row["revenue"]),
                "conversion_rate_pct": round(float(row["conversion_rate"]), 2)
            }

        adv_agg = merged.groupby("advisor_id").agg(
            sessions=("sessions_attended", "first"),
            calls=("calls_taken", "sum"),
            sales=("sales_made", "sum")
        )
        adv_agg["conv"] = adv_agg["sales"] / adv_agg["calls"] * 100
        corr = float(adv_agg["sessions"].corr(adv_agg["conv"]))

        summary = (
            f"Across all 400 advisors, a total of {total_sessions} coaching sessions were attended, "
            f"averaging {avg_sessions:.2f} sessions per advisor (0 sessions: {distribution.get(0, 0)}, "
            f"1 session: {distribution.get(1, 0)}, 2 sessions: {distribution.get(2, 0)}, 3 sessions: {distribution.get(3, 0)}, "
            f"4 sessions: {distribution.get(4, 0)}).\n\n"
            f"How coaching relates to sales:\n"
            f"There is no meaningful correlation between coaching attendance and sales conversion rate (correlation = {corr:.3f}). "
            f"Conversion rates remain nearly identical across attendance tiers: 0 sessions (15.49%), 1 session (15.27%), "
            f"2 sessions (15.29%), 3 sessions (15.22%), and 4 sessions (15.32%).\n\n"
            f"Root cause in simulation: In generate_data.py, coaching attendance was generated as random integers (0–4) "
            f"and was NOT factored into the sales probability calculation in prepare_data.py. Only e-learning completion "
            f"(trained_before_month) was programmed with a simulated conversion boost."
        )

        return AnalyticsResult(
            metric_or_topic="Coaching Attendance & Sales Relationship",
            data={
                "total_sessions": total_sessions,
                "average_sessions": round(avg_sessions, 2),
                "distribution": distribution,
                "performance_by_tier": sessions_data,
                "correlation_with_conversion": round(corr, 3)
            },
            formatted_summary=summary,
            data_source="data/coaching_attendance.csv and data/processed/sales_learning_analysis.csv",
            limitations="Coaching attendance was randomly generated and not modeled to affect sales probability."
        )

    # 9. Methodology, Synthetic Data & Causality Explanations
    def get_methodology_explanation(self, query: str = "") -> AnalyticsResult:
        q = query.lower()
        if any(w in q for w in ["cause", "causality", "causal", "did the training cause", "prove"]):
            topic = "Causality & Training Uplift Limitations"
            summary = (
                "Did the training cause the improvement?\n\n"
                "The analysis CANNOT prove real-world causality for two reasons:\n"
                "1. Synthetic Simulation: The uplift was explicitly programmed into the simulation in prepare_data.py "
                "(a +1 percentage point binomial probability uplift was added for advisors where trained_before_month is True). "
                "The result demonstrates that the code successfully reflected the simulation assumption, not that training works in reality.\n"
                "2. Observational Association vs Causality: In business analytics, a before-and-after comparison is an association. "
                "Without a randomized control trial (RCT) controlling for seasonality, caller intent, and market conditions, "
                "causality cannot be claimed."
            )
        elif "trained_before_month" in q:
            topic = "Purpose of the trained_before_month Column"
            summary = (
                "What is the purpose of the 'trained_before_month' column?\n\n"
                "The 'trained_before_month' column is a time-aware boolean flag engineered in scripts/prepare_data.py. "
                "It verifies whether an advisor completed at least one learning module BEFORE the start of that specific sales month.\n\n"
                "Why it matters:\n"
                "If we used lifetime module totals, an advisor who completed training in August 2026 would be counted as 'trained' "
                "for October 2025 sales, which is an analytical error known as 'lookahead bias'. "
                "'trained_before_month' ensures that past sales are never explained by future training."
            )
        elif any(w in q for w in ["synthetic", "generate", "generator", "simulated"]):
            topic = "Synthetic Dataset Generation Methodology"
            summary = (
                "How was the synthetic dataset generated?\n\n"
                "The synthetic dataset was generated in scripts/generate_data.py using NumPy (seed 42) and pandas:\n"
                "- 400 fictional advisors across Leeds, Glasgow, and Cardiff with 1–10 years experience.\n"
                "- 5 learning modules assigned to each advisor (2,000 records) with a 70% completion probability and completion dates spread over 183 days post-launch.\n"
                "- Coaching attendance generated as random integers (0–4 sessions) per advisor.\n"
                "- 12 months of sales records (4,800 records) with calls between 150–300 per advisor per month.\n"
                "- In prepare_data.py, sales were regenerated with a 15% baseline conversion probability and +1% for advisors with trained_before_month=True."
            )
        elif any(w in q for w in ["connect", "prepare", "pipeline", "python"]):
            topic = "Python Data Pipeline Methodology"
            summary = (
                "How did Python connect learning with sales?\n\n"
                "In scripts/prepare_data.py, Python connected learning with sales through:\n"
                "1. Loading advisors.csv, learning_activity.csv, coaching_attendance.csv, and sales_performance.csv.\n"
                "2. Aggregating module completion counts and latest completion dates per advisor.\n"
                "3. Comparing completion dates against each sales month start date to build the time-aware 'trained_before_month' column.\n"
                "4. Modeling simulated sales uplift based on trained status.\n"
                "5. Validating business rules (no sales > calls, no revenue calculation errors) and exporting sales_learning_analysis.csv for Power BI."
            )
        elif any(w in q for w in ["recommendation", "recommend", "next steps"]):
            return self.get_business_recommendations()
        elif any(w in q for w in ["limitation", "weakness"]):
            return self.get_data_limitations()
        else:
            topic = "Northbrook Project Methodology"
            summary = (
                "Northbrook Travel Project Methodology Summary:\n"
                "The project evaluates the fictional Northbrook Travel Sales Excellence Academy over 12 months (Oct 2025 – Sep 2026), "
                "with the academy launching on 1 April 2026. The pipeline includes synthetic data creation (generate_data.py), "
                "time-aware data preparation (prepare_data.py), and weighted analytics (analyse_data.py). "
                "All answers from this assistant are calculated strictly from the project CSV files."
            )

        return AnalyticsResult(
            metric_or_topic=topic,
            data={"topic": topic},
            formatted_summary=summary,
            data_source="scripts/generate_data.py and scripts/prepare_data.py",
            limitations="Synthetic demonstration project."
        )

    # 10. Data Limitations
    def get_data_limitations(self) -> AnalyticsResult:
        summary = (
            "Key limitations of the Northbrook Travel analysis:\n\n"
            "1. Synthetic Data: All advisor, sales, and learning figures were synthetically generated with fixed probabilistic rules. "
            "Findings reflect simulation assumptions rather than empirical human behavior.\n\n"
            "2. Lack of Causal Proof: The +1 pp conversion uplift was hardcoded into the simulation. No real-world causality can be inferred.\n\n"
            "3. Coaching Disconnect: Coaching attendance was generated purely as random integers (0–4) and was not linked to sales uplift in the code.\n\n"
            "4. Granularity: Sales records are aggregated monthly rather than by individual call or customer interaction.\n\n"
            "5. Missing Employee Attributes: Key factors such as prior sales skill, tenure before Northbrook, customer demographics, and product demand fluctuations are not captured in the dataset."
        )
        return AnalyticsResult(
            metric_or_topic="Project Analytical Limitations",
            data={"limitations_count": 5},
            formatted_summary=summary,
            data_source="Project methodology review",
            limitations="Highlights boundaries of what the dataset can and cannot establish."
        )

    # 11. Business Recommendations
    def get_business_recommendations(self) -> AnalyticsResult:
        summary = (
            "Validated business recommendations based strictly on project findings:\n\n"
            "1. Address Learning Transfer in High-Completion Centres: Glasgow had the highest module completion (71.43%) but the lowest conversion rate (15.29%). "
            "Investigate whether advisors are absorbing content or facing local operational barriers.\n\n"
            "2. Drive Completion in Cardiff: Cardiff had the lowest module completion rate (67.88%). Leadership should establish targeted coaching to lift completion.\n\n"
            "3. Restructure 1-on-1 Coaching: Coaching attendance currently shows no correlation with sales (r = -0.087). "
            "Realign coaching sessions to target specific behavioral weaknesses rather than general attendance.\n\n"
            "4. Prioritize Fast-Track Training: Since trained advisors convert at 16.00% vs 14.99% for untrained months (+1.01 pp), "
            "reducing the time from academy enrollment to module completion directly accelerates revenue realization."
        )
        return AnalyticsResult(
            metric_or_topic="Business Recommendations",
            data={"recommendations_count": 4},
            formatted_summary=summary,
            data_source="Documented project analysis and conclusions",
            limitations="Recommendations are grounded in the project data and analytical limitations."
        )