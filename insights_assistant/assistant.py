"""
assistant.py - Northbrook Insights AI Assistant Core Engine

Orchestrates question interpretation, conversational context tracking,
deterministic analytical routing, data grounding, and natural language synthesis.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

from analytics import NorthbrookAnalytics, AnalyticsResult

# Load environment configuration
load_dotenv()


@dataclass
class ConversationContext:
    last_centre: Optional[str] = None
    last_module: Optional[str] = None
    last_metric: Optional[str] = None
    last_topic: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    def reset(self):
        self.last_centre = None
        self.last_module = None
        self.last_metric = None
        self.last_topic = None
        self.history.clear()


@dataclass
class AssistantResponse:
    answer: str
    topic: str
    data_source: str
    limitations: str
    is_supported: bool = True
    analytics_result: Optional[AnalyticsResult] = None
    clarification_needed: bool = False


class NorthbrookAssistant:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        analytics_engine: Optional[NorthbrookAnalytics] = None
    ):
        self.analytics = analytics_engine or NorthbrookAnalytics()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.context = ConversationContext()
        self._init_gemini_client()

    def _init_gemini_client(self):
        """Initialise Google GenAI client if API key is provided."""
        self.gemini_client = None
        if self.api_key and self.api_key.strip():
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.api_key.strip())
            except Exception as e:
                print(f"[Warning] Could not initialize Gemini client: {e}")
                self.gemini_client = None

    def update_credentials(self, api_key: str, model_name: Optional[str] = None):
        """Update API key and reinitialize client."""
        self.api_key = api_key.strip()
        if model_name:
            self.model_name = model_name
        self._init_gemini_client()

    def ask(self, user_question: str) -> AssistantResponse:
        """Process user question and return grounded response."""
        q = user_question.strip()
        if not q:
            return AssistantResponse(
                answer="Please enter a question about Northbrook Travel's sales performance or academy analysis.",
                topic="Empty Question",
                data_source="None",
                limitations="",
                is_supported=False
            )

        # 1. Guard against prompt injection / requests to invent data
        if self._is_prompt_injection(q):
            return AssistantResponse(
                answer=(
                    "I am strictly programmed to answer questions regarding the Northbrook Travel Sales Excellence Academy "
                    "analysis using verified project datasets. I cannot ignore my safety rules or invent fictional figures."
                ),
                topic="System Guardrail Triggered",
                data_source="System Policy",
                limitations="Adherence to verified project scope.",
                is_supported=False
            )

        # 2. Check for out-of-scope questions (Situation C)
        if self._is_out_of_scope(q):
            return AssistantResponse(
                answer="I'm designed to answer questions about the Northbrook Travel Sales Excellence Academy analysis. Please ask me something related to the project.",
                topic="Out of Scope (Situation C)",
                data_source="Project Scope Policy",
                limitations="",
                is_supported=False
            )

        # 3. Check for questions requesting missing data (Situation A)
        missing_data_resp = self._check_missing_data(q)
        if missing_data_resp:
            return missing_data_resp

        # 4. Check for unsupported analytical requests (Situation B)
        unsupported_resp = self._check_unsupported_analysis(q)
        if unsupported_resp:
            return unsupported_resp

        # 5. Check for unknown contact centres
        unknown_centre_resp = self._check_unknown_centre(q)
        if unknown_centre_resp:
            return unknown_centre_resp

        # 6. Check for ambiguous questions needing clarification (Situation E)
        if self._is_ambiguous(q):
            return AssistantResponse(
                answer="Could you clarify whether you're asking about sales conversion rate, total revenue, or academy module completion?",
                topic="Clarification Requested (Situation E)",
                data_source="User Clarification Needed",
                limitations="",
                is_supported=True,
                clarification_needed=True
            )

        # 7. Resolve conversational references (e.g. 'its revenue', 'Cardiff and Leeds')
        resolved_query, detected_centre, detected_compare = self._resolve_context(q)

        # 8. Route to supported deterministic analytical function
        analytics_result, topic = self._route_analytics(resolved_query, detected_centre, detected_compare)

        # Update conversational memory
        if detected_centre:
            self.context.last_centre = detected_centre
        elif analytics_result and analytics_result.data.get("highest_revenue_centre"):
            self.context.last_centre = analytics_result.data["highest_revenue_centre"]

        self.context.last_topic = topic
        self.context.history.append({"role": "user", "content": q})

        # 9. Synthesize response (via Gemini if available, or grounded deterministic template)
        final_answer = self._synthesize_answer(q, analytics_result)
        self.context.history.append({"role": "assistant", "content": final_answer})

        return AssistantResponse(
            answer=final_answer,
            topic=analytics_result.metric_or_topic,
            data_source=analytics_result.data_source,
            limitations=analytics_result.limitations,
            is_supported=analytics_result.fully_supported,
            analytics_result=analytics_result
        )

    # Context Resolution & Pronoun Handling
    def _resolve_context(self, q: str) -> Tuple[str, Optional[str], Optional[str]]:
        q_lower = q.lower()
        centre_match = None
        compare_match = None

        centres = ["leeds", "glasgow", "cardiff"]
        mentioned = [c for c in centres if re.search(r"\b" + c + r"\b", q_lower)]

        if len(mentioned) >= 2:
            centre_match = mentioned[0].capitalize()
            compare_match = mentioned[1].capitalize()
        elif len(mentioned) == 1:
            centre_match = mentioned[0].capitalize()
            if any(w in q_lower for w in ["compare", "versus", "vs", "against"]) and self.context.last_centre:
                if self.context.last_centre.lower() != centre_match.lower():
                    compare_match = centre_match
                    centre_match = self.context.last_centre

        # Resolve pronouns ("its", "that centre")
        if not centre_match:
            if any(p in q_lower for p in ["its", "it", "that centre", "this centre", "the winner"]):
                centre_match = self.context.last_centre

        return q, centre_match, compare_match

    # Unknown Centre Check
    def _check_unknown_centre(self, q: str) -> Optional[AssistantResponse]:
        q_lower = q.lower()
        common_other_cities = ["london", "manchester", "birmingham", "bristol", "edinburgh", "liverpool", "newcastle", "belfast"]
        for city in common_other_cities:
            if re.search(r"\b" + city + r"\b", q_lower):
                return AssistantResponse(
                    answer=f"Contact centre '{city.capitalize()}' is not in the Northbrook dataset. The valid contact centres are Leeds, Glasgow, and Cardiff.",
                    topic="Unknown Contact Centre",
                    data_source="data/advisors.csv",
                    limitations="Valid centres in project: Leeds, Glasgow, Cardiff.",
                    is_supported=False
                )
        return None

    # Security & Guardrails
    def _is_prompt_injection(self, q: str) -> bool:
        injection_patterns = [
            r"\bignore\b.*(previous|prior)?.*(instruction|rule|prompt)",
            r"\byou are now\b",
            r"\bpretend you are\b",
            r"\bsystem prompt\b",
            r"\bjailbreak\b",
            r"\bdisregard\b",
            r"\b(invent|make up|fabricate|hallucinate)\b.*(number|stat|figure|revenue|conversion|rate|data|metric)?",
            r"\b(invent|make up|fabricate)\b"
        ]
        q_lower = q.lower()
        return any(re.search(pat, q_lower) for pat in injection_patterns)

    # Situation C: Out of Scope
    def _is_out_of_scope(self, q: str) -> bool:
        q_lower = q.lower()
        out_of_scope_patterns = [
            r"\b(weather|temperature|forecast for)\b",
            r"\b(recipe|cooking|bake)\b",
            r"\b(stock market|stocks|crypto|bitcoin|ethereum)\b",
            r"\b(football|world cup|premier league|champions league)\b",
            r"\b(prime minister|president|election)\b",
            r"\bwrite a (poem|song|story|joke)\b",
            r"\bcapital of\b"
        ]
        if any(re.search(pat, q_lower) for pat in out_of_scope_patterns):
            if not any(k in q_lower for k in ["northbrook", "sales", "academy", "conversion", "advisor", "centre"]):
                return True
        return False

    # Situation A: Missing Information (with word boundary regex)
    def _check_missing_data(self, q: str) -> Optional[AssistantResponse]:
        q_lower = q.lower()
        missing_checks = [
            (r"\b(turnover|attrition|quit|quitting|resigned|resignation)\b", "employee turnover and resignation data was not tracked"),
            (r"\b(salary|salaries|pay|bonus|bonuses|commission|compensation)\b", "advisor compensation, salaries, and commission records were not included"),
            (r"\b(gender|age|ages|ethnicity|demographics?)\b", "advisor demographic data (age, gender, ethnicity) was not collected"),
            (r"\b(customer feedback|survey comments?|verbatims?|nps comments?|unhappy reasons?)\b", "qualitative customer survey comments or feedback verbatims were not captured (only numerical satisfaction scores 1–5 exist)"),
            (r"\b(flights?|hotels?|destinations?|packages?)\b", "specific travel product lines (destinations, flight vs hotel packages) were not broken down in the dataset"),
            (r"\b(refunds?|cancellations?|cancelled bookings?)\b", "booking cancellation and customer refund rates were not recorded")
        ]
        for pattern, explanation in missing_checks:
            if re.search(pattern, q_lower):
                return AssistantResponse(
                    answer=f"I can't establish that from the current Northbrook dataset. We would need additional information to analyse it reliably. Specifically, {explanation}.",
                    topic="Missing Data (Situation A)",
                    data_source="None (Uncollected)",
                    limitations=f"Missing data: {explanation}",
                    is_supported=False
                )
        return None

    # Situation B: Unsupported Analytical Operations
    def _check_unsupported_analysis(self, q: str) -> Optional[AssistantResponse]:
        q_lower = q.lower()
        unsupported_checks = [
            (r"\b(regression|multivariate|ols|p-value|hypothesis test|statistical significance)\b", "statistical significance testing and multivariate regression modeling"),
            (r"\b(predict future|machine learning model|time series forecast|forecast next year)\b", "predictive forecasting and future machine learning modeling"),
            (r"\b(clustering|k-means|persona segmentation)\b", "unsupervised clustering and advisor persona segmentation")
        ]
        for pattern, desc in unsupported_checks:
            if re.search(pattern, q_lower):
                return AssistantResponse(
                    answer=f"That analysis is not yet supported in this prototype. It would require an additional calculation and validation step for {desc}.",
                    topic="Unsupported Analysis (Situation B)",
                    data_source="None (Feature not implemented)",
                    limitations=f"Unsupported analytical capability: {desc}",
                    is_supported=False
                )
        return None

    # Situation E: Ambiguous Question
    def _is_ambiguous(self, q: str) -> bool:
        q_clean = q.lower().strip("?!. ")
        ambiguous_queries = [
            "performance", "tell me about performance", "how did they do",
            "what is the rate", "tell me about numbers", "overview", "results"
        ]
        return q_clean in ambiguous_queries

    # Deterministic Routing to Analytics Engine
    def _route_analytics(
        self,
        q: str,
        centre: Optional[str] = None,
        compare_with: Optional[str] = None
    ) -> Tuple[AnalyticsResult, str]:
        q_lower = q.lower()

        # 1. Centre specific or comparison
        if centre or compare_with:
            if "revenue" in q_lower and centre and not compare_with:
                res = self.analytics.get_centre_performance(centre=centre)
                return res, "Centre Revenue"
            res = self.analytics.get_centre_performance(centre=centre, compare_with=compare_with)
            return res, "Centre Performance"

        if any(w in q_lower for w in ["which centre", "best centre", "top centre", "highest revenue centre", "centres"]):
            return self.analytics.get_centre_performance(), "Centre Performance"

        # 2. Before vs After Academy
        if any(w in q_lower for w in ["before and after", "before/after", "after the academy", "after academy", "academy launched", "launch", "period"]):
            return self.analytics.get_period_comparison(), "Period Comparison"

        # 3. Training status comparison / trained_before_month
        if any(w in q_lower for w in ["trained advisor", "training status", "trained vs untrained", "trained versus untrained", "previously trained"]):
            return self.analytics.get_training_status_comparison(), "Training Status Comparison"

        if "trained_before_month" in q_lower:
            return self.analytics.get_methodology_explanation("trained_before_month"), "Column Purpose"

        # 4. Causality & simulation assumptions
        if any(w in q_lower for w in ["cause", "causality", "did the training cause", "prove"]):
            return self.analytics.get_methodology_explanation("causality"), "Causality & Limitations"

        # 5. Synthetic generation methodology & Python pipeline
        if any(w in q_lower for w in ["synthetic", "dataset generated", "data generated", "data was generated", "how was the dataset", "simulation", "simulate"]):
            return self.analytics.get_methodology_explanation("synthetic"), "Methodology"

        if any(w in q_lower for w in ["python", "connect", "pipeline", "prepare_data"]):
            return self.analytics.get_methodology_explanation("connect"), "Methodology"

        # 6. Learning modules & completion
        if any(w in q_lower for w in ["highest completion", "best module", "completion rate", "module completed", "completed modules", "percentage of modules"]):
            modules = ["customer empathy", "upselling insurance", "needs discovery", "compliance essentials", "objection handling"]
            for m in modules:
                if m in q_lower:
                    return self.analytics.get_module_completion_summary(module_name=m), "Module Completion"
            return self.analytics.get_module_completion_summary(), "Module Completion"

        # 7. Advisor module distribution
        if any(w in q_lower for w in ["how many advisors completed", "advisors completed their", "advisor completion", "completed all"]):
            return self.analytics.get_advisor_completion_distribution(), "Advisor Completion"

        # 8. Coaching attendance & sales
        if any(w in q_lower for w in ["coaching", "sessions attended", "session"]):
            return self.analytics.get_coaching_summary(), "Coaching Attendance"

        # 9. Monthly trends
        if any(w in q_lower for w in ["monthly", "month by month", "trend", "over time"]):
            return self.analytics.get_monthly_trends(), "Monthly Trends"

        # 10. Recommendations & limitations
        if any(w in q_lower for w in ["recommendation", "recommend", "advice", "next step"]):
            return self.analytics.get_business_recommendations(), "Recommendations"

        if any(w in q_lower for w in ["limitation", "weakness"]):
            return self.analytics.get_data_limitations(), "Limitations"

        # 11. Overall conversion & revenue
        if any(w in q_lower for w in ["conversion rate", "overall conversion", "revenue", "how much revenue", "total sales", "overall sales", "percentage"]):
            return self.analytics.get_overall_summary(), "Overall Summary"

        # Default fallback: Overall summary
        return self.analytics.get_overall_summary(), "Overall Summary"

    # Natural Language Synthesis Layer (Grounded in Verified Results)
    def _synthesize_answer(self, user_question: str, res: AnalyticsResult) -> str:
        if self.gemini_client:
            try:
                system_instruction = (
                    "You are the Northbrook Insights AI Assistant for Northbrook Travel's Sales Excellence Academy. "
                    "Your goal is to answer the user's question clearly, concisely, and professionally.\n\n"
                    "CRITICAL GROUNDING CONSTRAINTS:\n"
                    "1. Rely EXCLUSIVELY on the provided VERIFIED ANALYTICS RESULT. Do NOT calculate, extrapolate, or introduce new numbers.\n"
                    "2. If the user asks whether training caused sales improvements, clearly distinguish between the simulated assumption and true real-world causality.\n"
                    "3. If asked about a centre, comparison, or module, use the exact figures given.\n"
                    "4. Maintain a professional, business-friendly tone suitable for senior leadership."
                )

                prompt = (
                    f"User Question: {user_question}\n\n"
                    f"VERIFIED ANALYTICS RESULT:\n"
                    f"Topic: {res.metric_or_topic}\n"
                    f"Verified Summary: {res.formatted_summary}\n"
                    f"Data Source: {res.data_source}\n"
                    f"Limitations: {res.limitations}\n\n"
                    "Please craft a direct, business-ready answer based strictly on the above verified data."
                )

                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"system_instruction": system_instruction}
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"[Notice] Gemini generation error ({e}). Using verified deterministic summary.")

        return res.formatted_summary