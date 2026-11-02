# Northbrook Insights AI Assistant

A lightweight, strictly data-grounded AI analytics assistant prototype built for **Northbrook Travel** as an optional extension to the Thrive Learning Insights technical project.

---

## 🎯 Purpose & Scope

The purpose of this assistant is to allow business users and leadership to ask natural-language questions about the **Sales Excellence Academy** and receive accurate, business-ready answers based strictly on verified project data and documented methodology.

> **Key Design Philosophy:**  
> The highest priority is **factual reliability**, not the appearance of intelligence.  
> The Large Language Model (Gemini) is used **exclusively for natural-language interpretation and wording synthesis**.  
> **All calculations, metric definitions, and aggregations are computed deterministically in Python using pandas.**  
> If an answer cannot be established from the dataset, the assistant explicitly states that rather than hallucinating.

---

## 🏗️ Architecture & Grounding Mechanism

```
  User Question
       │
       ▼
┌───────────────────────────────────────────────┐
│ 1. Assistant Engine (assistant.py)           │
│    - Resolves conversational context / follow-ups│
│    - Checks safety & prompt injection guards  │
│    - Verifies project scope (Situations A–F)  │
│    - Identifies requested metric & parameters │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ 2. Analytics Engine (analytics.py)            │
│    - Allowlist of validated pandas functions  │
│    - Computes weighted conversion & revenue   │
│    - Produces structured AnalyticsResult      │
└──────────────────────┬────────────────────────┘
                       │ Verified Structured Result
                       ▼
┌───────────────────────────────────────────────┐
│ 3. Grounded Synthesis Layer                   │
│    - With Gemini API: Models explain verified │
│      results in concise business English      │
│    - Offline / No API Key: Verified local     │
│      deterministic templates deliver 100%     │
│      reliable answers without external calls  │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ 4. Streamlit UI (app.py)                      │
│    - Conversational chat bubbles & history    │
│    - Expandable "Verified Source" badge       │
│    - Permanent Synthetic Data Notice Banner   │
└───────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
thrive-learning-insights/
├── data/
│   ├── advisors.csv
│   ├── learning_activity.csv
│   ├── coaching_attendance.csv
│   ├── sales_performance.csv
│   └── processed/
│       └── sales_learning_analysis.csv
├── scripts/
│   ├── generate_data.py
│   ├── prepare_data.py
│   └── analyse_data.py
├── DashBoard.pbix
└── insights_assistant/             <-- New Chatbot Prototype
    ├── app.py                      # Streamlit UI & conversation session state
    ├── analytics.py                # Deterministic calculation engine & data loader
    ├── assistant.py                # Context tracking, routing & grounded answer generation
    ├── requirements.txt            # Python dependencies
    ├── .env.example                # Environment configuration template
    ├── README.md                   # Installation & interview guide
    └── tests/
        ├── test_analytics.py       # Calculation & edge-case unit tests
        └── test_assistant.py       # Context, fallbacks & prompt injection tests
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (or the existing project virtual environment at `.venv`)

### 2. Environment Setup

Open PowerShell or Command Prompt in the project folder:

```powershell
cd C:\Users\Dnyaneshwari\Desktop\thrive-learning-insights
```

If using the existing virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

Install requirements:
```powershell
pip install -r insights_assistant\requirements.txt
```

### 3. API Key Configuration (Optional)

The prototype operates in two modes:
1. **Live Gemini Mode:** Set your Gemini API key to enable natural-language interpretation and synthesis.
2. **Deterministic Offline Mode:** If no key is provided, the assistant runs in local deterministic mode, ensuring reliable answers during offline demos.

To configure an API key:
1. Copy `.env.example` to `.env` inside `insights_assistant/`:
   ```powershell
   Copy-Item insights_assistant\.env.example insights_assistant\.env
   ```
2. Open `insights_assistant\.env` and enter your key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```
*(Alternatively, enter your key directly into the Streamlit sidebar password field at runtime.)*

---

## 🖥️ Running the Application

Launch the Streamlit app:

```powershell
streamlit run insights_assistant\app.py
```

The browser will open automatically at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

A comprehensive 18-test automated suite verifies calculations, follow-ups, and fallbacks:

```powershell
python -m unittest discover -s insights_assistant\tests -p "test_*.py" -v
```

All 18 tests pass with verified assertions matching the project's data files.

---

## 📊 Key Verified Project Metrics

| Metric | Verified Value | Calculation / Formula |
| :--- | :--- | :--- |
| **Total Revenue** | £206,201,878.00 | `sum(revenue)` |
| **Total Calls Taken** | 1,078,083 | `sum(calls_taken)` |
| **Total Sales Made** | 165,176 | `sum(sales_made)` |
| **Weighted Conversion Rate** | 15.32% | `sum(sales_made) / sum(calls_taken) * 100` |
| **Weighted Insurance Attach** | 40.07% | `sum(insurance_sales) / sum(sales_made) * 100` |
| **Before Academy Conversion** | 15.01% | Oct 2025 – Mar 2026 (541,572 calls, 81,279 sales) |
| **After Academy Conversion** | 15.64% | Apr 2026 – Sep 2026 (536,511 calls, 83,897 sales) |
| **Academy Period Uplift** | +0.63 pp | Difference in period weighted conversion rate |
| **Trained Status Conversion** | 16.00% | `trained_before_month == True` (353,694 calls, 56,575 sales) |
| **Untrained Status Conversion**| 14.99% | `trained_before_month == False` (724,389 calls, 108,601 sales) |
| **Training Status Uplift** | +1.00 pp | Matches simulated +1 pp binomial uplift |
| **Highest Revenue Centre** | Leeds (£70.98M) | Cardiff: £67.69M, Glasgow: £67.53M |
| **Highest Conversion Centre** | Leeds (15.37%) | Cardiff: 15.31%, Glasgow: 15.29% |
| **Overall Module Completion** | 69.35% | 1,387 completed of 2,000 assignments |
| **Top Completed Module** | Customer Empathy (70.75%) | 283 / 400 completions |
| **Coaching Attendance** | 809 sessions (avg 2.02) | Correlation with conversion: r = -0.087 |

---

## 🛡️ Fallback Behavior (Situations A–F)

| Situation | Trigger Example | Chatbot Response Policy |
| :--- | :--- | :--- |
| **Situation A (Data Missing)** | *"What was advisor turnover?"* | Clearly states data was not collected: *"I can't establish that from the current Northbrook dataset... Specifically, employee turnover was not tracked."* |
| **Situation B (Unsupported Analysis)**| *"Run a multivariate regression"* | Honest prototype boundary: *"That analysis is not yet supported in this prototype. It would require an additional calculation step..."* |
| **Situation C (Out of Scope)** | *"What is the weather in Paris?"* | Polite redirection: *"I'm designed to answer questions about the Northbrook Travel Sales Excellence Academy analysis..."* |
| **Situation D (API Error)** | LLM API down or network error | Friendly fallback message + diagnostic logs; automatic switch to verified local engine. |
| **Situation E (Ambiguous Query)** | *"Tell me about performance"* | Asks a single clarifying question without guessing. |
| **Situation F (Causality Claim)** | *"Did the training cause the sales rise?"*| Explains that the uplift was deliberately modeled in `prepare_data.py` (synthetic assumption) and observational data cannot prove real-world causality without an RCT. |

---

## 🎤 Interview Demonstration Plan (60 Seconds)

1. **Opening (15s):**  
   *"To complement my Power BI dashboard, I built a lightweight Python and Streamlit prototype called Northbrook Insights AI Assistant. Its purpose is to allow business users to ask conversational questions without the risk of LLM hallucination."*
2. **Live Interaction (30s):**  
   - Ask: *"Which centre performed best?"* -> Show answer (Leeds: £70.98M revenue, 15.37% conversion).  
   - Follow-up: *"What was its revenue?"* -> Highlight conversational pronoun resolution ("its" = Leeds).  
   - Follow-up: *"Did the training cause this improvement?"* -> Highlight the synthetic data caveat and causality explanation.
3. **Closing (15s):**  
   - Show the expandable **"Verified Calculation & Data Source"** badge.  
   - *"Notice that all calculations are performed deterministically in pandas. The LLM only formats the text, guaranteeing 100% factual reliability."*