"""
app.py - Northbrook Insights AI Assistant Streamlit Application

A lightweight, professional conversational analytics interface for
Northbrook Travel's Sales Excellence Academy evaluation.
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Ensure local module imports work
import sys
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analytics import NorthbrookAnalytics
from assistant import NorthbrookAssistant

# Load environment configuration
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Northbrook Insights AI Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished business aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .source-tag {
        font-size: 0.8rem;
        color: #64748B;
        font-style: italic;
    }
    .stChatMessage {
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "assistant" not in st.session_state:
    st.session_state.analytics = NorthbrookAnalytics()
    st.session_state.assistant = NorthbrookAssistant(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        analytics_engine=st.session_state.analytics
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Assistant Settings")
    st.markdown("Configure your AI model and connection parameters.")

    # API Key management
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Enter your Google Gemini API key. If left blank, the assistant runs in deterministic verified analytics mode."
    )

    model_options = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    selected_model = st.selectbox(
        "Model Selection",
        options=model_options,
        index=0,
        help="Choose the Gemini model for question interpretation and natural language wording."
    )

    if api_key_input != st.session_state.assistant.api_key or selected_model != st.session_state.assistant.model_name:
        st.session_state.assistant.update_credentials(api_key_input, selected_model)

    # Connection Status
    if st.session_state.assistant.gemini_client:
        st.success(f"🟢 Connected to Gemini ({selected_model})")
    else:
        st.info("🔵 Deterministic Analytics Mode (Verified Local Engine)")

    st.divider()

    # Dataset Status Card
    st.subheader("📊 Verified Datasets")
    st.markdown("""
    - **Advisors:** 400 records (Leeds, Glasgow, Cardiff)
    - **Monthly Sales:** 4,800 records (12 months)
    - **Learning Activity:** 2,000 module records (5 modules)
    - **Coaching Attendance:** 400 records (809 sessions)
    - **Prepared Dataset:** `sales_learning_analysis.csv`
    """)

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.assistant.context.reset()
        st.rerun()

    st.markdown("""
    ---
    **Thrive Learning Insights**  
    *Northbrook Travel Sales Excellence Academy*
    """)


# Header Section
st.markdown('<div class="main-header">Northbrook Insights AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about the Sales Excellence Academy and sales performance</div>', unsafe_allow_html=True)

# Mandatory Synthetic Data Notice Banner
st.markdown(
    '<div class="disclaimer-banner">ℹ️ Synthetic demonstration data. Training uplift is simulated and does not establish causality.</div>',
    unsafe_allow_html=True
)

# Optional Example Questions
st.markdown("**Quick questions to explore:**")
col1, col2, col3 = st.columns(3)

example_q = None
with col1:
    if st.button("📈 Overall Conversion & Revenue", use_container_width=True):
        example_q = "What was the overall sales conversion rate and revenue?"
with col2:
    if st.button("🎓 Before vs After Academy", use_container_width=True):
        example_q = "Did conversion improve after the academy launched?"
with col3:
    if st.button("🏆 Contact Centre Comparison", use_container_width=True):
        example_q = "Which centre achieved the highest revenue and conversion rate?"

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "data_source" in msg and msg["data_source"]:
            with st.expander("🔍 Verified Calculation & Data Source", expanded=False):
                st.markdown(f"**Data Source:** `{msg['data_source']}`")
                if msg.get("limitations"):
                    st.markdown(f"**Limitations & Context:** {msg['limitations']}")

# User Chat Input
user_input = st.chat_input("Ask a question about Northbrook sales, training, coaching, centres or methodology...")

question_to_process = example_q or user_input

if question_to_process:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": question_to_process})
    with st.chat_message("user"):
        st.markdown(question_to_process)

    # Process assistant response with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Analyzing verified project data..."):
            try:
                response = st.session_state.assistant.ask(question_to_process)
                st.markdown(response.answer)

                if response.data_source and response.data_source != "None":
                    with st.expander("🔍 Verified Calculation & Data Source", expanded=False):
                        st.markdown(f"**Topic:** {response.topic}")
                        st.markdown(f"**Data Source:** `{response.data_source}`")
                        if response.limitations:
                            st.markdown(f"**Limitations & Context:** {response.limitations}")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "data_source": response.data_source,
                    "limitations": response.limitations
                })
            except Exception as e:
                error_msg = "The AI assistant is temporarily unavailable. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "data_source": "Error",
                    "limitations": str(e)
                })